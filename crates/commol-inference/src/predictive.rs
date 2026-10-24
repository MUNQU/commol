//! The predictive of a parameter sample: the run of every draw, its bands,
//! and every draw's value of a set of observations.
//!
//! Every band has the layout of [`Band`] and comes from one reducer, which
//! reduces each column of values across the draws with their weights. With
//! observation noise ([`ObservationNoise`]) the values of observations are
//! draws of what an observation would show, so their interval is a
//! prediction interval; without noise ([`NoNoise`]) it is a band of the
//! model. [`DistributionNoise`] draws from the observation models of the
//! problem. A stochastic engine runs every replicate of each draw; each
//! replicate is a draw of the predictive with an equal share of the draw's
//! weight ([`share_among_replicates`]), and has a seed of its own
//! ([`run_sets`]): draw `d` takes a seed of its own from the seed of the
//! caller ([`draw_seeds`]), and its replicates are the replicates of that
//! seed, so the predictive draws fresh noise for every draw instead of the
//! same replicates for all.

use commol_calibration::likelihood::log_location;
use commol_calibration::{
    CalibrationProblem, EvaluationError, ForwardModel, Likelihood, Noise, ObservedDataPoint,
};
use commol_core::SimulationEngine;
use commol_random::sample::{binomial, gamma, normal, poisson};
use commol_random::{Generator, replicate_seed};
use rayon::prelude::*;

use crate::sample::ParameterSample;
use crate::statistics;
use commol_random::Stream;

/// Below this share of extra variance, `φμ`, a negative binomial draw is a
/// Poisson draw.
const POISSON_LIMIT: f64 = 1e-6;

/// What observations show for the values a draw predicts: the port through
/// which a method adds observation noise to the draws of a predictive.
///
/// The port sees one draw's whole vector of observations at once, so a
/// method can draw noise that is shared or dependent across observations,
/// such as blocks of residuals.
pub trait ObservationNoise: Sync {
    /// Reject observations this noise cannot serve. Called once, before
    /// any draw.
    ///
    /// # Errors
    ///
    /// Returns a message that names the first observation it cannot serve.
    fn check(&self, _observations: &[ObservedDataPoint]) -> Result<(), String> {
        Ok(())
    }

    /// Write into `out` what `observations` show when a draw with parameter
    /// `values` predicts `predicted`, one value per observation, drawn with
    /// `generator`.
    fn observe(
        &self,
        observations: &[ObservedDataPoint],
        values: &[f64],
        predicted: &[f64],
        generator: &mut Generator,
        out: &mut [f64],
    );
}

/// No observation noise: an observation shows the predicted value.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoNoise;

impl ObservationNoise for NoNoise {
    fn observe(
        &self,
        _observations: &[ObservedDataPoint],
        _values: &[f64],
        predicted: &[f64],
        _generator: &mut Generator,
        out: &mut [f64],
    ) {
        out.copy_from_slice(predicted);
    }
}

/// Observation noise of the observation models of a problem: each draw
/// draws every observation from the model of its series around the draw's
/// predicted mean, with the noise parameters of the draw's values.
///
/// A mean outside the means a model allows draws at the nearest allowed one
/// ([`Noise::nearest_mean`]).
#[derive(Debug, Clone)]
pub struct DistributionNoise {
    likelihood: Likelihood,
}

impl DistributionNoise {
    /// The noise of the observation models of `problem`.
    ///
    /// # Errors
    ///
    /// Returns a message when the problem has a loss instead of observation
    /// models.
    pub fn new<E: SimulationEngine>(problem: &CalibrationProblem<E>) -> Result<Self, String> {
        problem
            .discrepancy()
            .likelihood()
            .cloned()
            .map(|likelihood| Self { likelihood })
            .ok_or_else(|| {
                "DistributionNoise() needs observation models; the problem has a loss. Use \
                 ResidualNoise(...) for the residuals of a fit"
                    .to_string()
            })
    }

    /// The likelihood of the models.
    pub fn likelihood(&self) -> &Likelihood {
        &self.likelihood
    }
}

impl ObservationNoise for DistributionNoise {
    fn check(&self, observations: &[ObservedDataPoint]) -> Result<(), String> {
        self.likelihood.terms(observations).map(|_| ())
    }

    fn observe(
        &self,
        observations: &[ObservedDataPoint],
        values: &[f64],
        predicted: &[f64],
        generator: &mut Generator,
        out: &mut [f64],
    ) {
        let terms = self
            .likelihood
            .terms(observations)
            .expect("checked before the draws");
        for ((&term, &mean), shown) in terms.iter().zip(predicted).zip(out.iter_mut()) {
            *shown = draw_observation(self.likelihood.noise(term, values), mean, generator);
        }
    }
}

/// One observed value drawn from `noise` around `mean`, at the nearest mean
/// the model allows.
///
/// A normal and a log-normal scale a standard normal draw; a Student-t
/// divides a normal draw by the root of a chi-square draw over its degrees
/// of freedom; a negative binomial is a Poisson draw of a gamma mean, and a
/// Poisson draw when `φμ` is below `1e-6`.
pub(crate) fn draw_observation(noise: Noise, mean: f64, generator: &mut Generator) -> f64 {
    let mean = noise.nearest_mean(mean);
    match noise {
        Noise::Normal { sd } => mean + sd * normal(generator),
        Noise::LogNormal { sigma, center } => {
            libm::exp(log_location(mean, sigma, center) + sigma * normal(generator))
        }
        Noise::StudentT { df, scale } => {
            let z = normal(generator);
            let chi_square = gamma(generator, 0.5 * df, 0.5);
            mean + scale * z / (chi_square / df).sqrt()
        }
        Noise::Poisson => poisson(generator, mean) as f64,
        Noise::NegativeBinomial { overdispersion } => {
            if overdispersion * mean < POISSON_LIMIT {
                poisson(generator, mean) as f64
            } else {
                let shape = 1.0 / overdispersion;
                let rate = shape / mean;
                let rate_of_counts = gamma(generator, shape, rate);
                poisson(generator, rate_of_counts) as f64
            }
        }
        Noise::Binomial { trials } => {
            if trials == 0 {
                0.0
            } else {
                binomial(generator, trials, mean / trials as f64) as f64
            }
        }
    }
}

/// Why a predictive could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictiveError {
    /// An input does not fit the forward model, before any draw runs: the
    /// sample is not a sample of its parameters, a position is one its
    /// engine rejects, or an observation is one it cannot read.
    Mismatch(String),
    /// A draw has no run; `draw` is its index in the sample.
    Draw {
        /// Index of the draw in the sample.
        draw: usize,
        /// Why it has no run.
        error: EvaluationError,
    },
    /// A quantity is NaN at some points of a design, so a method that
    /// reduces it across the points has no value for it.
    NotANumber {
        /// Index of the quantity, in the order of the quantities of the
        /// method.
        quantity: usize,
        /// Index of each point where it is NaN, in the order of the design.
        points: Vec<usize>,
        /// The first run where it is NaN, when the method ran the points.
        run: Option<Box<NanRun>>,
    },
}

/// The first run where a quantity is NaN: what runs it again.
#[derive(Debug, Clone, PartialEq)]
pub struct NanRun {
    /// Index of the point, in the order of the design.
    pub draw: usize,
    /// Index of the replicate at the point, 0 for a deterministic engine;
    /// None when no replicate is NaN but the replicates of the point
    /// combine to no value, as a ratio can
    /// ([`crate::quantity::Undefined::Point`]).
    pub replicate: Option<usize>,
    /// The seed of the replicate, for a stochastic engine.
    pub seed: Option<u64>,
    /// The values of the inputs at the point.
    pub inputs: Vec<f64>,
}

impl std::fmt::Display for PredictiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mismatch(message) => f.write_str(message),
            Self::Draw { draw, error } => write!(f, "Draw {draw}: {error}"),
            Self::NotANumber {
                quantity,
                points,
                run,
            } => {
                let first = run.as_ref().map_or(String::new(), |run| {
                    let seed = run
                        .seed
                        .map_or(String::new(), |seed| format!(" of seed {seed}"));
                    let replicate = run.replicate.map_or(
                        " (no replicate is NaN, but a ratio has no value there: no replicate has \
                         a denominator above 0 under within_positive, or the mean denominator \
                         is 0 under pooled)"
                            .to_string(),
                        |replicate| format!(", replicate {replicate}{seed}"),
                    );
                    format!(
                        ", first at the point {} of the inputs {:?}{replicate}",
                        run.draw, run.inputs
                    )
                });
                write!(
                    f,
                    "The quantity {quantity} (counted from 0, in the order of the labels) is NaN \
                     at the points {points:?} of the design{first}, and a method that reduces it \
                     across the points needs a number at each; a Propagation of the same inputs \
                     and seed gives the value of each run"
                )
            }
        }
    }
}

pub(crate) fn check_names<E: SimulationEngine>(
    forward: &ForwardModel<E>,
    sample: &ParameterSample,
) -> Result<(), PredictiveError> {
    let names: Vec<&str> = forward.parameters().iter().map(|p| p.id.as_str()).collect();
    if sample
        .names()
        .iter()
        .map(String::as_str)
        .eq(names.iter().copied())
    {
        Ok(())
    } else {
        Err(PredictiveError::Mismatch(format!(
            "The sample has the parameters {:?}; the forward model has {:?}, in this order",
            sample.names(),
            names
        )))
    }
}

/// The run of every draw of a sample at a set of positions.
#[derive(Debug, Clone, PartialEq)]
pub struct Predictive {
    positions: Vec<f64>,
    /// `[draw][position][output]`; a replicate is a draw.
    trajectories: Vec<Vec<Vec<f64>>>,
    weights: Vec<f64>,
    /// The seed of each draw of the sample ([`draw_seeds`]).
    draw_seeds: Vec<u64>,
}

/// A band: four values at every position and output, `[position][output]`.
///
/// The band of a predictive and of a sensitivity analysis reduces draws
/// ([`Band::of`]); the band of the delta method
/// ([`crate::local::delta_band`]) has the same layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    /// The weighted mean, `[position][output]`.
    pub mean: Vec<Vec<f64>>,
    /// The weighted median.
    pub median: Vec<Vec<f64>>,
    /// The lower quantile, `(1 − c)/2`.
    pub lower: Vec<Vec<f64>>,
    /// The upper quantile, `(1 + c)/2`.
    pub upper: Vec<Vec<f64>>,
}

impl Band {
    /// The band of `confidence_level` of `values`, `[draw][position][output]`,
    /// with the weight of each draw: at every position and output, the
    /// weighted mean, median and central interval across the draws.
    ///
    /// # Errors
    ///
    /// Returns a message when `confidence_level` is not strictly between 0
    /// and 1, or a value is NaN.
    pub fn of(
        values: &[Vec<Vec<f64>>],
        weights: &[f64],
        confidence_level: f64,
    ) -> Result<Self, String> {
        let positions = values.first().map_or(0, Vec::len);
        let outputs = values
            .first()
            .and_then(|rows| rows.first())
            .map_or(0, Vec::len);
        Self::reduce(
            positions,
            outputs,
            weights,
            confidence_level,
            |draw, position, output| values[draw][position][output],
        )
    }

    /// The reducer of every band: at each of `positions` and `outputs`, the
    /// weighted mean, median and central interval of `value(draw, position,
    /// output)` across the draws of `weights`.
    fn reduce(
        positions: usize,
        outputs: usize,
        weights: &[f64],
        confidence_level: f64,
        value: impl Fn(usize, usize, usize) -> f64,
    ) -> Result<Self, String> {
        let [low, high] = statistics::interval_levels(confidence_level)?;
        let table = vec![vec![0.0; outputs]; positions];
        let mut band = Self {
            mean: table.clone(),
            median: table.clone(),
            lower: table.clone(),
            upper: table,
        };
        let mut column = vec![0.0; weights.len()];
        for position in 0..positions {
            for output in 0..outputs {
                for (draw, slot) in column.iter_mut().enumerate() {
                    *slot = value(draw, position, output);
                }
                let q = statistics::quantiles(&column, weights, &[low, 0.5, high])?;
                band.mean[position][output] = statistics::mean(&column, weights);
                band.lower[position][output] = q[0];
                band.median[position][output] = q[1];
                band.upper[position][output] = q[2];
            }
        }
        Ok(band)
    }
}

/// Every replicate of every draw of `runs`, `[draw][replicate]`, draw after
/// draw, and its weight: the weight of its draw shared equally among the
/// draw's replicates, `weight / replicates`.
pub fn share_among_replicates<T>(
    runs: impl IntoIterator<Item = Vec<T>>,
    weights: &[f64],
) -> (Vec<T>, Vec<f64>) {
    let mut items = Vec::new();
    let mut shares = Vec::new();
    for (replicates, &weight) in runs.into_iter().zip(weights) {
        let share = weight / replicates.len() as f64;
        shares.extend(std::iter::repeat_n(share, replicates.len()));
        items.extend(replicates);
    }
    (items, shares)
}

impl Predictive {
    /// Run every draw of `sample` at `positions` through `forward`, the
    /// forward model of a calibration problem or of an analysis, in
    /// parallel; the replicates of a stochastic engine are seeded from
    /// `seed` ([`run_sets`]).
    ///
    /// # Errors
    ///
    /// [`PredictiveError::Mismatch`], before any draw runs, when the
    /// sample's parameters are not the forward model's, or the positions are
    /// not sorted and unique or one is a position the engine rejects
    /// ([`commol_core::check_positions`]); [`PredictiveError::Draw`] when a
    /// draw has no run.
    pub fn new<E>(
        forward: &ForwardModel<E>,
        sample: &ParameterSample,
        positions: &[f64],
        seed: u64,
    ) -> Result<Self, PredictiveError>
    where
        E: SimulationEngine + Send + Sync,
    {
        check_names(forward, sample)?;
        let engine = forward.base_engine();
        commol_core::check_positions(positions, 0.0, |position| engine.check_position(position))
            .map_err(PredictiveError::Mismatch)?;
        let runs = run_sets(forward, sample.values(), positions, Some(seed))?;
        let (trajectories, weights) = share_among_replicates(runs, sample.weights());
        Ok(Self {
            positions: positions.to_vec(),
            trajectories,
            weights,
            draw_seeds: draw_seeds(seed, sample.values().len()),
        })
    }

    /// The trajectory of every draw, `[draw][position][output]`.
    pub fn trajectories(&self) -> &[Vec<Vec<f64>>] {
        &self.trajectories
    }

    /// The weight of every draw; they sum to 1.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The seed of each draw of the sample, in its order ([`draw_seeds`]):
    /// with a stochastic engine, replicate `i` of draw `j` runs with
    /// `replicate_seed(draw_seeds[j], i)`.
    pub fn draw_seeds(&self) -> &[u64] {
        &self.draw_seeds
    }

    /// The band of `confidence_level`: at every position and output, the
    /// weighted mean, median and central interval across the draws.
    ///
    /// # Errors
    ///
    /// Returns a message when `confidence_level` is not strictly between 0
    /// and 1, or a value is NaN.
    pub fn band(&self, confidence_level: f64) -> Result<Band, String> {
        Band::of(&self.trajectories, &self.weights, confidence_level)
    }
}

/// The band of one series of a predictive: four values at every position,
/// the one output of a [`Band`].
#[derive(Debug, Clone, PartialEq)]
pub struct SeriesBand {
    /// The weighted mean.
    pub mean: Vec<f64>,
    /// The weighted median.
    pub median: Vec<f64>,
    /// The lower quantile, `(1 − c)/2`.
    pub lower: Vec<f64>,
    /// The upper quantile, `(1 + c)/2`.
    pub upper: Vec<f64>,
}

impl SeriesBand {
    /// The band of the one output of `band`.
    fn of_one_output(band: Band) -> Self {
        let column = |table: Vec<Vec<f64>>| table.into_iter().map(|row| row[0]).collect();
        Self {
            mean: column(band.mean),
            median: column(band.median),
            lower: column(band.lower),
            upper: column(band.upper),
        }
    }
}

impl Predictive {
    /// The band of the sum of `outputs`, or of its running total along the
    /// positions when `cumulative`: each draw is reduced to the series first,
    /// then the series are reduced across draws with the weights.
    ///
    /// # Errors
    ///
    /// Returns a message when `confidence_level` is not strictly between 0
    /// and 1, an output index is out of range, or a value is NaN.
    pub fn series_band(
        &self,
        outputs: &[usize],
        cumulative: bool,
        confidence_level: f64,
    ) -> Result<SeriesBand, String> {
        let width = self
            .trajectories
            .first()
            .and_then(|rows| rows.first())
            .map_or(0, Vec::len);
        if let Some(&bad) = outputs.iter().find(|&&i| i >= width) {
            return Err(format!(
                "Output {bad} is out of range; a row has {width} outputs"
            ));
        }
        let series: Vec<Vec<f64>> = self
            .trajectories
            .iter()
            .map(|rows| {
                let mut total = 0.0;
                rows.iter()
                    .map(|row| {
                        let value: f64 = outputs.iter().map(|&i| row[i]).sum();
                        if cumulative {
                            total += value;
                            total
                        } else {
                            value
                        }
                    })
                    .collect()
            })
            .collect();
        let band = Band::reduce(
            self.positions.len(),
            1,
            &self.weights,
            confidence_level,
            |draw, position, _| series[draw][position],
        )?;
        Ok(SeriesBand::of_one_output(band))
    }
}

/// Every draw's value of a set of observations, and the seed they were
/// drawn with.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationDraws {
    /// `[draw][observation]`; a replicate is a draw.
    pub values: Vec<Vec<f64>>,
    /// The weight of every draw; they sum to 1.
    pub weights: Vec<f64>,
    /// The seed of the noise and of the replicates of the draws; a score
    /// that draws random numbers, such as [`crate::scores::pit`], takes a
    /// stream of it.
    pub seed: u64,
}

impl ObservationDraws {
    /// The band of each observation across the draws, as a [`SeriesBand`]
    /// with one entry per observation.
    ///
    /// # Errors
    ///
    /// Returns a message when `confidence_level` is not strictly between 0
    /// and 1, or a value is NaN.
    pub fn band(&self, confidence_level: f64) -> Result<SeriesBand, String> {
        let n = self.values.first().map_or(0, Vec::len);
        let band = Band::reduce(n, 1, &self.weights, confidence_level, |draw, i, _| {
            self.values[draw][i]
        })?;
        Ok(SeriesBand::of_one_output(band))
    }
}

/// Every draw's value of `observations`, run through `forward` and read
/// through the observation operator, with `noise` drawn from the stream
/// [`Stream::ObservationNoise`] of `seed`; the replicates of a stochastic engine are seeded from
/// `seed` as those of [`Predictive::new`]. The draws keep `seed`.
///
/// The observations need not be the problem's: held-out data are read the
/// same way, and a forward model with no observations reads them too.
///
/// # Errors
///
/// Returns an error when the sample's parameters are not the forward
/// model's, an observation is at a position the engine rejects, names an
/// unknown output or factor or is one `noise` cannot serve, or a draw has no
/// run.
pub fn observation_draws<E>(
    forward: &ForwardModel<E>,
    sample: &ParameterSample,
    observations: &[ObservedDataPoint],
    noise: &dyn ObservationNoise,
    seed: u64,
) -> Result<ObservationDraws, PredictiveError>
where
    E: SimulationEngine + Send + Sync,
{
    check_names(forward, sample)?;
    noise
        .check(observations)
        .map_err(PredictiveError::Mismatch)?;
    let runs = observe_sets(forward, sample.values(), observations, Some(seed))?;
    let mut generator = Generator::from_seed(Stream::ObservationNoise.seed(seed));
    let shown: Vec<Vec<Vec<f64>>> = runs
        .into_iter()
        .zip(sample.values())
        .map(|(replicates, values)| {
            replicates
                .into_iter()
                .map(|predicted| {
                    let mut shown = vec![0.0; predicted.len()];
                    noise.observe(observations, values, &predicted, &mut generator, &mut shown);
                    shown
                })
                .collect()
        })
        .collect();
    let (values, weights) = share_among_replicates(shown, sample.weights());
    Ok(ObservationDraws {
        values,
        weights,
        seed,
    })
}

/// The values of `observations` for each parameter set of `sets`, read
/// through the forward model observed at them
/// ([`ForwardModel::with_observations`]), in parallel:
/// `[set][replicate][observation]`. `seed` seeds the replicates as
/// [`run_sets`] does.
///
/// # Errors
///
/// Returns an error when the engine rejects a position of an observation,
/// an observation names an unknown output or factor, or a set has no run;
/// [`PredictiveError::Draw`] names the set's index.
pub(crate) fn observe_sets<E>(
    forward: &ForwardModel<E>,
    sets: &[Vec<f64>],
    observations: &[ObservedDataPoint],
    seed: Option<u64>,
) -> Result<Vec<Vec<Vec<f64>>>, PredictiveError>
where
    E: SimulationEngine + Send + Sync,
{
    let observed = forward
        .with_observations(observations)
        .map_err(PredictiveError::Mismatch)?;
    let operator = observed.observation();
    Ok(run_sets(&observed, sets, observed.schedule(), seed)?
        .into_iter()
        .zip(sets)
        .map(|(replicates, values)| {
            replicates
                .iter()
                .map(|rows| {
                    let mut out = vec![0.0; operator.count()];
                    operator.apply(rows, values, &mut out);
                    out
                })
                .collect()
        })
        .collect())
}

/// The seed of each of `count` draws of a sample run from `seed`: draw `j`
/// takes `Stream::PredictiveDraws.replicate(seed, j)`, which depends on
/// neither `count` nor the number of replicates.
pub fn draw_seeds(seed: u64, count: usize) -> Vec<u64> {
    (0..count as u64)
        .map(|draw| Stream::PredictiveDraws.replicate(seed, draw))
        .collect()
}

/// The runs of each parameter set of `sets` at `positions`, in parallel,
/// `[set][replicate][position][output]`.
///
/// With `seed`, the runs of a stochastic engine follow one tree of seeds:
/// set `d` takes the seed `s_d` of [`draw_seeds`], and its replicate `r`
/// runs with `replicate_seed(s_d, r)`, the seed replicate `r` of
/// `Simulation.replicates` takes from `s_d`. So every set draws fresh
/// noise, the runs repeat from `seed`, and the runs of a set depend on
/// neither the number of sets nor the number of replicates. Without,
/// every set runs the replicates of the forward model, the same seeds for
/// all (common random numbers), as a method that compares sets needs. A
/// deterministic engine runs once either way.
///
/// # Errors
///
/// [`PredictiveError::Draw`] with the index of the first set that has no
/// run.
pub fn run_sets<E>(
    forward: &ForwardModel<E>,
    sets: &[Vec<f64>],
    positions: &[f64],
    seed: Option<u64>,
) -> Result<Vec<Vec<Vec<Vec<f64>>>>, PredictiveError>
where
    E: SimulationEngine + Send + Sync,
{
    let replicates = forward.replicates().map(|r| r.seeds.len());
    let seeds = seed.map(|seed| draw_seeds(seed, sets.len()));
    sets.par_iter()
        .enumerate()
        .map(|(draw, values)| {
            let runs = match (replicates, &seeds) {
                (Some(count), Some(seeds)) => (0..count as u64)
                    .map(|r| {
                        let seed = replicate_seed(seeds[draw], r);
                        forward.run_once_at(values, positions, Some(seed))
                    })
                    .collect::<Result<Vec<_>, _>>(),
                _ => forward.run_at(values, positions),
            };
            runs.map_err(|error| PredictiveError::Draw { draw, error })
        })
        .collect()
}
