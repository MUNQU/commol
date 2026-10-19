//! The predictive of a parameter sample: the run of every draw, its bands,
//! and every draw's value of a set of observations.
//!
//! A band of the model reduces the trajectories draw by draw with the
//! weights. With observation noise ([`ObservationNoise`]) the values of
//! observations are draws of what an observation would show, so their
//! interval is a prediction interval; without noise ([`NoNoise`]) it is a
//! band of the model. [`DistributionNoise`] draws from the observation
//! models of the problem. A stochastic engine runs every replicate of each draw;
//! each replicate is a draw of the predictive with an equal share of the
//! draw's weight, and has a seed of its own ([`run_sets`]): replicate `r` of
//! draw `d` is seeded from the seed of the caller, so the predictive draws
//! fresh noise for every draw instead of the same replicates for all.

use commol_calibration::likelihood::log_location;
use commol_calibration::{
    CalibrationProblem, EvaluationError, ForwardModel, Likelihood, Noise, ObservationOperator,
    ObservedDataPoint,
};
use commol_core::SimulationEngine;
use commol_random::sample::{binomial, poisson};
use commol_random::{Generator, replicate_seed, stream_seed};
use rayon::prelude::*;

use crate::distribution::{Distribution, Family, open_uniform};
use crate::sample::ParameterSample;
use crate::special::normal_quantile;
use crate::statistics;

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
/// Continuous models invert their distribution function at a uniform number
/// strictly between 0 and 1; a Student-t divides a normal draw by the root
/// of a chi-square draw over its degrees of freedom; a negative binomial is
/// a Poisson draw of a gamma mean, and a Poisson draw when `φμ` is below
/// `1e-6`.
pub(crate) fn draw_observation(noise: Noise, mean: f64, generator: &mut Generator) -> f64 {
    let mean = noise.nearest_mean(mean);
    match noise {
        Noise::Normal { sd } => mean + sd * normal_quantile(open_uniform(generator)),
        Noise::LogNormal { sigma, center } => libm::exp(
            log_location(mean, sigma, center) + sigma * normal_quantile(open_uniform(generator)),
        ),
        Noise::StudentT { df, scale } => {
            let z = normal_quantile(open_uniform(generator));
            let chi_square = gamma(0.5 * df, 0.5, generator);
            mean + scale * z / (chi_square / df).sqrt()
        }
        Noise::Poisson => poisson(generator, mean) as f64,
        Noise::NegativeBinomial { overdispersion } => {
            if overdispersion * mean < POISSON_LIMIT {
                poisson(generator, mean) as f64
            } else {
                let shape = 1.0 / overdispersion;
                let rate = shape / mean;
                let rate_of_counts = gamma(shape, rate, generator);
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

/// A gamma draw of `shape` and `rate`, by inversion.
fn gamma(shape: f64, rate: f64, generator: &mut Generator) -> f64 {
    Distribution::new(Family::Gamma { shape, rate }, 0.0, f64::INFINITY)
        .expect("a gamma of positive shape and rate has mass above 0")
        .sample(generator)
}

/// Why a predictive could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictiveError {
    /// The sample is not a sample of the forward model's parameters.
    Mismatch(String),
    /// A draw has no run; `draw` is its index in the sample.
    Draw {
        /// Index of the draw in the sample.
        draw: usize,
        /// Why it has no run.
        error: EvaluationError,
    },
}

impl std::fmt::Display for PredictiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mismatch(message) => f.write_str(message),
            Self::Draw { draw, error } => write!(f, "Draw {draw}: {error}"),
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
}

/// A band of a predictive: four values at every position and output.
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

impl Predictive {
    /// Run every draw of `sample` at `positions` through `forward`, the
    /// forward model of a calibration problem or of an analysis, in
    /// parallel; the replicates of a stochastic engine are seeded from
    /// `seed` ([`run_sets`]).
    ///
    /// # Errors
    ///
    /// Returns an error when the sample's parameters are not the forward
    /// model's, or a draw has no run.
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
        let runs = run_sets(forward, sample.values(), positions, Some(seed))?;
        let mut trajectories = Vec::new();
        let mut weights = Vec::new();
        for (replicates, &weight) in runs.into_iter().zip(sample.weights()) {
            let share = weight / replicates.len() as f64;
            for rows in replicates {
                trajectories.push(rows);
                weights.push(share);
            }
        }
        Ok(Self {
            positions: positions.to_vec(),
            trajectories,
            weights,
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

    /// The band of `confidence_level`: at every position and output, the
    /// weighted mean, median and central interval across the draws.
    ///
    /// # Errors
    ///
    /// Returns a message when `confidence_level` is not strictly between 0
    /// and 1.
    pub fn band(&self, confidence_level: f64) -> Result<Band, String> {
        let [low, high] = statistics::interval_levels(confidence_level)?;
        let outputs = self
            .trajectories
            .first()
            .and_then(|rows| rows.first())
            .map_or(0, Vec::len);
        let mut band = Band {
            mean: vec![vec![0.0; outputs]; self.positions.len()],
            median: vec![vec![0.0; outputs]; self.positions.len()],
            lower: vec![vec![0.0; outputs]; self.positions.len()],
            upper: vec![vec![0.0; outputs]; self.positions.len()],
        };
        let mut values = vec![0.0; self.trajectories.len()];
        for position in 0..self.positions.len() {
            for output in 0..outputs {
                for (slot, rows) in values.iter_mut().zip(&self.trajectories) {
                    *slot = rows[position][output];
                }
                let q = statistics::quantiles(&values, &self.weights, &[low, 0.5, high]);
                band.mean[position][output] = statistics::mean(&values, &self.weights);
                band.lower[position][output] = q[0];
                band.median[position][output] = q[1];
                band.upper[position][output] = q[2];
            }
        }
        Ok(band)
    }
}

/// The band of one series of a predictive: four values at every position.
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

impl Predictive {
    /// The band of the sum of `outputs`, or of its running total along the
    /// positions when `cumulative`: each draw is reduced to the series first,
    /// then the series are reduced across draws with the weights.
    ///
    /// # Errors
    ///
    /// Returns a message when `confidence_level` is not strictly between 0
    /// and 1, or an output index is out of range.
    pub fn series_band(
        &self,
        outputs: &[usize],
        cumulative: bool,
        confidence_level: f64,
    ) -> Result<SeriesBand, String> {
        let [low, high] = statistics::interval_levels(confidence_level)?;
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
        let mut band = SeriesBand {
            mean: Vec::new(),
            median: Vec::new(),
            lower: Vec::new(),
            upper: Vec::new(),
        };
        for position in 0..self.positions.len() {
            let values: Vec<f64> = series.iter().map(|s| s[position]).collect();
            let q = statistics::quantiles(&values, &self.weights, &[low, 0.5, high]);
            band.mean.push(statistics::mean(&values, &self.weights));
            band.lower.push(q[0]);
            band.median.push(q[1]);
            band.upper.push(q[2]);
        }
        Ok(band)
    }
}

/// Every draw's value of a set of observations.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationDraws {
    /// `[draw][observation]`; a replicate is a draw.
    pub values: Vec<Vec<f64>>,
    /// The weight of every draw; they sum to 1.
    pub weights: Vec<f64>,
}

impl ObservationDraws {
    /// The band of each observation across the draws, as a [`SeriesBand`]
    /// with one entry per observation.
    ///
    /// # Errors
    ///
    /// Returns a message when `confidence_level` is not strictly between 0
    /// and 1.
    pub fn band(&self, confidence_level: f64) -> Result<SeriesBand, String> {
        let [low, high] = statistics::interval_levels(confidence_level)?;
        let n = self.values.first().map_or(0, Vec::len);
        let mut band = SeriesBand {
            mean: Vec::new(),
            median: Vec::new(),
            lower: Vec::new(),
            upper: Vec::new(),
        };
        for i in 0..n {
            let values: Vec<f64> = self.values.iter().map(|row| row[i]).collect();
            let q = statistics::quantiles(&values, &self.weights, &[low, 0.5, high]);
            band.mean.push(statistics::mean(&values, &self.weights));
            band.lower.push(q[0]);
            band.median.push(q[1]);
            band.upper.push(q[2]);
        }
        Ok(band)
    }
}

/// Every draw's value of `observations`, run through `forward` and read
/// through the observation operator, with `noise` drawn from the generator
/// seeded with `seed`; the replicates of a stochastic engine are seeded from
/// `seed` as those of [`Predictive::new`].
///
/// The observations need not be the problem's: held-out data are read the
/// same way, and a forward model with no observations reads them too.
///
/// # Errors
///
/// Returns an error when the sample's parameters are not the forward
/// model's, an
/// observation names an unknown output or factor or is one `noise` cannot
/// serve, or a draw has no run.
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
    let mut generator = Generator::from_seed(seed);
    let mut draws = ObservationDraws {
        values: Vec::new(),
        weights: Vec::new(),
    };
    for ((replicates, &weight), values) in
        runs.into_iter().zip(sample.weights()).zip(sample.values())
    {
        let share = weight / replicates.len() as f64;
        for predicted in replicates {
            let mut shown = vec![0.0; predicted.len()];
            noise.observe(observations, values, &predicted, &mut generator, &mut shown);
            draws.values.push(shown);
            draws.weights.push(share);
        }
    }
    Ok(draws)
}

/// The values of `observations` for each parameter set of `sets`, read
/// through the observation operator, in parallel: `[set][replicate][observation]`.
/// `seed` seeds the replicates as [`run_sets`] does.
///
/// # Errors
///
/// Returns an error when an observation names an unknown output or factor,
/// or a set has no run; [`PredictiveError::Draw`] names the set's index.
pub(crate) fn observe_sets<E>(
    forward: &ForwardModel<E>,
    sets: &[Vec<f64>],
    observations: &[ObservedDataPoint],
    seed: Option<u64>,
) -> Result<Vec<Vec<Vec<f64>>>, PredictiveError>
where
    E: SimulationEngine + Send + Sync,
{
    let mut schedule: Vec<f64> = observations
        .iter()
        .flat_map(|obs| std::iter::once(obs.at).chain(obs.window.map(|w| obs.at - w)))
        .collect();
    schedule.sort_by(f64::total_cmp);
    schedule.dedup();
    let operator = ObservationOperator::new(
        observations,
        forward.parameters(),
        &forward.base_engine().output_names(),
        &schedule,
    )
    .map_err(PredictiveError::Mismatch)?;
    Ok(run_sets(forward, sets, &schedule, seed)?
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

/// The stream of the seeds of the replicates of a set of draws
/// ([`commol_random::stream_seed`]); the streams 1 to 3 belong to the
/// designs, the replicates and the bootstrap of a sensitivity analysis.
pub const DRAWS_STREAM: u64 = 4;

/// The runs of each parameter set of `sets` at `positions`, in parallel,
/// `[set][replicate][position][output]`.
///
/// With `seed`, replicate `r` of set `d` of a stochastic engine runs with
/// its own seed, the replicate `d · R + r` of the stream of draws of `seed`,
/// so every set draws fresh noise and the runs repeat from `seed`. Without,
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
    let base = seed.map(|seed| stream_seed(seed, DRAWS_STREAM));
    sets.par_iter()
        .enumerate()
        .map(|(draw, values)| {
            let runs = match (replicates, base) {
                (Some(count), Some(base)) => (0..count)
                    .map(|r| {
                        let index = (draw * count + r) as u64;
                        forward.run_once_at(values, positions, Some(replicate_seed(base, index)))
                    })
                    .collect::<Result<Vec<_>, _>>(),
                _ => forward.run_at(values, positions),
            };
            runs.map_err(|error| PredictiveError::Draw { draw, error })
        })
        .collect()
}
