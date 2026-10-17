//! The predictive of a parameter sample: the run of every draw, its bands,
//! and every draw's value of a set of observations.
//!
//! A band of the model reduces the trajectories draw by draw with the
//! weights. With observation noise ([`ObservationNoise`]) the values of
//! observations are draws of what an observation would show, so their
//! interval is a prediction interval; without noise ([`NoNoise`]) it is a
//! band of the model. A stochastic engine runs every replicate of each draw;
//! each replicate is a draw of the predictive with an equal share of the
//! draw's weight.

use commol_calibration::{
    CalibrationProblem, EvaluationError, ObservationOperator, ObservedDataPoint,
};
use commol_core::SimulationEngine;
use commol_random::Generator;
use rayon::prelude::*;

use crate::sample::ParameterSample;
use crate::statistics;

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

/// Why a predictive could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictiveError {
    /// The sample is not a sample of this problem's parameters.
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

fn check_names<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    sample: &ParameterSample,
) -> Result<(), PredictiveError> {
    let names = problem.parameter_names();
    if names.as_slice() == sample.names() {
        Ok(())
    } else {
        Err(PredictiveError::Mismatch(format!(
            "The sample has the parameters {:?}; the problem has {:?}, in this order",
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
    /// Run every draw of `sample` at `positions`, in parallel.
    ///
    /// # Errors
    ///
    /// Returns an error when the sample's parameters are not the problem's,
    /// or a draw has no run.
    pub fn new<E>(
        problem: &CalibrationProblem<E>,
        sample: &ParameterSample,
        positions: &[f64],
    ) -> Result<Self, PredictiveError>
    where
        E: SimulationEngine + Send + Sync,
    {
        check_names(problem, sample)?;
        let runs = sample
            .values()
            .par_iter()
            .enumerate()
            .map(|(draw, values)| {
                problem
                    .forward()
                    .run_at(values, positions)
                    .map_err(|error| PredictiveError::Draw { draw, error })
            })
            .collect::<Result<Vec<_>, _>>()?;
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

    /// The positions of the trajectories.
    pub fn positions(&self) -> &[f64] {
        &self.positions
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

/// Every draw's value of `observations`, read through the observation
/// operator, with `noise` drawn from the generator seeded with `seed`.
///
/// The observations need not be the problem's: held-out data are read the
/// same way.
///
/// # Errors
///
/// Returns an error when the sample's parameters are not the problem's, an
/// observation names an unknown output or factor or is one `noise` cannot
/// serve, or a draw has no run.
pub fn observation_draws<E>(
    problem: &CalibrationProblem<E>,
    sample: &ParameterSample,
    observations: &[ObservedDataPoint],
    noise: &dyn ObservationNoise,
    seed: u64,
) -> Result<ObservationDraws, PredictiveError>
where
    E: SimulationEngine + Send + Sync,
{
    check_names(problem, sample)?;
    let mut schedule: Vec<f64> = observations
        .iter()
        .flat_map(|obs| std::iter::once(obs.at).chain(obs.window.map(|w| obs.at - w)))
        .collect();
    schedule.sort_by(f64::total_cmp);
    schedule.dedup();
    let forward = problem.forward();
    let operator = ObservationOperator::new(
        observations,
        forward.parameters(),
        &forward.base_engine().output_names(),
        &schedule,
    )
    .map_err(PredictiveError::Mismatch)?;
    noise
        .check(observations)
        .map_err(PredictiveError::Mismatch)?;
    let runs = sample
        .values()
        .par_iter()
        .enumerate()
        .map(|(draw, values)| {
            let replicates = forward
                .run_at(values, &schedule)
                .map_err(|error| PredictiveError::Draw { draw, error })?;
            Ok(replicates
                .iter()
                .map(|rows| {
                    let mut out = vec![0.0; operator.len()];
                    operator.apply(rows, values, &mut out);
                    out
                })
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>, PredictiveError>>()?;
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
