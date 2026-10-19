//! Point predictions and their spread at any observations, and the target
//! points of the intervals.

use commol_calibration::{CalibrationProblem, ObservedDataPoint};
use commol_core::SimulationEngine;

use crate::predictive::{NoNoise, PredictiveError, observation_draws, observe_sets};
use crate::sample::ParameterSample;
use crate::statistics;

/// What gives the point prediction of a variant that does not refit.
#[derive(Debug, Clone, Copy)]
pub enum Predictor<'a> {
    /// The run of one parameter vector, in problem order.
    Fit(&'a [f64]),
    /// The weighted median of the predictive of a sample; its weighted
    /// standard deviation is the spread.
    Sample(&'a ParameterSample),
}

/// The point prediction at each point, and its spread when the predictor
/// has one.
#[derive(Debug, Clone, PartialEq)]
pub struct Predictions {
    /// The point prediction.
    pub centre: Vec<f64>,
    /// The spread of the predictions, or None for a fit.
    pub spread: Option<Vec<f64>>,
}

/// The prediction of `predictor` at `points`.
///
/// # Errors
///
/// Returns an error when a point names an unknown output or factor, the
/// sample is not one of the problem's parameters, or a run fails.
pub fn predict<E>(
    problem: &CalibrationProblem<E>,
    predictor: Predictor<'_>,
    points: &[ObservedDataPoint],
) -> Result<Predictions, PredictiveError>
where
    E: SimulationEngine + Send + Sync,
{
    match predictor {
        Predictor::Fit(values) => {
            let fit = problem.applied_parameter_values(values);
            let centre = set_predictions(problem, &[fit], points)?.remove(0);
            Ok(Predictions {
                centre,
                spread: None,
            })
        }
        Predictor::Sample(sample) => {
            let draws = observation_draws(problem.forward(), sample, points, &NoNoise, 0)?;
            let mut centre = Vec::with_capacity(points.len());
            let mut spread = Vec::with_capacity(points.len());
            for i in 0..points.len() {
                let values: Vec<f64> = draws.values.iter().map(|row| row[i]).collect();
                centre.push(statistics::quantiles(&values, &draws.weights, &[0.5])[0]);
                spread.push(statistics::covariance(&values, &values, &draws.weights).sqrt());
            }
            Ok(Predictions {
                centre,
                spread: Some(spread),
            })
        }
    }
}

/// The prediction of each parameter set at `points`, `[set][point]`; with a
/// stochastic engine, the mean of the replicates.
///
/// # Errors
///
/// As [`predict`]; [`PredictiveError::Draw`] names the index of the set.
pub fn set_predictions<E>(
    problem: &CalibrationProblem<E>,
    sets: &[Vec<f64>],
    points: &[ObservedDataPoint],
) -> Result<Vec<Vec<f64>>, PredictiveError>
where
    E: SimulationEngine + Send + Sync,
{
    let runs = observe_sets(problem.forward(), sets, points, None)?;
    Ok(runs
        .into_iter()
        .map(|replicates| {
            let count = replicates.len() as f64;
            (0..points.len())
                .map(|i| replicates.iter().map(|row| row[i]).sum::<f64>() / count)
                .collect()
        })
        .collect())
}

/// The target points of the intervals: each series of `series`, in order
/// of first appearance, at each of `positions`. A target copies the first
/// observation of its series (its outputs, factor and window) with the
/// position changed.
pub fn targets(series: &[ObservedDataPoint], positions: &[f64]) -> Vec<ObservedDataPoint> {
    let mut templates: Vec<&ObservedDataPoint> = Vec::new();
    for obs in series {
        if !templates.iter().any(|t| t.label == obs.label) {
            templates.push(obs);
        }
    }
    templates
        .into_iter()
        .flat_map(|template| {
            positions.iter().map(move |&at| ObservedDataPoint {
                at,
                value: 0.0,
                weight: 1.0,
                ..template.clone()
            })
        })
        .collect()
}

/// The standard deviation, with equal weights, of each column of `rows`,
/// `[member][point]`: the spread of an ensemble of fits at each point.
pub fn ensemble_spread(rows: &[Vec<f64>]) -> Vec<f64> {
    let count = rows.len();
    let weights = vec![1.0 / count as f64; count];
    let points = rows.first().map_or(0, Vec::len);
    (0..points)
        .map(|i| {
            let values: Vec<f64> = rows.iter().map(|row| row[i]).collect();
            statistics::covariance(&values, &values, &weights).sqrt()
        })
        .collect()
}

/// The mean, with equal weights, of each column of `rows`, `[member][point]`.
pub fn ensemble_mean(rows: &[&Vec<f64>], points: usize) -> Vec<f64> {
    let count = rows.len() as f64;
    (0..points)
        .map(|i| rows.iter().map(|row| row[i]).sum::<f64>() / count)
        .collect()
}
