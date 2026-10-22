//! Split conformal: score held-out observations against one predictor.

use commol_calibration::{CalibrationProblem, ObservedDataPoint};
use commol_core::SimulationEngine;

use crate::conformal::intervals::{ConformalRun, PredictionIntervals, quantile_lookup};
use crate::conformal::predict::{Predictor, predict, targets};
use crate::conformal::quantile::group_quantiles;
use crate::conformal::score::{ConformalSettings, scales};
use crate::error::MethodError;

/// Split conformal intervals of `predictor` at `positions`, for each series
/// of `calibration`: the scores of the held-out `calibration` observations
/// give the quantile of each group.
///
/// When the scores of the held-out observations and of a new one are
/// exchangeable, the interval of the new one covers it with probability at
/// least `settings.confidence_level`.
///
/// # Errors
///
/// Returns an error when `calibration` is empty, a point cannot be
/// predicted, or the score has no scale at a point.
pub fn split_conformal<E>(
    problem: &CalibrationProblem<E>,
    predictor: Predictor<'_>,
    calibration: &[ObservedDataPoint],
    positions: &[f64],
    settings: ConformalSettings,
) -> Result<ConformalRun, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    if calibration.is_empty() {
        return Err(MethodError::Invalid(
            "calibration_data is empty; split conformal scores held-out observations".to_string(),
        ));
    }
    let targets = targets(calibration, positions);
    let points: Vec<ObservedDataPoint> = calibration.iter().chain(&targets).cloned().collect();
    let predictions = predict(problem, predictor, &points)?;
    let scales = scales(
        settings.score,
        problem.observed_data(),
        &points,
        predictions.spread.as_deref(),
    )?;
    let held_out = calibration.len();
    let labels: Vec<String> = calibration.iter().map(|obs| obs.label.clone()).collect();
    let scores: Vec<f64> = calibration
        .iter()
        .enumerate()
        .map(|(i, obs)| {
            settings
                .score
                .of(obs.value, predictions.centre[i], scales[i])
        })
        .collect();
    let quantiles = group_quantiles(&labels, &scores, settings.by, settings.confidence_level);
    let intervals = PredictionIntervals::symmetric(
        settings.confidence_level,
        targets.clone(),
        predictions.centre[held_out..].to_vec(),
        &scales[held_out..],
        quantile_lookup(&targets, &quantiles, settings.by),
    );
    Ok(ConformalRun {
        intervals,
        scores: labels.into_iter().zip(scores).collect(),
        failed: Vec::new(),
        not_converged: 0,
    })
}
