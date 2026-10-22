//! Adaptive conformal inference (Gibbs and Candès 2021): a working level
//! that moves with each miss, so the long-run share of misses tends to `α`
//! with no assumption on the data.

use std::collections::HashMap;

use commol_calibration::{CalibrationProblem, ObservedDataPoint};
use commol_core::SimulationEngine;

use crate::conformal::intervals::{ConformalRun, PredictionIntervals};
use crate::conformal::predict::{Predictor, predict};
use crate::conformal::quantile::conformal_quantile;
use crate::conformal::score::{ConformalSettings, scales};
use crate::error::MethodError;

/// The working level of adaptive conformal inference: after each interval,
/// `α_{t+1} = α_t + γ (α − err_t)`, where `err_t` is 1 for a miss and 0
/// for a cover.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveLevel {
    target: f64,
    step: f64,
    alpha: f64,
}

impl AdaptiveLevel {
    /// A working level that starts at, and aims for,
    /// `α = 1 − confidence_level`, with the step `γ = step`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when `confidence_level` is not strictly
    /// between 0 and 1 ([`crate::statistics::interval_levels`]) or `step` is
    /// not positive and finite.
    pub fn new(confidence_level: f64, step: f64) -> Result<Self, MethodError> {
        crate::statistics::interval_levels(confidence_level).map_err(MethodError::Invalid)?;
        if !(step > 0.0 && step.is_finite()) {
            return Err(MethodError::Invalid(format!(
                "step is {step}; it must be positive"
            )));
        }
        Ok(Self {
            target: 1.0 - confidence_level,
            step,
            alpha: 1.0 - confidence_level,
        })
    }

    /// The working `α_t`; it can leave `[0, 1]`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The level of the next interval, `1 − α_t`.
    pub fn level(&self) -> f64 {
        1.0 - self.alpha
    }

    /// Move the working level after an interval that covered or missed.
    pub fn update(&mut self, covered: bool) {
        let error = if covered { 0.0 } else { 1.0 };
        self.alpha += self.step * (self.target - error);
    }
}

/// Adaptive conformal intervals of `predictor` at each point of `stream`,
/// in order of position.
///
/// The interval of a point is the conformal quantile, at the working level
/// of its group, of the scores of the earlier points of its group; then the
/// point is scored and the level moves. With [`crate::conformal::Pooling::Series`]
/// each series has its own working level and scores; with
/// [`crate::conformal::Pooling::All`] they are shared.
///
/// # Errors
///
/// Returns an error when `step` is not positive, a point cannot be
/// predicted, or the score has no scale at a point.
pub fn adaptive_conformal<E>(
    problem: &CalibrationProblem<E>,
    predictor: Predictor<'_>,
    stream: &[ObservedDataPoint],
    step: f64,
    settings: ConformalSettings,
) -> Result<ConformalRun, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    AdaptiveLevel::new(settings.confidence_level, step)?;
    let mut order: Vec<usize> = (0..stream.len()).collect();
    order.sort_by(|&a, &b| stream[a].at.total_cmp(&stream[b].at));
    let predictions = predict(problem, predictor, stream)?;
    let scales = scales(
        settings.score,
        problem.observed_data(),
        stream,
        predictions.spread.as_deref(),
    )?;
    let mut groups: HashMap<&str, (AdaptiveLevel, Vec<f64>)> = HashMap::new();
    let mut intervals = PredictionIntervals {
        confidence_level: settings.confidence_level,
        targets: Vec::with_capacity(stream.len()),
        prediction: Vec::with_capacity(stream.len()),
        lower: Vec::with_capacity(stream.len()),
        upper: Vec::with_capacity(stream.len()),
    };
    let mut scores = Vec::with_capacity(stream.len());
    for i in order {
        let point = &stream[i];
        let (level, seen) = groups
            .entry(settings.by.group(&point.label))
            .or_insert_with(|| {
                let level =
                    AdaptiveLevel::new(settings.confidence_level, step).expect("checked above");
                (level, Vec::new())
            });
        let centre = predictions.centre[i];
        let q = conformal_quantile(seen, level.level());
        let (lower, upper) = (centre - q * scales[i], centre + q * scales[i]);
        level.update(lower <= point.value && point.value <= upper);
        let score = settings.score.of(point.value, centre, scales[i]);
        seen.push(score);
        scores.push((point.label.clone(), score));
        intervals.targets.push(point.clone());
        intervals.prediction.push(centre);
        intervals.lower.push(lower);
        intervals.upper.push(upper);
    }
    Ok(ConformalRun {
        intervals,
        scores,
        failed: Vec::new(),
        not_converged: 0,
    })
}
