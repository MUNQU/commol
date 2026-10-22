//! The result of every conformal variant: prediction intervals at target
//! points, and the scores they came from.

use std::collections::HashMap;

use commol_calibration::ObservedDataPoint;

use crate::conformal::score::Pooling;

/// Prediction intervals at a set of target points.
#[derive(Debug, Clone)]
pub struct PredictionIntervals {
    /// The level of the intervals, `1 − α`.
    pub confidence_level: f64,
    /// Where each interval is: a series (label) and a position, read as the
    /// series' observations define it.
    pub targets: Vec<ObservedDataPoint>,
    /// The point prediction at each target.
    pub prediction: Vec<f64>,
    /// The lower end of each interval; `−∞` when unbounded.
    pub lower: Vec<f64>,
    /// The upper end of each interval; `+∞` when unbounded.
    pub upper: Vec<f64>,
}

impl PredictionIntervals {
    /// The intervals `prediction ± q · scale` at `targets`, where `q` is the
    /// quantile of the target's group (`quantiles`, keyed by
    /// [`Pooling::group`]). A group with no quantile has unbounded
    /// intervals.
    pub fn symmetric(
        confidence_level: f64,
        targets: Vec<ObservedDataPoint>,
        prediction: Vec<f64>,
        scales: &[f64],
        quantile_of: impl Fn(usize) -> Option<f64>,
    ) -> Self {
        let (lower, upper) = prediction
            .iter()
            .zip(scales)
            .enumerate()
            .map(|(index, (&centre, &scale))| match quantile_of(index) {
                Some(q) if q.is_finite() => (centre - q * scale, centre + q * scale),
                _ => (f64::NEG_INFINITY, f64::INFINITY),
            })
            .unzip();
        Self {
            confidence_level,
            targets,
            prediction,
            lower,
            upper,
        }
    }
}

/// The quantile of each target's group, from the quantiles of
/// [`crate::conformal::group_quantiles`].
pub fn quantile_lookup<'a>(
    targets: &'a [ObservedDataPoint],
    quantiles: &'a HashMap<String, f64>,
    by: Pooling,
) -> impl Fn(usize) -> Option<f64> + 'a {
    move |index| quantiles.get(by.group(&targets[index].label)).copied()
}

/// The intervals of a conformal method and the scores they came from.
#[derive(Debug, Clone)]
pub struct ConformalRun {
    /// The intervals at the target points.
    pub intervals: PredictionIntervals,
    /// Each score, with the label of its series, in the order the method
    /// took them.
    pub scores: Vec<(String, f64)>,
    /// The index and reason of each refit that failed; a failed refit gives
    /// no score.
    pub failed: Vec<(usize, String)>,
    /// The number of refits kept although their optimizer did not converge.
    pub not_converged: usize,
}
