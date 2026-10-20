//! The delta method: the band of every output that a local approximation
//! gives by linearizing the model at the fit, with no run of its draws.
//!
//! With `J_u = J_θ · D` the Jacobian of an output at a position on the
//! unbounded scale, its variance is `J_u Σ J_uᵀ`; the band is the prediction
//! at the fit plus and minus the quantile of a normal, or of a Student t
//! with degrees of freedom, times its standard deviation.

use commol_calibration::{CalibrationProblem, ObservedDataPoint};
use commol_core::SimulationEngine;

use super::LocalError;
use super::curvature::sensitivity_error;
use crate::sensitivity::{FiniteDifferences, Matrix, SensitivityError, SensitivitySource};
use crate::special::{normal_quantile, student_t_quantile};
use crate::statistics::interval_levels;
use crate::transform::Transform;

/// The linearized band of every output at some positions.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaBand {
    /// The output names, one per row of each table.
    pub outputs: Vec<String>,
    /// The prediction at the fit: one row per output, one value per
    /// position.
    pub center: Vec<Vec<f64>>,
    /// The lower end of the central interval, as `center`.
    pub lower: Vec<Vec<f64>>,
    /// The upper end of the central interval, as `center`.
    pub upper: Vec<Vec<f64>>,
}

/// The band of `confidence_level` of every output of `problem` at
/// `positions`, linearized at `fit`, for a normal of `covariance` on the
/// unbounded scale, or a Student t with `degrees_of_freedom`.
///
/// # Errors
///
/// [`LocalError::Refused`] when the confidence level is not strictly between
/// 0 and 1, a position is one the engine rejects, or the engine is
/// stochastic; [`LocalError::Failed`] when the derivatives fail.
pub fn delta_band<E>(
    problem: &CalibrationProblem<E>,
    fit: &[f64],
    covariance: &Matrix,
    degrees_of_freedom: Option<f64>,
    positions: &[f64],
    confidence_level: f64,
) -> Result<DeltaBand, LocalError>
where
    E: SimulationEngine + Send + Sync,
{
    let [_, upper_level] = interval_levels(confidence_level).map_err(LocalError::Refused)?;
    let quantile = match degrees_of_freedom {
        Some(df) => student_t_quantile(upper_level, df),
        None => normal_quantile(upper_level),
    };
    let outputs = problem.forward().base_engine().output_names();
    let observations: Vec<ObservedDataPoint> = outputs
        .iter()
        .flat_map(|output| {
            positions
                .iter()
                .map(move |&at| ObservedDataPoint::new(at, output.clone(), 0.0))
        })
        .collect();
    // A position the engine rejects is the caller's.
    let source =
        FiniteDifferences::of_observations(problem, observations).map_err(|error| match error {
            SensitivityError::Failed(message) => LocalError::Refused(message),
            error => sensitivity_error(error),
        })?;
    let sensitivities = source.jacobian(fit).map_err(sensitivity_error)?;
    let derivatives = Transform::new(&problem.parameter_bounds()).derivatives(fit);
    let p = fit.len();
    let mut center = Vec::with_capacity(outputs.len());
    let mut lower = Vec::with_capacity(outputs.len());
    let mut upper = Vec::with_capacity(outputs.len());
    for (o, _) in outputs.iter().enumerate() {
        let mut centers = Vec::with_capacity(positions.len());
        let mut lows = Vec::with_capacity(positions.len());
        let mut highs = Vec::with_capacity(positions.len());
        for t in 0..positions.len() {
            let row = o * positions.len() + t;
            let gradient: Vec<f64> = (0..p)
                .map(|j| sensitivities.jacobian[(row, j)] * derivatives[j])
                .collect();
            let variance: f64 = (0..p)
                .flat_map(|j| (0..p).map(move |k| (j, k)))
                .map(|(j, k)| gradient[j] * covariance[(j, k)] * gradient[k])
                .sum();
            let half = quantile * variance.max(0.0).sqrt();
            let prediction = sensitivities.predictions[row];
            centers.push(prediction);
            lows.push(prediction - half);
            highs.push(prediction + half);
        }
        center.push(centers);
        lower.push(lows);
        upper.push(highs);
    }
    Ok(DeltaBand {
        outputs,
        center,
        lower,
        upper,
    })
}
