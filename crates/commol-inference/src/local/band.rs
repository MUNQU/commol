//! The delta method: the band of every output that a local approximation
//! gives by linearizing the model at the fit, with no run of its draws.
//!
//! With `J_u = J_θ · D` the Jacobian of an output at a position on the
//! unbounded scale, its variance is `J_u Σ J_uᵀ`; the band is the prediction
//! at the fit plus and minus the quantile of a normal, or of a Student t
//! with degrees of freedom, times its standard deviation. It has the layout
//! of every band ([`Band`]): the prediction at the fit is the mean and the
//! median, as the normal and the Student t are symmetric.

use commol_calibration::{CalibrationProblem, ObservedDataPoint};
use commol_core::{CONFIDENCE_LEVEL, SimulationEngine};
use commol_statistics::statistics::interval_levels;

use crate::error::MethodError;
use crate::predictive::Band;
use crate::sensitivity::{FiniteDifferences, Matrix, SensitivityError, SensitivitySource};
use crate::special::{normal_quantile, student_t_quantile};
use crate::transform::Transform;

/// The linearized band of every output at some positions.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaBand {
    /// The output names, one per column of each table of the band.
    pub outputs: Vec<String>,
    /// The band, `[position][output]`: the prediction at the fit as the mean
    /// and the median, and the ends of the central interval.
    pub band: Band,
}

/// The band of `confidence_level` of every output of `problem` at
/// `positions`, linearized at `fit`, for a normal of `covariance` on the
/// unbounded scale, or a Student t with `degrees_of_freedom`.
///
/// # Errors
///
/// [`MethodError::Invalid`] when the confidence level is not strictly between
/// 0 and 1, a position is one the engine rejects, or the engine is
/// stochastic; [`MethodError::Failed`] when the derivatives fail.
pub fn delta_band<E>(
    problem: &CalibrationProblem<E>,
    fit: &[f64],
    covariance: &Matrix,
    degrees_of_freedom: Option<f64>,
    positions: &[f64],
    confidence_level: f64,
) -> Result<DeltaBand, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    CONFIDENCE_LEVEL
        .check(confidence_level)
        .map_err(MethodError::Invalid)?;
    let upper_level = interval_levels(confidence_level)[1];
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
            SensitivityError::Failed(message) => MethodError::Invalid(message),
            error => MethodError::Sensitivity(error),
        })?;
    let sensitivities = source.jacobian(fit).map_err(MethodError::Sensitivity)?;
    let derivatives = Transform::new(&problem.parameter_bounds()).derivatives(fit);
    let p = fit.len();
    let table = vec![vec![0.0; outputs.len()]; positions.len()];
    let mut band = Band {
        mean: table.clone(),
        median: table.clone(),
        lower: table.clone(),
        upper: table,
    };
    for o in 0..outputs.len() {
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
            band.mean[t][o] = prediction;
            band.median[t][o] = prediction;
            band.lower[t][o] = prediction - half;
            band.upper[t][o] = prediction + half;
        }
    }
    Ok(DeltaBand { outputs, band })
}
