//! The delta method: the band of every output that a local approximation
//! gives by linearizing the model at the fit, with no run of its draws.
//!
//! With `J_u = J_θ · D` the Jacobian of an output at a position on the
//! unbounded scale, its variance is `J_u Σ J_uᵀ`; the band is the prediction
//! at the fit plus and minus the quantile of a normal, or of a Student t, times
//! its standard deviation. The Student t of a sandwich has the degrees of
//! freedom of the direction `J_u` in the blocks of the sandwich
//! ([`BlockShares::degrees_of_freedom`]), so each output at each position has
//! its own. It has the layout of every band ([`Band`]): the prediction at the
//! fit is the mean and the median, as the normal and the Student t are
//! symmetric.

use commol_calibration::{CalibrationProblem, ObservedDataPoint};
use commol_core::{CONFIDENCE_LEVEL, SimulationEngine};
use commol_statistics::statistics::interval_levels;

use super::approximation::{CurvatureKind, curvature_at};
use super::shares::BlockShares;
use crate::error::MethodError;
use crate::predictive::Band;
use crate::resampling::BlockScheme;
use crate::sensitivity::{FiniteDifferences, Matrix, SensitivityError, SensitivitySource};
use crate::transform::Transform;
use commol_statistics::special::{normal_quantile, student_t_quantile};

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
/// `positions`, linearized at `fit`, the value of each parameter by name,
/// for a normal of `covariance` on the unbounded scale, or, with the
/// `block_length` of a sandwich, a Student t whose degrees of freedom each
/// output at each position reads from the Gauss–Newton curvature of the blocks
/// at `fit` ([`BlockShares::degrees_of_freedom`]). The rows and the columns of
/// `covariance` are the parameters `names`, which are those of the problem, in
/// its order.
///
/// # Errors
///
/// [`MethodError::Invalid`] when `names` are not the parameters of the
/// problem, the fit has no value of one, the covariance has not one row and
/// one column per parameter, the confidence level is not strictly between 0
/// and 1, a position is one the engine rejects, or the engine is
/// stochastic; [`MethodError::Failed`] when the derivatives fail; with a
/// `block_length`, as [`curvature_at`], [`BlockScheme::new`] and
/// [`BlockShares::new`].
pub fn delta_band<E>(
    problem: &CalibrationProblem<E>,
    names: &[String],
    fit: &[(String, f64)],
    covariance: &[Vec<f64>],
    block_length: Option<usize>,
    positions: &[f64],
    confidence_level: f64,
) -> Result<DeltaBand, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    let parameters = problem.parameter_names();
    if names != parameters.as_slice() {
        return Err(MethodError::Invalid(format!(
            "The result has the parameters {names:?}; the problem has {parameters:?}"
        )));
    }
    let fit = parameters
        .iter()
        .map(|name| {
            fit.iter()
                .find(|(given, _)| given == name)
                .map(|(_, value)| *value)
                .ok_or_else(|| MethodError::Invalid(format!("The fit has no value of '{name}'")))
        })
        .collect::<Result<Vec<f64>, _>>()?;
    let p = parameters.len();
    if covariance.len() != p || covariance.iter().any(|row| row.len() != p) {
        return Err(MethodError::Invalid(format!(
            "The covariance needs {p} rows of {p} values, one per parameter"
        )));
    }
    let covariance = Matrix::from_fn(p, p, |j, k| covariance[j][k]);
    let fit = fit.as_slice();
    CONFIDENCE_LEVEL
        .check(confidence_level)
        .map_err(MethodError::Invalid)?;
    let upper_level = interval_levels(confidence_level)[1];
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
    let spread = match block_length {
        Some(length) => {
            let curvature = curvature_at(problem, fit, CurvatureKind::GaussNewton)?;
            let blocks = BlockScheme::new(problem.observed_data(), length)?;
            let shares = BlockShares::new(&curvature, &blocks)?;
            Some((curvature.scales().to_vec(), shares))
        }
        None => None,
    };
    let normal = normal_quantile(upper_level);
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
            let quantile = match &spread {
                Some((scales, shares)) => student_t_quantile(
                    upper_level,
                    shares.degrees_of_freedom(&in_units(&gradient, scales)),
                ),
                None => normal,
            };
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

/// The direction of the gradient `gradient` on the unbounded scale in the
/// units of a curvature of `scales`, `g_j / s_j`, times a power of two that
/// brings its largest entry to between 1/2 and 1, read by exponents so that no
/// quotient leaves the range of a number: the degrees of freedom of a
/// direction do not depend on its size.
fn in_units(gradient: &[f64], scales: &[f64]) -> Vec<f64> {
    let parts: Vec<(f64, i32)> = gradient
        .iter()
        .zip(scales)
        .map(|(&entry, &scale)| {
            let (entry, entry_exponent) = libm::frexp(entry);
            let (scale, scale_exponent) = libm::frexp(scale);
            let (fraction, exponent) = libm::frexp(entry / scale);
            (fraction, exponent + entry_exponent - scale_exponent)
        })
        .collect();
    let largest = parts
        .iter()
        .filter(|(fraction, _)| *fraction != 0.0)
        .map(|&(_, exponent)| exponent)
        .max()
        .unwrap_or(0);
    parts
        .into_iter()
        .map(|(fraction, exponent)| libm::ldexp(fraction, exponent - largest))
        .collect()
}
