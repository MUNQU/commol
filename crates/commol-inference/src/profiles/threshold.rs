//! The thresholds of a profile: how far the statistic may rise above the fit
//! inside a pointwise interval.
//!
//! | Threshold | Statistic | Assumes |
//! |---|---|---|
//! | [`chi_square`] | `−log L` of observation models | Wilks: the statistic is `½ χ²₁` |
//! | [`beale`] | A sum of squares | Independent Gaussian errors of one variance |
//! | [`bootstrap_thresholds`] | Any loss | That the bootstrap mimics the data |

use commol_calibration::{CalibrationProblem, LossConfig, Subproblem, optimize_subproblem};
use commol_core::SimulationEngine;
use commol_random::replicate_seed;
use rayon::prelude::*;

use crate::error::MethodError;
use crate::profiles::tracer::ProfiledQuantity;
use crate::resampling::{FitUnit, ReplicateRunner, Resampler};
use crate::special::{f_one_quantile, normal_quantile};
use crate::statistics;

/// `½ χ²₁(confidence_level)`: the threshold of `−log L` for a pointwise
/// interval of `confidence_level` (Wilks).
///
/// # Errors
///
/// [`MethodError::Invalid`] when the problem has a loss instead of
/// observation models, or `confidence_level` is not strictly between 0 and
/// 1 ([`statistics::interval_levels`]).
pub fn chi_square<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    confidence_level: f64,
) -> Result<f64, MethodError> {
    statistics::interval_levels(confidence_level).map_err(MethodError::Invalid)?;
    if problem.discrepancy().likelihood().is_none() {
        return Err(MethodError::Invalid(
            "The chi-square threshold reads −log L and needs observation models; the problem \
             has a loss. With a loss, use Threshold.beale (a sum of squares of independent \
             Gaussian errors) or Threshold.bootstrap_calibrated"
                .to_string(),
        ));
    }
    let z = normal_quantile((1.0 + confidence_level) / 2.0);
    Ok(z * z / 2.0)
}

/// Beale's threshold of a sum of squares, for one parameter or prediction:
/// `S(θ̂) · F(1, n − p; confidence_level) / (n − p)`, with `n` the
/// observations of positive weight and `p` the parameters. It is exact for
/// a linear model with independent Gaussian errors of one variance.
///
/// # Errors
///
/// [`MethodError::Invalid`] when the problem has no sum of squares (another
/// loss, or observation models), it has no more observations than
/// parameters, or `confidence_level` is not strictly between 0 and 1.
pub fn beale<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    fit_loss: f64,
    confidence_level: f64,
) -> Result<f64, MethodError> {
    statistics::interval_levels(confidence_level).map_err(MethodError::Invalid)?;
    if problem.discrepancy().loss() != Some(LossConfig::SumSquaredError) {
        return Err(MethodError::Invalid(
            "Beale's threshold reads a sum of squares; the problem needs the loss 'sse'"
                .to_string(),
        ));
    }
    let n = problem
        .discrepancy()
        .weights()
        .iter()
        .filter(|&&weight| weight > 0.0)
        .count();
    let p = problem.num_parameters();
    if n <= p {
        return Err(MethodError::Invalid(format!(
            "Beale's threshold needs more observations than parameters: {n} observations, \
             {p} parameters"
        )));
    }
    let nu = (n - p) as f64;
    Ok(fit_loss * f_one_quantile(nu, confidence_level) / nu)
}

/// The thresholds of a bootstrap, one per quantity.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapThresholds {
    /// The quantile `confidence_level` of the rises of each quantity.
    pub thresholds: Vec<f64>,
    /// The rise of each quantity on each replicate that did not fail, in
    /// the order of `replicates`.
    pub rises: Vec<Vec<f64>>,
    /// The replicates that did not fail.
    pub replicates: Vec<usize>,
    /// The index and reason of each replicate that failed: its fit, or the
    /// fit of a quantity held at its center.
    pub failed: Vec<(usize, String)>,
    /// The number of replicates kept although the optimizer of their fit
    /// did not converge.
    pub not_converged: usize,
}

/// The thresholds of `quantities`, held at their values `centers` at the
/// fit, calibrated by the bootstrap (prepivoting, Beran 1988).
///
/// Replicate `b` draws its data with `resampler` from the generator of
/// `replicate_seed(runner.seed(), b)` ([`ReplicateRunner::perturbations`]),
/// so it is the replicate of that bootstrap. It is fitted
/// ([`ReplicateRunner::fit_all`], giving `θ̂_b` and its loss `L_b(θ̂_b)`),
/// and each quantity is held at its center on the replicate's data from
/// `θ̂_b`: the rise is `PL_b(center) − L_b(θ̂_b)`, at least 0. The threshold
/// of a quantity is the quantile `confidence_level` of its rises.
/// Replicates run in parallel.
///
/// # Errors
///
/// [`MethodError::Invalid`] when `confidence_level` is not strictly between
/// 0 and 1 or the lengths differ; [`MethodError::Failed`] when every
/// replicate fails ([`FitUnit::require`]).
pub fn bootstrap_thresholds<E>(
    runner: &ReplicateRunner<'_, E>,
    resampler: &dyn Resampler,
    replicates: usize,
    quantities: &[ProfiledQuantity],
    centers: &[f64],
    confidence_level: f64,
) -> Result<BootstrapThresholds, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    statistics::interval_levels(confidence_level).map_err(MethodError::Invalid)?;
    if quantities.len() != centers.len() {
        return Err(MethodError::Invalid(format!(
            "One center per quantity: {} quantities, {} centers",
            quantities.len(),
            centers.len()
        )));
    }
    let perturbations = runner.perturbations(resampler, replicates);
    let fits = runner.fit_all(&perturbations);
    let outcomes: Vec<(usize, bool, Result<Vec<f64>, String>)> = fits
        .fits
        .par_iter()
        .map(|(index, fit)| {
            let index = *index;
            let rises = runner
                .replicate_problem(&perturbations[index])
                .map_err(|error| error.to_string())
                .and_then(|problem| {
                    let seed = replicate_seed(replicate_seed(runner.seed(), index as u64), 0);
                    quantities
                        .iter()
                        .zip(centers)
                        .enumerate()
                        .map(|(q, (quantity, &center))| {
                            let held = hold(&problem, quantity, center)?;
                            let mut start = fit.best_parameters.clone();
                            if let ProfiledQuantity::Parameter(j) = quantity {
                                start[*j] = center;
                            }
                            let refit = optimize_subproblem(
                                &held,
                                runner.config(),
                                &start,
                                Some(replicate_seed(seed, q as u64)),
                            )?;
                            Ok((refit.fit.final_loss - fit.final_loss).max(0.0))
                        })
                        .collect()
                });
            (index, fit.converged, rises)
        })
        .collect();
    let mut rises = vec![Vec::new(); quantities.len()];
    let mut kept = Vec::new();
    let mut failed = fits.failed;
    let mut not_converged = 0;
    for (index, converged, outcome) in outcomes {
        match outcome {
            Ok(values) => {
                kept.push(index);
                not_converged += usize::from(!converged);
                for (column, value) in rises.iter_mut().zip(values) {
                    column.push(value);
                }
            }
            Err(reason) => failed.push((index, reason)),
        }
    }
    failed.sort_by_key(|(index, _)| *index);
    FitUnit::Replicate.require(kept.len(), &failed, 1)?;
    let weights = vec![1.0; kept.len()];
    let thresholds = rises
        .iter()
        .map(|column| {
            statistics::quantiles(column, &weights, &[confidence_level])
                .expect("a rise of the profile of a fit is a number")[0]
        })
        .collect();
    Ok(BootstrapThresholds {
        thresholds,
        rises,
        replicates: kept,
        failed,
        not_converged,
    })
}

/// The subproblem of `problem` that holds `quantity` at `value`.
fn hold<'a, E: SimulationEngine>(
    problem: &'a CalibrationProblem<E>,
    quantity: &ProfiledQuantity,
    value: f64,
) -> Result<Subproblem<'a, E>, String> {
    match quantity {
        ProfiledQuantity::Parameter(index) => Subproblem::fix_parameter(problem, *index, value),
        ProfiledQuantity::Prediction(spec) => Subproblem::fix_prediction(problem, spec, value),
    }
}
