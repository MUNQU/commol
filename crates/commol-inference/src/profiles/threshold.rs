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
use commol_random::{Generator, replicate_seed};
use rayon::prelude::*;

use crate::profiles::tracer::ProfiledQuantity;
use crate::resampling::{ReplicateRunner, Resampler};
use crate::special::{f_one_quantile, normal_quantile};
use crate::statistics;

fn check_level(level: f64) -> Result<(), String> {
    if level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        Err(format!("A level is strictly between 0 and 1, got {level}"))
    }
}

/// `½ χ²₁(level)`: the threshold of `−log L` for a pointwise interval of
/// level `level` (Wilks).
///
/// # Errors
///
/// Returns a message when the problem has a loss instead of observation
/// models, or `level` is not strictly between 0 and 1.
pub fn chi_square<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    level: f64,
) -> Result<f64, String> {
    check_level(level)?;
    if problem.discrepancy().likelihood().is_none() {
        return Err(
            "The chi-square threshold reads −log L and needs observation models; the problem \
             has a loss. With a loss, use Threshold.beale (a sum of squares of independent \
             Gaussian errors) or Threshold.bootstrap_calibrated"
                .to_string(),
        );
    }
    let z = normal_quantile((1.0 + level) / 2.0);
    Ok(z * z / 2.0)
}

/// Beale's threshold of a sum of squares, for one parameter or prediction:
/// `S(θ̂) · F(1, n − p; level) / (n − p)`, with `n` the observations of
/// positive weight and `p` the parameters. It is exact for a linear model
/// with independent Gaussian errors of one variance.
///
/// # Errors
///
/// Returns a message when the problem has no sum of squares (another loss,
/// or observation models), it has no more observations than parameters, or
/// `level` is not strictly between 0 and 1.
pub fn beale<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    fit_loss: f64,
    level: f64,
) -> Result<f64, String> {
    check_level(level)?;
    if problem.discrepancy().loss() != Some(LossConfig::SumSquaredError) {
        return Err(
            "Beale's threshold reads a sum of squares; the problem needs the loss 'sse'"
                .to_string(),
        );
    }
    let n = problem
        .discrepancy()
        .weights()
        .iter()
        .filter(|&&weight| weight > 0.0)
        .count();
    let p = problem.num_parameters();
    if n <= p {
        return Err(format!(
            "Beale's threshold needs more observations than parameters: {n} observations, \
             {p} parameters"
        ));
    }
    let nu = (n - p) as f64;
    Ok(fit_loss * f_one_quantile(nu, level) / nu)
}

/// The thresholds of a bootstrap, one per quantity.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapThresholds {
    /// The quantile `level` of the rises of each quantity.
    pub thresholds: Vec<f64>,
    /// The rise of each quantity on each replicate that did not fail, in
    /// the order of `replicates`.
    pub rises: Vec<Vec<f64>>,
    /// The replicates that did not fail.
    pub replicates: Vec<usize>,
    /// The index and reason of each replicate that failed.
    pub failed: Vec<(usize, String)>,
}

/// The thresholds of `quantities`, held at their values `centers` at the
/// fit, calibrated by the bootstrap (prepivoting, Beran 1988).
///
/// Replicate `b` draws its data with `resampler` from the generator of
/// `replicate_seed(runner.seed(), b)`, as [`ReplicateRunner::run`] does, so
/// it is the replicate of that bootstrap. It is fitted
/// ([`ReplicateRunner::fit_one`], giving `θ̂_b` and its loss `L_b(θ̂_b)`),
/// and each quantity is held at its center on the replicate's data from
/// `θ̂_b`: the rise is `PL_b(center) − L_b(θ̂_b)`, at least 0. The threshold
/// of a quantity is the quantile `level` of its rises. Replicates run in
/// parallel.
///
/// # Errors
///
/// Returns a message when `level` is not strictly between 0 and 1, the
/// lengths differ, or every replicate fails.
pub fn bootstrap_thresholds<E>(
    runner: &ReplicateRunner<'_, E>,
    resampler: &dyn Resampler,
    replicates: usize,
    quantities: &[ProfiledQuantity],
    centers: &[f64],
    level: f64,
) -> Result<BootstrapThresholds, String>
where
    E: SimulationEngine + Send + Sync,
{
    check_level(level)?;
    if quantities.len() != centers.len() {
        return Err(format!(
            "One center per quantity: {} quantities, {} centers",
            quantities.len(),
            centers.len()
        ));
    }
    let outcomes: Vec<(usize, Result<Vec<f64>, String>)> = (0..replicates)
        .into_par_iter()
        .map(|index| {
            let mut generator = Generator::from_seed(replicate_seed(runner.seed(), index as u64));
            let perturbation = resampler.replicate(&mut generator);
            let rises = runner.fit_one(index, &perturbation).and_then(|fit| {
                let problem = runner.replicate_problem(&perturbation)?;
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
            (index, rises)
        })
        .collect();
    let mut rises = vec![Vec::new(); quantities.len()];
    let mut kept = Vec::new();
    let mut failed = Vec::new();
    for (index, outcome) in outcomes {
        match outcome {
            Ok(values) => {
                kept.push(index);
                for (column, value) in rises.iter_mut().zip(values) {
                    column.push(value);
                }
            }
            Err(reason) => failed.push((index, reason)),
        }
    }
    if kept.is_empty() {
        return Err(crate::resampling::runner::every_replicate_failed(&failed));
    }
    let weights = vec![1.0; kept.len()];
    let thresholds = rises
        .iter()
        .map(|column| statistics::quantiles(column, &weights, &[level])[0])
        .collect();
    Ok(BootstrapThresholds {
        thresholds,
        rises,
        replicates: kept,
        failed,
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
