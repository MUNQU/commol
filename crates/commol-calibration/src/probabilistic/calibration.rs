//! Parallel calibration execution.
//!
//! This module provides functions for running multiple calibration attempts
//! in parallel and generating predictions.

use rayon::prelude::*;

use crate::calibration_problem::CalibrationProblem;
use crate::optimization::{optimize_with_history, OptimizationConfig};
use crate::types::CalibrationResultWithHistory;
use commol_core::SimulationEngine;

use super::error::{CalibrationError, CalibrationResult};

/// How much optimizer evaluation history to retain after each calibration run.
#[derive(Debug, Clone)]
pub enum EvaluationRetention {
    /// Keep every recorded evaluation.
    All,
    /// Keep only the best evaluation from each run.
    BestPerRun,
    /// Keep the top-k evaluations by loss from each run.
    TopKPerRun(usize),
}

/// Run multiple calibration attempts in parallel with different seeds.
///
/// This function runs N calibration attempts in parallel, each with a different
/// random seed. This is the performance-critical bottleneck of probabilistic
/// calibration, so we use Rayon for parallelization.
///
/// # Arguments
/// * `base_problem` - Template calibration problem (will be cloned for each run)
/// * `optimization_config` - Optimization algorithm configuration
/// * `n_runs` - Number of calibration runs to perform
/// * `seed` - Base random seed (each run gets seed + run_index)
///
/// # Returns
/// Vector of calibration results with evaluation history
///
/// # Errors
/// Returns `CalibrationError::AllRunsFailed` if all calibration runs fail
pub fn run_multiple_calibrations<E: SimulationEngine + Send + Sync>(
    base_problem: &CalibrationProblem<E>,
    optimization_config: &OptimizationConfig,
    n_runs: usize,
    seed: u64,
    evaluation_retention: EvaluationRetention,
) -> CalibrationResult<Vec<CalibrationResultWithHistory>> {
    // Generate indexed seeds for each run
    // The index is used to restore deterministic order after parallel execution
    let indexed_seeds: Vec<(usize, u64)> = (0..n_runs)
        .map(|i| (i, seed.wrapping_add(i as u64)))
        .collect();

    // Run calibrations in parallel using Rayon
    // We collect (index, result) tuples to restore deterministic order after parallel execution
    let mut indexed_results: Vec<(usize, Result<CalibrationResultWithHistory, String>)> =
        indexed_seeds
            .par_iter()
            .map(|&(idx, run_seed)| {
                // Clone the problem for this run (each thread gets its own copy)
                let problem_clone = base_problem.clone();

                // Modify optimization config with this seed (only affects PSO)
                let mut opt_config = optimization_config.clone();
                if let OptimizationConfig::ParticleSwarm(ref mut ps_config) = opt_config {
                    ps_config.seed = Some(run_seed);
                }

                // Run optimization with history tracking
                let result = optimize_with_history(problem_clone, opt_config).map(|mut result| {
                    apply_evaluation_retention(&mut result, &evaluation_retention);
                    result
                });
                (idx, result)
            })
            .collect();

    // Sort by original index to restore deterministic order
    // This is critical for reproducibility: Rayon's parallel execution order is non-deterministic,
    // but by sorting by the original index, we ensure results are always in the same order
    indexed_results.sort_by_key(|(idx, _)| *idx);

    // Collect successful results and report failures
    let mut successful_results = Vec::new();
    let mut failed_count = 0;

    for (_, result) in indexed_results {
        match result {
            Ok(r) => successful_results.push(r),
            Err(e) => {
                eprintln!("Warning: Calibration run failed: {}", e);
                failed_count += 1;
            }
        }
    }

    if successful_results.is_empty() {
        return Err(CalibrationError::AllRunsFailed(failed_count));
    }

    if failed_count > 0 {
        eprintln!(
            "Completed {}/{} calibration runs successfully ({} failed)",
            successful_results.len(),
            n_runs,
            failed_count
        );
    }

    Ok(successful_results)
}

fn apply_evaluation_retention(
    result: &mut CalibrationResultWithHistory,
    retention: &EvaluationRetention,
) {
    match retention {
        EvaluationRetention::All => {}
        EvaluationRetention::BestPerRun => {
            if let Some(best) = result
                .evaluations
                .iter()
                .min_by(|a, b| {
                    a.loss
                        .partial_cmp(&b.loss)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned()
            {
                result.evaluations = vec![best];
            }
        }
        EvaluationRetention::TopKPerRun(k) => {
            if *k == 0 || result.evaluations.len() <= *k {
                return;
            }
            result.evaluations.select_nth_unstable_by(*k, |a, b| {
                a.loss
                    .partial_cmp(&b.loss)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            result.evaluations.truncate(*k);
            result.evaluations.sort_by(|a, b| {
                a.loss
                    .partial_cmp(&b.loss)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }
}

/// Internal error type for parallel prediction generation.
///
/// This allows us to distinguish between parameter setting errors and simulation errors
/// within the parallel execution, then convert to the appropriate CalibrationError.
enum PredictionError {
    ParameterSetFailed { name: String, reason: String },
    SimulationFailed(String),
}

/// Generate predictions for multiple parameter sets in parallel.
///
/// This is a performance-critical operation that runs simulations for each
/// parameter set in parallel using Rayon.
///
/// # Arguments
/// * `base_engine` - Base simulation engine (will be cloned for each parameter set)
/// * `parameter_sets` - Vector of parameter values for each parameter set
/// * `parameter_names` - Names of parameters (must match engine parameter names)
/// * `time_steps` - Number of time steps to simulate
///
/// # Returns
/// Vector of predictions, where each prediction is a 2D array [time_step][compartment_idx]
///
/// # Errors
/// - `CalibrationError::ParameterSetFailed` if setting a parameter fails
/// - `CalibrationError::SimulationFailed` if simulation fails
pub fn generate_predictions_parallel<E: SimulationEngine + Send + Sync>(
    base_engine: &E,
    parameter_sets: Vec<Vec<f64>>,
    parameter_names: Vec<String>,
    time_steps: u32,
) -> CalibrationResult<Vec<Vec<Vec<f64>>>> {
    // Run simulations in parallel using Rayon
    let predictions: Vec<Result<Vec<Vec<f64>>, PredictionError>> = parameter_sets
        .par_iter()
        .map(|params| {
            // Clone the engine for this thread
            let mut engine = base_engine.clone();

            // Reset to initial conditions (engine may have stale state from previous runs)
            engine.reset();

            // Update parameter values
            for (param_name, &param_value) in parameter_names.iter().zip(params.iter()) {
                engine.set_parameter(param_name, param_value).map_err(|e| {
                    PredictionError::ParameterSetFailed {
                        name: param_name.clone(),
                        reason: e.to_string(),
                    }
                })?;
            }

            // Run simulation - this returns [time_step][compartment]
            engine
                .run(time_steps)
                .map_err(|e| PredictionError::SimulationFailed(e.to_string()))
        })
        .collect();

    // Collect successful results
    let mut all_predictions = Vec::new();
    for result in predictions {
        match result {
            Ok(pred) => all_predictions.push(pred),
            Err(PredictionError::ParameterSetFailed { name, reason }) => {
                return Err(CalibrationError::ParameterSetFailed { name, reason });
            }
            Err(PredictionError::SimulationFailed(reason)) => {
                return Err(CalibrationError::SimulationFailed(reason));
            }
        }
    }

    Ok(all_predictions)
}

/// Generate predictions for calibration parameter sets, applying both model
/// parameters and calibrated initial conditions.
pub fn generate_calibrated_predictions_parallel<E: SimulationEngine + Send + Sync>(
    base_problem: &CalibrationProblem<E>,
    parameter_sets: Vec<Vec<f64>>,
    time_steps: u32,
) -> CalibrationResult<Vec<Vec<Vec<f64>>>> {
    let predictions: Vec<Result<Vec<Vec<f64>>, PredictionError>> = parameter_sets
        .par_iter()
        .map(|params| {
            let mut engine = base_problem.configured_engine(params).map_err(|reason| {
                PredictionError::ParameterSetFailed {
                    name: "calibration_parameters".to_string(),
                    reason,
                }
            })?;

            engine
                .run(time_steps)
                .map_err(|e| PredictionError::SimulationFailed(e.to_string()))
        })
        .collect();

    collect_prediction_results(predictions)
}

/// Generate compact predictions for selected `(time_step, compartment_idx)` points.
pub fn generate_predictions_at_points_parallel<E: SimulationEngine + Send + Sync>(
    base_engine: &E,
    parameter_sets: Vec<Vec<f64>>,
    parameter_names: Vec<String>,
    metric_points: Vec<(u32, usize)>,
) -> CalibrationResult<Vec<Vec<f64>>> {
    let mut time_steps: Vec<u32> = metric_points.iter().map(|&(step, _)| step).collect();
    time_steps.sort_unstable();
    time_steps.dedup();

    let point_time_indices: Vec<usize> = metric_points
        .iter()
        .map(|&(step, _)| {
            time_steps
                .binary_search(&step)
                .expect("metric point time step must be present")
        })
        .collect();

    let predictions: Vec<Result<Vec<f64>, PredictionError>> = parameter_sets
        .par_iter()
        .map(|params| {
            let mut engine = base_engine.clone();
            engine.reset();

            for (param_name, &param_value) in parameter_names.iter().zip(params.iter()) {
                engine.set_parameter(param_name, param_value).map_err(|e| {
                    PredictionError::ParameterSetFailed {
                        name: param_name.clone(),
                        reason: e.to_string(),
                    }
                })?;
            }

            let mut sparse_results = Vec::with_capacity(time_steps.len());
            engine
                .run_at_steps_into_buffer(&time_steps, &mut sparse_results)
                .map_err(|e| PredictionError::SimulationFailed(e.to_string()))?;

            metric_points
                .iter()
                .zip(&point_time_indices)
                .map(|(&(_, compartment_idx), &time_idx)| {
                    sparse_results
                        .get(time_idx)
                        .and_then(|row| row.get(compartment_idx))
                        .copied()
                        .ok_or_else(|| {
                            PredictionError::SimulationFailed(format!(
                                "Missing compact prediction at sparse row {} compartment {}",
                                time_idx, compartment_idx
                            ))
                        })
                })
                .collect()
        })
        .collect();

    let mut all_predictions = Vec::new();
    for result in predictions {
        match result {
            Ok(pred) => all_predictions.push(pred),
            Err(PredictionError::ParameterSetFailed { name, reason }) => {
                return Err(CalibrationError::ParameterSetFailed { name, reason });
            }
            Err(PredictionError::SimulationFailed(reason)) => {
                return Err(CalibrationError::SimulationFailed(reason));
            }
        }
    }

    Ok(all_predictions)
}

/// Generate compact predictions for calibration parameter sets, applying both
/// model parameters and calibrated initial conditions.
pub fn generate_calibrated_predictions_at_points_parallel<E: SimulationEngine + Send + Sync>(
    base_problem: &CalibrationProblem<E>,
    parameter_sets: Vec<Vec<f64>>,
    metric_points: Vec<(u32, usize)>,
) -> CalibrationResult<Vec<Vec<f64>>> {
    let mut time_steps: Vec<u32> = metric_points.iter().map(|&(step, _)| step).collect();
    time_steps.sort_unstable();
    time_steps.dedup();

    let point_time_indices: Vec<usize> = metric_points
        .iter()
        .map(|&(step, _)| {
            time_steps
                .binary_search(&step)
                .expect("metric point time step must be present")
        })
        .collect();

    let predictions: Vec<Result<Vec<f64>, PredictionError>> = parameter_sets
        .par_iter()
        .map(|params| {
            let mut engine = base_problem.configured_engine(params).map_err(|reason| {
                PredictionError::ParameterSetFailed {
                    name: "calibration_parameters".to_string(),
                    reason,
                }
            })?;

            let mut sparse_results = Vec::with_capacity(time_steps.len());
            engine
                .run_at_steps_into_buffer(&time_steps, &mut sparse_results)
                .map_err(|e| PredictionError::SimulationFailed(e.to_string()))?;

            metric_points
                .iter()
                .zip(&point_time_indices)
                .map(|(&(_, compartment_idx), &time_idx)| {
                    sparse_results
                        .get(time_idx)
                        .and_then(|row| row.get(compartment_idx))
                        .copied()
                        .ok_or_else(|| {
                            PredictionError::SimulationFailed(format!(
                                "Missing compact prediction at sparse row {} compartment {}",
                                time_idx, compartment_idx
                            ))
                        })
                })
                .collect()
        })
        .collect();

    collect_prediction_results(predictions)
}

fn collect_prediction_results<T>(
    predictions: Vec<Result<T, PredictionError>>,
) -> CalibrationResult<Vec<T>> {
    let mut all_predictions = Vec::with_capacity(predictions.len());
    for result in predictions {
        match result {
            Ok(pred) => all_predictions.push(pred),
            Err(PredictionError::ParameterSetFailed { name, reason }) => {
                return Err(CalibrationError::ParameterSetFailed { name, reason });
            }
            Err(PredictionError::SimulationFailed(reason)) => {
                return Err(CalibrationError::SimulationFailed(reason));
            }
        }
    }

    Ok(all_predictions)
}
