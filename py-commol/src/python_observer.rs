//! Custom argmin observer that writes to Python's stdout/stderr
//!
//! This module provides a custom observer for argmin optimization that properly
//! integrates with Python's output streams. This is necessary because Rust's
//! println! and eprintln! write to OS-level stdout/stderr, which don't appear
//! in Python environments (especially Jupyter notebooks).

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{KV, State};
use commol_calibration::{CalibrationProblem, OptimizationConfig};
use commol_core::SimulationEngine;
use pyo3::prelude::*;

// Constants for table formatting
const HEADER_SEPARATOR_WIDTH: usize = 76;
const DEFAULT_HEADER_INTERVAL: u64 = 100;

/// Write a message to Python's stdout
///
/// This function writes to Python's sys.stdout instead of Rust's stdout,
/// which is necessary for proper display in Python environments.
fn write_to_python_stdout(message: &str) {
    Python::with_gil(|py| {
        if let Err(e) = py
            .import("sys")
            .and_then(|sys| sys.getattr("stdout"))
            .and_then(|stdout| {
                stdout.call_method1("write", (format!("{}\n", message),))?;
                stdout.call_method0("flush")
            })
        {
            // Fallback to eprintln if Python stdout fails
            eprintln!("Failed to write to Python stdout: {}", e);
            eprintln!("{}", message);
        }
    });
}

/// Write an optimization header to Python's stdout
fn write_optimization_header(header: &str) {
    Python::with_gil(|py| {
        let _ = py
            .import("sys")
            .and_then(|sys| sys.getattr("stdout"))
            .and_then(|stdout| {
                stdout.call_method1("write", (header,))?;
                stdout.call_method0("flush")
            });
    });
}

/// Observer that writes optimization progress to Python's sys.stdout in table format
pub struct PythonObserver {
    /// Number of iterations between header repeats
    header_interval: u64,
    /// Last iteration where header was printed
    last_header_iter: u64,
}

impl PythonObserver {
    /// Create a new Python observer with default header interval
    pub fn new() -> Self {
        Self {
            header_interval: DEFAULT_HEADER_INTERVAL,
            last_header_iter: 0,
        }
    }

    /// Create a new Python observer with custom header interval
    pub fn with_header_interval(header_interval: u64) -> Self {
        Self {
            header_interval,
            last_header_iter: 0,
        }
    }

    /// Print the table header
    fn print_header(&self) {
        let separator = "=".repeat(HEADER_SEPARATOR_WIDTH);
        write_to_python_stdout(&separator);
        write_to_python_stdout(&format!(
            "{:>12} | {:>14} | {:>16} | {:>16}",
            "Iteration", "Time (s)", "Obj. Evaluations", "Best Objective"
        ));
        write_to_python_stdout(&separator);
    }
}

impl Default for PythonObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl<I> Observe<I> for PythonObserver
where
    I: State,
    <I as State>::Float: std::fmt::LowerExp,
{
    /// Called when observation should occur
    fn observe_iter(&mut self, state: &I, _kv: &KV) -> Result<(), argmin::core::Error> {
        let iter = state.get_iter();

        // Print header at start and every N iterations
        if iter == 0 || (iter > 0 && (iter - self.last_header_iter) >= self.header_interval) {
            self.print_header();
            self.last_header_iter = iter;
        }

        // Extract values from state
        let best_obj = state.get_best_cost();
        let time = state.get_time().map(|d| d.as_secs_f64()).unwrap_or(0.0);

        // Get objective evaluation count from function counts
        let obj_counts = state.get_func_counts();
        let obj_evaluations = obj_counts.get("cost_count").copied().unwrap_or(0);

        // Print data row in table format
        let output = format!(
            "{:>12} | {:>14.6} | {:>16} | {:>16}",
            iter,
            time,
            obj_evaluations,
            format!("{:.6e}", best_obj),
        );

        write_to_python_stdout(&output);
        Ok(())
    }
}

/// Helper struct to hold correction parameters extracted before moving problem
struct CorrectionParams {
    fixed_ic_sum: f64,
    auto_calc_ic_idx: Option<usize>,
    param_types: Vec<commol_calibration::CalibrationParameterType>,
}

impl CorrectionParams {
    /// Apply corrections to parameter values
    fn apply(&self, mut param_values: Vec<f64>) -> Vec<f64> {
        if let Some(idx) = self.auto_calc_ic_idx {
            let calibrated_ic_sum: f64 = param_values
                .iter()
                .enumerate()
                .filter(|(param_idx, _)| {
                    self.param_types[*param_idx]
                        == commol_calibration::CalibrationParameterType::InitialCondition
                        && *param_idx != idx
                })
                .map(|(_, value)| value)
                .sum();

            let auto_calculated_value = (1.0 - self.fixed_ic_sum - calibrated_ic_sum).max(0.0);
            param_values[idx] = auto_calculated_value;
        }
        param_values
    }
}

/// Run optimization with Python observer for verbose output
///
/// This function runs the same optimization as `commol_calibration::optimize`,
/// but with a custom observer that writes to Python's stdout instead of Rust's.
pub fn optimize_with_python_observer<E: SimulationEngine>(
    problem: CalibrationProblem<E>,
    config: OptimizationConfig,
    header_interval: u64,
) -> Result<commol_calibration::CalibrationResult, String> {
    use argmin::core::Executor;
    use argmin::solver::neldermead::NelderMead;
    use argmin::solver::particleswarm::ParticleSwarm;

    let initial_params = problem.initial_parameters();
    let parameter_names = problem.parameter_names();

    // Extract correction info BEFORE moving problem into executor
    let (fixed_ic_sum, auto_calc_ic_idx, param_types) = problem.get_parameter_fix_info();
    let correction_params = CorrectionParams {
        fixed_ic_sum,
        auto_calc_ic_idx,
        param_types,
    };

    match config {
        OptimizationConfig::NelderMead(nm_config) => {
            // Create simplex vertices
            let n = initial_params.len();
            let mut vertices = vec![initial_params.clone()];
            for i in 0..n {
                let mut vertex = initial_params.clone();
                vertex[i] *= 1.1;
                vertices.push(vertex);
            }

            // Build solver
            let mut solver = NelderMead::new(vertices)
                .with_sd_tolerance(nm_config.sd_tolerance)
                .map_err(|e| format!("Failed to set sd_tolerance: {}", e))?;

            if let Some(alpha) = nm_config.alpha {
                solver = solver
                    .with_alpha(alpha)
                    .map_err(|e| format!("Failed to set alpha: {}", e))?;
            }
            if let Some(gamma) = nm_config.gamma {
                solver = solver
                    .with_gamma(gamma)
                    .map_err(|e| format!("Failed to set gamma: {}", e))?;
            }
            if let Some(rho) = nm_config.rho {
                solver = solver
                    .with_rho(rho)
                    .map_err(|e| format!("Failed to set rho: {}", e))?;
            }
            if let Some(sigma) = nm_config.sigma {
                solver = solver
                    .with_sigma(sigma)
                    .map_err(|e| format!("Failed to set sigma: {}", e))?;
            }

            // Write header to Python stdout
            let header = format!(
                "=== Nelder-Mead Optimization (Verbose Mode) ===\n\
                 Parameters: {:?}\n\
                 Initial values: {:?}\n\
                 Max iterations: {}\n\
                 SD tolerance: {}\n\
                 ===============================================\n",
                parameter_names, initial_params, nm_config.max_iterations, nm_config.sd_tolerance
            );
            write_optimization_header(&header);

            let executor = Executor::new(problem, solver)
                .configure(|state| state.max_iters(nm_config.max_iterations).counting(true))
                .timer(true)
                .add_observer(
                    PythonObserver::with_header_interval(header_interval),
                    ObserverMode::Always,
                );

            let result = executor
                .run()
                .map_err(|e| format!("Optimization failed: {}", e))?;

            let state = result.state();

            let best_params = state.best_param.clone().unwrap_or(initial_params.clone());
            let corrected_params = correction_params.apply(best_params);

            // Print final calibrated parameters
            let separator = "=".repeat(HEADER_SEPARATOR_WIDTH);
            write_to_python_stdout(&separator);
            write_to_python_stdout("Optimization Complete - Calibrated Parameters:");
            write_to_python_stdout(&separator);
            for (name, value) in parameter_names.iter().zip(corrected_params.iter()) {
                write_to_python_stdout(&format!("  {:<20} = {:.6e}", name, value));
            }
            write_to_python_stdout(&separator);

            Ok(commol_calibration::CalibrationResult {
                best_parameters: corrected_params,
                parameter_names,
                final_loss: state.best_cost,
                iterations: state.iter as usize,
                converged: state.termination_status.terminated(),
                termination_reason: format!("{:?}", state.termination_status),
            })
        }

        OptimizationConfig::ParticleSwarm(ps_config) => {
            // Get bounds
            let bounds = problem.parameter_bounds();
            let lower_bound: Vec<f64> = bounds.iter().map(|(min, _)| *min).collect();
            let upper_bound: Vec<f64> = bounds.iter().map(|(_, max)| *max).collect();

            // Build solver
            let mut solver =
                ParticleSwarm::new((lower_bound, upper_bound), ps_config.num_particles);

            // Apply inertia strategy if provided
            solver = match &ps_config.inertia_strategy {
                Some(commol_calibration::InertiaWeightStrategy::Constant(w)) => solver
                    .with_inertia_factor(*w)
                    .map_err(|e| format!("Failed to set inertia_factor: {}", e))?,
                Some(commol_calibration::InertiaWeightStrategy::Chaotic { w_min, w_max }) => solver
                    .with_chaotic_inertia(*w_min, *w_max)
                    .map_err(|e| format!("Failed to set chaotic_inertia: {}", e))?,
                None => solver,
            };

            // Apply acceleration strategy if provided
            solver = match &ps_config.acceleration_strategy {
                Some(commol_calibration::AccelerationStrategy::Constant { cognitive, social }) => {
                    solver = solver
                        .with_cognitive_factor(*cognitive)
                        .map_err(|e| format!("Failed to set cognitive_factor: {}", e))?;
                    solver
                        .with_social_factor(*social)
                        .map_err(|e| format!("Failed to set social_factor: {}", e))?
                }
                Some(commol_calibration::AccelerationStrategy::TimeVarying {
                    c1_initial,
                    c1_final,
                    c2_initial,
                    c2_final,
                }) => solver
                    .with_tvac(
                        *c1_initial,
                        *c1_final,
                        *c2_initial,
                        *c2_final,
                        ps_config.max_iterations as usize,
                    )
                    .map_err(|e| format!("Failed to set TVAC: {}", e))?,
                None => solver,
            };

            // Apply initialization strategy
            let argmin_init_strategy = match ps_config.initialization_strategy {
                commol_calibration::InitializationStrategy::UniformRandom => {
                    argmin::solver::particleswarm::InitializationStrategy::UniformRandom
                }
                commol_calibration::InitializationStrategy::LatinHypercube => {
                    argmin::solver::particleswarm::InitializationStrategy::LatinHypercube
                }
                commol_calibration::InitializationStrategy::OppositionBased => {
                    argmin::solver::particleswarm::InitializationStrategy::OppositionBased
                }
            };
            solver = solver.with_initialization_strategy(argmin_init_strategy);

            // Apply velocity clamping if enabled
            if let Some(clamp_factor) = ps_config.velocity_clamp_factor {
                solver = solver
                    .with_velocity_clamping(clamp_factor)
                    .map_err(|e| format!("Failed to set velocity_clamping: {}", e))?;
            }

            // Apply velocity mutation if enabled
            if let Some(threshold) = ps_config.velocity_mutation_threshold {
                solver = solver
                    .with_velocity_mutation(threshold)
                    .map_err(|e| format!("Failed to set velocity_mutation: {}", e))?;
            }

            // Apply mutation if enabled
            if !matches!(
                ps_config.mutation_strategy,
                commol_calibration::MutationStrategy::None
            ) && !matches!(
                ps_config.mutation_application,
                commol_calibration::MutationApplication::None
            ) {
                let argmin_mutation_strategy = match ps_config.mutation_strategy {
                    commol_calibration::MutationStrategy::None => {
                        argmin::solver::particleswarm::MutationStrategy::None
                    }
                    commol_calibration::MutationStrategy::Gaussian(std_dev) => {
                        argmin::solver::particleswarm::MutationStrategy::Gaussian(std_dev)
                    }
                    commol_calibration::MutationStrategy::Cauchy(scale) => {
                        argmin::solver::particleswarm::MutationStrategy::Cauchy(scale)
                    }
                };

                let argmin_mutation_application = match ps_config.mutation_application {
                    commol_calibration::MutationApplication::None => {
                        argmin::solver::particleswarm::MutationApplication::None
                    }
                    commol_calibration::MutationApplication::GlobalBestOnly => {
                        argmin::solver::particleswarm::MutationApplication::GlobalBestOnly
                    }
                    commol_calibration::MutationApplication::AllParticles => {
                        argmin::solver::particleswarm::MutationApplication::AllParticles
                    }
                    commol_calibration::MutationApplication::BelowAverage => {
                        argmin::solver::particleswarm::MutationApplication::BelowAverage
                    }
                };

                solver = solver
                    .with_mutation(
                        argmin_mutation_strategy,
                        ps_config.mutation_probability,
                        argmin_mutation_application,
                    )
                    .map_err(|e| format!("Failed to set mutation: {}", e))?;
            }

            // Write header to Python stdout
            let mut header = format!(
                "=== Particle Swarm Optimization (Verbose Mode) ===\n\
                 Parameters: {:?}\n\
                 Bounds: {:?}\n\
                 Num particles: {}\n\
                 Max iterations: {}\n",
                parameter_names, bounds, ps_config.num_particles, ps_config.max_iterations
            );
            if let Some(target) = ps_config.target_cost {
                header.push_str(&format!("Target cost: {}\n", target));
            }
            header.push_str("===================================================\n");
            write_optimization_header(&header);

            let executor = Executor::new(problem, solver)
                .configure(|state| {
                    let mut state = state.max_iters(ps_config.max_iterations).counting(true);
                    if let Some(target) = ps_config.target_cost {
                        state = state.target_cost(target);
                    }
                    state
                })
                .timer(true)
                .add_observer(
                    PythonObserver::with_header_interval(header_interval),
                    ObserverMode::Always,
                );

            let result = executor
                .run()
                .map_err(|e| format!("Optimization failed: {}", e))?;

            let state = result.state();

            let (best_params, best_cost) = match &state.best_individual {
                Some(particle) => (particle.position.clone(), particle.cost),
                None => (initial_params.clone(), f64::INFINITY),
            };

            let corrected_params = correction_params.apply(best_params);

            // Print final calibrated parameters
            let separator = "=".repeat(HEADER_SEPARATOR_WIDTH);
            write_to_python_stdout(&separator);
            write_to_python_stdout("Optimization Complete - Calibrated Parameters:");
            write_to_python_stdout(&separator);
            for (name, value) in parameter_names.iter().zip(corrected_params.iter()) {
                write_to_python_stdout(&format!("  {:<20} = {:.6e}", name, value));
            }
            write_to_python_stdout(&separator);

            Ok(commol_calibration::CalibrationResult {
                best_parameters: corrected_params,
                parameter_names,
                final_loss: best_cost,
                iterations: state.iter as usize,
                converged: state.termination_status.terminated(),
                termination_reason: format!("{:?}", state.termination_status),
            })
        }
    }
}
