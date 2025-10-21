//! Custom argmin observer that writes to Python's stdout/stderr
//!
//! This module provides a custom observer for argmin optimization that properly
//! integrates with Python's output streams. This is necessary because Rust's
//! println! and eprintln! write to OS-level stdout/stderr, which don't appear
//! in Python environments (especially Jupyter notebooks).

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{KV, State};
use epimodel_calibration::{CalibrationProblem, OptimizationConfig};
use epimodel_core::SimulationEngine;
use pyo3::prelude::*;

/// Observer that writes optimization progress to Python's sys.stdout in table format
pub struct PythonObserver {
    /// Number of iterations between header repeats
    header_interval: u64,
    /// Last iteration where header was printed
    last_header_iter: u64,
}

impl PythonObserver {
    /// Create a new Python observer with default header interval of 100
    pub fn new() -> Self {
        Self {
            header_interval: 100,
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

    /// Write a line to Python's stdout
    fn write_to_python(&self, message: &str) {
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

    /// Print the table header
    fn print_header(&self) {
        let separator = "=".repeat(92);
        self.write_to_python(&separator);
        self.write_to_python(&format!(
            "{:>12} | {:>14} | {:>16} | {:>16} | {:>16}",
            "Iteration", "Time (s)", "Objective", "Best Objective", "Obj. Evaluations"
        ));
        self.write_to_python(&separator);
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
        let obj = state.get_cost();
        let best_obj = state.get_best_cost();
        let time = state.get_time().map(|d| d.as_secs_f64()).unwrap_or(0.0);

        // Get objective evaluation count from function counts
        let obj_counts = state.get_func_counts();
        let obj_evaluations = obj_counts.get("cost_count").copied().unwrap_or(0);

        // Print data row in table format
        let output = format!(
            "{:>12} | {:>14.6} | {:>16} | {:>16} | {:>16}",
            iter,
            time,
            format!("{:.6e}", obj),
            format!("{:.6e}", best_obj),
            obj_evaluations,
        );

        self.write_to_python(&output);
        Ok(())
    }
}

/// Run optimization with Python observer for verbose output
///
/// This function runs the same optimization as `epimodel_calibration::optimize`,
/// but with a custom observer that writes to Python's stdout instead of Rust's.
pub fn optimize_with_python_observer<E: SimulationEngine>(
    problem: CalibrationProblem<E>,
    config: OptimizationConfig,
    header_interval: u64,
) -> Result<epimodel_calibration::CalibrationResult, String> {
    use argmin::core::Executor;
    use argmin::solver::neldermead::NelderMead;
    use argmin::solver::particleswarm::ParticleSwarm;

    let initial_params = problem.initial_parameters();
    let parameter_names = problem.parameter_names();

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
            Python::with_gil(|py| {
                let _ = py
                    .import("sys")
                    .and_then(|sys| sys.getattr("stdout"))
                    .and_then(|stdout| {
                        let header = format!(
                            "=== Nelder-Mead Optimization (Verbose Mode) ===\n\
                             Parameters: {:?}\n\
                             Initial values: {:?}\n\
                             Max iterations: {}\n\
                             SD tolerance: {}\n\
                             ===============================================\n",
                            parameter_names,
                            initial_params,
                            nm_config.max_iterations,
                            nm_config.sd_tolerance
                        );
                        stdout.call_method1("write", (header,))?;
                        stdout.call_method0("flush")
                    });
            });

            let executor = Executor::new(problem, solver)
                .configure(|state| state.max_iters(nm_config.max_iterations))
                .add_observer(
                    PythonObserver::with_header_interval(header_interval),
                    ObserverMode::Always,
                );

            let result = executor
                .run()
                .map_err(|e| format!("Optimization failed: {}", e))?;

            let state = result.state();

            Ok(epimodel_calibration::CalibrationResult {
                best_parameters: state.best_param.clone().unwrap_or(initial_params),
                parameter_names,
                final_loss: state.best_cost,
                iterations: state.iter as usize,
                converged: state.termination_status.terminated(),
                termination_reason: format!("{:?}", state.termination_status),
            })
        }

        OptimizationConfig::ParticleSwarm(ps_config) => {
            // Get bounds
            let bounds = problem.get_parameter_bounds();
            let lower_bound: Vec<f64> = bounds.iter().map(|(min, _)| *min).collect();
            let upper_bound: Vec<f64> = bounds.iter().map(|(_, max)| *max).collect();

            // Build solver
            let mut solver =
                ParticleSwarm::new((lower_bound, upper_bound), ps_config.num_particles);

            if let Some(inertia) = ps_config.inertia_factor {
                solver = solver
                    .with_inertia_factor(inertia)
                    .map_err(|e| format!("Failed to set inertia_factor: {}", e))?;
            }
            if let Some(cognitive) = ps_config.cognitive_factor {
                solver = solver
                    .with_cognitive_factor(cognitive)
                    .map_err(|e| format!("Failed to set cognitive_factor: {}", e))?;
            }
            if let Some(social) = ps_config.social_factor {
                solver = solver
                    .with_social_factor(social)
                    .map_err(|e| format!("Failed to set social_factor: {}", e))?;
            }

            // Write header to Python stdout
            Python::with_gil(|py| {
                let _ = py
                    .import("sys")
                    .and_then(|sys| sys.getattr("stdout"))
                    .and_then(|stdout| {
                        let mut header = format!(
                            "=== Particle Swarm Optimization (Verbose Mode) ===\n\
                             Parameters: {:?}\n\
                             Bounds: {:?}\n\
                             Num particles: {}\n\
                             Max iterations: {}\n",
                            parameter_names,
                            bounds,
                            ps_config.num_particles,
                            ps_config.max_iterations
                        );
                        if let Some(target) = ps_config.target_cost {
                            header.push_str(&format!("Target cost: {}\n", target));
                        }
                        header.push_str("===================================================\n");
                        stdout.call_method1("write", (header,))?;
                        stdout.call_method0("flush")
                    });
            });

            let executor = Executor::new(problem, solver)
                .configure(|state| {
                    let mut state = state.max_iters(ps_config.max_iterations);
                    if let Some(target) = ps_config.target_cost {
                        state = state.target_cost(target);
                    }
                    state
                })
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

            Ok(epimodel_calibration::CalibrationResult {
                best_parameters: best_params,
                parameter_names,
                final_loss: best_cost,
                iterations: state.iter as usize,
                converged: state.termination_status.terminated(),
                termination_reason: format!("{:?}", state.termination_status),
            })
        }
    }
}
