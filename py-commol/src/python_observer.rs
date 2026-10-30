//! The argmin observer of a verbose optimization: it writes its progress to
//! Python's `sys.stdout`, where a notebook shows it.
//!
//! The library never writes to the standard error of the process; a write
//! that fails is dropped.

use argmin::core::observers::Observe;
use argmin::core::{KV, State};
use commol_calibration::{
    CalibrationProblem, NelderMeadState, OptimizationConfig, ParticleSwarmState, Progress, Starts,
};
use commol_core::SimulationEngine;
use pyo3::prelude::*;

// Constants for table formatting
const HEADER_SEPARATOR_WIDTH: usize = 76;

/// Write a message to Python's stdout
///
/// This function writes to Python's sys.stdout instead of Rust's stdout,
/// which is necessary for proper display in Python environments.
fn write_to_python_stdout(message: &str) {
    write_optimization_header(&format!("{message}\n"));
}

/// Write an optimization header to Python's stdout
fn write_optimization_header(header: &str) {
    Python::attach(|py| {
        let _ = py
            .import("sys")
            .and_then(|sys| sys.getattr("stdout"))
            .and_then(|stdout| {
                stdout.call_method1("write", (header,))?;
                stdout.call_method0("flush")
            });
    });
}

/// Observer that writes optimization progress to Python's sys.stdout in
/// table format, one table per stage that `progress` names
#[derive(Clone)]
pub struct PythonObserver {
    /// The iterations between two headers of each stage that writes
    progress: Progress,
    /// Last iteration where header was printed
    last_header_iter: u64,
}

impl PythonObserver {
    /// An observer that writes the stages of `progress`
    pub fn new(progress: Progress) -> Self {
        Self {
            progress,
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

impl Observe<NelderMeadState> for PythonObserver {
    fn observe_iter(
        &mut self,
        state: &NelderMeadState,
        _kv: &KV,
    ) -> Result<(), argmin::core::Error> {
        if let Some(header_interval) = self.progress.nelder_mead {
            self.write_row(state, header_interval);
        }
        Ok(())
    }
}

impl Observe<ParticleSwarmState> for PythonObserver {
    fn observe_iter(
        &mut self,
        state: &ParticleSwarmState,
        _kv: &KV,
    ) -> Result<(), argmin::core::Error> {
        if let Some(header_interval) = self.progress.particle_swarm {
            self.write_row(state, header_interval);
        }
        Ok(())
    }
}

impl PythonObserver {
    /// Write the row of `state`, after a header at its first iteration and
    /// every `header_interval` iterations
    fn write_row<I>(&mut self, state: &I, header_interval: u64)
    where
        I: State,
        <I as State>::Float: std::fmt::LowerExp,
    {
        let iter = state.get_iter();

        // Print header at start and every N iterations
        if iter == 0 || (iter > 0 && (iter - self.last_header_iter) >= header_interval) {
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
    }
}

/// Run the starts of `points` seeded with `seed`, with an observer of each
/// that writes to Python's stdout instead of Rust's
///
/// It runs `commol_calibration::optimize_starts_with_observer`, so a verbose
/// calibration takes the starts and the seeds of one that is not; the
/// caller passes one start.
pub fn optimize_with_python_observer<E: SimulationEngine + Send + Sync>(
    problem: &CalibrationProblem<E>,
    config: &OptimizationConfig,
    points: Vec<Vec<f64>>,
    seed: u64,
    progress: Progress,
) -> Result<Starts, String> {
    let initial_params = problem.initial_parameters();
    let parameter_names = problem.parameter_names();

    let header = match config {
        OptimizationConfig::NelderMead(nm_config) => format!(
            "=== Nelder-Mead Optimization (Verbose Mode) ===\n\
             Parameters: {:?}\n\
             Initial values: {:?}\n\
             Max iterations: {}\n\
             SD tolerance: {}\n\
             Initial simplex size: {}\n\
             ===============================================\n",
            parameter_names,
            initial_params,
            nm_config.max_iterations,
            nm_config.sd_tolerance,
            nm_config.initial_simplex_size
        ),
        OptimizationConfig::ParticleSwarm(ps_config) => {
            let mut header = format!(
                "=== Particle Swarm Optimization (Verbose Mode) ===\n\
                 Parameters: {:?}\n\
                 Bounds: {:?}\n\
                 Particles: {}\n\
                 Max iterations: {}\n",
                parameter_names,
                problem.parameter_bounds(),
                ps_config.particles,
                ps_config.max_iterations
            );
            if let Some(polish) = &ps_config.polish {
                header.push_str(&format!(
                    "Polish: Nelder-Mead from the best point of the swarm\n\
                     Polish max iterations: {}\n\
                     Polish SD tolerance: {}\n",
                    polish.max_iterations, polish.sd_tolerance
                ));
            }
            header.push_str("===================================================\n");
            header
        }
    };
    write_optimization_header(&header);

    let starts =
        commol_calibration::optimize_starts_with_observer(problem, config, points, seed, || {
            PythonObserver::new(progress)
        })?;
    let result = starts.best();

    // Print final calibrated parameters
    let separator = "=".repeat(HEADER_SEPARATOR_WIDTH);
    write_to_python_stdout(&separator);
    write_to_python_stdout("Optimization Complete - Calibrated Parameters:");
    write_to_python_stdout(&separator);
    for (name, value) in result
        .parameter_names
        .iter()
        .zip(result.best_parameters.iter())
    {
        write_to_python_stdout(&format!("  {:<20} = {:.6e}", name, value));
    }
    write_to_python_stdout(&separator);

    Ok(starts)
}
