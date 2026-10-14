//! # Commol Calibration
//!
//! Model-agnostic calibration and parameter estimation.
//!
//! This crate provides tools for calibrating any model that implements the
//! `SimulationEngine` trait from `commol-core`. It uses the `argmin` optimization
//! library to find parameter values that minimize the difference between model
//! predictions and observed data.
//!
//! ## Features
//!
//! - **Model-agnostic**: Works with the engine of any model kind; the engine
//!   resolves and writes the values of its kind, such as initial conditions
//! - **Multiple algorithms**: Nelder-Mead, L-BFGS, Particle Swarm Optimization
//! - **Flexible loss functions**: SSE, RMSE, MAE
//! - **Type-safe**: Leverages Rust's type system for compile-time guarantees
//!
//! ## Example
//!
//! ```rust,ignore
//! use commol_calibration::{
//!     CalibrationProblem, optimize,
//!     OptimizationConfig, OptimizationAlgorithm,
//!     types::*,
//! };
//! use commol_difference::DifferenceEquations;
//!
//! // Create your model
//! let model = create_model();
//! let engine = DifferenceEquations::from_model(&model);
//!
//! // Define observed data
//! let observed_data = vec![
//!     ObservedDataPoint::new(10.0, 1, 501.0),  // position 10, compartment B, value=501
//!     ObservedDataPoint::new(20.0, 1, 823.0),
//!     ObservedDataPoint::new(30.0, 1, 654.0),
//! ];
//!
//! // Define parameters to calibrate
//! let params = vec![
//!     CalibrationParameter::new("k1".to_string(), 0.0, 1.0),
//!     CalibrationParameter::new("k2".to_string(), 0.0, 0.5),
//! ];
//!
//! // Create calibration problem
//! let problem = CalibrationProblem::new(
//!     engine,
//!     observed_data,
//!     params,
//!     LossConfig::SumSquaredError,
//! )?;
//!
//! // Configure and run optimization
//! let config = OptimizationConfig::new()
//!     .with_algorithm(OptimizationAlgorithm::NelderMead)
//!     .with_max_iterations(1000)
//!     .with_tolerance(1e-6);
//!
//! let result = optimize(problem, config)?;
//!
//! println!("Best parameters: {:?}", result.best_parameters);
//! println!("Final loss: {}", result.final_loss);
//! println!("Converged: {}", result.converged);
//! ```

#![warn(missing_docs)]

pub mod calibration_problem;
pub mod optimization;
pub mod probabilistic;
pub mod types;
pub mod values;

// Re-export commonly used items
pub use calibration_problem::{check_positions, CalibrationProblem};
pub use optimization::{
    converged, optimize, optimize_with_history, optimize_with_observer, AccelerationStrategy,
    InertiaWeightStrategy, InitializationStrategy, MutationApplication, MutationStrategy,
    NelderMeadConfig, NelderMeadState, NoObserver, OptimizationAlgorithm, OptimizationConfig,
    ParticleSwarmConfig, ParticleSwarmState,
};
pub use probabilistic::{
    deduplicate_evaluations, generate_calibrated_predictions_at_points_parallel,
    generate_calibrated_predictions_parallel, generate_predictions_at_points_parallel,
    generate_predictions_parallel, run_multiple_calibrations, select_cluster_representatives,
    select_compact_ensemble, CentralLossMetric, ClusterRepresentativeConfig, EnsembleAlgorithm,
    EnsembleSelectionConfig, EnsembleSelectionResult, EnsembleSizeMode, EvaluationRetention,
    ParetoSolution, RepresentativeSelectionConfig,
};
pub use probabilistic::{CalibrationError, CalibrationResult as CalibrationResultType};
pub use types::{
    CalibrationConstraint, CalibrationEvaluation, CalibrationParameter, CalibrationParameterType,
    CalibrationResult, CalibrationResultWithHistory, LossConfig, ObservedDataPoint,
};
pub use values::{check_model_values, configured_engine};
