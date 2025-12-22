//! # Commol Calibration
//!
//! Model-agnostic calibration and parameter estimation for compartment models.
//!
//! This crate provides tools for calibrating any model that implements the
//! `SimulationEngine` trait from `epimodel-core`. It uses the `argmin` optimization
//! library to find parameter values that minimize the difference between model
//! predictions and observed data.
//!
//! ## Features
//!
//! - **Model-agnostic**: Works with DifferenceEquations, NetworkModel, or any future model type
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
//! let model = create_sir_model();
//! let engine = DifferenceEquations::from_model(&model);
//!
//! // Define observed data
//! let observed_data = vec![
//!     ObservedDataPoint::new(10, 1, 501.0),  // time=10, compartment I, value=501
//!     ObservedDataPoint::new(20, 1, 823.0),
//!     ObservedDataPoint::new(30, 1, 654.0),
//! ];
//!
//! // Define parameters to calibrate
//! let params = vec![
//!     CalibrationParameter::new("beta".to_string(), 0.0, 1.0),
//!     CalibrationParameter::new("gamma".to_string(), 0.0, 0.5),
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

pub mod calibration_problem;
pub mod optimization;
pub mod probabilistic;
pub mod types;

// Re-export commonly used items
pub use calibration_problem::CalibrationProblem;
pub use optimization::{
    optimize, optimize_with_history, AccelerationStrategy, InertiaWeightStrategy,
    InitializationStrategy, MutationApplication, MutationStrategy, NelderMeadConfig,
    OptimizationAlgorithm, OptimizationConfig, ParticleSwarmConfig,
};
pub use probabilistic::{
    deduplicate_evaluations, generate_predictions_parallel, run_multiple_calibrations,
    select_cluster_representatives, select_optimal_ensemble, ClusterRepresentativeConfig,
    EnsembleSelectionConfig, EnsembleSelectionResult, EnsembleSizeMode, OptimalEnsembleConfig,
    ParetoSolution,
};
pub use probabilistic::{CalibrationError, CalibrationResult as CalibrationResultType};
pub use types::{
    CalibrationConstraint, CalibrationEvaluation, CalibrationParameter, CalibrationParameterType,
    CalibrationResult, CalibrationResultWithHistory, LossConfig, ObservedDataPoint,
};
