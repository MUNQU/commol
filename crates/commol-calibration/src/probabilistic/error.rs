//! Error types for probabilistic calibration.
//!
//! This module provides structured error types using `thiserror` for better
//! error handling and more informative error messages.

use thiserror::Error;

/// Errors that can occur during probabilistic calibration operations.
#[derive(Debug, Error)]
pub enum CalibrationError {
    /// All calibration runs failed during parallel execution.
    #[error("All {0} calibration runs failed")]
    AllRunsFailed(usize),

    /// Mismatch between evaluations count and cluster labels count.
    #[error("Evaluations count ({evaluations}) doesn't match labels count ({labels})")]
    EvaluationLabelMismatch { evaluations: usize, labels: usize },

    /// Not enough candidates for ensemble selection.
    #[error(
        "Need at least {required} candidate parameter sets for ensemble selection, got {actual}"
    )]
    InsufficientCandidates { required: usize, actual: usize },

    /// NSGA-II solver creation failed.
    #[error("Failed to create NSGA-II solver: {0}")]
    NsgaSolverCreation(String),

    /// NSGA-II optimization failed.
    #[error("NSGA-II optimization failed: {0}")]
    NsgaOptimization(String),

    /// No population in the NSGA-II final state.
    #[error("No population in final state")]
    EmptyPopulation,

    /// Empty Pareto front after optimization.
    #[error("Empty Pareto front")]
    EmptyParetoFront,

    /// Selected ensemble has no parameter sets.
    #[error("Selected solution has no parameter sets")]
    EmptyEnsemble,

    /// Failed to set a parameter value.
    #[error("Failed to set parameter '{name}': {reason}")]
    ParameterSetFailed { name: String, reason: String },

    /// Simulation failed during prediction generation.
    #[error("Simulation failed: {0}")]
    SimulationFailed(String),
}

/// Result type alias for probabilistic calibration operations.
pub type CalibrationResult<T> = Result<T, CalibrationError>;
