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

    /// Failed to set a parameter value.
    #[error("Failed to set parameter '{name}': {reason}")]
    ParameterSetFailed { name: String, reason: String },

    /// Simulation failed during prediction generation.
    #[error("Simulation failed: {0}")]
    SimulationFailed(String),

    /// Ensemble subset selection failed.
    #[error("Ensemble selection failed: {0}")]
    EnsembleSelectionFailed(String),
}

/// Result type alias for probabilistic calibration operations.
pub type CalibrationResult<T> = Result<T, CalibrationError>;
