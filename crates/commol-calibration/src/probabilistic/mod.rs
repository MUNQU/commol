//! Probabilistic calibration - performance-critical components.
//!
//! This module provides Rust implementations of the performance-critical parts
//! of probabilistic calibration:
//!
//! 1. **Parallel execution** of multiple calibration runs ([`calibration`])
//! 2. **NSGA-II ensemble selection** ([`ensemble`])
//! 3. **Cluster-based representative selection** ([`clustering`])
//! 4. **Evaluation deduplication** ([`deduplication`])
//!
//! Python orchestrates the overall workflow using sklearn for K-means clustering,
//! while Rust handles the computationally intensive operations.
//!
//! # Module Structure
//!
//! - [`error`] - Error types for probabilistic calibration
//! - [`config`] - Configuration for ensemble selection algorithms
//! - [`calibration`] - Parallel calibration and prediction generation
//! - [`deduplication`] - Grid-based spatial deduplication
//! - [`ensemble`] - NSGA-II multi-objective ensemble selection
//! - [`clustering`] - Cluster representative selection with various methods
//! - [`utils`] - Shared utility functions

mod calibration;
mod clustering;
mod config;
mod deduplication;
mod ensemble;
pub mod error;
mod utils;

// Re-export public API
pub use calibration::{generate_predictions_parallel, run_multiple_calibrations};
pub use clustering::{select_cluster_representatives, ClusterRepresentativeConfig};
pub use config::EnsembleSelectionConfig;
pub use deduplication::deduplicate_evaluations;
pub use ensemble::{
    select_optimal_ensemble, EnsembleSelectionResult, EnsembleSizeMode, OptimalEnsembleConfig,
    ParetoSolution,
};
pub use error::{CalibrationError, CalibrationResult};
