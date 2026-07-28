//! Probabilistic calibration - performance-critical components.
//!
//! This module provides Rust implementations of the performance-critical parts
//! of probabilistic calibration:
//!
//! 1. **Parallel execution** of multiple calibration runs ([`calibration`])
//! 2. **Cluster-based representative selection** ([`clustering`])
//! 3. **Evaluation deduplication** ([`deduplication`])
//!
//! Python orchestrates the overall workflow using sklearn for K-means clustering,
//! while Rust handles the computationally intensive operations.
//!
//! # Module Structure
//!
//! - [`error`] - Error types for probabilistic calibration
//! - [`config`] - Configuration for representative selection algorithms
//! - [`calibration`] - Parallel calibration and prediction generation
//! - [`deduplication`] - Grid-based spatial deduplication
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
pub use calibration::{
    generate_calibrated_predictions_at_points_parallel, generate_calibrated_predictions_parallel,
    generate_predictions_at_points_parallel, generate_predictions_parallel,
    run_multiple_calibrations, EvaluationRetention,
};
pub use clustering::{select_cluster_representatives, ClusterRepresentativeConfig};
pub use config::RepresentativeSelectionConfig;
pub use deduplication::deduplicate_evaluations;
pub use ensemble::{
    select_compact_ensemble, CentralLossMetric, EnsembleAlgorithm, EnsembleSelectionConfig,
    EnsembleSelectionResult, EnsembleSizeMode, ParetoSolution,
};
pub use error::{CalibrationError, CalibrationResult};
