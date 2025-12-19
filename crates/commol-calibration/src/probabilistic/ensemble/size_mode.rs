//! Ensemble size constraint modes.
//!
//! This module defines how ensemble size constraints are handled during
//! NSGA-II optimization.

/// Ensemble size constraint mode.
#[derive(Debug, Clone)]
pub enum EnsembleSizeMode {
    /// Fixed ensemble size - must select exactly this many parameter sets.
    Fixed { size: usize },
    /// Bounded ensemble size - must select between min and max parameter sets.
    Bounded { min: usize, max: usize },
    /// Automatic ensemble size - algorithm determines optimal size without boundaries.
    Automatic,
}
