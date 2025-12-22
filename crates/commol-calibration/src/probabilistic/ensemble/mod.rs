//! Ensemble selection using NSGA-II multi-objective optimization.
//!
//! This module provides functionality for selecting an optimal ensemble of
//! parameter sets that balances narrow confidence intervals with good coverage
//! of observed data.

mod problem;
mod selection;
mod size_mode;

pub use selection::{
    select_optimal_ensemble, EnsembleSelectionResult, OptimalEnsembleConfig, ParetoSolution,
};
pub use size_mode::EnsembleSizeMode;
