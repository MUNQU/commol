//! Ensemble selection algorithm dispatch.
//!
//! This module defines the shared result types and routes ensemble selection
//! requests to the configured algorithm implementation.

use super::size_mode::EnsembleSizeMode;
use crate::probabilistic::config::{EnsembleAlgorithm, EnsembleSelectionConfig};
use crate::probabilistic::error::CalibrationResult;
use crate::types::CalibrationEvaluation;

mod greedy;
mod nsga2;
mod preference;
#[cfg(test)]
mod tests;

use greedy::select_greedy_ensemble;
use nsga2::select_nsga2_ensemble;

/// Configuration for optimal ensemble selection.
pub struct OptimalEnsembleConfig<'a> {
    pub population_size: usize,
    pub generations: usize,
    pub confidence_level: f64,
    pub seed: u64,
    pub pareto_preference: f64,
    pub size_mode: EnsembleSizeMode,
    pub ensemble_config: &'a EnsembleSelectionConfig,
}

/// Information about a Pareto front solution
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    /// Ensemble size (number of selected parameter sets)
    pub ensemble_size: usize,
    /// Normalized CI width objective [0, 1]
    pub ci_width: f64,
    /// Coverage percentage [0, 1]
    pub coverage: f64,
    /// Size constraint penalty [0, infinity]
    pub size_penalty: f64,
    /// Indices of selected parameter sets
    pub selected_indices: Vec<usize>,
}

/// Result from ensemble selection including Pareto front
#[derive(Debug, Clone)]
pub struct EnsembleSelectionResult {
    /// The selected ensemble (indices of parameter sets)
    pub selected_ensemble: Vec<usize>,
    /// All solutions from the Pareto front
    pub pareto_front: Vec<ParetoSolution>,
    /// Index in pareto_front that was selected
    pub selected_pareto_index: usize,
}

/// Select an optimal ensemble from candidates using the configured algorithm.
///
/// # Arguments
/// * `candidates` - Candidate parameter evaluations with predictions
/// * `observed_data` - Observed data points as (time_step, compartment_idx, value)
/// * `config` - Configuration for optimal ensemble selection
///
/// # Returns
/// EnsembleSelectionResult containing the selected ensemble and Pareto front.
///
/// # Errors
/// - `CalibrationError::InsufficientCandidates` if fewer than 2 candidates provided
/// - `CalibrationError::EmptyPopulation` if no population in final state
/// - `CalibrationError::EmptyParetoFront` if Pareto front is empty
/// - `CalibrationError::EmptyEnsemble` if selected ensemble has no parameter sets
pub fn select_optimal_ensemble(
    candidates: Vec<CalibrationEvaluation>,
    observed_data: Vec<(usize, usize, f64)>,
    config: &OptimalEnsembleConfig,
) -> CalibrationResult<EnsembleSelectionResult> {
    match config.ensemble_config.ensemble_algorithm {
        EnsembleAlgorithm::GreedyLocalSearch => {
            select_greedy_ensemble(candidates, observed_data, config)
        }
        EnsembleAlgorithm::Nsga2 => select_nsga2_ensemble(candidates, observed_data, config),
    }
}
