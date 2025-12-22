//! Ensemble selection using NSGA-II optimization.
//!
//! This module provides the main entry point for selecting an optimal ensemble
//! of parameter sets using multi-objective optimization.

use argmin::core::Executor;
use argmin::solver::nsgaii::{Individual, NsgaII};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use super::problem::EnsembleSelectionProblem;
use super::size_mode::EnsembleSizeMode;
use crate::probabilistic::config::EnsembleSelectionConfig;
use crate::probabilistic::error::{CalibrationError, CalibrationResult};
use crate::types::CalibrationEvaluation;

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

/// Run NSGA-II to select optimal ensemble from candidates.
///
/// Uses NSGA-II multi-objective optimization to find a Pareto-optimal ensemble
/// of parameter sets that balances narrow confidence intervals with good coverage
/// of observed data.
///
/// # Arguments
/// * `candidates` - Candidate parameter evaluations with predictions
/// * `observed_data` - Observed data points as (time_step, compartment_idx, value)
/// * `config` - Configuration for optimal ensemble selection
///
/// # Returns
/// EnsembleSelectionResult containing selected ensemble and full Pareto front
///
/// # Errors
/// - `CalibrationError::InsufficientCandidates` if fewer than 2 candidates provided
/// - `CalibrationError::NsgaSolverCreation` if NSGA-II solver creation fails
/// - `CalibrationError::NsgaOptimization` if optimization fails
/// - `CalibrationError::EmptyPopulation` if no population in final state
/// - `CalibrationError::EmptyParetoFront` if Pareto front is empty
/// - `CalibrationError::EmptyEnsemble` if selected ensemble has no parameter sets
pub fn select_optimal_ensemble(
    candidates: Vec<CalibrationEvaluation>,
    observed_data: Vec<(usize, usize, f64)>,
    config: &OptimalEnsembleConfig,
) -> CalibrationResult<EnsembleSelectionResult> {
    let n_candidates = candidates.len();

    if n_candidates < 2 {
        return Err(CalibrationError::InsufficientCandidates {
            required: 2,
            actual: n_candidates,
        });
    }

    // Create ensemble selection problem
    let problem = EnsembleSelectionProblem::new(
        candidates,
        observed_data,
        config.confidence_level,
        config.size_mode.clone(),
        config.ensemble_config,
    );

    // Create bounds (0 to 1 for each candidate - binary selection)
    let bounds = vec![(0.0, 1.0); n_candidates];

    // Create NSGA-II solver with RNG
    let mut solver = NsgaII::new(bounds, config.population_size)
        .map_err(|e| CalibrationError::NsgaSolverCreation(e.to_string()))?
        .with_rng(SmallRng::seed_from_u64(config.seed));

    // Configure crossover and mutation (crossover probability from config)
    solver = solver
        .with_crossover_probability(config.ensemble_config.nsga_crossover_probability)
        .with_mutation_probability(1.0 / n_candidates as f64);

    // Run NSGA-II optimization
    let result = Executor::new(problem, solver)
        .configure(|state| state.max_iters(config.generations as u64))
        .run()
        .map_err(|e| CalibrationError::NsgaOptimization(e.to_string()))?;

    // Extract Pareto front
    let state = result.state();
    let pareto_front = state
        .population
        .as_ref()
        .ok_or(CalibrationError::EmptyPopulation)?;

    if pareto_front.is_empty() {
        return Err(CalibrationError::EmptyParetoFront);
    }

    // Filter Pareto front to only include solutions satisfying size constraints
    let valid_indices: Vec<usize> = pareto_front
        .iter()
        .enumerate()
        .filter_map(|(idx, individual)| {
            // Count selected indices
            let ensemble_size = individual
                .position
                .iter()
                .filter(|&&val| val >= 0.5)
                .count();

            // Check if size satisfies constraints
            let is_valid = match &config.size_mode {
                EnsembleSizeMode::Fixed { size } => ensemble_size == *size,
                EnsembleSizeMode::Bounded { min, max } => {
                    ensemble_size >= *min && ensemble_size <= *max
                }
                EnsembleSizeMode::Automatic => true, // No constraint
            };

            if is_valid {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    // Select from valid solutions, or fall back to all if none are valid
    let selected_idx = if valid_indices.is_empty() {
        // No valid solutions - select from all (fallback)
        select_from_pareto_front_by_preference(pareto_front, config.pareto_preference)
    } else {
        // Create a filtered slice of the Pareto front with only valid solutions
        let valid_solutions: Vec<Individual<Vec<f64>, f64>> = valid_indices
            .iter()
            .map(|&idx| pareto_front[idx].clone())
            .collect();
        let local_idx =
            select_from_pareto_front_by_preference(&valid_solutions, config.pareto_preference);
        valid_indices[local_idx]
    };

    let selected_solution = &pareto_front[selected_idx];

    // Convert binary encoding to indices
    let selected_indices: Vec<usize> = selected_solution
        .position
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| if val >= 0.5 { Some(i) } else { None })
        .collect();

    if selected_indices.is_empty() {
        return Err(CalibrationError::EmptyEnsemble);
    }

    // Build Pareto front information
    let pareto_solutions: Vec<ParetoSolution> = pareto_front
        .iter()
        .map(|individual| {
            let ensemble_size = individual
                .position
                .iter()
                .filter(|&&val| val >= 0.5)
                .count();

            let indices: Vec<usize> = individual
                .position
                .iter()
                .enumerate()
                .filter_map(|(i, &val)| if val >= 0.5 { Some(i) } else { None })
                .collect();

            ParetoSolution {
                ensemble_size,
                ci_width: individual.objectives[0],
                coverage: 1.0 - individual.objectives[1], // Convert neg_coverage back to coverage
                size_penalty: individual.objectives[2],
                selected_indices: indices,
            }
        })
        .collect();

    Ok(EnsembleSelectionResult {
        selected_ensemble: selected_indices,
        pareto_front: pareto_solutions,
        selected_pareto_index: selected_idx,
    })
}

/// Select solution from Pareto front based on preference parameter.
///
/// Uses a preference value to interpolate between different selection strategies:
/// - preference = 0.0: Select solution with minimum CI width (narrow uncertainty)
/// - preference = 1.0: Select solution with maximum coverage (best data fit)
/// - preference = 0.5: Balanced selection using weighted combination
/// - Other values: Smooth interpolation using weighted objective combination
///
/// # Arguments
/// * `pareto_front` - The Pareto-optimal solutions
/// * `preference` - Preference value in [0.0, 1.0]
///
/// # Returns
/// Index of selected solution in the Pareto front
fn select_from_pareto_front_by_preference(
    pareto_front: &[Individual<Vec<f64>, f64>],
    preference: f64,
) -> usize {
    if pareto_front.is_empty() {
        return 0;
    }

    if pareto_front.len() == 1 {
        return 0;
    }

    // Clamp preference to [0, 1]
    let preference = preference.clamp(0.0, 1.0);

    // For extreme values, use direct objective minimization
    if preference <= 0.05 {
        // Strongly prefer narrow CI width (minimize objective 0)
        return pareto_front
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.objectives[0]
                    .partial_cmp(&b.objectives[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    if preference >= 0.95 {
        // Strongly prefer high coverage (minimize objective 1 = negative coverage)
        return pareto_front
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.objectives[1]
                    .partial_cmp(&b.objectives[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    // For middle values, use weighted combination
    // Weight interpretation:
    //   preference = 0.0 -> weight = 1.0 (all CI width)
    //   preference = 0.5 -> weight = 0.5 (balanced)
    //   preference = 1.0 -> weight = 0.0 (all coverage)
    let weight = 1.0 - preference;

    pareto_front
        .iter()
        .enumerate()
        .map(|(idx, ind)| {
            // Weighted sum: weight * ci_width + (1 - weight) * neg_coverage
            // Lower score is better (both objectives are minimization)
            let score = weight * ind.objectives[0] + (1.0 - weight) * ind.objectives[1];
            (idx, score)
        })
        .min_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
