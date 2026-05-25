//! NSGA-II ensemble selection.

use argmin::core::Executor;
use argmin::solver::nsgaii::{Individual, NsgaII};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use super::preference::{select_objective_point_by_preference, ObjectivePoint};
use super::{EnsembleSelectionResult, OptimalEnsembleConfig, ParetoSolution};
use crate::probabilistic::ensemble::problem::EnsembleSelectionProblem;
use crate::probabilistic::ensemble::size_mode::EnsembleSizeMode;
use crate::probabilistic::error::{CalibrationError, CalibrationResult};
use crate::types::CalibrationEvaluation;

pub(super) fn select_nsga2_ensemble(
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

    let problem = EnsembleSelectionProblem::new(
        candidates,
        observed_data,
        config.confidence_level,
        config.size_mode.clone(),
        config.ensemble_config,
    );

    let bounds = vec![(0.0, 1.0); n_candidates];
    let mut solver = NsgaII::new(bounds, config.population_size)
        .map_err(|e| CalibrationError::SolverCreation(e.to_string()))?
        .with_rng(SmallRng::seed_from_u64(config.seed));

    solver = solver
        .with_crossover_probability(config.ensemble_config.crossover_probability)
        .with_mutation_probability(1.0 / n_candidates as f64);

    let result = Executor::new(problem, solver)
        .configure(|state| state.max_iters(config.generations as u64))
        .run()
        .map_err(|e| CalibrationError::EnsembleSelectionFailed(e.to_string()))?;

    let state = result.state();
    let pareto_front = state
        .population
        .as_ref()
        .ok_or(CalibrationError::EmptyPopulation)?;

    if pareto_front.is_empty() {
        return Err(CalibrationError::EmptyParetoFront);
    }

    let problem = result
        .problem
        .problem
        .as_ref()
        .expect("ensemble selection problem must be present after optimization");

    let valid_indices: Vec<usize> = pareto_front
        .iter()
        .enumerate()
        .filter_map(|(idx, individual)| {
            let ensemble_size = problem
                .selected_indices_from_param(&individual.position)
                .len();

            let is_valid = match &config.size_mode {
                EnsembleSizeMode::Fixed { size } => ensemble_size == *size,
                EnsembleSizeMode::Bounded { min, max } => {
                    ensemble_size >= *min && ensemble_size <= *max
                }
                EnsembleSizeMode::Automatic => true,
            };

            is_valid.then_some(idx)
        })
        .collect();

    let selected_idx = if valid_indices.is_empty() {
        select_individual_by_preference(
            pareto_front,
            config.pareto_preference,
            config.confidence_level,
        )
    } else {
        let valid_solutions: Vec<Individual<Vec<f64>, f64>> = valid_indices
            .iter()
            .map(|&idx| pareto_front[idx].clone())
            .collect();
        let local_idx = select_individual_by_preference(
            &valid_solutions,
            config.pareto_preference,
            config.confidence_level,
        );
        valid_indices[local_idx]
    };

    let selected_solution = &pareto_front[selected_idx];
    let selected_indices = problem.selected_indices_from_param(&selected_solution.position);

    if selected_indices.is_empty() {
        return Err(CalibrationError::EmptyEnsemble);
    }

    let pareto_solutions: Vec<ParetoSolution> = pareto_front
        .iter()
        .map(|individual| {
            let indices = problem.selected_indices_from_param(&individual.position);
            ParetoSolution {
                ensemble_size: indices.len(),
                ci_width: individual.objectives[0],
                coverage: 1.0 - individual.objectives[1],
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

fn select_individual_by_preference(
    individuals: &[Individual<Vec<f64>, f64>],
    preference: f64,
    confidence_level: f64,
) -> usize {
    let points: Vec<_> = individuals
        .iter()
        .map(|individual| ObjectivePoint {
            ci_width: individual.objectives[0],
            coverage: 1.0 - individual.objectives[1],
        })
        .collect();
    select_objective_point_by_preference(&points, preference, confidence_level)
}
