//! Greedy local-search ensemble selection.

use argmin::solver::greedysubsetselection::{solve, solve_pareto, ParetoSelection};

use super::preference::select_pareto_solution_by_preference;
use super::{EnsembleSelectionResult, OptimalEnsembleConfig, ParetoSolution};
use crate::probabilistic::ensemble::problem::EnsembleSelectionProblem;
use crate::probabilistic::ensemble::size_mode::EnsembleSizeMode;
use crate::probabilistic::error::{CalibrationError, CalibrationResult};
use crate::types::CalibrationEvaluation;

pub(super) fn select_greedy_ensemble(
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
    let n_starts = 1;
    let max_local_iterations = 1;

    let pareto_solutions = match &config.size_mode {
        EnsembleSizeMode::Fixed { size } => {
            let subset = solve(&problem, *size, n_starts, config.seed, max_local_iterations)
                .map_err(|e| CalibrationError::EnsembleSelectionFailed(e.to_string()))?;
            vec![greedy_subset_to_pareto_solution(&problem, subset.indices)]
        }
        EnsembleSizeMode::Bounded { min, max } => {
            let pareto = solve_pareto(
                &problem,
                *min..=*max,
                n_starts,
                config.seed,
                max_local_iterations,
                ParetoSelection::MinCost,
            )
            .map_err(|e| CalibrationError::EnsembleSelectionFailed(e.to_string()))?;
            pareto
                .solutions
                .into_iter()
                .map(|subset| greedy_subset_to_pareto_solution(&problem, subset.indices))
                .collect()
        }
        EnsembleSizeMode::Automatic => {
            let pareto = solve_pareto(
                &problem,
                2..=n_candidates,
                n_starts,
                config.seed,
                max_local_iterations,
                ParetoSelection::MinCost,
            )
            .map_err(|e| CalibrationError::EnsembleSelectionFailed(e.to_string()))?;
            pareto
                .solutions
                .into_iter()
                .map(|subset| greedy_subset_to_pareto_solution(&problem, subset.indices))
                .collect()
        }
    };

    if pareto_solutions.is_empty() {
        return Err(CalibrationError::EmptyParetoFront);
    }

    let selected_pareto_index = select_pareto_solution_by_preference(
        &pareto_solutions,
        config.pareto_preference,
        config.confidence_level,
    );
    let selected_ensemble = pareto_solutions[selected_pareto_index]
        .selected_indices
        .clone();

    if selected_ensemble.is_empty() {
        return Err(CalibrationError::EmptyEnsemble);
    }

    Ok(EnsembleSelectionResult {
        selected_ensemble,
        pareto_front: pareto_solutions,
        selected_pareto_index,
    })
}

fn greedy_subset_to_pareto_solution(
    problem: &EnsembleSelectionProblem,
    selected_indices: Vec<usize>,
) -> ParetoSolution {
    let (ci_width, coverage) = problem.evaluate_selected_indices(&selected_indices);
    ParetoSolution {
        ensemble_size: selected_indices.len(),
        ci_width,
        coverage,
        size_penalty: 0.0,
        selected_indices,
    }
}
