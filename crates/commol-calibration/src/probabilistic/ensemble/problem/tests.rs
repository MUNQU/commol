use argmin::core::{CostFunction, MultiObjectiveCostFunction};
use argmin::solver::greedysubsetselection::SubsetSelectionCost;

use super::*;
use crate::probabilistic::config::EnsembleSelectionConfig;

fn candidate(seed: f64) -> CalibrationEvaluation {
    CalibrationEvaluation {
        parameters: vec![seed],
        loss: seed,
        predictions: vec![vec![seed, seed + 1.0], vec![seed + 2.0, seed + 3.0]],
    }
}

fn problem(size_mode: EnsembleSizeMode) -> EnsembleSelectionProblem {
    EnsembleSelectionProblem::new(
        (0..8).map(|idx| candidate(idx as f64)).collect(),
        vec![(0, 0, 3.0), (1, 1, 5.0)],
        0.95,
        size_mode,
        &EnsembleSelectionConfig::default().with_parallel_objective_threshold(usize::MAX),
    )
}

#[test]
fn repair_keeps_fixed_and_bounded_evaluations_in_bounds() {
    let params = vec![
        vec![0.9, 0.8, 0.7, 0.6, 0.55, 0.45, 0.4, 0.3],
        vec![0.9, 0.1, 0.2, 0.3, 0.4, 0.49, 0.51, 0.52],
        vec![0.2, 0.7, 0.1, 0.8, 0.3, 0.9, 0.6, 0.4],
    ];

    let fixed = problem(EnsembleSizeMode::Fixed { size: 3 });
    let bounded = problem(EnsembleSizeMode::Bounded { min: 3, max: 5 });

    for param in &params {
        let fixed_selected = fixed.selected_indices_from_param(param);
        assert_eq!(fixed_selected.len(), 3);
        let fixed_objectives = fixed.objectives(param).unwrap();
        assert_ne!(fixed_objectives, vec![1.0, 1.0, fixed_objectives[2]]);

        let bounded_selected = bounded.selected_indices_from_param(param);
        assert!((3..=5).contains(&bounded_selected.len()));
        let bounded_objectives = bounded.objectives(param).unwrap();
        assert_ne!(bounded_objectives, vec![1.0, 1.0, bounded_objectives[2]]);
    }
}

#[test]
fn parallel_and_serial_ci_width_match() {
    let problem = problem(EnsembleSizeMode::Automatic);
    let selected = vec![0, 2, 4, 6];
    let serial = EnsembleSelectionProblem::calculate_ci_width_from_points_serial(
        &problem.ci_point_predictions,
        &selected,
        problem.lower_percentile,
        problem.upper_percentile,
    );
    let parallel = EnsembleSelectionProblem::calculate_ci_width_from_points_parallel(
        &problem.ci_point_predictions,
        &selected,
        problem.lower_percentile,
        problem.upper_percentile,
    );
    assert!((serial - parallel).abs() <= 1e-12);
}

#[test]
#[should_panic(expected = "prediction time-step count does not match first candidate")]
fn point_major_predictions_rejects_mismatched_shapes() {
    let candidates = vec![
        candidate(1.0),
        CalibrationEvaluation {
            parameters: vec![2.0],
            loss: 2.0,
            predictions: vec![vec![2.0, 3.0]],
        },
    ];
    let _ = EnsembleSelectionProblem::point_major_predictions(&candidates, &[0]);
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "actual {actual} vs expected {expected}"
    );
}

fn rebuild_cost(problem: &EnsembleSelectionProblem, selection: &[usize]) -> f64 {
    let mut sorted = selection.to_vec();
    sorted.sort_unstable();
    problem.cost(&sorted).unwrap()
}

#[test]
fn incremental_cost_with_added_matches_rebuild() {
    let problem = problem(EnsembleSizeMode::Automatic);
    let current = vec![0, 2, 4];
    for candidate_idx in [1usize, 3, 5, 6, 7] {
        let mut rebuilt = current.clone();
        rebuilt.push(candidate_idx);
        rebuilt.sort_unstable();
        let expected = problem.cost(&rebuilt).unwrap();
        let actual = problem.cost_with_added(&current, candidate_idx).unwrap();
        assert_close(actual, expected);
    }
}

#[test]
fn incremental_cost_with_removed_matches_rebuild() {
    let problem = problem(EnsembleSizeMode::Automatic);
    let current = vec![0, 2, 4, 6];
    for &removed in &current {
        let rebuilt: Vec<usize> = current.iter().copied().filter(|&i| i != removed).collect();
        let expected = problem.cost(&rebuilt).unwrap();
        let actual = problem.cost_with_removed(&current, removed).unwrap();
        assert_close(actual, expected);
    }
}

#[test]
fn incremental_cost_with_swapped_matches_rebuild() {
    let problem = problem(EnsembleSizeMode::Automatic);
    let current = vec![0, 2, 4, 6];
    for &removed in &current {
        for added in [1usize, 3, 5, 7] {
            let mut rebuilt: Vec<usize> =
                current.iter().copied().filter(|&i| i != removed).collect();
            rebuilt.push(added);
            rebuilt.sort_unstable();
            let expected = problem.cost(&rebuilt).unwrap();
            let actual = problem.cost_with_swapped(&current, removed, added).unwrap();
            assert_close(actual, expected);
        }
    }
}

#[test]
fn incremental_cache_invalidates_across_problems() {
    let problem_a = problem(EnsembleSizeMode::Automatic);
    let candidates_b: Vec<_> = (0..8).map(|idx| candidate((idx as f64) + 10.0)).collect();
    let problem_b = EnsembleSelectionProblem::new(
        candidates_b,
        vec![(0, 0, 13.0), (1, 1, 15.0)],
        0.95,
        EnsembleSizeMode::Automatic,
        &EnsembleSelectionConfig::default().with_parallel_objective_threshold(usize::MAX),
    );

    let current = vec![0, 2, 4];
    let _ = problem_a.cost_with_added(&current, 1).unwrap();
    let actual = problem_b.cost_with_added(&current, 1).unwrap();
    let expected = rebuild_cost(&problem_b, &[0, 1, 2, 4]);
    assert_close(actual, expected);
}
