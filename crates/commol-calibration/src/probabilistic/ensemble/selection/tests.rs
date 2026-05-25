use std::time::Instant;

use super::*;
use crate::probabilistic::config::{CIWidthScope, EnsembleAlgorithm};

fn candidate(seed: f64) -> CalibrationEvaluation {
    CalibrationEvaluation {
        parameters: vec![seed],
        loss: seed,
        predictions: vec![vec![seed], vec![seed + 1.0], vec![seed + 2.0]],
    }
}

#[test]
fn greedy_local_search_dispatch_returns_valid_fixed_ensemble() {
    let candidates: Vec<_> = (0..6).map(|idx| candidate(idx as f64)).collect();
    let ensemble_config = EnsembleSelectionConfig::default()
        .with_ensemble_algorithm(EnsembleAlgorithm::GreedyLocalSearch)
        .with_parallel_objective_threshold(usize::MAX);
    let config = OptimalEnsembleConfig {
        population_size: 4,
        generations: 4,
        confidence_level: 0.95,
        seed: 7,
        pareto_preference: 0.5,
        size_mode: EnsembleSizeMode::Fixed { size: 3 },
        ensemble_config: &ensemble_config,
    };

    let result = select_optimal_ensemble(candidates, vec![(1, 0, 2.0)], &config).unwrap();

    assert_eq!(result.selected_ensemble.len(), 3);
    assert_eq!(result.pareto_front.len(), 1);
    assert_eq!(result.pareto_front[0].ensemble_size, 3);
}

fn synthetic_candidate(
    candidate_idx: usize,
    time_steps: usize,
    compartments: usize,
) -> CalibrationEvaluation {
    let phase = candidate_idx as f64 * 0.073;
    let scale = 1.0 + (candidate_idx % 17) as f64 * 0.003;
    let predictions = (0..time_steps)
        .map(|time_step| {
            (0..compartments)
                .map(|compartment_idx| {
                    let trend = 100.0
                        + time_step as f64 * (1.4 + compartment_idx as f64 * 0.08)
                        + compartment_idx as f64 * 18.0;
                    let wave = ((time_step as f64 * 0.17) + phase + compartment_idx as f64 * 0.31)
                        .sin()
                        * (4.0 + compartment_idx as f64);
                    trend * scale + wave
                })
                .collect()
        })
        .collect();

    CalibrationEvaluation {
        parameters: vec![phase, scale],
        loss: (candidate_idx as f64 % 23.0) / 23.0,
        predictions,
    }
}

fn synthetic_fixture() -> (Vec<CalibrationEvaluation>, Vec<(usize, usize, f64)>) {
    let time_steps = 48;
    let compartments = 4;
    let candidates: Vec<_> = (0..160)
        .map(|idx| synthetic_candidate(idx, time_steps, compartments))
        .collect();
    let observed_data = (0..time_steps)
        .flat_map(|time_step| {
            (0..compartments).map(move |compartment_idx| {
                let trend = 100.0
                    + time_step as f64 * (1.4 + compartment_idx as f64 * 0.08)
                    + compartment_idx as f64 * 18.0;
                let wave = ((time_step as f64 * 0.17) + compartment_idx as f64 * 0.31).sin()
                    * (4.0 + compartment_idx as f64);
                (time_step, compartment_idx, trend + wave)
            })
        })
        .collect();

    (candidates, observed_data)
}

#[test]
#[ignore = "performance check; run explicitly with --ignored --nocapture"]
fn greedy_local_search_outperforms_nsga2_on_synthetic_fixture() {
    let (candidates, observed_data) = synthetic_fixture();
    let greedy_ensemble_config = EnsembleSelectionConfig::default()
        .with_ensemble_algorithm(EnsembleAlgorithm::GreedyLocalSearch)
        .with_ci_width_scope(CIWidthScope::ObservedPoints)
        .with_parallel_objective_threshold(usize::MAX);
    let nsga_ensemble_config = EnsembleSelectionConfig::default()
        .with_ensemble_algorithm(EnsembleAlgorithm::Nsga2)
        .with_ci_width_scope(CIWidthScope::ObservedPoints)
        .with_parallel_objective_threshold(usize::MAX);

    let greedy_config = OptimalEnsembleConfig {
        population_size: 100,
        generations: 100,
        confidence_level: 0.95,
        seed: 13,
        pareto_preference: 0.5,
        size_mode: EnsembleSizeMode::Fixed { size: 24 },
        ensemble_config: &greedy_ensemble_config,
    };
    let nsga_config = OptimalEnsembleConfig {
        population_size: 100,
        generations: 100,
        confidence_level: 0.95,
        seed: 13,
        pareto_preference: 0.5,
        size_mode: EnsembleSizeMode::Fixed { size: 24 },
        ensemble_config: &nsga_ensemble_config,
    };

    let start = Instant::now();
    let greedy = select_optimal_ensemble(candidates.clone(), observed_data.clone(), &greedy_config)
        .expect("greedy selection should succeed");
    let greedy_elapsed = start.elapsed();

    let start = Instant::now();
    let nsga = select_optimal_ensemble(candidates, observed_data, &nsga_config)
        .expect("NSGA-II selection should succeed");
    let nsga_elapsed = start.elapsed();

    let speedup = nsga_elapsed.as_secs_f64() / greedy_elapsed.as_secs_f64();
    println!(
        "greedy={:?} nsga2={:?} speedup={:.2}x greedy_size={} nsga2_size={} greedy_ci={:.6} nsga2_ci={:.6} greedy_cov={:.6} nsga2_cov={:.6}",
        greedy_elapsed,
        nsga_elapsed,
        speedup,
        greedy.selected_ensemble.len(),
        nsga.selected_ensemble.len(),
        greedy.pareto_front[greedy.selected_pareto_index].ci_width,
        nsga.pareto_front[nsga.selected_pareto_index].ci_width,
        greedy.pareto_front[greedy.selected_pareto_index].coverage,
        nsga.pareto_front[nsga.selected_pareto_index].coverage,
    );

    assert_eq!(greedy.selected_ensemble.len(), 24);
    assert_eq!(nsga.selected_ensemble.len(), 24);
    assert!(
        speedup >= 2.0,
        "expected greedy ({greedy_elapsed:?}) to be at least 2x faster than NSGA-II ({nsga_elapsed:?}); got {speedup:.2}x"
    );
}
