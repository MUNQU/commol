//! Fit-gated greedy local-search ensemble selection.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use super::{
    central_loss_limit, diagnostics, CompactSelectionProblem, EnsembleSelectionConfig,
    EnsembleSelectionResult, EnsembleSizeMode, Metrics,
};
use crate::probabilistic::error::{CalibrationError, CalibrationResult};

pub(super) fn select_greedy(
    problem: &CompactSelectionProblem,
    config: &EnsembleSelectionConfig,
) -> CalibrationResult<EnsembleSelectionResult> {
    if config.central_fit_max_loss_ratio < 1.0 {
        return Err(CalibrationError::EnsembleSelectionFailed(
            "central_fit_max_loss_ratio must be at least 1.0".to_string(),
        ));
    }
    if config.search_beam_width < 2 {
        return Err(CalibrationError::EnsembleSelectionFailed(
            "search_beam_width must be at least 2".to_string(),
        ));
    }
    let (min_size, max_size) =
        problem.size_range(&config.size_mode, problem.candidate_count().min(25))?;
    let best_index = problem
        .losses
        .iter()
        .enumerate()
        .min_by(|(a_index, a), (b_index, b)| {
            a.partial_cmp(b)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a_index.cmp(b_index))
        })
        .map(|(index, _)| index)
        .ok_or_else(|| {
            CalibrationError::EnsembleSelectionFailed("No candidates available".to_string())
        })?;
    let central_limit = central_loss_limit(problem, config.central_fit_max_loss_ratio);
    let initial = vec![best_index];
    let initial_metrics = problem.metrics(&initial);
    let mut metrics_cache = HashMap::<Vec<usize>, Metrics>::new();
    metrics_cache.insert(initial.clone(), initial_metrics);
    let mut beam = vec![initial];
    let mut feasible_ensembles: Vec<(Vec<usize>, Metrics)> = Vec::new();
    let mut max_feasible_size = usize::from(initial_metrics.central_loss <= central_limit);
    let mut best_explored_coverage = initial_metrics.coverage;
    let mut rejected_single_additions = 0usize;

    for size in 2..=max_size {
        let expanded = expand_states(&beam, problem.candidate_count());
        if expanded.is_empty() {
            break;
        }
        let mut level_feasible = Vec::new();
        let mut level_best_coverage: f64 = 0.0;
        for selected in &expanded {
            let metrics = *metrics_cache
                .entry(selected.clone())
                .or_insert_with(|| problem.metrics(selected));
            level_best_coverage = level_best_coverage.max(metrics.coverage);
            if metrics.central_loss <= central_limit {
                level_feasible.push((selected.clone(), metrics));
            } else if size == 2 {
                rejected_single_additions += 1;
            }
        }
        best_explored_coverage = best_explored_coverage.max(level_best_coverage);
        if !level_feasible.is_empty() {
            max_feasible_size = max_feasible_size.max(size);
            if size >= min_size {
                feasible_ensembles.extend(level_feasible);
            }
        }
        beam = prune_search_beam(
            &expanded,
            &metrics_cache,
            central_limit,
            config.search_beam_width,
        );
    }

    if feasible_ensembles.is_empty() {
        return Err(CalibrationError::EnsembleSelectionFailed(format!(
            "Unable to construct a fit-gated ensemble of at least {} members; maximum feasible size was {}",
            min_size, max_feasible_size
        )));
    }
    let (selected_indices, selected_metrics) =
        choose_feasible(&feasible_ensembles, &config.size_mode);
    Ok(EnsembleSelectionResult {
        selected_ensemble: selected_indices,
        pareto_front: Vec::new(),
        selected_pareto_index: None,
        ci_width: selected_metrics.standardized_ci_width,
        coverage: selected_metrics.coverage,
        diagnostics: diagnostics([
            ("n_candidates", problem.candidate_count() as f64),
            ("search_beam_width", config.search_beam_width as f64),
            ("central_loss_limit", central_limit),
            ("n_evaluated_subsets", metrics_cache.len() as f64),
            (
                "n_feasible_subsets",
                metrics_cache
                    .values()
                    .filter(|metrics| metrics.central_loss <= central_limit)
                    .count() as f64,
            ),
            ("max_feasible_ensemble_size", max_feasible_size as f64),
            (
                "n_single_additions_rejected_by_central_fit",
                rejected_single_additions as f64,
            ),
            ("best_explored_coverage", best_explored_coverage),
        ]),
    })
}

fn expand_states(beam: &[Vec<usize>], n_candidates: usize) -> HashSet<Vec<usize>> {
    let mut expanded = HashSet::new();
    for selected in beam {
        for candidate_index in 0..n_candidates {
            if selected.binary_search(&candidate_index).is_err() {
                let mut next = selected.clone();
                next.push(candidate_index);
                next.sort_unstable();
                expanded.insert(next);
            }
        }
    }
    expanded
}

fn prune_search_beam(
    candidates: &HashSet<Vec<usize>>,
    metrics_cache: &HashMap<Vec<usize>, Metrics>,
    central_limit: f64,
    beam_width: usize,
) -> Vec<Vec<usize>> {
    let mut feasible = Vec::new();
    let mut infeasible = Vec::new();
    for indices in candidates {
        let metrics = metrics_cache[indices];
        if metrics.central_loss <= central_limit {
            feasible.push(indices.clone());
        } else {
            infeasible.push(indices.clone());
        }
    }

    feasible.sort_by(|a, b| compare_feasible(a, b, metrics_cache));
    infeasible.sort_by(|a, b| compare_infeasible(a, b, metrics_cache, central_limit));
    let mut coverage_bridges = infeasible.clone();
    coverage_bridges.sort_by(|a, b| compare_coverage_bridge(a, b, metrics_cache, central_limit));

    let feasible_slots = feasible.len().min((beam_width / 2).max(1));
    let infeasible_slots = infeasible
        .len()
        .min(beam_width.saturating_sub(feasible_slots));
    let fit_bridge_slots = infeasible_slots.div_ceil(2);
    let mut selected_bridges = infeasible[..fit_bridge_slots].to_vec();
    let selected_bridge_set: HashSet<Vec<usize>> = selected_bridges.iter().cloned().collect();
    for indices in coverage_bridges {
        if selected_bridges.len() >= infeasible_slots {
            break;
        }
        if !selected_bridge_set.contains(&indices) {
            selected_bridges.push(indices);
        }
    }

    let mut selected = feasible[..feasible_slots].to_vec();
    selected.extend(selected_bridges);
    if selected.len() < beam_width {
        let remaining = beam_width - selected.len();
        selected.extend(
            feasible
                [feasible_slots..feasible_slots + remaining.min(feasible.len() - feasible_slots)]
                .iter()
                .cloned(),
        );
    }
    if selected.len() < beam_width {
        let remaining = beam_width - selected.len();
        selected.extend(
            infeasible[infeasible_slots
                ..infeasible_slots + remaining.min(infeasible.len() - infeasible_slots)]
                .iter()
                .cloned(),
        );
    }
    selected
}

fn compare_feasible(
    a: &[usize],
    b: &[usize],
    metrics_cache: &HashMap<Vec<usize>, Metrics>,
) -> Ordering {
    let a_metrics = metrics_cache[a];
    let b_metrics = metrics_cache[b];
    b_metrics
        .minimum_series_coverage
        .partial_cmp(&a_metrics.minimum_series_coverage)
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            b_metrics
                .coverage
                .partial_cmp(&a_metrics.coverage)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| {
            a_metrics
                .standardized_ci_width
                .partial_cmp(&b_metrics.standardized_ci_width)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| {
            a_metrics
                .central_loss
                .partial_cmp(&b_metrics.central_loss)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| a.cmp(b))
}

fn compare_infeasible(
    a: &[usize],
    b: &[usize],
    metrics_cache: &HashMap<Vec<usize>, Metrics>,
    central_limit: f64,
) -> Ordering {
    let a_metrics = metrics_cache[a];
    let b_metrics = metrics_cache[b];
    (a_metrics.central_loss - central_limit)
        .partial_cmp(&(b_metrics.central_loss - central_limit))
        .unwrap_or(Ordering::Equal)
        .then_with(|| compare_feasible(a, b, metrics_cache))
}

fn compare_coverage_bridge(
    a: &[usize],
    b: &[usize],
    metrics_cache: &HashMap<Vec<usize>, Metrics>,
    central_limit: f64,
) -> Ordering {
    let a_metrics = metrics_cache[a];
    let b_metrics = metrics_cache[b];
    b_metrics
        .minimum_series_coverage
        .partial_cmp(&a_metrics.minimum_series_coverage)
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            b_metrics
                .coverage
                .partial_cmp(&a_metrics.coverage)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| {
            a_metrics
                .standardized_ci_width
                .partial_cmp(&b_metrics.standardized_ci_width)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| {
            (a_metrics.central_loss - central_limit)
                .partial_cmp(&(b_metrics.central_loss - central_limit))
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| a.cmp(b))
}

fn choose_feasible(
    feasible: &[(Vec<usize>, Metrics)],
    size_mode: &EnsembleSizeMode,
) -> (Vec<usize>, Metrics) {
    feasible
        .iter()
        .min_by(|(a_indices, a_metrics), (b_indices, b_metrics)| {
            b_metrics
                .minimum_series_coverage
                .partial_cmp(&a_metrics.minimum_series_coverage)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    b_metrics
                        .coverage
                        .partial_cmp(&a_metrics.coverage)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| {
                    a_metrics
                        .standardized_ci_width
                        .partial_cmp(&b_metrics.standardized_ci_width)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| {
                    a_metrics
                        .central_loss
                        .partial_cmp(&b_metrics.central_loss)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| {
                    if matches!(size_mode, EnsembleSizeMode::Fixed { .. }) {
                        Ordering::Equal
                    } else {
                        a_indices.len().cmp(&b_indices.len())
                    }
                })
                .then_with(|| a_indices.cmp(b_indices))
        })
        .map(|(indices, metrics)| (indices.clone(), *metrics))
        .expect("feasible ensemble list must not be empty")
}
