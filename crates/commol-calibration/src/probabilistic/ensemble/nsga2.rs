//! NSGA-II ensemble selection.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::HashSet;

use super::{
    diagnostics, CompactSelectionProblem, EnsembleSelectionConfig, EnsembleSelectionResult,
    Metrics, ParetoSolution,
};
use crate::probabilistic::error::{CalibrationError, CalibrationResult};

#[derive(Clone)]
struct Individual {
    genes: Vec<bool>,
    indices: Vec<usize>,
    metrics: Metrics,
    rank: usize,
    crowding: f64,
}

pub(super) fn select_nsga2(
    problem: &CompactSelectionProblem,
    config: &EnsembleSelectionConfig,
) -> CalibrationResult<EnsembleSelectionResult> {
    if config.population_size < 4 || config.generations == 0 {
        return Err(CalibrationError::EnsembleSelectionFailed(
            "NSGA-II population_size must be at least 4 and generations must be positive"
                .to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&config.crossover_probability)
        || !(0.0..=1.0).contains(&config.pareto_preference)
    {
        return Err(CalibrationError::EnsembleSelectionFailed(
            "crossover_probability and pareto_preference must be in [0, 1]".to_string(),
        ));
    }
    let (min_size, max_size) = problem.size_range(&config.size_mode, problem.candidate_count())?;
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut population = initialize_population(
        problem,
        min_size,
        max_size,
        config.population_size,
        &mut rng,
    );

    for _ in 0..config.generations {
        assign_rank_and_crowding(&mut population);
        let mut offspring = Vec::with_capacity(config.population_size);
        while offspring.len() < config.population_size {
            let first = tournament_index(&population, &mut rng);
            let second = tournament_index(&population, &mut rng);
            let (mut first_genes, mut second_genes) = crossover(
                &population[first].genes,
                &population[second].genes,
                config.crossover_probability,
                &mut rng,
            );
            mutate(&mut first_genes, problem.candidate_count(), &mut rng);
            mutate(&mut second_genes, problem.candidate_count(), &mut rng);
            repair_genes(&mut first_genes, min_size, max_size, &mut rng);
            repair_genes(&mut second_genes, min_size, max_size, &mut rng);
            offspring.push(make_individual(problem, first_genes));
            if offspring.len() < config.population_size {
                offspring.push(make_individual(problem, second_genes));
            }
        }

        population.extend(offspring);
        assign_rank_and_crowding(&mut population);
        population = next_generation(population, config.population_size);
    }

    assign_rank_and_crowding(&mut population);
    let pareto_front = unique_pareto_front(population);
    if pareto_front.is_empty() {
        return Err(CalibrationError::EnsembleSelectionFailed(
            "NSGA-II produced an empty feasible Pareto front".to_string(),
        ));
    }
    let selected_pareto_index = select_pareto_by_preference(
        &pareto_front,
        config.pareto_preference,
        config.confidence_level,
    );
    let pareto_count = pareto_front.len() as f64;
    let selected = pareto_front[selected_pareto_index].clone();
    let max_explored_size = pareto_front
        .iter()
        .map(|solution| solution.ensemble_size)
        .max()
        .unwrap_or(0);

    Ok(EnsembleSelectionResult {
        selected_ensemble: selected.selected_indices.clone(),
        ci_width: selected.ci_width,
        coverage: selected.coverage,
        selected_pareto_index: Some(selected_pareto_index),
        pareto_front,
        diagnostics: diagnostics([
            ("n_candidates", problem.candidate_count() as f64),
            ("population_size", config.population_size as f64),
            ("generations", config.generations as f64),
            ("n_pareto_solutions", pareto_count),
            ("max_explored_ensemble_size", max_explored_size as f64),
            ("max_feasible_ensemble_size", max_explored_size as f64),
            ("selected_central_loss", selected.central_loss),
        ]),
    })
}

fn initialize_population(
    problem: &CompactSelectionProblem,
    min_size: usize,
    max_size: usize,
    population_size: usize,
    rng: &mut SmallRng,
) -> Vec<Individual> {
    let mut population = Vec::with_capacity(population_size);
    let mut seen = HashSet::<Vec<usize>>::new();
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
        .unwrap_or(0);

    let mut best_genes = vec![false; problem.candidate_count()];
    best_genes[best_index] = true;
    repair_genes(&mut best_genes, min_size, max_size, rng);
    seen.insert(indices_from_genes(&best_genes));
    population.push(make_individual(problem, best_genes));

    let mut attempts = 0usize;
    while population.len() < population_size {
        attempts = attempts.saturating_add(1);
        let target_size = if min_size == max_size {
            min_size
        } else {
            rng.random_range(min_size..=max_size)
        };
        let mut genes = vec![false; problem.candidate_count()];
        while genes.iter().filter(|&&selected| selected).count() < target_size {
            genes[rng.random_range(0..problem.candidate_count())] = true;
        }
        let indices = indices_from_genes(&genes);
        // Small candidate pools can have fewer unique subsets than the
        // requested population. Fill the remaining slots with duplicates
        // after a bounded number of attempts instead of looping forever.
        if seen.insert(indices) || attempts > population_size.saturating_mul(10).max(10) {
            population.push(make_individual(problem, genes));
        }
    }
    population
}

fn make_individual(problem: &CompactSelectionProblem, genes: Vec<bool>) -> Individual {
    let indices = indices_from_genes(&genes);
    let metrics = problem.metrics(&indices);
    Individual {
        genes,
        indices,
        metrics,
        rank: 0,
        crowding: 0.0,
    }
}

fn indices_from_genes(genes: &[bool]) -> Vec<usize> {
    genes
        .iter()
        .enumerate()
        .filter_map(|(index, &selected)| selected.then_some(index))
        .collect()
}

fn repair_genes(genes: &mut [bool], min_size: usize, max_size: usize, rng: &mut SmallRng) {
    let mut selected = indices_from_genes(genes);
    while selected.len() > max_size {
        let position = rng.random_range(0..selected.len());
        let index = selected.swap_remove(position);
        genes[index] = false;
    }
    while selected.len() < min_size {
        let index = rng.random_range(0..genes.len());
        if !genes[index] {
            genes[index] = true;
            selected.push(index);
        }
    }
}

fn crossover(
    first: &[bool],
    second: &[bool],
    probability: f64,
    rng: &mut SmallRng,
) -> (Vec<bool>, Vec<bool>) {
    if first.len() < 2 || rng.random::<f64>() >= probability {
        return (first.to_vec(), second.to_vec());
    }
    let split = rng.random_range(1..first.len());
    let mut first_child = first[..split].to_vec();
    first_child.extend_from_slice(&second[split..]);
    let mut second_child = second[..split].to_vec();
    second_child.extend_from_slice(&first[split..]);
    (first_child, second_child)
}

fn mutate(genes: &mut [bool], n_candidates: usize, rng: &mut SmallRng) {
    let probability = 1.0 / n_candidates as f64;
    for gene in genes {
        if rng.random::<f64>() < probability {
            *gene = !*gene;
        }
    }
}

fn dominates(first: &Individual, second: &Individual) -> bool {
    let first_objectives = first.metrics.objective();
    let second_objectives = second.metrics.objective();
    let no_worse = first_objectives
        .iter()
        .zip(second_objectives.iter())
        .all(|(first, second)| first <= second);
    let strictly_better = first_objectives
        .iter()
        .zip(second_objectives.iter())
        .any(|(first, second)| first < second);
    no_worse && strictly_better
}

fn assign_rank_and_crowding(population: &mut [Individual]) -> Vec<Vec<usize>> {
    let count = population.len();
    let mut dominated = vec![Vec::<usize>::new(); count];
    let mut domination_counts = vec![0usize; count];
    let mut fronts = Vec::<Vec<usize>>::new();

    for first in 0..count {
        for second in (first + 1)..count {
            if dominates(&population[first], &population[second]) {
                dominated[first].push(second);
                domination_counts[second] += 1;
            } else if dominates(&population[second], &population[first]) {
                dominated[second].push(first);
                domination_counts[first] += 1;
            }
        }
        if domination_counts[first] == 0 {
            population[first].rank = 0;
        }
    }

    let first_front: Vec<usize> = (0..count)
        .filter(|&index| domination_counts[index] == 0)
        .collect();
    fronts.push(first_front);
    let mut front_index = 0;
    while front_index < fronts.len() && !fronts[front_index].is_empty() {
        let mut next = Vec::new();
        for &first in &fronts[front_index] {
            for &second in &dominated[first] {
                domination_counts[second] -= 1;
                if domination_counts[second] == 0 {
                    population[second].rank = front_index + 1;
                    next.push(second);
                }
            }
        }
        if !next.is_empty() {
            fronts.push(next);
        }
        front_index += 1;
    }

    for front in &fronts {
        assign_crowding(population, front);
    }
    fronts
}

fn assign_crowding(population: &mut [Individual], front: &[usize]) {
    for &index in front {
        population[index].crowding = 0.0;
    }
    if front.len() <= 2 {
        for &index in front {
            population[index].crowding = f64::INFINITY;
        }
        return;
    }
    for objective_index in 0..2 {
        let mut ordered = front.to_vec();
        ordered.sort_by(|&a, &b| {
            population[a].metrics.objective()[objective_index]
                .partial_cmp(&population[b].metrics.objective()[objective_index])
                .unwrap_or(Ordering::Equal)
        });
        let first_value = population[ordered[0]].metrics.objective()[objective_index];
        let last_value = population[*ordered.last().unwrap()].metrics.objective()[objective_index];
        population[ordered[0]].crowding = f64::INFINITY;
        population[*ordered.last().unwrap()].crowding = f64::INFINITY;
        let range = last_value - first_value;
        if range <= f64::EPSILON {
            continue;
        }
        for window in ordered.windows(3) {
            let previous = population[window[0]].metrics.objective()[objective_index];
            let next = population[window[2]].metrics.objective()[objective_index];
            if population[window[1]].crowding.is_finite() {
                population[window[1]].crowding += (next - previous) / range;
            }
        }
    }
}

fn tournament_index(population: &[Individual], rng: &mut SmallRng) -> usize {
    let first = rng.random_range(0..population.len());
    let second = rng.random_range(0..population.len());
    let first_individual = &population[first];
    let second_individual = &population[second];
    if first_individual.rank < second_individual.rank
        || (first_individual.rank == second_individual.rank
            && first_individual.crowding > second_individual.crowding)
    {
        first
    } else {
        second
    }
}

fn next_generation(mut population: Vec<Individual>, population_size: usize) -> Vec<Individual> {
    let fronts = assign_rank_and_crowding(&mut population);
    let mut next = Vec::with_capacity(population_size);
    for front in fronts {
        if next.len() + front.len() <= population_size {
            next.extend(front.into_iter().map(|index| population[index].clone()));
        } else {
            let mut remaining: Vec<Individual> = front
                .into_iter()
                .map(|index| population[index].clone())
                .collect();
            remaining.sort_by(|a, b| {
                b.crowding
                    .partial_cmp(&a.crowding)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.indices.cmp(&b.indices))
            });
            next.extend(remaining.into_iter().take(population_size - next.len()));
            break;
        }
    }
    next
}

fn unique_pareto_front(population: Vec<Individual>) -> Vec<ParetoSolution> {
    let mut front = Vec::<Individual>::new();
    for candidate in population {
        if front.iter().any(|existing| {
            existing.indices == candidate.indices || dominates(existing, &candidate)
        }) {
            continue;
        }
        front.retain(|existing| !dominates(&candidate, existing));
        front.push(candidate);
    }
    front.sort_by(|a, b| {
        a.metrics
            .standardized_ci_width
            .partial_cmp(&b.metrics.standardized_ci_width)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b.metrics
                    .coverage
                    .partial_cmp(&a.metrics.coverage)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| a.indices.cmp(&b.indices))
    });
    front
        .into_iter()
        .map(|individual| ParetoSolution {
            ensemble_size: individual.indices.len(),
            ci_width: individual.metrics.standardized_ci_width,
            coverage: individual.metrics.coverage,
            central_loss: individual.metrics.central_loss,
            selected_indices: individual.indices,
        })
        .collect()
}

fn select_pareto_by_preference(
    pareto_front: &[ParetoSolution],
    preference: f64,
    confidence_level: f64,
) -> usize {
    if pareto_front.len() <= 1 {
        return 0;
    }
    if preference <= 0.05 {
        return pareto_front
            .iter()
            .enumerate()
            .min_by(|(a_index, a), (b_index, b)| {
                a.ci_width
                    .partial_cmp(&b.ci_width)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a_index.cmp(b_index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
    }
    if preference >= 0.95 {
        return pareto_front
            .iter()
            .enumerate()
            .max_by(|(a_index, a), (b_index, b)| {
                a.coverage
                    .partial_cmp(&b.coverage)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| b_index.cmp(a_index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
    }

    let max_coverage = pareto_front
        .iter()
        .map(|solution| solution.coverage)
        .fold(f64::NEG_INFINITY, f64::max);
    let coverage_floor = confidence_level.min(max_coverage);
    let eligible: Vec<usize> = pareto_front
        .iter()
        .enumerate()
        .filter_map(|(index, solution)| {
            (solution.coverage + f64::EPSILON >= coverage_floor).then_some(index)
        })
        .collect();
    let (min_ci, max_ci) = pareto_front.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min_value, max_value), solution| {
            (
                min_value.min(solution.ci_width),
                max_value.max(solution.ci_width),
            )
        },
    );
    let (min_neg_coverage, max_neg_coverage) = pareto_front.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min_value, max_value), solution| {
            let value = 1.0 - solution.coverage;
            (min_value.min(value), max_value.max(value))
        },
    );
    let ci_range = max_ci - min_ci;
    let coverage_range = max_neg_coverage - min_neg_coverage;
    let weight = 1.0 - preference;

    eligible
        .into_iter()
        .min_by(|&a, &b| {
            let score = |index: usize| {
                let solution = &pareto_front[index];
                let normalized_ci = if ci_range > f64::EPSILON {
                    (solution.ci_width - min_ci) / ci_range
                } else {
                    0.0
                };
                let neg_coverage = 1.0 - solution.coverage;
                let normalized_coverage = if coverage_range > f64::EPSILON {
                    (neg_coverage - min_neg_coverage) / coverage_range
                } else {
                    0.0
                };
                weight * normalized_ci + (1.0 - weight) * normalized_coverage
            };
            score(a)
                .partial_cmp(&score(b))
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        })
        .unwrap_or(0)
}
