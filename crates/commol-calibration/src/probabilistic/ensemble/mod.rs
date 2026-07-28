//! Ensemble selection on compact, observation-space candidate predictions.
//!
//! The Python probabilistic workflow transforms each representative into one
//! prediction value per observed quantity. Keeping both selection algorithms
//! on that same representation avoids objective drift between the Rust and
//! Python implementations.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::types::CalibrationEvaluation;

use super::error::{CalibrationError, CalibrationResult};

mod greedy;
mod nsga2;

/// Algorithm used for final ensemble subset selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleAlgorithm {
    /// Fit-gated bridge-aware beam search.
    GreedyLocalSearch,
    /// Multi-objective evolutionary search.
    Nsga2,
}

/// Valid ensemble-size constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnsembleSizeMode {
    /// Select exactly `size` candidates.
    Fixed { size: usize },
    /// Select between `min` and `max` candidates.
    Bounded { min: usize, max: usize },
    /// Let the selected algorithm choose a size between two and all candidates.
    Automatic,
}

/// Loss metric used to score the ensemble median prediction (`central_loss`).
///
/// It mirrors the optimizer's loss so the central-fit gate compares two numbers
/// measured the same way. Each variant reproduces the corresponding
/// `LossConfig` formula from the calibration problem: the sum-of-squares
/// variants weight residuals, while RMSE and MAE ignore observation weights,
/// exactly as the optimizer does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CentralLossMetric {
    /// Weighted sum of squared residuals (matches `SumSquaredError`/`WeightedSSE`).
    WeightedSumOfSquares,
    /// Root mean squared residual, ignoring weights (matches `RootMeanSquaredError`).
    RootMeanSquared,
    /// Mean absolute residual, ignoring weights (matches `MeanAbsoluteError`).
    MeanAbsolute,
}

/// Configuration shared by the Rust selection backends.
#[derive(Debug, Clone)]
pub struct EnsembleSelectionConfig {
    pub algorithm: EnsembleAlgorithm,
    pub confidence_level: f64,
    pub seed: u64,
    pub size_mode: EnsembleSizeMode,
    pub central_loss_metric: CentralLossMetric,
    pub central_fit_max_loss_ratio: f64,
    pub search_beam_width: usize,
    pub population_size: usize,
    pub generations: usize,
    pub crossover_probability: f64,
    pub pareto_preference: f64,
}

/// A compact Pareto-front solution.
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    pub ensemble_size: usize,
    pub ci_width: f64,
    pub coverage: f64,
    pub central_loss: f64,
    pub selected_indices: Vec<usize>,
}

/// Result returned by either selection backend.
#[derive(Debug, Clone)]
pub struct EnsembleSelectionResult {
    pub selected_ensemble: Vec<usize>,
    pub pareto_front: Vec<ParetoSolution>,
    pub selected_pareto_index: Option<usize>,
    pub ci_width: f64,
    pub coverage: f64,
    pub diagnostics: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    central_loss: f64,
    coverage: f64,
    minimum_series_coverage: f64,
    standardized_ci_width: f64,
}

impl Metrics {
    fn objective(self) -> [f64; 2] {
        [self.standardized_ci_width, 1.0 - self.coverage]
    }
}

struct CompactSelectionProblem {
    /// Candidate-major prediction matrix: candidate -> observed quantity.
    predictions: Vec<Vec<f64>>,
    losses: Vec<f64>,
    observed_values: Vec<f64>,
    /// Raw per-observation weights (applied only by the weighted-SSE metric).
    weights: Vec<f64>,
    /// Per-observation normalization factors (applied by every metric).
    normalization: Vec<f64>,
    central_loss_metric: CentralLossMetric,
    series_groups: Vec<usize>,
    series_counts: Vec<usize>,
    lower_percentile: f64,
    upper_percentile: f64,
}

impl CompactSelectionProblem {
    fn new(
        candidates: Vec<CalibrationEvaluation>,
        observed_values: Vec<f64>,
        weights: Vec<f64>,
        normalization: Vec<f64>,
        series_ids: Vec<String>,
        confidence_level: f64,
        central_loss_metric: CentralLossMetric,
    ) -> CalibrationResult<Self> {
        if candidates.len() < 2 {
            return Err(CalibrationError::EnsembleSelectionFailed(format!(
                "At least two candidates are required, got {}",
                candidates.len()
            )));
        }
        if observed_values.is_empty() {
            return Err(CalibrationError::EnsembleSelectionFailed(
                "At least one observed quantity is required".to_string(),
            ));
        }
        if weights.len() != observed_values.len() {
            return Err(CalibrationError::EnsembleSelectionFailed(format!(
                "Observation weights ({}) do not match observed values ({})",
                weights.len(),
                observed_values.len()
            )));
        }
        if normalization.len() != observed_values.len() {
            return Err(CalibrationError::EnsembleSelectionFailed(format!(
                "Normalization factors ({}) do not match observed values ({})",
                normalization.len(),
                observed_values.len()
            )));
        }
        if series_ids.len() != observed_values.len() {
            return Err(CalibrationError::EnsembleSelectionFailed(format!(
                "Series identifiers ({}) do not match observed values ({})",
                series_ids.len(),
                observed_values.len()
            )));
        }
        if !(0.0..1.0).contains(&confidence_level) {
            return Err(CalibrationError::EnsembleSelectionFailed(format!(
                "Confidence level must be in (0, 1), got {}",
                confidence_level
            )));
        }

        let n_observations = observed_values.len();
        if observed_values.iter().any(|value| !value.is_finite()) {
            return Err(CalibrationError::EnsembleSelectionFailed(
                "Observed values must be finite".to_string(),
            ));
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(CalibrationError::EnsembleSelectionFailed(
                "Observation weights must be finite".to_string(),
            ));
        }
        if normalization.iter().any(|factor| !factor.is_finite()) {
            return Err(CalibrationError::EnsembleSelectionFailed(
                "Normalization factors must be finite".to_string(),
            ));
        }
        let mut predictions = Vec::with_capacity(candidates.len());
        let mut losses = Vec::with_capacity(candidates.len());
        for (candidate_index, candidate) in candidates.into_iter().enumerate() {
            if !candidate.loss.is_finite() {
                return Err(CalibrationError::EnsembleSelectionFailed(format!(
                    "Candidate {} has a non-finite loss",
                    candidate_index
                )));
            }
            if candidate.predictions.len() != n_observations {
                return Err(CalibrationError::EnsembleSelectionFailed(format!(
                    "Candidate {} has {} compact predictions, expected {}",
                    candidate_index,
                    candidate.predictions.len(),
                    n_observations
                )));
            }
            let mut compact = Vec::with_capacity(n_observations);
            for (observation_index, row) in candidate.predictions.into_iter().enumerate() {
                let Some(value) = row.first().copied() else {
                    return Err(CalibrationError::EnsembleSelectionFailed(format!(
                        "Candidate {} has an empty compact prediction at observation {}",
                        candidate_index, observation_index
                    )));
                };
                if !value.is_finite() {
                    return Err(CalibrationError::EnsembleSelectionFailed(format!(
                        "Candidate {} has a non-finite prediction at observation {}",
                        candidate_index, observation_index
                    )));
                }
                compact.push(value);
            }
            predictions.push(compact);
            losses.push(candidate.loss);
        }

        let mut series_lookup = HashMap::<String, usize>::new();
        let mut series_groups = Vec::with_capacity(series_ids.len());
        for series_id in series_ids {
            let next_group = series_lookup.len();
            let group = *series_lookup.entry(series_id).or_insert(next_group);
            series_groups.push(group);
        }
        let mut series_counts = vec![0usize; series_lookup.len()];
        for &group in &series_groups {
            series_counts[group] += 1;
        }

        Ok(Self {
            predictions,
            losses,
            observed_values,
            weights,
            normalization,
            central_loss_metric,
            series_groups,
            series_counts,
            lower_percentile: (1.0 - confidence_level) * 50.0,
            upper_percentile: (1.0 + confidence_level) * 50.0,
        })
    }

    fn candidate_count(&self) -> usize {
        self.predictions.len()
    }

    fn metrics(&self, selected_indices: &[usize]) -> Metrics {
        let mut covered = 0usize;
        let mut covered_by_series = vec![0usize; self.series_counts.len()];
        let mut total_width = 0.0;
        // Accumulator interpreted by `central_loss_metric`: a running sum of
        // squared residuals for the SSE/RMSE metrics, or of absolute residuals
        // for MAE. It is reduced to a scalar loss after the loop.
        let mut central_accumulator = 0.0;

        for observation_index in 0..self.observed_values.len() {
            let values: Vec<f64> = selected_indices
                .iter()
                .map(|&candidate_index| self.predictions[candidate_index][observation_index])
                .collect();
            let lower = percentile(&values, self.lower_percentile);
            let upper = percentile(&values, self.upper_percentile);
            let median = percentile(&values, 50.0);
            let observed = self.observed_values[observation_index];
            let is_covered = lower <= observed && observed <= upper;
            if is_covered {
                covered += 1;
                covered_by_series[self.series_groups[observation_index]] += 1;
            }
            let normalizer = observed.abs().max(1.0);
            total_width += (upper - lower) / normalizer;

            let normalization = self.normalization[observation_index];
            let residual = median - observed;
            central_accumulator += match self.central_loss_metric {
                CentralLossMetric::WeightedSumOfSquares => {
                    let error = self.weights[observation_index] * normalization * residual;
                    error * error
                }
                CentralLossMetric::RootMeanSquared => {
                    let error = normalization * residual;
                    error * error
                }
                CentralLossMetric::MeanAbsolute => (normalization * residual).abs(),
            };
        }

        let n_observations = self.observed_values.len() as f64;
        let central_loss = match self.central_loss_metric {
            CentralLossMetric::WeightedSumOfSquares => central_accumulator,
            CentralLossMetric::RootMeanSquared => (central_accumulator / n_observations).sqrt(),
            CentralLossMetric::MeanAbsolute => central_accumulator / n_observations,
        };

        let coverage = covered as f64 / n_observations;
        let minimum_series_coverage = covered_by_series
            .iter()
            .zip(&self.series_counts)
            .map(|(&covered_count, &series_count)| covered_count as f64 / series_count as f64)
            .fold(1.0, f64::min);

        Metrics {
            central_loss,
            coverage,
            minimum_series_coverage,
            standardized_ci_width: total_width / n_observations,
        }
    }

    fn size_range(
        &self,
        mode: &EnsembleSizeMode,
        automatic_max: usize,
    ) -> CalibrationResult<(usize, usize)> {
        let (min_size, max_size) = match mode {
            EnsembleSizeMode::Fixed { size } => (*size, *size),
            EnsembleSizeMode::Bounded { min, max } => (*min, *max),
            EnsembleSizeMode::Automatic => (2, automatic_max),
        };
        if min_size < 2 {
            return Err(CalibrationError::EnsembleSelectionFailed(format!(
                "Minimum ensemble size must be at least 2, got {}",
                min_size
            )));
        }
        if max_size < min_size {
            return Err(CalibrationError::EnsembleSelectionFailed(format!(
                "Maximum ensemble size ({}) must be at least the minimum ({})",
                max_size, min_size
            )));
        }
        if min_size > self.candidate_count() {
            return Err(CalibrationError::EnsembleSelectionFailed(format!(
                "Minimum ensemble size ({}) exceeds candidate count ({})",
                min_size,
                self.candidate_count()
            )));
        }
        Ok((min_size, max_size.min(self.candidate_count())))
    }
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = (percentile / 100.0).clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
    }
}

fn central_loss_limit(problem: &CompactSelectionProblem, ratio: f64) -> f64 {
    let best_loss = problem.losses.iter().copied().fold(f64::INFINITY, f64::min);
    best_loss * ratio + f64::EPSILON
}

fn diagnostics(entries: impl IntoIterator<Item = (&'static str, f64)>) -> Vec<(String, f64)> {
    entries
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

/// Select an ensemble using the configured Rust backend.
pub fn select_compact_ensemble(
    candidates: Vec<CalibrationEvaluation>,
    observed_values: Vec<f64>,
    weights: Vec<f64>,
    normalization: Vec<f64>,
    series_ids: Vec<String>,
    config: &EnsembleSelectionConfig,
) -> CalibrationResult<EnsembleSelectionResult> {
    let problem = CompactSelectionProblem::new(
        candidates,
        observed_values,
        weights,
        normalization,
        series_ids,
        config.confidence_level,
        config.central_loss_metric,
    )?;

    // Each backend validates only the parameters it actually consumes, so a
    // caller running one algorithm never has to satisfy the other's
    // constraints (see `greedy::select_greedy` and `nsga2::select_nsga2`).
    match config.algorithm {
        EnsembleAlgorithm::GreedyLocalSearch => greedy::select_greedy(&problem, config),
        EnsembleAlgorithm::Nsga2 => nsga2::select_nsga2(&problem, config),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(loss: f64, values: &[f64]) -> CalibrationEvaluation {
        CalibrationEvaluation {
            parameters: vec![loss],
            loss,
            predictions: values.iter().map(|value| vec![*value]).collect(),
        }
    }

    fn config(algorithm: EnsembleAlgorithm) -> EnsembleSelectionConfig {
        EnsembleSelectionConfig {
            algorithm,
            confidence_level: 0.95,
            seed: 7,
            size_mode: EnsembleSizeMode::Fixed { size: 2 },
            central_loss_metric: CentralLossMetric::WeightedSumOfSquares,
            central_fit_max_loss_ratio: 2.0,
            search_beam_width: 8,
            population_size: 12,
            generations: 10,
            crossover_probability: 0.9,
            pareto_preference: 0.5,
        }
    }

    #[test]
    fn greedy_selection_respects_fixed_size_and_fit_gate() {
        let candidates = vec![
            candidate(1.0, &[10.0, 10.0]),
            candidate(1.1, &[9.0, 11.0]),
            candidate(4.0, &[1.0, 20.0]),
        ];
        let result = select_compact_ensemble(
            candidates,
            vec![10.0, 10.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec!["series".to_string(), "series".to_string()],
            &config(EnsembleAlgorithm::GreedyLocalSearch),
        )
        .expect("greedy selection should succeed");
        assert_eq!(result.selected_ensemble.len(), 2);
        assert!(result.pareto_front.is_empty());
    }

    #[test]
    fn nsga2_returns_a_pareto_solution() {
        let candidates = vec![
            candidate(1.0, &[10.0, 10.0]),
            candidate(1.1, &[9.0, 11.0]),
            candidate(1.2, &[8.0, 12.0]),
            candidate(2.0, &[7.0, 13.0]),
        ];
        let result = select_compact_ensemble(
            candidates,
            vec![10.0, 10.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec!["series".to_string(), "series".to_string()],
            &config(EnsembleAlgorithm::Nsga2),
        )
        .expect("NSGA-II selection should succeed");
        assert_eq!(result.selected_ensemble.len(), 2);
        assert!(!result.pareto_front.is_empty());
        assert!(result.selected_pareto_index.is_some());
    }

    #[test]
    fn greedy_ignores_nsga2_only_parameters() {
        // NSGA-II-only knobs are invalid, but the greedy backend must not
        // validate them (C-2: each backend validates only what it consumes).
        let mut greedy = config(EnsembleAlgorithm::GreedyLocalSearch);
        greedy.population_size = 0;
        greedy.generations = 0;
        greedy.crossover_probability = 5.0;
        greedy.pareto_preference = -1.0;
        let candidates = vec![candidate(1.0, &[10.0, 10.0]), candidate(1.1, &[9.0, 11.0])];
        let result = select_compact_ensemble(
            candidates,
            vec![10.0, 10.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec!["series".to_string(), "series".to_string()],
            &greedy,
        )
        .expect("greedy selection should ignore NSGA-II-only parameters");
        assert_eq!(result.selected_ensemble.len(), 2);
    }

    #[test]
    fn nsga2_ignores_greedy_only_parameters() {
        // Greedy-only knobs are invalid, but the NSGA-II backend must not
        // validate them.
        let mut nsga2 = config(EnsembleAlgorithm::Nsga2);
        nsga2.search_beam_width = 0;
        nsga2.central_fit_max_loss_ratio = 0.5;
        let candidates = vec![
            candidate(1.0, &[10.0, 10.0]),
            candidate(1.1, &[9.0, 11.0]),
            candidate(1.2, &[8.0, 12.0]),
            candidate(2.0, &[7.0, 13.0]),
        ];
        let result = select_compact_ensemble(
            candidates,
            vec![10.0, 10.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec!["series".to_string(), "series".to_string()],
            &nsga2,
        )
        .expect("NSGA-II selection should ignore greedy-only parameters");
        assert_eq!(result.selected_ensemble.len(), 2);
    }

    #[test]
    fn central_loss_matches_configured_metric() {
        // Members [8, 8] and [12, 12] give a per-observation median of 10; with
        // observed values of 7 the residual is 3 at both observations.
        let make = || vec![candidate(1.0, &[8.0, 8.0]), candidate(1.0, &[12.0, 12.0])];
        let observed = vec![7.0, 7.0];
        let weights = vec![1.0, 1.0];
        let normalization = vec![1.0, 1.0];
        let series = || vec!["s".to_string(), "s".to_string()];
        let indices = [0usize, 1usize];

        let problem = |metric| {
            CompactSelectionProblem::new(
                make(),
                observed.clone(),
                weights.clone(),
                normalization.clone(),
                series(),
                0.95,
                metric,
            )
            .expect("problem construction should succeed")
        };

        let sse = problem(CentralLossMetric::WeightedSumOfSquares)
            .metrics(&indices)
            .central_loss;
        let rmse = problem(CentralLossMetric::RootMeanSquared)
            .metrics(&indices)
            .central_loss;
        let mae = problem(CentralLossMetric::MeanAbsolute)
            .metrics(&indices)
            .central_loss;

        assert!((sse - 18.0).abs() < 1e-9); // 3^2 + 3^2
        assert!((rmse - 3.0).abs() < 1e-9); // sqrt((9 + 9) / 2)
        assert!((mae - 3.0).abs() < 1e-9); // (3 + 3) / 2
    }
}
