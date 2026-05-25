//! Ensemble selection objective problem.
//!
//! This module defines the multi-objective optimization problem for selecting
//! an ensemble of parameter sets that balances narrow confidence intervals
//! with good coverage of observed data.

use argmin::core::{CostFunction, MultiObjectiveCostFunction};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use super::size_mode::EnsembleSizeMode;
use crate::probabilistic::config::{CIWidthScope, EnsembleSelectionConfig};
use crate::probabilistic::utils::percentile_unstable;
use crate::types::CalibrationEvaluation;

mod incremental;
#[cfg(test)]
mod tests;

thread_local! {
    static PERCENTILE_SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

static NEXT_PROBLEM_ID: AtomicU64 = AtomicU64::new(1);

/// Ensemble selection objective problem.
///
/// This multi-objective optimization problem selects an ensemble of parameter sets
/// that balances:
/// 1. Narrow confidence intervals (minimize CI width)
/// 2. Good coverage of observed data (maximize coverage)
/// 3. Ensemble size constraints (penalty for violating size constraints)
pub(crate) struct EnsembleSelectionProblem {
    /// Candidate parameter sets to choose from.
    pub candidates: Vec<CalibrationEvaluation>,

    /// Point-major prediction values for CI width:
    /// metric point -> candidate value.
    ci_point_predictions: Vec<Vec<f64>>,

    /// Point-major prediction values for observed coverage:
    /// observed point -> candidate value.
    coverage_point_predictions: Vec<Vec<f64>>,

    /// Observed data points: (time_step, compartment_idx, value).
    observed_values: Vec<f64>,

    /// Lower percentile for CI calculation (e.g., 2.5 for 95% CI).
    lower_percentile: f64,

    /// Upper percentile for CI calculation (e.g., 97.5 for 95% CI).
    upper_percentile: f64,

    /// Normalization bounds for CI width objective.
    min_ci_width: f64,
    max_ci_width: f64,

    /// Ensemble size constraint mode.
    size_mode: EnsembleSizeMode,

    /// Minimum metric points before percentile evaluation uses Rayon.
    parallel_objective_threshold: usize,

    /// Identifier used to detect stale entries in the per-thread incremental
    /// evaluation cache. Each problem instance gets a unique non-zero id.
    problem_id: u64,
}

impl EnsembleSelectionProblem {
    pub fn new(
        candidates: Vec<CalibrationEvaluation>,
        observed_data: Vec<(usize, usize, f64)>,
        confidence_level: f64,
        size_mode: EnsembleSizeMode,
        config: &EnsembleSelectionConfig,
    ) -> Self {
        let lower_percentile = (1.0 - confidence_level) / 2.0 * 100.0;
        let upper_percentile = (1.0 + confidence_level) / 2.0 * 100.0;

        let ci_point_indices =
            Self::ci_point_indices(&candidates, &observed_data, config.ci_width_scope);
        let coverage_point_indices: Vec<usize> = observed_data
            .iter()
            .map(|&(time_step, compartment_idx, _)| {
                Self::point_index(&candidates, time_step, compartment_idx)
            })
            .collect();
        let ci_point_predictions = Self::point_major_predictions(&candidates, &ci_point_indices);
        let coverage_point_predictions =
            Self::point_major_predictions(&candidates, &coverage_point_indices);
        let observed_values = observed_data.iter().map(|&(_, _, value)| value).collect();

        // Compute normalization bounds for CI width.
        let (min_ci_width, max_ci_width) = Self::compute_ci_width_bounds(
            &candidates,
            &ci_point_predictions,
            lower_percentile,
            upper_percentile,
            config.ci_margin_factor,
            &config.ci_sample_sizes,
        );

        Self {
            candidates,
            ci_point_predictions,
            coverage_point_predictions,
            observed_values,
            lower_percentile,
            upper_percentile,
            min_ci_width,
            max_ci_width,
            size_mode,
            parallel_objective_threshold: config.parallel_objective_threshold,
            problem_id: NEXT_PROBLEM_ID.fetch_add(1, AtomicOrdering::Relaxed),
        }
    }

    /// Compute min and max CI width bounds for normalization.
    ///
    /// Strategy:
    /// - Min: CI width from 2 most similar candidates (smallest ensemble)
    /// - Max: CI width from diverse sample of candidates (largest ensemble)
    fn compute_ci_width_bounds(
        candidates: &[CalibrationEvaluation],
        ci_point_predictions: &[Vec<f64>],
        lower_percentile: f64,
        upper_percentile: f64,
        ci_margin_factor: f64,
        ci_sample_sizes: &[usize],
    ) -> (f64, f64) {
        use rand::seq::SliceRandom;

        if candidates.len() < 2 {
            return (0.0, 1.0);
        }

        // Min CI: Select 2 most similar candidates (by loss)
        let mut sorted_by_loss: Vec<usize> = (0..candidates.len()).collect();
        sorted_by_loss.sort_by(|&a, &b| {
            candidates[a]
                .loss
                .partial_cmp(&candidates[b].loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let min_ensemble = vec![sorted_by_loss[0], sorted_by_loss[1]];
        let min_ci = Self::calculate_ci_width_from_points(
            ci_point_predictions,
            &min_ensemble,
            lower_percentile,
            upper_percentile,
        );

        // Max CI: Try various ensemble sizes and configurations
        let mut max_ci = min_ci;
        let mut rng = SmallRng::seed_from_u64(42);

        // Try all candidates if feasible (< 100)
        if candidates.len() <= 100 {
            let all_indices: Vec<usize> = (0..candidates.len()).collect();
            let ci = Self::calculate_ci_width_from_points(
                ci_point_predictions,
                &all_indices,
                lower_percentile,
                upper_percentile,
            );
            max_ci = max_ci.max(ci);
        }

        // Try random samples of different sizes from config
        for &sample_size in ci_sample_sizes {
            let actual_size = sample_size.min(candidates.len());
            if actual_size >= candidates.len() {
                continue;
            }

            // Create a pool of indices and shuffle it
            let mut all_indices: Vec<usize> = (0..candidates.len()).collect();
            all_indices.shuffle(&mut rng);

            // Take first sample_size indices
            let random_indices: Vec<usize> = all_indices.into_iter().take(actual_size).collect();

            let ci = Self::calculate_ci_width_from_points(
                ci_point_predictions,
                &random_indices,
                lower_percentile,
                upper_percentile,
            );
            max_ci = max_ci.max(ci);
        }

        // Add margin to avoid edge cases (configurable)
        let range = max_ci - min_ci;
        let min_bound = (min_ci - ci_margin_factor * range).max(0.0);
        let max_bound = max_ci + ci_margin_factor * range;

        // Ensure bounds are valid
        if max_bound <= min_bound {
            (min_bound, min_bound + 1.0)
        } else {
            (min_bound, max_bound)
        }
    }

    fn point_index(
        candidates: &[CalibrationEvaluation],
        time_step: usize,
        compartment_idx: usize,
    ) -> usize {
        let n_compartments = candidates
            .first()
            .and_then(|candidate| candidate.predictions.first())
            .map(Vec::len)
            .unwrap_or(0);
        time_step * n_compartments + compartment_idx
    }

    fn ci_point_indices(
        candidates: &[CalibrationEvaluation],
        observed_data: &[(usize, usize, f64)],
        scope: CIWidthScope,
    ) -> Vec<usize> {
        let Some(first_prediction) = candidates.first().map(|candidate| &candidate.predictions)
        else {
            return Vec::new();
        };
        let n_time_steps = first_prediction.len();
        let n_compartments = first_prediction.first().map(Vec::len).unwrap_or(0);

        match scope {
            CIWidthScope::ObservedPoints => {
                let mut points: Vec<usize> = observed_data
                    .iter()
                    .map(|&(time_step, compartment_idx, _)| {
                        time_step * n_compartments + compartment_idx
                    })
                    .collect();
                points.sort_unstable();
                points.dedup();
                points
            }
            CIWidthScope::ObservedStepsAllCompartments => {
                let mut observed_steps: Vec<usize> = observed_data
                    .iter()
                    .map(|&(time_step, _, _)| time_step)
                    .collect();
                observed_steps.sort_unstable();
                observed_steps.dedup();

                observed_steps
                    .into_iter()
                    .flat_map(|time_step| {
                        (0..n_compartments).map(move |compartment_idx| {
                            time_step * n_compartments + compartment_idx
                        })
                    })
                    .collect()
            }
            CIWidthScope::FullTrajectory => (0..n_time_steps * n_compartments).collect(),
        }
    }

    fn point_major_predictions(
        candidates: &[CalibrationEvaluation],
        point_indices: &[usize],
    ) -> Vec<Vec<f64>> {
        let Some(first_predictions) = candidates.first().map(|candidate| &candidate.predictions)
        else {
            return Vec::new();
        };
        let n_time_steps = first_predictions.len();
        let n_compartments = first_predictions.first().map(Vec::len).unwrap_or(0);
        if n_time_steps == 0 || n_compartments == 0 {
            return Vec::new();
        }

        for (candidate_idx, candidate) in candidates.iter().enumerate() {
            assert_eq!(
                candidate.predictions.len(),
                n_time_steps,
                "candidate {candidate_idx} prediction time-step count does not match first candidate"
            );
            for (time_step, step) in candidate.predictions.iter().enumerate() {
                assert_eq!(
                    step.len(),
                    n_compartments,
                    "candidate {candidate_idx} prediction compartment count at time step {time_step} does not match first candidate"
                );
            }
        }

        point_indices
            .iter()
            .map(|&point_idx| {
                let time_step = point_idx / n_compartments;
                let compartment_idx = point_idx % n_compartments;
                assert!(
                    time_step < n_time_steps,
                    "prediction point {point_idx} is outside the prediction matrix shape ({n_time_steps} time steps x {n_compartments} compartments)"
                );
                candidates
                    .iter()
                    .map(|candidate| candidate.predictions[time_step][compartment_idx])
                    .collect()
            })
            .collect()
    }

    fn calculate_ci_width_from_points(
        point_predictions: &[Vec<f64>],
        selected_indices: &[usize],
        lower_percentile: f64,
        upper_percentile: f64,
    ) -> f64 {
        if selected_indices.len() < 2 || point_predictions.is_empty() {
            f64::MAX
        } else {
            Self::calculate_ci_width_from_points_serial(
                point_predictions,
                selected_indices,
                lower_percentile,
                upper_percentile,
            )
        }
    }

    fn calculate_ci_width_from_points_serial(
        point_predictions: &[Vec<f64>],
        selected_indices: &[usize],
        lower_percentile: f64,
        upper_percentile: f64,
    ) -> f64 {
        let (total_width, count) = point_predictions.iter().fold(
            (0.0, 0usize),
            |(total_width, count), candidate_values| {
                let width = Self::point_ci_width(
                    candidate_values,
                    selected_indices,
                    lower_percentile,
                    upper_percentile,
                );
                match width {
                    Some(width) => (total_width + width, count + 1),
                    None => (total_width, count),
                }
            },
        );

        if count > 0 {
            total_width / count as f64
        } else {
            f64::MAX
        }
    }

    fn calculate_ci_width_from_points_parallel(
        point_predictions: &[Vec<f64>],
        selected_indices: &[usize],
        lower_percentile: f64,
        upper_percentile: f64,
    ) -> f64 {
        let (total_width, count) = point_predictions
            .par_iter()
            .map(|candidate_values| {
                match Self::point_ci_width(
                    candidate_values,
                    selected_indices,
                    lower_percentile,
                    upper_percentile,
                ) {
                    Some(width) => (width, 1usize),
                    None => (0.0, 0usize),
                }
            })
            .reduce(|| (0.0, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));

        if count > 0 {
            total_width / count as f64
        } else {
            f64::MAX
        }
    }

    fn point_ci_width(
        candidate_values: &[f64],
        selected_indices: &[usize],
        lower_percentile: f64,
        upper_percentile: f64,
    ) -> Option<f64> {
        PERCENTILE_SCRATCH.with(|scratch| {
            let mut values = scratch.borrow_mut();
            values.clear();
            values.extend(
                selected_indices
                    .iter()
                    .filter_map(|&candidate_idx| candidate_values.get(candidate_idx).copied()),
            );

            if values.is_empty() {
                return None;
            }

            let lower = percentile_unstable(&mut values[..], lower_percentile);
            let upper = percentile_unstable(&mut values[..], upper_percentile);
            Some(upper - lower)
        })
    }

    /// Calculate average confidence interval width for selected parameter sets.
    fn calculate_ci_width(&self, selected_indices: &[usize]) -> f64 {
        if self.ci_point_predictions.len() >= self.parallel_objective_threshold
            && rayon::current_num_threads() > 1
        {
            Self::calculate_ci_width_from_points_parallel(
                &self.ci_point_predictions,
                selected_indices,
                self.lower_percentile,
                self.upper_percentile,
            )
        } else {
            Self::calculate_ci_width_from_points(
                &self.ci_point_predictions,
                selected_indices,
                self.lower_percentile,
                self.upper_percentile,
            )
        }
    }

    /// Calculate coverage percentage for selected parameter sets.
    fn calculate_coverage(&self, selected_indices: &[usize]) -> f64 {
        if self.observed_values.is_empty() || selected_indices.len() < 2 {
            return 0.0;
        }

        let covered_count: usize = if self.coverage_point_predictions.len()
            >= self.parallel_objective_threshold
            && rayon::current_num_threads() > 1
        {
            self.coverage_point_predictions
                .par_iter()
                .zip(&self.observed_values)
                .map(|(candidate_values, &observed_value)| {
                    usize::from(Self::point_covers_observed(
                        candidate_values,
                        selected_indices,
                        observed_value,
                        self.lower_percentile,
                        self.upper_percentile,
                    ))
                })
                .sum()
        } else {
            self.coverage_point_predictions
                .iter()
                .zip(&self.observed_values)
                .map(|(candidate_values, &observed_value)| {
                    usize::from(Self::point_covers_observed(
                        candidate_values,
                        selected_indices,
                        observed_value,
                        self.lower_percentile,
                        self.upper_percentile,
                    ))
                })
                .sum()
        };

        covered_count as f64 / self.observed_values.len() as f64
    }

    pub(crate) fn evaluate_selected_indices(&self, selected_indices: &[usize]) -> (f64, f64) {
        let ci_width = self.calculate_ci_width(selected_indices);
        let coverage = self.calculate_coverage(selected_indices);
        let normalized_ci_width = if self.max_ci_width > self.min_ci_width {
            ((ci_width - self.min_ci_width) / (self.max_ci_width - self.min_ci_width))
                .clamp(0.0, 1.0)
        } else {
            0.5
        };

        (normalized_ci_width, coverage)
    }

    fn point_covers_observed(
        candidate_values: &[f64],
        selected_indices: &[usize],
        observed_value: f64,
        lower_percentile: f64,
        upper_percentile: f64,
    ) -> bool {
        PERCENTILE_SCRATCH.with(|scratch| {
            let mut values = scratch.borrow_mut();
            values.clear();
            values.extend(
                selected_indices
                    .iter()
                    .filter_map(|&candidate_idx| candidate_values.get(candidate_idx).copied()),
            );

            if values.is_empty() {
                return false;
            }

            let lower = percentile_unstable(&mut values[..], lower_percentile);
            let upper = percentile_unstable(&mut values[..], upper_percentile);
            observed_value >= lower && observed_value <= upper
        })
    }

    pub(crate) fn selected_indices_from_param(&self, param: &[f64]) -> Vec<usize> {
        let mut selected_indices: Vec<usize> = param
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val >= 0.5 { Some(i) } else { None })
            .collect();

        if selected_indices.len() >= 2 {
            self.repair_to_valid_size(&mut selected_indices, param);
        }

        selected_indices
    }

    fn repair_to_valid_size(&self, selected: &mut Vec<usize>, param: &[f64]) {
        match &self.size_mode {
            EnsembleSizeMode::Fixed { size } => {
                self.repair_to_bounds(selected, param, *size, *size)
            }
            EnsembleSizeMode::Bounded { min, max } => {
                self.repair_to_bounds(selected, param, *min, *max)
            }
            EnsembleSizeMode::Automatic => {}
        }
    }

    fn repair_to_bounds(&self, selected: &mut Vec<usize>, param: &[f64], min: usize, max: usize) {
        while selected.len() > max {
            let remove_pos = selected
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, &b)| {
                    let a_confidence = (param.get(a).copied().unwrap_or(0.5) - 0.5).abs();
                    let b_confidence = (param.get(b).copied().unwrap_or(0.5) - 0.5).abs();
                    a_confidence
                        .partial_cmp(&b_confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.cmp(&b))
                })
                .map(|(pos, _)| pos)
                .expect("selected is non-empty while above max");
            selected.swap_remove(remove_pos);
        }

        while selected.len() < min {
            let selected_set: HashSet<usize> = selected.iter().copied().collect();
            let Some((candidate_idx, _)) = param
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx < self.candidates.len() && !selected_set.contains(idx))
                .max_by(|(a_idx, &a), (b_idx, &b)| {
                    a.partial_cmp(&b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| b_idx.cmp(a_idx))
                })
            else {
                break;
            };
            selected.push(candidate_idx);
        }

        selected.sort_unstable();
    }
}

impl MultiObjectiveCostFunction for EnsembleSelectionProblem {
    type Param = Vec<f64>; // Binary vector (0 or 1) indicating selected parameter sets
    type Output = Vec<f64>; // [normalized CI width, normalized negative coverage, size penalty]

    fn objectives(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        // Convert continuous values to binary (threshold at 0.5)
        let mut selected_indices: Vec<usize> = param
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val >= 0.5 { Some(i) } else { None })
            .collect();

        let ensemble_size = selected_indices.len();

        // Need at least 2 parameter sets for meaningful statistics
        if ensemble_size < 2 {
            return Ok(vec![1.0, 1.0, 1.0]); // Worst possible normalized values
        }

        self.repair_to_valid_size(&mut selected_indices, param);

        // Calculate raw objectives
        let ci_width = self.calculate_ci_width(&selected_indices);
        let coverage = self.calculate_coverage(&selected_indices);

        // Normalize CI width to [0, 1]
        let normalized_ci_width = if self.max_ci_width > self.min_ci_width {
            ((ci_width - self.min_ci_width) / (self.max_ci_width - self.min_ci_width))
                .clamp(0.0, 1.0)
        } else {
            0.5 // If all CI widths are the same, use middle value
        };

        // Normalize coverage to [0, 1] and negate for minimization
        // Coverage is already in [0, 1], so just negate
        let normalized_neg_coverage = 1.0 - coverage; // Convert maximization to minimization

        // Return normalized objectives: [CI width, negative coverage, size penalty]
        // All values in [0, 1], all minimization
        Ok(vec![normalized_ci_width, normalized_neg_coverage, 0.0])
    }

    fn num_objectives(&self) -> usize {
        3
    }
}

impl CostFunction for EnsembleSelectionProblem {
    type Param = Vec<usize>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        if param.len() < 2 {
            return Ok(f64::INFINITY);
        }

        let (normalized_ci_width, coverage) = self.evaluate_selected_indices(param);
        Ok(normalized_ci_width + (1.0 - coverage))
    }
}
