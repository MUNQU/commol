//! NSGA-II ensemble selection problem definition.
//!
//! This module defines the multi-objective optimization problem for selecting
//! an ensemble of parameter sets that balances narrow confidence intervals
//! with good coverage of observed data.

use argmin::core::MultiObjectiveCostFunction;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use super::size_mode::EnsembleSizeMode;
use crate::probabilistic::config::EnsembleSelectionConfig;
use crate::probabilistic::utils::percentile;
use crate::types::CalibrationEvaluation;

/// NSGA-II ensemble selection problem.
///
/// This multi-objective optimization problem selects an ensemble of parameter sets
/// that balances:
/// 1. Narrow confidence intervals (minimize CI width)
/// 2. Good coverage of observed data (maximize coverage)
/// 3. Ensemble size constraints (penalty for violating size constraints)
pub(crate) struct EnsembleSelectionProblem {
    /// Candidate parameter sets to choose from.
    pub candidates: Vec<CalibrationEvaluation>,

    /// Observed data points: (time_step, compartment_idx, value).
    observed_data: Vec<(usize, usize, f64)>,

    /// Lower percentile for CI calculation (e.g., 2.5 for 95% CI).
    lower_percentile: f64,

    /// Upper percentile for CI calculation (e.g., 97.5 for 95% CI).
    upper_percentile: f64,

    /// Normalization bounds for CI width objective.
    min_ci_width: f64,
    max_ci_width: f64,

    /// Ensemble size constraint mode.
    size_mode: EnsembleSizeMode,
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

        // Compute normalization bounds for CI width
        let (min_ci_width, max_ci_width) = Self::compute_ci_width_bounds(
            &candidates,
            lower_percentile,
            upper_percentile,
            config.ci_margin_factor,
            &config.ci_sample_sizes,
        );

        Self {
            candidates,
            observed_data,
            lower_percentile,
            upper_percentile,
            min_ci_width,
            max_ci_width,
            size_mode,
        }
    }

    /// Compute min and max CI width bounds for normalization.
    ///
    /// Strategy:
    /// - Min: CI width from 2 most similar candidates (smallest ensemble)
    /// - Max: CI width from diverse sample of candidates (largest ensemble)
    fn compute_ci_width_bounds(
        candidates: &[CalibrationEvaluation],
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
        let min_ci = Self::calculate_ci_width_static(
            candidates,
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
            let ci = Self::calculate_ci_width_static(
                candidates,
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

            let ci = Self::calculate_ci_width_static(
                candidates,
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

    /// Static version of CI width calculation (for use in bounds computation).
    pub fn calculate_ci_width_static(
        candidates: &[CalibrationEvaluation],
        selected_indices: &[usize],
        lower_percentile: f64,
        upper_percentile: f64,
    ) -> f64 {
        if selected_indices.len() < 2 {
            return f64::MAX;
        }

        let selected_predictions: Vec<&Vec<Vec<f64>>> = selected_indices
            .iter()
            .filter_map(|&i| candidates.get(i).map(|e| &e.predictions))
            .collect();

        if selected_predictions.is_empty() || selected_predictions[0].is_empty() {
            return f64::MAX;
        }

        let n_time_steps = selected_predictions[0].len();
        let n_compartments = selected_predictions[0][0].len();

        let mut total_width = 0.0;
        let mut count = 0;

        for t in 0..n_time_steps {
            for c in 0..n_compartments {
                let mut values: Vec<f64> = selected_predictions
                    .iter()
                    .filter_map(|pred| pred.get(t).and_then(|step| step.get(c)).copied())
                    .collect();

                if values.is_empty() {
                    continue;
                }

                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let lower = percentile(&values, lower_percentile);
                let upper = percentile(&values, upper_percentile);

                total_width += upper - lower;
                count += 1;
            }
        }

        if count > 0 {
            total_width / count as f64
        } else {
            f64::MAX
        }
    }

    /// Calculate average confidence interval width for selected parameter sets.
    fn calculate_ci_width(&self, selected_indices: &[usize]) -> f64 {
        if selected_indices.len() < 2 {
            return f64::MAX;
        }

        let selected_predictions: Vec<&Vec<Vec<f64>>> = selected_indices
            .iter()
            .filter_map(|&i| self.candidates.get(i).map(|e| &e.predictions))
            .collect();

        if selected_predictions.is_empty() || selected_predictions[0].is_empty() {
            return f64::MAX;
        }

        let n_time_steps = selected_predictions[0].len();
        let n_compartments = selected_predictions[0][0].len();

        let mut total_width = 0.0;
        let mut count = 0;

        for t in 0..n_time_steps {
            for c in 0..n_compartments {
                let mut values: Vec<f64> = selected_predictions
                    .iter()
                    .filter_map(|pred| pred.get(t).and_then(|step| step.get(c)).copied())
                    .collect();

                if values.is_empty() {
                    continue;
                }

                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let lower = percentile(&values, self.lower_percentile);
                let upper = percentile(&values, self.upper_percentile);

                total_width += upper - lower;
                count += 1;
            }
        }

        if count > 0 {
            total_width / count as f64
        } else {
            f64::MAX
        }
    }

    /// Calculate coverage percentage for selected parameter sets.
    fn calculate_coverage(&self, selected_indices: &[usize]) -> f64 {
        if self.observed_data.is_empty() || selected_indices.len() < 2 {
            return 0.0;
        }

        let selected_predictions: Vec<&Vec<Vec<f64>>> = selected_indices
            .iter()
            .filter_map(|&i| self.candidates.get(i).map(|e| &e.predictions))
            .collect();

        if selected_predictions.is_empty() {
            return 0.0;
        }

        let mut covered_count = 0;

        for &(time_step, compartment_idx, observed_value) in &self.observed_data {
            let mut values: Vec<f64> = selected_predictions
                .iter()
                .filter_map(|pred| {
                    pred.get(time_step)
                        .and_then(|step| step.get(compartment_idx))
                        .copied()
                })
                .collect();

            if values.is_empty() {
                continue;
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let lower = percentile(&values, self.lower_percentile);
            let upper = percentile(&values, self.upper_percentile);

            if observed_value >= lower && observed_value <= upper {
                covered_count += 1;
            }
        }

        covered_count as f64 / self.observed_data.len() as f64
    }
}

impl MultiObjectiveCostFunction for EnsembleSelectionProblem {
    type Param = Vec<f64>; // Binary vector (0 or 1) indicating selected parameter sets
    type Output = Vec<f64>; // [normalized CI width, normalized negative coverage, size penalty]

    fn objectives(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        // Convert continuous values to binary (threshold at 0.5)
        let selected_indices: Vec<usize> = param
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val >= 0.5 { Some(i) } else { None })
            .collect();

        let ensemble_size = selected_indices.len();

        // Need at least 2 parameter sets for meaningful statistics
        if ensemble_size < 2 {
            return Ok(vec![1.0, 1.0, 1.0]); // Worst possible normalized values
        }

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

        // Calculate size constraint penalty based on mode
        let size_penalty = match &self.size_mode {
            EnsembleSizeMode::Fixed { size } => {
                // Penalize deviation from target size
                let deviation = (ensemble_size as i32 - *size as i32).abs() as f64;
                let base_penalty = (deviation / self.candidates.len() as f64).min(1.0);
                // Multiply by 100 to ensure deviations are dominated
                (base_penalty * 100.0).min(100.0)
            }
            EnsembleSizeMode::Bounded { min, max } => {
                // Hard penalty for being outside bounds
                if ensemble_size < *min {
                    // Penalty proportional to how far below minimum
                    let violation = (*min - ensemble_size) as f64;
                    let base_penalty = (violation / *min as f64).min(1.0);
                    // Multiply by 100 to ensure violations are dominated
                    (base_penalty * 100.0).min(100.0)
                } else if ensemble_size > *max {
                    // Penalty proportional to how far above maximum
                    let violation = (ensemble_size - *max) as f64;
                    let base_penalty = (violation / *max as f64).min(1.0);
                    // Multiply by 100 to ensure violations are dominated
                    (base_penalty * 100.0).min(100.0)
                } else {
                    // No penalty within bounds
                    0.0
                }
            }
            EnsembleSizeMode::Automatic => {
                // No size penalty for automatic mode - let NSGA-II explore freely
                // The optimal size will be determined from the Pareto front later
                0.0
            }
        };

        // Return normalized objectives: [CI width, negative coverage, size penalty]
        // All values in [0, 1], all minimization
        Ok(vec![
            normalized_ci_width,
            normalized_neg_coverage,
            size_penalty,
        ])
    }

    fn num_objectives(&self) -> usize {
        3
    }
}
