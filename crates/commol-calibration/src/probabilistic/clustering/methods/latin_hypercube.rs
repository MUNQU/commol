//! Latin Hypercube Sampling selection method.
//!
//! This module provides Latin Hypercube Sampling based selection for uniform
//! coverage of the parameter space.

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::types::CalibrationEvaluation;

/// Select diverse representatives using Latin Hypercube Sampling.
///
/// This method divides the parameter space into strata and selects one representative
/// from each stratum, ensuring uniform coverage across all parameter dimensions.
///
/// Latin Hypercube Sampling provides better space-filling properties than random sampling
/// and more uniform coverage than crowding distance (which favors boundaries).
///
/// # Arguments
/// * `evaluations` - All evaluations with parameters and losses
/// * `sorted_by_loss` - Indices sorted by loss (ascending)
/// * `all_param_vectors` - Parameter vectors for all evaluations
/// * `n_elite` - Number of elite solutions already selected
/// * `n_remaining` - Number of additional solutions to select
/// * `rng` - Random number generator for stratified sampling
/// * `stratum_fit_weight` - Weight for stratum fit vs quality (higher = prioritize space-filling)
///
/// # Returns
/// Indices of selected solutions (excluding elite)
pub fn select_by_latin_hypercube(
    evaluations: &[CalibrationEvaluation],
    sorted_by_loss: &[usize],
    all_param_vectors: &[Vec<f64>],
    n_elite: usize,
    n_remaining: usize,
    rng: &mut SmallRng,
    stratum_fit_weight: f64,
) -> Vec<usize> {
    if n_remaining == 0 || sorted_by_loss.len() <= n_elite {
        return Vec::new();
    }

    let n_params = all_param_vectors[0].len();
    let candidates = &sorted_by_loss[n_elite..];

    if candidates.is_empty() {
        return Vec::new();
    }

    // If we need more samples than available candidates, just return all
    if n_remaining >= candidates.len() {
        return candidates.to_vec();
    }

    // Calculate parameter ranges for normalization
    let mut param_mins = vec![f64::INFINITY; n_params];
    let mut param_maxs = vec![f64::NEG_INFINITY; n_params];

    for &idx in candidates {
        for (i, &p) in all_param_vectors[idx].iter().enumerate() {
            param_mins[i] = param_mins[i].min(p);
            param_maxs[i] = param_maxs[i].max(p);
        }
    }

    let param_ranges: Vec<f64> = param_mins
        .iter()
        .zip(param_maxs.iter())
        .map(|(min, max)| max - min)
        .collect();

    // Create strata for each parameter dimension
    // Each dimension is divided into n_remaining intervals
    let strata: Vec<Vec<usize>> = (0..n_params)
        .map(|_| {
            let mut perm: Vec<usize> = (0..n_remaining).collect();
            perm.shuffle(rng);
            perm
        })
        .collect();

    // For each sample, determine which stratum it should come from in each dimension
    let mut selected_indices = Vec::with_capacity(n_remaining);

    for sample_idx in 0..n_remaining {
        // Get the target stratum for each dimension
        let target_strata: Vec<usize> = strata.iter().map(|dim| dim[sample_idx]).collect();

        // Find the candidate that best fits into these strata
        // We use a scoring system that rewards being in the correct stratum
        let mut best_score = f64::NEG_INFINITY;
        let mut best_candidate_idx = 0;

        for &cand_global_idx in candidates.iter() {
            // Skip if already selected
            if selected_indices.contains(&cand_global_idx) {
                continue;
            }

            let params = &all_param_vectors[cand_global_idx];

            // Calculate how well this candidate fits the target strata
            let mut stratum_score = 0.0;

            for (dim_idx, &target_stratum) in target_strata.iter().enumerate() {
                let param_value = params[dim_idx];

                // Normalize parameter value to [0, 1]
                let normalized = if param_ranges[dim_idx] > 1e-10 {
                    (param_value - param_mins[dim_idx]) / param_ranges[dim_idx]
                } else {
                    0.5
                };

                // Calculate which stratum this value falls into
                let actual_stratum =
                    ((normalized * n_remaining as f64).floor() as usize).min(n_remaining - 1);

                // Score: higher if in correct stratum, lower if far from correct stratum
                let stratum_distance = if actual_stratum == target_stratum {
                    0.0
                } else {
                    (actual_stratum as f64 - target_stratum as f64).abs()
                };

                // Accumulate inverse distance (closer = better)
                // Add small constant to avoid division by zero
                stratum_score += 1.0 / (stratum_distance + 0.1);
            }

            // Add quality bias: prefer better solutions among similar stratum fits
            // Use exponential decay to give slight preference to better quality
            let quality_score = (-evaluations[cand_global_idx].loss
                / evaluations[sorted_by_loss[0]].loss.max(1e-10))
            .exp();

            // Combined score: stratum fit (configurable weight) + quality (weight 1)
            // Heavily favor stratum fit, but use quality as tiebreaker
            let total_score = stratum_fit_weight * stratum_score + quality_score;

            if total_score > best_score {
                best_score = total_score;
                best_candidate_idx = cand_global_idx;
            }
        }

        selected_indices.push(best_candidate_idx);
    }

    selected_indices
}
