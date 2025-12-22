//! Maximin distance selection method.
//!
//! This module provides density-aware quality-weighted maximin distance selection
//! for diverse representative selection.

use crate::probabilistic::utils::parameter_distance_normalized;
use crate::types::CalibrationEvaluation;

/// Configuration for maximin distance selection.
pub struct MaximinConfig {
    pub n_elite: usize,
    pub n_remaining: usize,
    pub quality_temperature: f64,
    pub k_neighbors_min: usize,
    pub k_neighbors_max: usize,
    pub sparsity_weight: f64,
}

/// Select diverse representatives using density-aware quality-weighted maximin distance.
///
/// This method provides uniform coverage of the parameter space without boundary bias,
/// accounting for local density to avoid over-sampling from convergent regions.
///
/// It iteratively selects the candidate that maximizes a combination of:
/// 1. Distance from already selected points (diversity)
/// 2. Solution quality (lower loss)
/// 3. Local sparsity (density penalty to avoid convergent clusters)
///
/// # Arguments
/// * `evaluations` - All evaluations with parameters and losses
/// * `sorted_by_loss` - Indices sorted by loss (ascending)
/// * `all_param_vectors` - Parameter vectors for all evaluations
/// * `config` - Configuration for maximin distance selection
///
/// # Returns
/// Indices of selected solutions (excluding elite)
pub fn select_by_maximin_distance(
    evaluations: &[CalibrationEvaluation],
    sorted_by_loss: &[usize],
    all_param_vectors: &[Vec<f64>],
    config: &MaximinConfig,
) -> Vec<usize> {
    if config.n_remaining == 0 || sorted_by_loss.len() <= config.n_elite {
        return Vec::new();
    }

    // Calculate parameter ranges for normalization
    let n_params = all_param_vectors[0].len();
    let mut param_mins = vec![f64::INFINITY; n_params];
    let mut param_maxs = vec![f64::NEG_INFINITY; n_params];

    for params in all_param_vectors.iter() {
        for (i, &p) in params.iter().enumerate() {
            param_mins[i] = param_mins[i].min(p);
            param_maxs[i] = param_maxs[i].max(p);
        }
    }

    let param_ranges: Vec<f64> = param_mins
        .iter()
        .zip(param_maxs.iter())
        .map(|(min, max)| max - min)
        .collect();

    // Initialize selected set with elite solutions
    let mut selected: Vec<usize> = sorted_by_loss[..config.n_elite].to_vec();

    // Candidate pool (solutions not yet selected)
    let mut candidates: Vec<usize> = sorted_by_loss[config.n_elite..].to_vec();

    // Normalize losses for quality scoring
    let median_loss = if !evaluations.is_empty() {
        let mut losses: Vec<f64> = evaluations.iter().map(|e| e.loss).collect();
        losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        losses[losses.len() / 2]
    } else {
        1.0
    };

    // Pre-calculate local density for each candidate using k-nearest neighbors
    // This helps identify over-sampled convergent regions
    // Adaptive k using configurable bounds
    let k_neighbors = config
        .k_neighbors_max
        .min(sorted_by_loss.len() / 10)
        .max(config.k_neighbors_min);

    let mut local_densities: Vec<f64> = Vec::with_capacity(candidates.len());

    for &cand_idx in &candidates {
        // Calculate distances to all other evaluations (candidates + selected)
        let mut distances: Vec<f64> = sorted_by_loss
            .iter()
            .filter(|&&idx| idx != cand_idx)
            .map(|&idx| {
                parameter_distance_normalized(
                    &all_param_vectors[cand_idx],
                    &all_param_vectors[idx],
                    &param_ranges,
                )
            })
            .collect();

        if distances.is_empty() {
            local_densities.push(1.0);
            continue;
        }

        // Sort to find k nearest neighbors
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Use average distance to k nearest neighbors as density measure
        let k_actual = k_neighbors.min(distances.len());
        let avg_k_distance: f64 = distances.iter().take(k_actual).sum::<f64>() / k_actual as f64;

        // Density = 1 / avg_distance (higher in dense regions)
        // Add small epsilon to avoid division by zero
        let density = 1.0 / (avg_k_distance + 1e-10);
        local_densities.push(density);
    }

    // Normalize densities to [0, 1] range
    let max_density = local_densities
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_density = local_densities
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let density_range = max_density - min_density;

    let mut normalized_densities: Vec<f64> = if density_range > 1e-10 {
        local_densities
            .iter()
            .map(|&d| (d - min_density) / density_range)
            .collect()
    } else {
        vec![0.5; local_densities.len()] // All same density
    };

    // Iteratively select candidates with best quality × distance × sparsity score
    for _ in 0..config.n_remaining {
        if candidates.is_empty() {
            break;
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_candidate_idx = 0;

        for (cand_list_idx, &cand_global_idx) in candidates.iter().enumerate() {
            // Calculate minimum distance to any selected point
            let min_distance = selected
                .iter()
                .map(|&sel_idx| {
                    parameter_distance_normalized(
                        &all_param_vectors[cand_global_idx],
                        &all_param_vectors[sel_idx],
                        &param_ranges,
                    )
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f64::MAX);

            // Calculate quality score: exp(-loss / temperature)
            // Higher temperature = weaker quality preference (more diversity)
            // Lower temperature = stronger quality preference
            let normalized_loss = evaluations[cand_global_idx].loss / median_loss;
            let quality_score = (-normalized_loss / config.quality_temperature).exp();

            // Calculate sparsity bonus: prefer candidates in sparse regions
            // normalized_density is in [0, 1], where 1 = densest region
            // sparsity_factor = (1 - density) so sparse regions get higher scores
            let sparsity_factor = 1.0 - normalized_densities[cand_list_idx];

            // Apply sparsity with exponential weighting to amplify the effect
            // This strongly penalizes selection from convergent (dense) regions
            let density_weight = (config.sparsity_weight * sparsity_factor).exp(); // e^(weight*sparsity)

            // Combined score: quality × distance × density_weight
            // - quality_score: prefers better solutions
            // - min_distance: prefers points far from already selected
            // - density_weight: prefers points in sparse regions (avoids convergent clusters)
            let score = quality_score * min_distance * density_weight;

            if score > best_score {
                best_score = score;
                best_candidate_idx = cand_list_idx;
            }
        }

        // Add best candidate to selected set
        let selected_global_idx = candidates[best_candidate_idx];
        selected.push(selected_global_idx);

        // Remove from candidates list (and corresponding density value)
        candidates.remove(best_candidate_idx);
        normalized_densities.remove(best_candidate_idx);
    }

    // Return only the newly selected indices (excluding elite)
    selected[config.n_elite..].to_vec()
}
