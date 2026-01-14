//! Cluster representative selection.
//!
//! This module provides the main entry point for selecting diverse representatives
//! from clustered calibration evaluations.

use rand::rngs::SmallRng;
use rand::SeedableRng;

use super::distribution::{
    distribute_representatives_equal, distribute_representatives_proportional,
};
use super::methods::{
    maximin::MaximinConfig, select_by_crowding_distance, select_by_latin_hypercube,
    select_by_maximin_distance,
};
use crate::probabilistic::config::EnsembleSelectionConfig;
use crate::probabilistic::error::{CalibrationError, CalibrationResult};
use crate::probabilistic::utils::calculate_elite_count;
use crate::types::CalibrationEvaluation;

/// Configuration for cluster representative selection.
pub struct ClusterRepresentativeConfig<'a> {
    pub max_representatives: usize,
    pub elite_fraction: f64,
    pub strategy: &'a str,
    pub selection_method: &'a str,
    pub quality_temperature: f64,
    pub seed: u64,
    pub ensemble_config: &'a EnsembleSelectionConfig,
}

/// Select diverse representatives from clustered evaluations.
///
/// This function efficiently selects a maximum number of representatives across
/// all clusters, distributed proportionally to cluster size. Uses greedy maximum
/// minimum distance selection for diversity within each cluster.
///
/// # Arguments
/// * `evaluations` - All evaluations with parameters and losses
/// * `cluster_labels` - Cluster assignment for each evaluation (from KMeans)
/// * `config` - Configuration for cluster representative selection
///
/// # Returns
/// Indices of selected representative evaluations
///
/// # Errors
/// - `CalibrationError::EvaluationLabelMismatch` if evaluations and labels counts differ
pub fn select_cluster_representatives(
    evaluations: Vec<CalibrationEvaluation>,
    cluster_labels: Vec<usize>,
    config: &ClusterRepresentativeConfig,
) -> CalibrationResult<Vec<usize>> {
    if evaluations.is_empty() {
        return Ok(Vec::new());
    }

    if evaluations.len() != cluster_labels.len() {
        return Err(CalibrationError::EvaluationLabelMismatch {
            evaluations: evaluations.len(),
            labels: cluster_labels.len(),
        });
    }

    // Group evaluations by cluster
    let n_clusters = cluster_labels.iter().max().map(|&m| m + 1).unwrap_or(1);
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];

    for (idx, &label) in cluster_labels.iter().enumerate() {
        if label < n_clusters {
            clusters[label].push(idx);
        }
    }

    // Calculate cluster sizes and representatives per cluster
    let cluster_sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();
    let total_evaluations: usize = cluster_sizes.iter().sum();
    let non_empty_clusters = cluster_sizes.iter().filter(|&&s| s > 0).count();

    if non_empty_clusters == 0 {
        return Ok(Vec::new());
    }

    let max_reps = config.max_representatives.min(total_evaluations);

    // Distribute representatives based on strategy
    let reps_per_cluster = match config.strategy {
        "equal" => distribute_representatives_equal(&cluster_sizes, max_reps),
        "proportional" => distribute_representatives_proportional(&cluster_sizes, max_reps),
        _ => distribute_representatives_proportional(&cluster_sizes, max_reps),
    };

    // Select representatives from each cluster using elite + diversity selection
    let mut selected_indices = Vec::new();

    // Extract all parameter vectors once (for crowding distance calculation)
    let all_param_vectors: Vec<Vec<f64>> =
        evaluations.iter().map(|e| e.parameters.clone()).collect();

    for (cluster_id, cluster_indices) in clusters.iter().enumerate() {
        if cluster_indices.is_empty() {
            continue;
        }

        let n_to_select = reps_per_cluster[cluster_id].min(cluster_indices.len());

        // Sort by loss (lower is better)
        let mut sorted_by_loss = cluster_indices.clone();
        sorted_by_loss.sort_by(|&a, &b| {
            evaluations[a]
                .loss
                .partial_cmp(&evaluations[b].loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // If we need to select all solutions, just take them all
        if n_to_select >= sorted_by_loss.len() {
            selected_indices.extend(sorted_by_loss);
            continue;
        }

        // Calculate how many elite solutions to include
        let n_elite = calculate_elite_count(n_to_select, config.elite_fraction);

        // Include n_elite best solutions by loss (if any)
        if n_elite > 0 {
            let elite_solutions = &sorted_by_loss[..n_elite];
            selected_indices.extend_from_slice(elite_solutions);

            // If we've selected enough, continue to next cluster
            if n_elite >= n_to_select {
                continue;
            }
        }

        // Select remaining solutions by diversity method
        let n_remaining = n_to_select - n_elite;
        let remaining_candidates = &sorted_by_loss[n_elite..];

        if remaining_candidates.is_empty() {
            continue;
        }

        // Select diverse solutions based on method
        let diverse_solutions: Vec<usize> = match config.selection_method {
            "latin_hypercube" => {
                // Latin Hypercube Sampling for stratified space-filling selection
                let mut rng = SmallRng::seed_from_u64(config.seed.wrapping_add(cluster_id as u64));
                select_by_latin_hypercube(
                    &evaluations,
                    &sorted_by_loss,
                    &all_param_vectors,
                    n_elite,
                    n_remaining,
                    &mut rng,
                    config.ensemble_config.stratum_fit_weight,
                )
            }
            "maximin_distance" => {
                // Quality-weighted maximin distance selection
                let maximin_config = MaximinConfig {
                    n_elite,
                    n_remaining,
                    quality_temperature: config.quality_temperature,
                    k_neighbors_min: config.ensemble_config.k_neighbors_min,
                    k_neighbors_max: config.ensemble_config.k_neighbors_max,
                    sparsity_weight: config.ensemble_config.sparsity_weight,
                };
                select_by_maximin_distance(
                    &evaluations,
                    &sorted_by_loss,
                    &all_param_vectors,
                    &maximin_config,
                )
            }
            "crowding_distance" => {
                // NSGA-II crowding distance selection
                select_by_crowding_distance(&all_param_vectors, remaining_candidates, n_remaining)
            }
            _ => {
                // Default to crowding distance
                select_by_crowding_distance(&all_param_vectors, remaining_candidates, n_remaining)
            }
        };

        selected_indices.extend(diverse_solutions);
    }

    Ok(selected_indices)
}
