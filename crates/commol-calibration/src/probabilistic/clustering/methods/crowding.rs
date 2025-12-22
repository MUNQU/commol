//! Crowding distance selection method.
//!
//! This module provides NSGA-II style crowding distance selection for diverse
//! representative selection in parameter space.

/// Calculate crowding distance for solutions in parameter space.
///
/// Uses the NSGA-II crowding distance metric, measuring how "crowded" each solution
/// is by its neighbors along each parameter dimension. Higher values indicate more isolated
/// solutions (better for diversity).
///
/// # Arguments
/// * `param_vectors` - Parameter vectors for all solutions
/// * `indices` - Indices to calculate crowding distance for (subset of solutions)
///
/// # Returns
/// Vector of (index, crowding_distance) tuples, sorted by index
pub fn calculate_crowding_distance_parameter_space(
    param_vectors: &[Vec<f64>],
    indices: &[usize],
) -> Vec<(usize, f64)> {
    if indices.is_empty() {
        return Vec::new();
    }

    if indices.len() == 1 {
        // Single solution has infinite crowding distance
        return vec![(indices[0], f64::INFINITY)];
    }

    let n_params = param_vectors[0].len();
    let n_solutions = indices.len();

    // Initialize crowding distances to 0
    let mut crowding_distances: Vec<f64> = vec![0.0; n_solutions];

    // For each parameter dimension
    for (param_idx, _) in param_vectors.iter().enumerate().take(n_params) {
        // Create list of (local_idx, global_idx, param_value)
        let mut sorted_by_param: Vec<(usize, usize, f64)> = indices
            .iter()
            .enumerate()
            .map(|(local_idx, &global_idx)| {
                (local_idx, global_idx, param_vectors[global_idx][param_idx])
            })
            .collect();

        // Sort by parameter value
        sorted_by_param.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Get parameter range
        let param_min = sorted_by_param[0].2;
        let param_max = sorted_by_param[n_solutions - 1].2;
        let param_range = param_max - param_min;

        // Boundary solutions get infinite crowding distance
        crowding_distances[sorted_by_param[0].0] = f64::INFINITY;
        crowding_distances[sorted_by_param[n_solutions - 1].0] = f64::INFINITY;

        // For interior solutions, add normalized distance to neighbors
        if param_range > 1e-10 {
            for i in 1..(n_solutions - 1) {
                let local_idx = sorted_by_param[i].0;
                let prev_val = sorted_by_param[i - 1].2;
                let next_val = sorted_by_param[i + 1].2;

                // Skip if already infinite (boundary in another dimension)
                if crowding_distances[local_idx].is_infinite() {
                    continue;
                }

                // Add normalized distance contribution from this dimension
                crowding_distances[local_idx] += (next_val - prev_val) / param_range;
            }
        }
    }

    // Return as (global_index, crowding_distance) tuples
    indices
        .iter()
        .enumerate()
        .map(|(local_idx, &global_idx)| (global_idx, crowding_distances[local_idx]))
        .collect()
}

/// Select diverse solutions using crowding distance.
///
/// Uses NSGA-II crowding distance in parameter space to select diverse solutions.
pub fn select_by_crowding_distance(
    all_param_vectors: &[Vec<f64>],
    remaining_candidates: &[usize],
    n_remaining: usize,
) -> Vec<usize> {
    let crowding_distances =
        calculate_crowding_distance_parameter_space(all_param_vectors, remaining_candidates);

    // Sort by crowding distance (descending - higher is better for diversity)
    let mut candidates_with_crowding: Vec<(usize, f64)> = crowding_distances;
    candidates_with_crowding.sort_by(|a, b| {
        // Sort descending by crowding distance (infinity first, then larger values)
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Select top n_remaining by crowding distance
    candidates_with_crowding
        .iter()
        .take(n_remaining)
        .map(|(idx, _)| *idx)
        .collect()
}
