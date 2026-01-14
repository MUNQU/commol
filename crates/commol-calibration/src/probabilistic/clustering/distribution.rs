//! Distribution strategies for cluster representatives.
//!
//! This module provides strategies for distributing a fixed number of
//! representatives across clusters.

/// Distribute representatives equally across non-empty clusters.
///
/// Each non-empty cluster gets the same base number of representatives,
/// with leftovers distributed to the first clusters.
pub fn distribute_representatives_equal(cluster_sizes: &[usize], max_reps: usize) -> Vec<usize> {
    let n_clusters = cluster_sizes.len();
    let non_empty_clusters = cluster_sizes.iter().filter(|&&s| s > 0).count();

    if non_empty_clusters == 0 {
        return vec![0; n_clusters];
    }

    let base_reps_per_cluster = max_reps / non_empty_clusters;
    let leftover = max_reps % non_empty_clusters;

    let mut reps: Vec<usize> = cluster_sizes
        .iter()
        .map(|&s| if s > 0 { base_reps_per_cluster } else { 0 })
        .collect();

    // Distribute leftover representatives to first clusters
    let mut leftover_count = 0;
    for i in 0..n_clusters {
        if leftover_count >= leftover {
            break;
        }
        if cluster_sizes[i] > 0 {
            reps[i] += 1;
            leftover_count += 1;
        }
    }

    // Ensure we don't exceed cluster size
    for i in 0..n_clusters {
        reps[i] = reps[i].min(cluster_sizes[i]);
    }

    reps
}

/// Distribute representatives proportionally to cluster sizes.
///
/// Larger clusters get more representatives. Each non-empty cluster gets
/// at least one representative to ensure coverage.
pub fn distribute_representatives_proportional(
    cluster_sizes: &[usize],
    max_reps: usize,
) -> Vec<usize> {
    let n_clusters = cluster_sizes.len();
    let total_evaluations: usize = cluster_sizes.iter().sum();
    let non_empty_clusters = cluster_sizes.iter().filter(|&&s| s > 0).count();

    if non_empty_clusters == 0 {
        return vec![0; n_clusters];
    }

    // Allocate 1 representative to each non-empty cluster first
    let mut reps: Vec<usize> = cluster_sizes
        .iter()
        .map(|&s| if s > 0 { 1 } else { 0 })
        .collect();
    let remaining_reps = max_reps.saturating_sub(non_empty_clusters);

    // Distribute remaining representatives proportionally
    if remaining_reps > 0 && total_evaluations > non_empty_clusters {
        for i in 0..n_clusters {
            if cluster_sizes[i] > 0 {
                let proportion = cluster_sizes[i] as f64 / total_evaluations as f64;
                let additional = (remaining_reps as f64 * proportion).floor() as usize;
                reps[i] += additional;
            }
        }

        // Handle rounding errors - give leftovers to largest clusters
        let allocated: usize = reps.iter().sum();
        if allocated < max_reps {
            let mut cluster_indices: Vec<usize> = (0..n_clusters).collect();
            cluster_indices.sort_by_key(|&i| std::cmp::Reverse(cluster_sizes[i]));

            for &i in &cluster_indices {
                if reps.iter().sum::<usize>() >= max_reps {
                    break;
                }
                if cluster_sizes[i] > 0 {
                    reps[i] += 1;
                }
            }
        }
    }

    reps
}
