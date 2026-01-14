//! Utility functions for probabilistic calibration.
//!
//! This module provides shared utility functions used across the probabilistic
//! calibration submodules.

/// Calculate percentile of sorted values using linear interpolation.
///
/// # Arguments
/// * `sorted_values` - Slice of values sorted in ascending order
/// * `p` - Percentile to calculate (0-100)
///
/// # Returns
/// The interpolated percentile value
pub fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    if sorted_values.len() == 1 {
        return sorted_values[0];
    }

    let n = sorted_values.len();
    let idx = (p / 100.0) * (n - 1) as f64;
    let lower_idx = idx.floor() as usize;
    let upper_idx = idx.ceil() as usize;

    if lower_idx == upper_idx {
        sorted_values[lower_idx]
    } else {
        let weight = idx - lower_idx as f64;
        sorted_values[lower_idx] * (1.0 - weight) + sorted_values[upper_idx] * weight
    }
}

/// Calculate Euclidean distance between two parameter vectors in normalized space.
///
/// Normalizes by parameter ranges to give equal weight to all parameters.
///
/// # Arguments
/// * `params1` - First parameter vector
/// * `params2` - Second parameter vector
/// * `param_ranges` - Range (max - min) for each parameter for normalization
///
/// # Returns
/// Normalized Euclidean distance
pub fn parameter_distance_normalized(
    params1: &[f64],
    params2: &[f64],
    param_ranges: &[f64],
) -> f64 {
    params1
        .iter()
        .zip(params2.iter())
        .zip(param_ranges.iter())
        .map(|((p1, p2), range)| {
            let diff = p1 - p2;
            let normalized_diff = if *range > 1e-10 { diff / range } else { diff };
            normalized_diff * normalized_diff
        })
        .sum::<f64>()
        .sqrt()
}

/// Calculate the number of elite solutions to include.
///
/// # Arguments
/// * `n_to_select` - Total number of solutions to select
/// * `elite_fraction` - Fraction of solutions to select by quality (0.0-1.0)
///
/// # Returns
/// Number of elite solutions (best by loss) to include
pub fn calculate_elite_count(n_to_select: usize, elite_fraction: f64) -> usize {
    if elite_fraction >= 1.0 {
        n_to_select
    } else if elite_fraction <= 0.0 {
        0
    } else {
        ((n_to_select as f64 * elite_fraction).ceil() as usize).min(n_to_select)
    }
}
