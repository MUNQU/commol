//! Deduplication of calibration evaluations.
//!
//! This module provides efficient deduplication of parameter sets using
//! grid-based spatial hashing.

use crate::types::CalibrationEvaluation;
use std::collections::HashMap;

/// Deduplicate evaluations using grid-based spatial hashing - O(n log n) due to sorting.
///
/// Uses a spatial grid to group similar parameter sets, then only compares
/// within the same and neighboring grid cells. This avoids O(n^2) comparisons.
///
/// Evaluations are sorted by loss before deduplication to ensure deterministic results.
///
/// # Arguments
/// * `evaluations` - Vector of calibration evaluations to deduplicate
/// * `tolerance` - Relative tolerance for considering parameters as duplicates
///
/// # Returns
/// Vector of unique evaluations
pub fn deduplicate_evaluations(
    mut evaluations: Vec<CalibrationEvaluation>,
    tolerance: f64,
) -> Vec<CalibrationEvaluation> {
    if evaluations.is_empty() {
        return Vec::new();
    }

    // Sort evaluations by loss to ensure deterministic deduplication order
    // This is critical because HashMap iteration order is non-deterministic
    evaluations.sort_by(|a, b| {
        a.loss
            .partial_cmp(&b.loss)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let cell_size = tolerance * 2.0;
    let mut grid: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();
    let mut unique = Vec::with_capacity(evaluations.len());

    for eval in evaluations {
        let grid_coords: Vec<i64> = eval
            .parameters
            .iter()
            .map(|&p| (p / cell_size).floor() as i64)
            .collect();

        let n_dims = grid_coords.len();
        let mut is_duplicate = false;
        let total_neighbors = 3_usize.pow(n_dims as u32);

        'neighbor_loop: for neighbor_idx in 0..total_neighbors {
            let mut neighbor_coords = grid_coords.clone();
            let mut idx = neighbor_idx;

            for coord in neighbor_coords.iter_mut() {
                let offset = (idx % 3) as i64 - 1;
                *coord += offset;
                idx /= 3;
            }

            if let Some(indices) = grid.get(&neighbor_coords) {
                for &unique_idx in indices {
                    let unique_eval: &CalibrationEvaluation = &unique[unique_idx];

                    let all_close = eval
                        .parameters
                        .iter()
                        .zip(unique_eval.parameters.iter())
                        .all(|(p1, p2)| {
                            let max_abs = p1.abs().max(p2.abs()).max(1e-10);
                            let rel_diff = (p1 - p2).abs() / max_abs;
                            rel_diff < tolerance
                        });

                    if all_close {
                        is_duplicate = true;
                        break 'neighbor_loop;
                    }
                }
            }
        }

        if !is_duplicate {
            let unique_idx = unique.len();
            unique.push(eval);
            grid.entry(grid_coords).or_default().push(unique_idx);
        }
    }

    unique
}
