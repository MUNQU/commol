//! Deduplication of calibration evaluations.
//!
//! This module provides efficient deduplication of parameter sets using
//! grid-based spatial hashing.

use crate::types::CalibrationEvaluation;
use std::collections::HashMap;

/// Deduplicate evaluations using indexed candidate lookup - O(n log n) due to sorting.
///
/// Uses one indexed parameter dimension to find a small candidate set, then
/// applies the full relative-tolerance comparison across all parameters. Avoid
/// enumerating neighboring cells across every dimension: with 16 parameters that
/// would require 3^16 grid probes per evaluation.
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

    let index_dim = widest_parameter_dimension(&evaluations);
    let cell_size = tolerance.max(f64::MIN_POSITIVE);
    let mut grid: HashMap<i64, Vec<usize>> = HashMap::new();
    let mut unique = Vec::with_capacity(evaluations.len());

    for eval in evaluations {
        let index_value = eval.parameters.get(index_dim).copied().unwrap_or(0.0);
        let grid_coord = grid_coordinate(index_value, cell_size);
        let neighbor_radius = neighbor_radius(index_value, tolerance, cell_size);
        let mut is_duplicate = false;

        'neighbor_loop: for neighbor_coord in
            grid_coord.saturating_sub(neighbor_radius)..=grid_coord.saturating_add(neighbor_radius)
        {
            if let Some(indices) = grid.get(&neighbor_coord) {
                for &unique_idx in indices {
                    let unique_eval: &CalibrationEvaluation = &unique[unique_idx];

                    if parameter_vectors_close(&eval.parameters, &unique_eval.parameters, tolerance)
                    {
                        is_duplicate = true;
                        break 'neighbor_loop;
                    }
                }
            }
        }

        if !is_duplicate {
            let unique_idx = unique.len();
            unique.push(eval);
            grid.entry(grid_coord).or_default().push(unique_idx);
        }
    }

    unique
}

fn widest_parameter_dimension(evaluations: &[CalibrationEvaluation]) -> usize {
    let Some(first) = evaluations.first() else {
        return 0;
    };
    let n_dims = first.parameters.len();
    if n_dims == 0 {
        return 0;
    }

    let mut min_values = vec![f64::INFINITY; n_dims];
    let mut max_values = vec![f64::NEG_INFINITY; n_dims];

    for eval in evaluations {
        for (idx, value) in eval.parameters.iter().copied().enumerate() {
            if value.is_finite() {
                min_values[idx] = min_values[idx].min(value);
                max_values[idx] = max_values[idx].max(value);
            }
        }
    }

    (0..n_dims)
        .max_by(|&left, &right| {
            let left_range = max_values[left] - min_values[left];
            let right_range = max_values[right] - min_values[right];
            left_range
                .partial_cmp(&right_range)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}

fn grid_coordinate(value: f64, cell_size: f64) -> i64 {
    (value / cell_size).floor() as i64
}

fn neighbor_radius(value: f64, tolerance: f64, cell_size: f64) -> i64 {
    ((tolerance * value.abs().max(1.0)) / cell_size).ceil() as i64 + 1
}

fn parameter_vectors_close(left: &[f64], right: &[f64], tolerance: f64) -> bool {
    left.len() == right.len()
        && left.iter().zip(right.iter()).all(|(p1, p2)| {
            let max_abs = p1.abs().max(p2.abs()).max(1e-10);
            let rel_diff = (p1 - p2).abs() / max_abs;
            rel_diff < tolerance
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluation(parameters: Vec<f64>, loss: f64) -> CalibrationEvaluation {
        CalibrationEvaluation {
            parameters,
            loss,
            predictions: Vec::new(),
        }
    }

    #[test]
    fn deduplicate_keeps_lowest_loss_duplicate() {
        let evaluations = vec![
            evaluation(vec![1.0, 2.0], 2.0),
            evaluation(vec![1.0 + 1e-7, 2.0 - 1e-7], 1.0),
        ];

        let unique = deduplicate_evaluations(evaluations, 1e-6);

        assert_eq!(unique.len(), 1);
        assert_eq!(unique[0].loss, 1.0);
    }

    #[test]
    fn deduplicate_handles_high_dimensional_inputs() {
        let mut evaluations = Vec::new();
        for idx in 0..10 {
            let mut parameters = vec![0.5; 20];
            parameters[0] += idx as f64 * 1e-4;
            evaluations.push(evaluation(parameters, idx as f64));
        }
        evaluations.push(evaluation(vec![0.5 + 1e-8; 20], -1.0));

        let unique = deduplicate_evaluations(evaluations, 1e-6);

        assert_eq!(unique.len(), 10);
        assert_eq!(unique[0].loss, -1.0);
    }
}
