//! Collinearity analysis after Brun et al. (2001).
//!
//! From the scaled sensitivities `S`: the sensitivity measure of parameter
//! `j` is `δ_msqr_j = ‖S_·j‖ / √n`, and the collinearity index of a subset `K`
//! of parameters is `γ_K = 1 / √λ_min(S̃_Kᵀ S̃_K)`, where `S̃` has the columns
//! of `S` scaled to unit length. `γ_K = 1` when the columns are orthogonal,
//! and grows without bound as they approach linear dependence: a change of
//! one parameter of `K` is then undone by changes of the others. The subsets
//! whose index is below a threshold are the parameters that can be fitted
//! together.
//!
//! Over several points (the draws of a sample), each index is computed at
//! each point and summarized by its weighted median.

use rayon::prelude::*;

use super::skippable;
use crate::error::MethodError;
use crate::sensitivity::{Matrix, SensitivitySource};
use crate::statistics::quantiles;

/// The size up to which every subset is evaluated; a greedy search goes
/// above it.
const EXHAUSTIVE: usize = 6;

/// A subset of parameters and its collinearity index.
#[derive(Debug, Clone, PartialEq)]
pub struct Subset {
    /// The parameter indices, ascending.
    pub parameters: Vec<usize>,
    /// The weighted median of its index over the points.
    pub gamma: f64,
    /// The share of the weight of the points where its index is below the
    /// threshold.
    pub share_below: f64,
}

/// The result of [`collinearity`].
#[derive(Debug, Clone, PartialEq)]
pub struct Collinearity {
    /// The weighted median of `δ_msqr` of each parameter.
    pub sensitivity: Vec<f64>,
    /// The weighted median of `γ` of each pair, `[j][k]`; 1 on the diagonal.
    pub pair_gamma: Vec<Vec<f64>>,
    /// The subset of lowest index of each size from 2 to the largest
    /// evaluated, by size.
    pub best_by_size: Vec<Subset>,
    /// The largest subsets whose index is below the threshold; single
    /// parameters when no pair is.
    pub admissible: Vec<Subset>,
    /// How many points were used.
    pub points: usize,
    /// Why each infeasible point was skipped.
    pub skipped: Vec<String>,
}

/// The collinearity index of the columns `subset` of the scaled
/// sensitivities `scaled`: infinite when a column is zero or the columns are
/// linearly dependent within rounding.
pub fn collinearity_index(scaled: &Matrix, subset: &[usize]) -> f64 {
    let rows = scaled.nrows();
    let mut normalized = Matrix::zeros(rows, subset.len());
    for (c, &j) in subset.iter().enumerate() {
        let column = scaled.column(j);
        let norm = column.norm();
        if norm == 0.0 {
            return f64::INFINITY;
        }
        normalized.set_column(c, &(column / norm));
    }
    index_of_normalized(&normalized)
}

fn index_of_normalized(normalized: &Matrix) -> f64 {
    let gram = normalized.transpose() * normalized;
    let eigenvalues = gram.symmetric_eigenvalues();
    let smallest = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let largest = eigenvalues.iter().copied().fold(0.0, f64::max);
    if smallest <= gram.nrows() as f64 * f64::EPSILON * largest {
        f64::INFINITY
    } else {
        1.0 / smallest.sqrt()
    }
}

/// Brun's indices of `source` at `points`, with weights `weights`, for
/// subsets up to `max_subset_size` parameters, against `threshold`.
///
/// # Errors
///
/// [`MethodError::Invalid`] when there is no point, the weights do
/// not match the points, or `max_subset_size` is below 2 or above the number
/// of parameters; [`MethodError::NoFeasiblePoint`] when every point
/// is infeasible; [`MethodError::Sensitivity`] when the
/// sensitivities fail for another reason.
pub fn collinearity<S: SensitivitySource>(
    source: &S,
    points: &[Vec<f64>],
    weights: &[f64],
    max_subset_size: usize,
    threshold: f64,
) -> Result<Collinearity, MethodError> {
    let count = source.parameters().len();
    if points.is_empty() || points.len() != weights.len() {
        return Err(MethodError::Invalid(format!(
            "Collinearity needs one weight per point, got {} points and {} weights",
            points.len(),
            weights.len()
        )));
    }
    if max_subset_size < 2 || max_subset_size > count {
        return Err(MethodError::Invalid(format!(
            "max_subset_size must be from 2 to the number of parameters ({count}), got \
             {max_subset_size}"
        )));
    }
    let mut columns: Vec<Matrix> = Vec::new();
    let mut used_weights = Vec::new();
    let mut sensitivity_draws: Vec<Vec<f64>> = vec![Vec::new(); count];
    let mut skipped = Vec::new();
    for (values, &weight) in points.iter().zip(weights) {
        let scaled = match source.scaled_jacobian(values) {
            Ok(sensitivities) => sensitivities.jacobian,
            Err(error) if skippable(&error) => {
                skipped.push(error.to_string());
                continue;
            }
            Err(error) => return Err(MethodError::Sensitivity(error)),
        };
        let rows = scaled.nrows() as f64;
        for (j, draws) in sensitivity_draws.iter_mut().enumerate() {
            draws.push(scaled.column(j).norm() / rows.sqrt());
        }
        columns.push(scaled);
        used_weights.push(weight);
    }
    if columns.is_empty() {
        return Err(MethodError::NoFeasiblePoint {
            points: points.len(),
            reasons: skipped,
        });
    }
    let total: f64 = used_weights.iter().sum();
    let used_weights: Vec<f64> = used_weights.iter().map(|w| w / total).collect();
    let median = |values: &[f64]| {
        quantiles(values, &used_weights, &[0.5])
            .expect("the norms of a finite Jacobian are numbers")[0]
    };
    let subset = |parameters: Vec<usize>| {
        let gammas: Vec<f64> = columns
            .iter()
            .map(|scaled| collinearity_index(scaled, &parameters))
            .collect();
        let share_below = gammas
            .iter()
            .zip(&used_weights)
            .filter(|(gamma, _)| **gamma < threshold)
            .map(|(_, w)| w)
            .sum();
        Subset {
            gamma: median(&gammas),
            share_below,
            parameters,
        }
    };

    let sensitivity: Vec<f64> = sensitivity_draws
        .iter()
        .map(|draws| median(draws))
        .collect();
    let mut pair_gamma = vec![vec![1.0; count]; count];
    let mut best_by_size: Vec<Subset> = Vec::new();
    let mut admissible: Vec<Subset> = (0..count)
        .map(|j| subset(vec![j]))
        .filter(|single| single.gamma < threshold)
        .collect();
    for size in 2..=max_subset_size.min(EXHAUSTIVE) {
        let evaluated: Vec<Subset> = combinations(count, size)
            .into_par_iter()
            .map(subset)
            .collect();
        if size == 2 {
            for pair in &evaluated {
                let (a, b) = (pair.parameters[0], pair.parameters[1]);
                pair_gamma[a][b] = pair.gamma;
                pair_gamma[b][a] = pair.gamma;
            }
        }
        let below: Vec<Subset> = evaluated
            .iter()
            .filter(|s| s.gamma < threshold)
            .cloned()
            .collect();
        if !below.is_empty() {
            admissible = below;
        }
        best_by_size.extend(lowest(evaluated));
    }
    while best_by_size
        .last()
        .is_some_and(|best| best.parameters.len() < max_subset_size)
    {
        let previous = best_by_size
            .last()
            .expect("checked above")
            .parameters
            .clone();
        let next = (0..count)
            .filter(|j| !previous.contains(j))
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|j| {
                let mut parameters = previous.clone();
                parameters.push(j);
                parameters.sort_unstable();
                subset(parameters)
            })
            .collect::<Vec<_>>();
        let Some(best) = lowest(next) else { break };
        if best.gamma < threshold {
            admissible = vec![best.clone()];
        }
        best_by_size.push(best);
    }
    Ok(Collinearity {
        sensitivity,
        pair_gamma,
        best_by_size,
        admissible,
        points: columns.len(),
        skipped,
    })
}

/// The subset of lowest index, the first on a tie.
fn lowest(subsets: Vec<Subset>) -> Option<Subset> {
    subsets
        .into_iter()
        .reduce(|best, s| if s.gamma < best.gamma { s } else { best })
}

/// Every subset of `size` of `0..count`, in lexicographic order.
fn combinations(count: usize, size: usize) -> Vec<Vec<usize>> {
    let mut all = Vec::new();
    debug_assert!(size <= count, "the caller checks the size of a subset");
    let mut current: Vec<usize> = (0..size).collect();
    loop {
        all.push(current.clone());
        let Some(i) = (0..size).rev().find(|&i| current[i] < count - size + i) else {
            return all;
        };
        current[i] += 1;
        for k in i + 1..size {
            current[k] = current[k - 1] + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_subset_of_a_size_is_listed_once() {
        assert_eq!(
            combinations(4, 2),
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        );
        assert_eq!(combinations(3, 3), [[0, 1, 2]]);
    }
}
