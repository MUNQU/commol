//! Sobol indices on the design of Saltelli (2010).
//!
//! The first-order index `S1` of an input is the share of the variance of a
//! quantity that the input explains alone; the total index `ST` adds every
//! interaction it takes part in. With `f_A`, `f_B` and `f_ABi` the values of
//! the quantity on the rows of `A`, `B` and `A_B⁽ⁱ⁾`, and `V` the variance of
//! `f_A` and `f_B` together:
//!
//! - `S1 = mean(f_B · (f_ABi − f_A)) / V`, the estimator of Saltelli (2010);
//! - `ST = mean((f_A − f_ABi)²) / (2 V)`, the estimator of Jansen (1999).
//!
//! The intervals are percentiles of a bootstrap of the rows: each replicate
//! draws `base` row indices with replacement, the same for every matrix, and
//! estimates the indices again. A replicate whose rows have no variance has
//! no indices, and then the interval is NaN: raise `base` rather than drop
//! it, since dropping such replicates would bias the interval.

use commol_random::Generator;
use rayon::prelude::*;

use super::check_numbers;
use crate::design::SaltelliDesign;
use crate::error::MethodError;
use crate::statistics::{interval_levels, quantiles};

/// The first-order and the total index of each input.
type Indices = (Vec<f64>, Vec<f64>);

/// The indices of one quantity and the lower and upper ends of the intervals
/// of its first-order and total indices.
type QuantityIndices = (Indices, [Vec<f64>; 2], [Vec<f64>; 2]);

/// The Sobol indices of every input for every quantity, `[quantity][input]`,
/// with their bootstrap intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct Sobol {
    /// The first-order indices; NaN for a quantity of no variance.
    pub first: Vec<Vec<f64>>,
    /// The total indices; NaN for a quantity of no variance.
    pub total: Vec<Vec<f64>>,
    /// The lower and upper ends of the interval of each first-order index.
    pub first_interval: [Vec<Vec<f64>>; 2],
    /// The lower and upper ends of the interval of each total index.
    pub total_interval: [Vec<Vec<f64>>; 2],
}

/// The Sobol indices of `design` from the quantities of its points,
/// `[point][quantity]`, with `replicates` bootstrap replicates for intervals at
/// `confidence_level`; `seed` seeds the bootstrap alone, so a caller gives
/// it a stream of its own ([`commol_random::stream_seed`]). The quantities
/// run in parallel.
///
/// # Errors
///
/// [`MethodError::Prediction`] of [`PredictiveError::NotANumber`](crate::predictive::PredictiveError::NotANumber) with the first quantity that is NaN at a
/// point, before any index; [`MethodError::Invalid`] when there is not
/// one row of quantities per point, `replicates` is below 2, or
/// `confidence_level` is not strictly between 0 and 1.
pub fn sobol(
    design: &SaltelliDesign,
    outputs: &[Vec<f64>],
    replicates: usize,
    confidence_level: f64,
    seed: u64,
) -> Result<Sobol, MethodError> {
    check_numbers(outputs)?;
    estimate(design, outputs, replicates, confidence_level, seed).map_err(MethodError::Invalid)
}

fn estimate(
    design: &SaltelliDesign,
    outputs: &[Vec<f64>],
    replicates: usize,
    confidence_level: f64,
    seed: u64,
) -> Result<Sobol, String> {
    if outputs.len() != design.points.len() {
        return Err(format!(
            "Sobol needs one row of quantities per point: {} points, {} rows",
            design.points.len(),
            outputs.len()
        ));
    }
    if replicates < 2 {
        return Err(format!(
            "Sobol needs at least 2 bootstrap replicates, got {replicates}"
        ));
    }
    let levels = interval_levels(confidence_level)?;
    let (base, inputs) = (design.base, design.inputs);
    let quantities = outputs.first().map_or(0, Vec::len);
    let all_rows: Vec<usize> = (0..base).collect();
    let mut generator = Generator::from_seed(seed);
    let resamples: Vec<Vec<usize>> = (0..replicates)
        .map(|_| {
            (0..base)
                .map(|_| ((generator.uniform() * base as f64) as usize).min(base - 1))
                .collect()
        })
        .collect();
    let equal = vec![1.0; replicates];
    let per_quantity: Vec<QuantityIndices> = (0..quantities)
        .into_par_iter()
        .map(|q| {
            let column: Vec<f64> = outputs.iter().map(|row| row[q]).collect();
            let value = |matrix: usize, row: usize| column[matrix * base + row];
            let estimate = indices(&value, inputs, &all_rows);
            let replicate_indices: Vec<Indices> = resamples
                .iter()
                .map(|rows| indices(&value, inputs, rows))
                .collect();
            let ends = |pick: &dyn Fn(&Indices) -> f64| -> [f64; 2] {
                let values: Vec<f64> = replicate_indices.iter().map(pick).collect();
                if values.iter().any(|v| v.is_nan()) {
                    return [f64::NAN; 2];
                }
                let ends = quantiles(&values, &equal, &levels)
                    .expect("a NaN index returned before the quantiles");
                [ends[0], ends[1]]
            };
            let (mut first_ends, mut total_ends) = ([vec![], vec![]], [vec![], vec![]]);
            for i in 0..inputs {
                let [low, high] = ends(&|r| r.0[i]);
                first_ends[0].push(low);
                first_ends[1].push(high);
                let [low, high] = ends(&|r| r.1[i]);
                total_ends[0].push(low);
                total_ends[1].push(high);
            }
            (estimate, first_ends, total_ends)
        })
        .collect();
    let mut result = Sobol {
        first: Vec::with_capacity(quantities),
        total: Vec::with_capacity(quantities),
        first_interval: [Vec::new(), Vec::new()],
        total_interval: [Vec::new(), Vec::new()],
    };
    for ((first, total), [first_low, first_high], [total_low, total_high]) in per_quantity {
        result.first.push(first);
        result.total.push(total);
        result.first_interval[0].push(first_low);
        result.first_interval[1].push(first_high);
        result.total_interval[0].push(total_low);
        result.total_interval[1].push(total_high);
    }
    Ok(result)
}

/// The first-order and total indices of each input on `rows` of the
/// design; `value(m, r)` is the quantity on row `r` of matrix `m` (0 is
/// `A`, 1 is `B`, `2 + i` is `A_B⁽ⁱ⁾`).
fn indices(value: &dyn Fn(usize, usize) -> f64, inputs: usize, rows: &[usize]) -> Indices {
    let count = rows.len() as f64;
    let pooled: Vec<f64> = rows
        .iter()
        .flat_map(|&r| [value(0, r), value(1, r)])
        .collect();
    let mean = pooled.iter().sum::<f64>() / pooled.len() as f64;
    let variance = pooled.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / pooled.len() as f64;
    if variance <= 0.0 {
        return (vec![f64::NAN; inputs], vec![f64::NAN; inputs]);
    }
    (0..inputs)
        .map(|i| {
            let (mut first, mut total) = (0.0, 0.0);
            for &r in rows {
                let (a, b, ab) = (value(0, r), value(1, r), value(2 + i, r));
                first += b * (ab - a);
                total += (a - ab).powi(2);
            }
            (first / count / variance, total / (2.0 * count) / variance)
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(base: usize) -> SaltelliDesign {
        SaltelliDesign {
            base,
            inputs: 1,
            points: vec![vec![0.0]; base * 3],
        }
    }

    #[test]
    fn sobol_checks_its_arguments() {
        let outputs = vec![vec![1.0]; 6];
        assert!(sobol(&design(2), &outputs[..5], 10, 0.9, 1).is_err());
        assert!(sobol(&design(2), &outputs, 1, 0.9, 1).is_err());
        assert!(sobol(&design(2), &outputs, 10, 1.0, 1).is_err());
    }

    #[test]
    fn a_quantity_of_no_variance_has_no_indices() {
        let result = sobol(&design(2), &vec![vec![1.0]; 6], 10, 0.9, 1).unwrap();
        assert!(result.first[0][0].is_nan() && result.total[0][0].is_nan());
        assert!(result.first_interval[0][0][0].is_nan());
    }

    #[test]
    fn the_indices_are_the_estimators_on_the_rows() {
        // One input that is the quantity: the rows of A_B⁽¹⁾ are those of B.
        let a = [0.0, 1.0];
        let b = [2.0, 3.0];
        let outputs: Vec<Vec<f64>> = a.iter().chain(&b).chain(&b).map(|v| vec![*v]).collect();
        let result = sobol(&design(2), &outputs, 10, 0.9, 1).unwrap();
        // f_A and f_B pooled are 0, 1, 2, 3: V = 1.25. S1: the mean of
        // 2·(2 − 0) and 3·(3 − 1) is 5, over V is 4. ST: the mean of 2² and
        // 2² is 4, over 2V is 1.6.
        assert!((result.first[0][0] - 4.0).abs() < 1e-12);
        assert!((result.total[0][0] - 1.6).abs() < 1e-12);
    }
}
