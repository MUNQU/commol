//! Weighted statistics of a sample: the one quantile convention of commol,
//! and the weighted mean, covariance, summary and effective size.
//!
//! # The quantile
//!
//! The quantile at level `p` is the inverse of the weighted empirical
//! distribution, interpolated linearly between the midpoints of the
//! cumulative weights. Sort the values with positive weight; member `i` sits
//! at `pᵢ = Cᵢ₋₁ + wᵢ/2`, where `Cᵢ` is the cumulative weight (the weights
//! sum to 1). Below the first midpoint the quantile is the smallest value,
//! above the last one the largest. With equal weights `pᵢ = (i − ½)/n`,
//! which is NumPy's `method="hazen"`.

/// Weights that sum to 1 from log weights of any scale.
///
/// # Panics
///
/// Panics when a log weight is NaN or `+∞`, or every log weight is `−∞`.
pub fn normalized_weights(log_weights: &[f64]) -> Vec<f64> {
    let max = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(max.is_finite(), "the log weights need one finite value");
    let weights: Vec<f64> = log_weights.iter().map(|&w| (w - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|w| w / total).collect()
}

/// Weights that sum to 1: each of `weights` divided by their sum.
///
/// # Errors
///
/// Returns a message when a weight is negative or not finite, or none is
/// positive.
pub fn normalize(weights: &[f64]) -> Result<Vec<f64>, String> {
    if let Some(weight) = weights.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
        return Err(format!(
            "A weight is a finite number of at least 0, got {weight}"
        ));
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Err("No weight is positive: a summary needs a run of positive weight".to_string());
    }
    Ok(weights.iter().map(|w| w / total).collect())
}

/// Check that every quantile level is in `[0, 1]`.
///
/// # Errors
///
/// Returns a message with the first level outside `[0, 1]`, NaN included.
pub fn check_levels(levels: &[f64]) -> Result<(), String> {
    match levels.iter().find(|level| !(0.0..=1.0).contains(*level)) {
        Some(level) => Err(format!("A quantile level is between 0 and 1, got {level}")),
        None => Ok(()),
    }
}

/// Values that are NaN where a reduction across them needs a number at
/// each.
#[derive(Debug, Clone, PartialEq)]
pub struct NotANumber {
    /// Index of each value that is NaN.
    pub points: Vec<usize>,
}

impl std::fmt::Display for NotANumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "The values at {:?} are NaN, and a quantile needs a number at each",
            self.points
        )
    }
}

impl std::error::Error for NotANumber {}

impl From<NotANumber> for String {
    fn from(error: NotANumber) -> Self {
        error.to_string()
    }
}

/// The weighted quantiles of `values` at each of `levels`, in `[0, 1]`
/// ([`check_levels`]).
///
/// # Errors
///
/// [`NotANumber`] with every value that is NaN, whatever its weight: a
/// quantile sorted by the sign bit of a NaN is no quantile.
///
/// # Panics
///
/// Panics when `values` and `weights` differ in length, or no weight is
/// positive.
pub fn quantiles(values: &[f64], weights: &[f64], levels: &[f64]) -> Result<Vec<f64>, NotANumber> {
    assert_eq!(values.len(), weights.len(), "one weight per value");
    let points: Vec<usize> = (0..values.len()).filter(|&i| values[i].is_nan()).collect();
    if !points.is_empty() {
        return Err(NotANumber { points });
    }
    let mut members: Vec<(f64, f64)> = values
        .iter()
        .zip(weights)
        .filter(|(_, w)| **w > 0.0)
        .map(|(&v, &w)| (v, w))
        .collect();
    assert!(!members.is_empty(), "a quantile needs a positive weight");
    members.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = members.iter().map(|(_, w)| w).sum();
    let mut cumulative = 0.0;
    let midpoints: Vec<f64> = members
        .iter()
        .map(|(_, w)| {
            let midpoint = (cumulative + w / 2.0) / total;
            cumulative += w;
            midpoint
        })
        .collect();
    Ok(levels
        .iter()
        .map(|&level| {
            let after = midpoints.partition_point(|&p| p < level);
            if after == 0 {
                members[0].0
            } else if after == members.len() {
                members[members.len() - 1].0
            } else {
                let (p0, p1) = (midpoints[after - 1], midpoints[after]);
                let (v0, v1) = (members[after - 1].0, members[after].0);
                if v0 == v1 {
                    // Also when both are infinite, where the difference is NaN.
                    v0
                } else {
                    v0 + (level - p0) / (p1 - p0) * (v1 - v0)
                }
            }
        })
        .collect())
}

/// The two quantiles of a central interval of `confidence_level`:
/// `(1 − c)/2` and `(1 + c)/2`.
///
/// The caller checks the level first (`commol_core::CONFIDENCE_LEVEL`):
/// `confidence_level` is strictly between 0 and 1.
pub fn interval_levels(confidence_level: f64) -> [f64; 2] {
    debug_assert!(
        confidence_level > 0.0 && confidence_level < 1.0,
        "the caller checks the confidence level, got {confidence_level}"
    );
    [
        (1.0 - confidence_level) / 2.0,
        (1.0 + confidence_level) / 2.0,
    ]
}

/// The weighted mean of `values`.
pub fn mean(values: &[f64], weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / total
}

/// The weighted summary of one column of values: the reduction of every
/// band and of every summary of runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// The weighted mean.
    pub mean: f64,
    /// The weighted median.
    pub median: f64,
    /// The lower end of the central interval, the quantile `(1 − c)/2`.
    pub lower: f64,
    /// The upper end of the central interval, the quantile `(1 + c)/2`.
    pub upper: f64,
    /// The smallest value of positive weight.
    pub minimum: f64,
    /// The largest value of positive weight.
    pub maximum: f64,
}

/// The weighted summary of `values`, with the ends `[lower, upper]` of the
/// central interval from [`interval_levels`].
///
/// # Errors
///
/// [`NotANumber`] with every value that is NaN ([`quantiles`]).
///
/// # Panics
///
/// Panics when `values` and `weights` differ in length, or no weight is
/// positive.
pub fn summary(
    values: &[f64],
    weights: &[f64],
    [lower, upper]: [f64; 2],
) -> Result<Summary, NotANumber> {
    let q = quantiles(values, weights, &[0.0, lower, 0.5, upper, 1.0])?;
    Ok(Summary {
        mean: mean(values, weights),
        median: q[2],
        lower: q[1],
        upper: q[3],
        minimum: q[0],
        maximum: q[4],
    })
}

/// The weighted covariance of two series, with the correction of
/// reliability weights: `Σ wᵢ(xᵢ − x̄)(yᵢ − ȳ) / (1 − Σ wᵢ²)` for weights
/// that sum to 1. With equal weights it is the sample covariance (divisor
/// `n − 1`). It is NaN for a sample of one effective member.
pub fn covariance(x: &[f64], y: &[f64], weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    let mean_x = mean(x, weights);
    let mean_y = mean(y, weights);
    let sum_squares: f64 = weights.iter().map(|w| (w / total).powi(2)).sum();
    let product: f64 = x
        .iter()
        .zip(y)
        .zip(weights)
        .map(|((a, b), w)| (w / total) * (a - mean_x) * (b - mean_y))
        .sum();
    let correction = 1.0 - sum_squares;
    if correction > 0.0 {
        product / correction
    } else {
        f64::NAN
    }
}

/// Kish's effective sample size: `(Σ wᵢ)² / Σ wᵢ²`.
pub fn effective_size(weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    let squares: f64 = weights.iter().map(|w| w * w).sum();
    total * total / squares
}

/// The Monte Carlo error of the quantile at `level` of equally weighted
/// draws: an order-statistic interval `[x₍ₗ₎, x₍ᵤ₎]` that holds the quantile
/// of the distribution the draws come from with about 95 % probability.
///
/// The ranks are `n p ∓ 1.96 √(n p (1 − p))`, rounded outward and kept
/// between 1 and `n`, by the normal approximation of the binomial count of
/// draws below the quantile.
///
/// # Panics
///
/// Panics when `values` is empty or `level` is not in `[0, 1]`.
pub fn quantile_error(values: &[f64], level: f64) -> (f64, f64) {
    assert!(!values.is_empty(), "a quantile needs a value");
    assert!((0.0..=1.0).contains(&level), "a level is in [0, 1]");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len() as f64;
    let centre = n * level;
    let half = 1.959_963_984_540_054 * (n * level * (1.0 - level)).sqrt();
    let rank = |r: f64| (r.clamp(1.0, n) as usize) - 1;
    (
        sorted[rank((centre - half).floor())],
        sorted[rank((centre + half).ceil())],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_normalize_to_one_when_one_is_positive() {
        assert_eq!(normalize(&[1.0, 3.0, 0.0]), Ok(vec![0.25, 0.75, 0.0]));
        assert_eq!(
            normalize(&[0.0, 0.0]),
            Err("No weight is positive: a summary needs a run of positive weight".to_string())
        );
        assert_eq!(
            normalize(&[1.0, -1.0]),
            Err("A weight is a finite number of at least 0, got -1".to_string())
        );
    }

    #[test]
    fn the_covariance_of_one_effective_member_is_not_a_number() {
        assert!(covariance(&[1.0, 5.0], &[2.0, 7.0], &[1.0, 0.0]).is_nan());
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn the_error_of_a_median_brackets_it_by_its_order_statistics() {
        let values: Vec<f64> = (1..=100).map(f64::from).collect();
        // 50 ∓ 1.96 · 5 = 40.2 and 59.8, rounded outward.
        assert_eq!(quantile_error(&values, 0.5), (40.0, 60.0));
        assert_eq!(quantile_error(&values, 0.0), (1.0, 1.0));
    }

    #[test]
    fn a_quantile_level_is_between_zero_and_one() {
        assert_eq!(check_levels(&[0.0, 0.5, 1.0]), Ok(()));
        for level in [-0.1, 1.5, f64::NAN] {
            assert_eq!(
                check_levels(&[0.5, level]),
                Err(format!("A quantile level is between 0 and 1, got {level}"))
            );
        }
    }

    #[test]
    fn equal_weights_give_the_hazen_quantile() {
        // NumPy: percentile([1, 2, 4, 8], [10, 50, 80], method="hazen").
        let values = [4.0, 1.0, 8.0, 2.0];
        let q = quantiles(&values, &[0.25; 4], &[0.1, 0.5, 0.8]).unwrap();
        assert!(close(q[0], 1.0), "{q:?}");
        assert!(close(q[1], 3.0), "{q:?}");
        assert!(close(q[2], 6.8), "{q:?}");
    }

    #[test]
    fn a_weighted_median_interpolates_between_midpoints() {
        // Midpoints 1/8 and 5/8: the median is 3/4 of the way to 1.
        let q = quantiles(&[0.0, 1.0], &[0.25, 0.75], &[0.5]).unwrap();
        assert!(close(q[0], 0.75), "{q:?}");
    }

    #[test]
    fn a_weight_of_zero_removes_a_member() {
        let with_zero = quantiles(&[1.0, 100.0, 3.0], &[0.5, 0.0, 0.5], &[0.2, 0.5, 0.8]).unwrap();
        let without = quantiles(&[1.0, 3.0], &[0.5, 0.5], &[0.2, 0.5, 0.8]).unwrap();
        assert_eq!(with_zero, without);
    }

    #[test]
    fn the_scale_of_the_weights_does_not_matter() {
        let a = quantiles(&[1.0, 2.0, 5.0], &[1.0, 2.0, 1.0], &[0.3, 0.6]).unwrap();
        let b = quantiles(&[1.0, 2.0, 5.0], &[0.25, 0.5, 0.25], &[0.3, 0.6]).unwrap();
        assert!(close(a[0], b[0]) && close(a[1], b[1]));
    }

    #[test]
    fn equal_weights_give_the_sample_covariance() {
        let x = [1.0, 2.0, 4.0];
        let y = [2.0, 1.0, 0.0];
        // Means 7/3 and 1; Σ(x − x̄)(y − ȳ) = −3, over n − 1 = 2.
        assert!(close(covariance(&x, &y, &[1.0 / 3.0; 3]), -1.5));
    }

    #[test]
    fn the_covariance_does_not_depend_on_the_scale_of_the_weights() {
        let x = [1.0, 2.0, 4.0];
        let y = [2.0, 1.0, 0.0];
        assert!(close(covariance(&x, &y, &[3.0; 3]), -1.5));
    }

    #[test]
    fn kish_size_counts_equal_weights_and_shrinks_with_unequal_ones() {
        assert!(close(effective_size(&[0.25; 4]), 4.0));
        assert!(close(effective_size(&[2.0; 4]), 4.0));
        assert!(close(effective_size(&[0.5, 0.25, 0.25]), 1.0 / 0.375));
    }

    #[test]
    fn a_central_interval_splits_the_rest_of_its_level_in_two() {
        assert_eq!(interval_levels(0.5), [0.25, 0.75]);
        assert!(close(interval_levels(0.9)[0], 0.05));
        assert!(close(interval_levels(0.9)[1], 0.95));
    }

    #[test]
    fn a_summary_reads_its_extremes_over_positive_weights() {
        let values = [5.0, 1.0, 100.0, 3.0];
        let weights = [1.0, 1.0, 0.0, 2.0];
        let summary = summary(&values, &weights, [0.25, 0.75]).unwrap();
        let q = quantiles(&values, &weights, &[0.5, 0.25, 0.75]).unwrap();
        assert_eq!((summary.minimum, summary.maximum), (1.0, 5.0));
        assert_eq!(
            [summary.median, summary.lower, summary.upper],
            [q[0], q[1], q[2]]
        );
        assert!(close(summary.mean, 3.0));
    }

    #[test]
    fn log_weights_of_any_scale_normalize_to_one() {
        let weights = normalized_weights(&[1000.0, 1000.0 + 2f64.ln()]);
        assert!(close(weights[0], 1.0 / 3.0) && close(weights[1], 2.0 / 3.0));
    }
}
