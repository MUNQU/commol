//! Weighted statistics of a sample: the one quantile convention of commol,
//! and the weighted mean, covariance and effective size.
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

/// The weighted quantiles of `values` at each of `levels`, in `[0, 1]`.
///
/// # Panics
///
/// Panics when `values` and `weights` differ in length, or no weight is
/// positive.
pub fn quantiles(values: &[f64], weights: &[f64], levels: &[f64]) -> Vec<f64> {
    assert_eq!(values.len(), weights.len(), "one weight per value");
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
    levels
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
                v0 + (level - p0) / (p1 - p0) * (v1 - v0)
            }
        })
        .collect()
}

/// The two quantiles of a central interval of `confidence_level`:
/// `(1 − c)/2` and `(1 + c)/2`.
///
/// # Errors
///
/// Returns a message when `confidence_level` is not strictly between 0 and
/// 1.
pub fn interval_levels(confidence_level: f64) -> Result<[f64; 2], String> {
    if !(confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(format!(
            "A confidence level is strictly between 0 and 1, got {confidence_level}; use 0.95 \
             for a 95% interval"
        ));
    }
    Ok([
        (1.0 - confidence_level) / 2.0,
        (1.0 + confidence_level) / 2.0,
    ])
}

/// The weighted mean of `values`.
pub fn mean(values: &[f64], weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / total
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

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn equal_weights_give_the_hazen_quantile() {
        // NumPy: percentile([1, 2, 4, 8], [10, 50, 80], method="hazen").
        let values = [4.0, 1.0, 8.0, 2.0];
        let q = quantiles(&values, &[0.25; 4], &[0.1, 0.5, 0.8]);
        assert!(close(q[0], 1.0), "{q:?}");
        assert!(close(q[1], 3.0), "{q:?}");
        assert!(close(q[2], 6.8), "{q:?}");
    }

    #[test]
    fn a_weighted_median_interpolates_between_midpoints() {
        // Midpoints 1/8 and 5/8: the median is 3/4 of the way to 1.
        let q = quantiles(&[0.0, 1.0], &[0.25, 0.75], &[0.5]);
        assert!(close(q[0], 0.75), "{q:?}");
    }

    #[test]
    fn a_weight_of_zero_removes_a_member() {
        let with_zero = quantiles(&[1.0, 100.0, 3.0], &[0.5, 0.0, 0.5], &[0.2, 0.5, 0.8]);
        let without = quantiles(&[1.0, 3.0], &[0.5, 0.5], &[0.2, 0.5, 0.8]);
        assert_eq!(with_zero, without);
    }

    #[test]
    fn the_scale_of_the_weights_does_not_matter() {
        let a = quantiles(&[1.0, 2.0, 5.0], &[1.0, 2.0, 1.0], &[0.3, 0.6]);
        let b = quantiles(&[1.0, 2.0, 5.0], &[0.25, 0.5, 0.25], &[0.3, 0.6]);
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
    fn kish_size_counts_equal_weights_and_shrinks_with_unequal_ones() {
        assert!(close(effective_size(&[0.25; 4]), 4.0));
        assert!(close(effective_size(&[0.5, 0.25, 0.25]), 1.0 / 0.375));
    }

    #[test]
    fn log_weights_of_any_scale_normalize_to_one() {
        let weights = normalized_weights(&[1000.0, 1000.0 + 2f64.ln()]);
        assert!(close(weights[0], 1.0 / 3.0) && close(weights[1], 2.0 / 3.0));
    }
}
