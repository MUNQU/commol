//! Pareto-smoothed importance sampling (Vehtari, Simpson, Gelman, Yao and
//! Gabry 2024): the largest importance weights replaced by the quantiles of
//! a generalized Pareto distribution fitted to them, and its shape `k̂`, which
//! says whether the weights have a finite variance.
//!
//! The tail holds the `M` largest weights, `M = ⌊3√S⌋` for `S > 225` draws
//! and `⌊S/5⌋` otherwise; the generalized Pareto fit is the empirical Bayes
//! estimate of Zhang and Stephens (2009) with a weak prior on `k` centred at
//! 0.5. Weight `r` of the tail, ascending, becomes the quantile
//! `(r + ½)/M` of the fit, kept at most the largest weight. `k̂ > 0.7` marks
//! an estimate as unreliable.

/// Above this `k̂` the smoothed weights do not give a reliable estimate.
pub const RELIABLE_K_HAT: f64 = 0.7;

/// The smallest tail that the generalized Pareto fit reads.
const SMALLEST_TAIL: usize = 5;
/// The prior of the scale of the fit: `b` is spread up to this many times
/// the inverse of the first quartile.
const PRIOR_SCALE: f64 = 3.0;
/// The weight of the prior of `k`, in draws.
const PRIOR_WEIGHT: f64 = 10.0;
/// The centre of the prior of `k`.
const PRIOR_CENTER: f64 = 0.5;

/// Smoothed log weights and the shape of their tail.
#[derive(Debug, Clone, PartialEq)]
pub struct Smoothed {
    /// The log weights, the tail smoothed, normalized so that the weights
    /// sum to 1.
    pub log_weights: Vec<f64>,
    /// The estimated shape `k̂` of the tail: NaN when the tail is not
    /// Pareto-like, `−∞` when every weight of the tail is equal.
    pub k_hat: f64,
    /// The number of weights in the tail, `M`.
    pub tail: usize,
}

impl Smoothed {
    /// Whether `k̂ ≤ 0.7`, so that the smoothed weights give a reliable
    /// estimate.
    pub fn reliable(&self) -> bool {
        self.k_hat <= RELIABLE_K_HAT
    }
}

/// The number of weights in the tail of `count` draws.
pub fn tail_length(count: usize) -> usize {
    let size = count as f64;
    if size > 225.0 {
        (3.0 * size.sqrt()).floor() as usize
    } else {
        count / 5
    }
}

/// Smooth the tail of `log_weights` of any scale; `−∞` is a weight of 0.
///
/// # Errors
///
/// Returns a message when there are fewer than 25 draws, so the tail has
/// fewer than 5, or no log weight is finite, or one is NaN or `+∞`.
pub fn smooth(log_weights: &[f64]) -> Result<Smoothed, String> {
    let count = log_weights.len();
    let tail = tail_length(count);
    if tail < SMALLEST_TAIL {
        return Err(format!(
            "Pareto smoothing needs at least 25 draws, so that its tail has 5; got {count}"
        ));
    }
    if log_weights
        .iter()
        .any(|w| w.is_nan() || *w == f64::INFINITY)
    {
        return Err("A log weight is NaN or +∞".to_string());
    }
    let largest = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !largest.is_finite() {
        return Err("Every importance weight is 0".to_string());
    }
    let mut shifted: Vec<f64> = log_weights.iter().map(|w| w - largest).collect();
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|&i, &j| shifted[i].total_cmp(&shifted[j]));
    let tail_indices = &order[count - tail..];
    let cutoff = shifted[order[count - tail - 1]];
    let draws_tail: Vec<f64> = tail_indices.iter().map(|&i| shifted[i]).collect();
    let max_tail = draws_tail[tail - 1];
    let k_hat = if max_tail - draws_tail[0] < f64::MIN_POSITIVE {
        f64::NEG_INFINITY
    } else {
        let cutoff_weight = cutoff.exp();
        let excess: Vec<f64> = draws_tail.iter().map(|w| w.exp() - cutoff_weight).collect();
        let (k, sigma) = generalized_pareto_fit(&excess);
        if k.is_finite() && sigma > 0.0 && sigma.is_finite() {
            for (rank, &index) in tail_indices.iter().enumerate() {
                let p = (rank as f64 + 0.5) / tail as f64;
                let smoothed = generalized_pareto_quantile(p, k, sigma, cutoff_weight).ln();
                shifted[index] = if smoothed > max_tail {
                    max_tail
                } else {
                    smoothed
                };
            }
        }
        k
    };
    let total = log_sum_exp(&shifted);
    Ok(Smoothed {
        log_weights: shifted.into_iter().map(|w| w - total).collect(),
        k_hat,
        tail,
    })
}

/// `log Σ exp(xᵢ)`, from the largest term.
pub(crate) fn log_sum_exp(values: &[f64]) -> f64 {
    let largest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !largest.is_finite() {
        return largest;
    }
    largest + values.iter().map(|v| (v - largest).exp()).sum::<f64>().ln()
}

/// The empirical Bayes estimate `(k, σ)` of a generalized Pareto
/// distribution of the ascending excesses `x` (Zhang and Stephens 2009),
/// with `k` shrunk to 0.5 by a prior worth 10 draws; `(NaN, NaN)` when the
/// first quartile is not above the smallest excess.
fn generalized_pareto_fit(x: &[f64]) -> (f64, f64) {
    let n = x.len();
    let size = n as f64;
    let candidates = 30 + size.sqrt() as usize;
    let first_quartile = x[(size / 4.0 + 0.5) as usize - 1];
    if first_quartile <= x[0] {
        return (f64::NAN, f64::NAN);
    }
    let mean_log = |b: f64| x.iter().map(|&xi| (-b * xi).ln_1p()).sum::<f64>() / size;
    let b: Vec<f64> = (1..=candidates)
        .map(|j| {
            (1.0 - (candidates as f64 / (j as f64 - 0.5)).sqrt()) / (PRIOR_SCALE * first_quartile)
                + 1.0 / x[n - 1]
        })
        .collect();
    let log_likelihood: Vec<f64> = b
        .iter()
        .map(|&bj| {
            let k = mean_log(bj);
            size * ((-bj / k).ln() - k - 1.0)
        })
        .collect();
    let total = log_sum_exp(&log_likelihood);
    let kept: Vec<(f64, f64)> = b
        .iter()
        .zip(&log_likelihood)
        .map(|(&bj, &l)| (bj, (l - total).exp()))
        .filter(|&(_, w)| w >= 10.0 * f64::EPSILON)
        .collect();
    let weight_sum: f64 = kept.iter().map(|(_, w)| w).sum();
    let b_post: f64 = kept.iter().map(|(bj, w)| bj * w / weight_sum).sum();
    let k = mean_log(b_post);
    let sigma = -k / b_post;
    let k = (size * k + PRIOR_WEIGHT * PRIOR_CENTER) / (size + PRIOR_WEIGHT);
    if k.is_nan() {
        return (f64::INFINITY, f64::NAN);
    }
    (k, sigma)
}

/// The quantile at `p` of the generalized Pareto distribution of shape `k`,
/// scale `σ` and location `μ`.
fn generalized_pareto_quantile(p: f64, k: f64, sigma: f64, mu: f64) -> f64 {
    if k == 0.0 {
        mu - sigma * (-p).ln_1p()
    } else {
        mu + sigma * (-k * (-p).ln_1p()).exp_m1() / k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tail_has_a_fifth_of_few_draws_and_three_root_of_many() {
        assert_eq!(tail_length(100), 20);
        assert_eq!(tail_length(225), 45);
        assert_eq!(tail_length(400), 60);
        assert_eq!(tail_length(1000), 94);
    }

    #[test]
    fn few_draws_or_no_weight_are_refused() {
        assert!(
            smooth(&[0.0; 24])
                .unwrap_err()
                .contains("at least 25 draws")
        );
        assert_eq!(
            smooth(&[f64::NEG_INFINITY; 30]).unwrap_err(),
            "Every importance weight is 0"
        );
        assert!(smooth(&[f64::NAN; 30]).is_err());
    }

    #[test]
    fn equal_weights_stay_equal() {
        let smoothed = smooth(&[2.0; 50]).unwrap();
        assert_eq!(smoothed.k_hat, f64::NEG_INFINITY);
        assert!(smoothed.reliable());
        for w in smoothed.log_weights {
            assert!((w + 50f64.ln()).abs() < 1e-12);
        }
    }

    #[test]
    fn a_tail_tied_at_its_cutoff_is_not_pareto_and_stays() {
        // The tail of 20 holds 5 weights equal to the cutoff: its first
        // quartile is not above its smallest excess, so the fit has no shape.
        let mut log_weights = vec![0.0; 85];
        log_weights.extend([1.0; 15]);
        let smoothed = smooth(&log_weights).unwrap();
        assert!(smoothed.k_hat.is_nan());
        assert!(!smoothed.reliable());
        let total = log_sum_exp(&log_weights);
        for (w, raw) in smoothed.log_weights.iter().zip(&log_weights) {
            assert!((w - (raw - total)).abs() < 1e-12);
        }
        assert_eq!(
            log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn a_shape_of_zero_is_the_exponential_quantile() {
        let q = generalized_pareto_quantile(0.5, 0.0, 2.0, 1.0);
        assert!((q - (1.0 + 2.0 * 2f64.ln())).abs() < 1e-12);
    }
}
