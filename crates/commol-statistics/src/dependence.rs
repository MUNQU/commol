//! The dependence of a series on a grid with gaps: its autocovariances, the
//! robust portmanteau of Dalla, Giraitis and Phillips (2022), and the
//! integrated autocorrelation time by the initial monotone sequence of Geyer
//! (1992).
//!
//! A series is one value or none (a gap) at each point of its grid, and its
//! values are centered first ([`centered`]). Each lag reads the pairs of
//! values that exist at that distance, and an autocovariance is the mean of
//! their products, weighted by their number: the amplitude-modulated
//! estimator of Parzen (1963) and Stoffer and Toloi (1992).
//!
//! # References
//!
//! - Dalla, V., Giraitis, L., & Phillips, P. C. B. (2022). Robust tests for
//!   white noise and cross-correlation. *Econometric Theory*, 38(5),
//!   913–941. <https://doi.org/10.1017/S0266466620000341>
//! - Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical
//!   Science*, 7(4), 473–483. <https://doi.org/10.1214/ss/1177011137>
//! - Parzen, E. (1963). On spectral analysis with missing observations and
//!   amplitude modulation. *Sankhyā A*, 25, 383–392.
//! - Stoffer, D. S., & Toloi, C. M. C. (1992). A note on the Ljung–Box–Pierce
//!   portmanteau statistic with missing data. *Statistics & Probability
//!   Letters*, 13(5), 391–396.

/// The values a series needs for its integrated time.
pub const TIME_VALUES: usize = 4;

/// The pairs a lag reads to enter the robust portmanteau.
pub const PORTMANTEAU_PAIRS: usize = 4;

/// Why a series has no robust portmanteau when no lag has
/// [`PORTMANTEAU_PAIRS`] pairs.
pub const FEW_PAIRS: &str = "every lag reads fewer than 4 pairs";
/// Why a series has no robust portmanteau when its lags of enough pairs
/// have no product other than 0.
pub const NO_PRODUCT: &str =
    "no lag of 4 pairs or more has a product other than 0, as of constant values";

/// Why a series has no integrated time.
pub const TOO_SHORT: &str = "it has fewer than 4 values";
/// Why a constant series has no autocorrelation and no integrated time.
pub const CONSTANT: &str = "its values are constant";

/// `series` minus the mean of its values; a gap stays a gap.
pub fn centered(series: &[Option<f64>]) -> Vec<Option<f64>> {
    let (sum, count) = series
        .iter()
        .flatten()
        .fold((0.0, 0usize), |(sum, count), value| {
            (sum + value, count + 1)
        });
    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    series.iter().map(|value| value.map(|v| v - mean)).collect()
}

/// The pairs of values of `series` that are `lag` points apart, as
/// `(earlier, later)`.
fn pairs(series: &[Option<f64>], lag: usize) -> impl Iterator<Item = (f64, f64)> + '_ {
    series
        .iter()
        .zip(series.iter().skip(lag))
        .filter_map(|(a, b)| Some(((*a)?, (*b)?)))
}

/// The autocovariance of a centered series at one lag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Autocovariance {
    /// `Σ e_t e_{t+k} / N_k`, the mean product of the pairs.
    pub value: f64,
    /// `N_k`, the pairs of values the lag reads.
    pub pairs: usize,
}

/// The autocovariance of the centered `series` at `lag`, over the pairs that
/// exist; at lag 0, the variance of its values. None when no pair is `lag`
/// apart.
pub fn autocovariance(series: &[Option<f64>], lag: usize) -> Option<Autocovariance> {
    let (sum, count) = pairs(series, lag).fold((0.0, 0usize), |(sum, count), (a, b)| {
        (sum + a * b, count + 1)
    });
    (count > 0).then(|| Autocovariance {
        value: sum / count as f64,
        pairs: count,
    })
}

/// The robust portmanteau statistic of a centered series.
#[derive(Debug, Clone, PartialEq)]
pub struct Portmanteau {
    /// `Q̃ = Σ_k (Σ_t e_t e_{t−k})² / Σ_t e_t² e_{t−k}²`.
    pub statistic: f64,
    /// The lags in the sum: those with [`PORTMANTEAU_PAIRS`] pairs or more
    /// and a product other than 0. Under independence `Q̃` is `χ²` of this
    /// many degrees of freedom.
    pub lags: usize,
    /// Each lag of fewer than [`PORTMANTEAU_PAIRS`] pairs, left out of the
    /// sum, as `(lag, pairs)`.
    pub few_pairs: Vec<(usize, usize)>,
    /// Why no lag is in the sum: [`FEW_PAIRS`] or [`NO_PRODUCT`]; None
    /// when one is.
    pub reason: Option<&'static str>,
}

/// The robust portmanteau `Q̃` of the centered `series` at the lags `1 …
/// max_lag`, the diagonal form of the statistic of Dalla, Giraitis and
/// Phillips (2022), for the null of independence: each lag's sum of products
/// over the pairs that exist, standardized by its own sum of squared
/// products, so heteroscedastic independent values are `χ²(h)` too. A lag
/// with fewer than [`PORTMANTEAU_PAIRS`] pairs, or whose products are all 0,
/// is left out.
pub fn robust_portmanteau(series: &[Option<f64>], max_lag: usize) -> Portmanteau {
    let mut statistic = 0.0;
    let mut lags = 0;
    let mut few_pairs = Vec::new();
    for lag in 1..=max_lag {
        let (sum, squares, count) = pairs(series, lag)
            .fold((0.0, 0.0, 0usize), |(sum, squares, count), (a, b)| {
                (sum + a * b, squares + (a * b) * (a * b), count + 1)
            });
        if count < PORTMANTEAU_PAIRS {
            few_pairs.push((lag, count));
        } else if squares > 0.0 {
            statistic += sum * sum / squares;
            lags += 1;
        }
    }
    let reason = (lags == 0).then_some(if few_pairs.len() == max_lag {
        FEW_PAIRS
    } else {
        NO_PRODUCT
    });
    Portmanteau {
        statistic,
        lags,
        few_pairs,
        reason,
    }
}

/// The integrated autocorrelation time of a series as its two moments, so
/// that times of several series can be averaged as a ratio of means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Time {
    /// `ĝ = −γ(0) + 2 Σ_{k<K} Γ_k`, the long-run variance.
    pub long_run_variance: f64,
    /// `γ(0)`, the variance.
    pub variance: f64,
    /// The largest lag in the sum, `2K − 1`; 0 when the sum is empty.
    pub cut_lag: usize,
}

impl Time {
    /// `τ = ĝ / γ(0)`, or None when it is not above 0.
    pub fn tau(&self) -> Option<f64> {
        let tau = self.long_run_variance / self.variance;
        (tau > 0.0).then_some(tau)
    }
}

/// The integrated autocorrelation time of the centered `series` by the
/// initial monotone sequence of Geyer (1992): the sums `Γ_k = γ(2k) +
/// γ(2k + 1)` of pairs of autocovariances, each at most the one before,
/// added until the first `Γ_k ≤ 0` or the first lag with no pair; `τ = −1 +
/// 2 Σ Γ_k / γ(0) = 1 + 2 Σ r(j)`.
///
/// # Errors
///
/// [`TOO_SHORT`] below [`TIME_VALUES`] values, and [`CONSTANT`] when
/// `γ(0) = 0`.
pub fn initial_monotone_time(series: &[Option<f64>]) -> Result<Time, &'static str> {
    if series.iter().flatten().count() < TIME_VALUES {
        return Err(TOO_SHORT);
    }
    let variance = autocovariance(series, 0).map_or(0.0, |gamma| gamma.value);
    if variance <= 0.0 {
        return Err(CONSTANT);
    }
    let mut sum = 0.0;
    let mut previous = f64::INFINITY;
    let mut k = 0;
    loop {
        let even = if k == 0 {
            Some(variance)
        } else {
            autocovariance(series, 2 * k).map(|gamma| gamma.value)
        };
        let odd = autocovariance(series, 2 * k + 1).map(|gamma| gamma.value);
        let (Some(even), Some(odd)) = (even, odd) else {
            break;
        };
        let gamma = (even + odd).min(previous);
        if gamma <= 0.0 {
            break;
        }
        sum += gamma;
        previous = gamma;
        k += 1;
    }
    Ok(Time {
        long_run_variance: -variance + 2.0 * sum,
        variance,
        cut_lag: if k == 0 { 0 } else { 2 * k - 1 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn some(values: &[f64]) -> Vec<Option<f64>> {
        values.iter().copied().map(Some).collect()
    }

    #[test]
    fn a_lag_reads_only_the_pairs_that_exist() {
        let series = [Some(1.0), None, Some(-1.0), Some(2.0)];
        // Lag 1: (−1, 2); lag 2: (1, −1); lag 3: (1, 2).
        assert_eq!(
            autocovariance(&series, 1),
            Some(Autocovariance {
                value: -2.0,
                pairs: 1
            })
        );
        assert_eq!(autocovariance(&series, 2).unwrap().value, -1.0);
        assert_eq!(autocovariance(&series, 4), None);
    }

    #[test]
    fn a_lag_with_no_product_or_few_pairs_leaves_the_portmanteau() {
        let series = some(&[1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]);
        // Lag 1: 6 products 0; lag 2: −1, 0, −1, 0, −1, so (−3)² / 3; lag
        // 3: 4 products 0; lag 4: 1, 0, 1, fewer than 4 pairs.
        assert_eq!(
            robust_portmanteau(&series, 4),
            Portmanteau {
                statistic: 3.0,
                lags: 1,
                few_pairs: vec![(4, 3)],
                reason: None,
            }
        );
        // Without lag 2, no lag is in the sum.
        let alone = some(&[1.0, 0.0, 0.0, 0.0, 0.0, -1.0]);
        assert_eq!(robust_portmanteau(&alone, 1).reason, Some(NO_PRODUCT));
        assert_eq!(robust_portmanteau(&alone, 1).few_pairs, vec![]);
        assert_eq!(
            robust_portmanteau(&some(&[1.0, 2.0]), 2).reason,
            Some(FEW_PAIRS)
        );
        // The reasons name the minimum.
        let minimum = PORTMANTEAU_PAIRS.to_string();
        assert!(FEW_PAIRS.contains(&minimum) && NO_PRODUCT.contains(&minimum));
    }

    #[test]
    fn a_short_or_constant_series_has_no_time() {
        assert_eq!(
            initial_monotone_time(&some(&[1.0, 2.0, 3.0])),
            Err(TOO_SHORT)
        );
        assert_eq!(
            initial_monotone_time(&some(&[0.0, 0.0, 0.0, 0.0])),
            Err(CONSTANT)
        );
    }

    #[test]
    fn alternating_values_cut_the_sum_at_once() {
        let time = initial_monotone_time(&centered(&some(&[1.0, -1.0, 1.0, -1.0]))).unwrap();
        // γ(0) = 1, γ(1) = −1: Γ_0 = 0, so the sum is empty and τ = −1.
        assert_eq!(time.cut_lag, 0);
        assert_eq!(time.tau(), None);
        assert_eq!(centered(&[None, Some(3.0)]), vec![None, Some(0.0)]);
    }
}
