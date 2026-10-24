//! The log density of each observation model at an observed value, given its
//! predicted mean and its noise.
//!
//! Every elementary function comes from `libm`, so a value is the same on
//! every platform.

use commol_statistics::numeric::{
    digamma, ln_gamma, log_normal_log_density, normal_log_density, student_t_log_density, trigamma,
    x_log_y, x_log1p_y,
};

/// Above this observed count, the negative binomial reads `ln Γ` instead of
/// the sum of its first terms, unless the overdispersion is below
/// [`SMALL_OVERDISPERSION`].
const SUM_TERMS: f64 = 1000.0;
/// Below this overdispersion, `ln Γ(y + 1/φ) − ln Γ(1/φ)` loses digits, so
/// the negative binomial sums its terms whatever the count.
const SMALL_OVERDISPERSION: f64 = 1e-8;

/// What the predicted value of a log-normal model is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Center {
    /// The mean: `log Y ~ N(log μ − σ²/2, σ²)`.
    Mean,
    /// The median: `log Y ~ N(log μ, σ²)`.
    Median,
}

/// An observation model with a number for each of its noise arguments: how
/// one observation scatters around its predicted mean `μ`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Noise {
    /// Normal of mean `μ` and standard deviation `sd`.
    Normal {
        /// The standard deviation, above 0.
        sd: f64,
    },
    /// Log-normal whose mean or median is `μ`, with `sigma` the standard
    /// deviation of `log Y`.
    LogNormal {
        /// The standard deviation of `log Y`, above 0.
        sigma: f64,
        /// Whether `μ` is the mean or the median.
        center: Center,
    },
    /// Student-t of center `μ`, `df` degrees of freedom and scale `scale`.
    StudentT {
        /// The degrees of freedom, above 0.
        df: f64,
        /// The scale, above 0.
        scale: f64,
    },
    /// Poisson of mean `μ`.
    Poisson,
    /// Negative binomial of mean `μ` and variance `μ + φμ²`; `φ = 0` is the
    /// Poisson.
    NegativeBinomial {
        /// The overdispersion `φ`, at least 0.
        overdispersion: f64,
    },
    /// Binomial of `trials` trials and mean `μ`, so the probability is
    /// `μ / trials`.
    Binomial {
        /// The number of trials.
        trials: u64,
    },
}

impl Noise {
    /// How far `mean` lies outside the means this model allows, or `None`
    /// inside them: a count needs a mean of at least 0, a binomial one of at
    /// most its trials, a log-normal one above 0.
    pub fn mean_outside(&self, mean: f64) -> Option<f64> {
        match *self {
            Self::Normal { .. } | Self::StudentT { .. } => None,
            Self::LogNormal { .. } => (mean <= 0.0).then_some(-mean),
            Self::Poisson | Self::NegativeBinomial { .. } => (mean < 0.0).then_some(-mean),
            Self::Binomial { trials } => {
                let trials = trials as f64;
                if mean < 0.0 {
                    Some(-mean)
                } else {
                    (mean > trials).then_some(mean - trials)
                }
            }
        }
    }

    /// The allowed mean nearest to `mean`: `mean` itself when
    /// [`Self::mean_outside`] allows it, the smallest positive number for a
    /// log-normal model, and the nearest end otherwise.
    pub fn nearest_mean(&self, mean: f64) -> f64 {
        match *self {
            Self::Normal { .. } | Self::StudentT { .. } => mean,
            Self::LogNormal { .. } => mean.max(f64::MIN_POSITIVE),
            Self::Poisson | Self::NegativeBinomial { .. } => mean.max(0.0),
            Self::Binomial { trials } => mean.clamp(0.0, trials as f64),
        }
    }

    /// The log density (or log probability) of `y` when the predicted mean
    /// is `mean`, which [`Self::mean_outside`] allows; `y` lies in the
    /// support of the model. `−∞` where the model gives `y` no mass.
    pub fn log_density(&self, y: f64, mean: f64) -> f64 {
        match *self {
            Self::Normal { sd } => {
                let z = (y - mean) / sd;
                normal_log_density(z * z, 1, libm::log(sd))
            }
            Self::LogNormal { sigma, center } => {
                log_normal_log_density(y, log_location(mean, sigma, center), sigma)
            }
            Self::StudentT { df, scale } => {
                let z = (y - mean) / scale;
                student_t_log_density(z * z, df, 1, libm::log(scale))
            }
            Self::Poisson => negative_binomial(y, mean, 0.0),
            Self::NegativeBinomial { overdispersion } => negative_binomial(y, mean, overdispersion),
            Self::Binomial { trials } => {
                let n = trials as f64;
                if n == 0.0 {
                    return 0.0;
                }
                let p = mean / n;
                ln_gamma(n + 1.0) - ln_gamma(y + 1.0) - ln_gamma(n - y + 1.0)
                    + x_log_y(y, p)
                    + x_log1p_y(n - y, -p)
            }
        }
    }

    /// The expected Fisher information of one observation at the mean
    /// `mean` with respect to its arguments `(μ, noise arguments…)`, the
    /// noise arguments in the order of the model's fields, row-major: `E[s sᵀ]`, with `s` the
    /// derivatives of `log p(y | μ, …)`.
    ///
    /// | Model | `μ` | Noise arguments |
    /// |---|---|---|
    /// | `Normal` | `1/σ²` | `2/σ²` |
    /// | `LogNormal` | `1/(μ²σ²)` | `2/σ²`; with a `Mean` center `1 + 2/σ²` and `−1/(μσ)` with `μ` |
    /// | `StudentT` | `(ν+1)/((ν+3)s²)` | `ν`: `¼(ψ'(ν/2) − ψ'((ν+1)/2)) − (ν+5)/(2ν(ν+1)(ν+3))`; `s`: `2ν/((ν+3)s²)`; both: `−2/((ν+1)(ν+3)s)` |
    /// | `Poisson` | `1/μ` | |
    /// | `NegativeBinomial` | `1/(μ + φμ²)` | `φ`: the variance of its score, summed over the counts |
    /// | `Binomial` | `1/(n p (1 − p))`, `p = μ/n` | |
    ///
    /// The mean is orthogonal to the noise arguments in every model but the
    /// log-normal of a `Mean` center, whose location `log μ − σ²/2` reads
    /// `σ`. A mean on the edge of the means of a count model gives an
    /// infinite information.
    pub fn information(&self, mean: f64) -> Vec<f64> {
        match *self {
            Self::Normal { sd } => vec![1.0 / (sd * sd), 0.0, 0.0, 2.0 / (sd * sd)],
            Self::LogNormal { sigma, center } => {
                let variance = sigma * sigma;
                let mean_mean = 1.0 / (mean * mean * variance);
                match center {
                    Center::Median => vec![mean_mean, 0.0, 0.0, 2.0 / variance],
                    Center::Mean => {
                        let cross = -1.0 / (mean * sigma);
                        vec![mean_mean, cross, cross, 1.0 + 2.0 / variance]
                    }
                }
            }
            Self::StudentT { df, scale } => {
                let nu = df;
                let s2 = scale * scale;
                let mean_mean = (nu + 1.0) / ((nu + 3.0) * s2);
                let df_df = 0.25 * (trigamma(0.5 * nu) - trigamma(0.5 * (nu + 1.0)))
                    - (nu + 5.0) / (2.0 * nu * (nu + 1.0) * (nu + 3.0));
                let df_scale = -2.0 / ((nu + 1.0) * (nu + 3.0) * scale);
                let scale_scale = 2.0 * nu / ((nu + 3.0) * s2);
                vec![
                    mean_mean,
                    0.0,
                    0.0,
                    0.0,
                    df_df,
                    df_scale,
                    0.0,
                    df_scale,
                    scale_scale,
                ]
            }
            Self::Poisson => vec![1.0 / mean],
            Self::NegativeBinomial { overdispersion } => vec![
                1.0 / (mean + overdispersion * mean * mean),
                0.0,
                0.0,
                dispersion_information(mean, overdispersion),
            ],
            Self::Binomial { trials } => {
                let n = trials as f64;
                if n == 0.0 {
                    return vec![0.0];
                }
                let p = mean / n;
                vec![1.0 / (n * p * (1.0 - p))]
            }
        }
    }

    /// The derivatives of `log p(y | μ, …)` at the mean `mean` with respect
    /// to the arguments `(μ, noise arguments…)` of [`Self::information`]:
    /// the scores of one observation, whose variance is
    /// [`Self::information`].
    pub fn scores(&self, y: f64, mean: f64) -> Vec<f64> {
        match *self {
            Self::Normal { sd } => {
                let z = (y - mean) / sd;
                vec![z / sd, (z * z - 1.0) / sd]
            }
            Self::LogNormal { sigma, center } => {
                let z = (libm::log(y) - log_location(mean, sigma, center)) / sigma;
                // The location `log μ − σ²/2` of a `Mean` center moves with σ.
                let location_slope = match center {
                    Center::Mean => -sigma,
                    Center::Median => 0.0,
                };
                vec![
                    z / (sigma * mean),
                    (z * z - 1.0) / sigma + z * location_slope / sigma,
                ]
            }
            Self::StudentT { df, scale } => {
                let d = y - mean;
                let z2 = d * d / (scale * scale);
                vec![
                    (df + 1.0) * d / (df * scale * scale + d * d),
                    0.5 * (digamma(0.5 * (df + 1.0)) - digamma(0.5 * df))
                        - 0.5 / df
                        - 0.5 * libm::log1p(z2 / df)
                        + (df + 1.0) * z2 / (2.0 * df * (df + z2)),
                    -1.0 / scale + (df + 1.0) * z2 / (scale * (df + z2)),
                ]
            }
            Self::Poisson => vec![y / mean - 1.0],
            Self::NegativeBinomial { overdispersion } => {
                let phi = overdispersion;
                vec![
                    (y - mean) / (mean * (1.0 + phi * mean)),
                    dispersion_score(y, mean, phi, dispersion_first_terms(y, phi)),
                ]
            }
            Self::Binomial { trials } => {
                let n = trials as f64;
                if n == 0.0 {
                    return vec![0.0];
                }
                vec![(y - mean) * n / (mean * (n - mean))]
            }
        }
    }

    /// The log density of `y` at a mean far from it: `distance` above `y`,
    /// or, for a binomial, half a trial from the end of its trials farther
    /// from `y`. [`crate::OptimizationObjective`] scales its penalty with it.
    pub(crate) fn far_log_density(&self, y: f64, distance: f64) -> f64 {
        match *self {
            Self::Binomial { trials } => {
                let n = trials as f64;
                let mean = if n == 0.0 {
                    0.0
                } else if y >= 0.5 * n {
                    0.5
                } else {
                    n - 0.5
                };
                self.log_density(y, mean)
            }
            _ => self.log_density(y, y.max(0.0) + distance),
        }
    }
}

/// The location of `log Y` of a log-normal model whose `center` is `mean`.
pub fn log_location(mean: f64, sigma: f64, center: Center) -> f64 {
    match center {
        Center::Mean => libm::log(mean) - 0.5 * sigma * sigma,
        Center::Median => libm::log(mean),
    }
}

/// The log probability of the count `y` under a negative binomial of mean
/// `mean ≥ 0` and overdispersion `φ ≥ 0`, written so that `φ = 0` is the
/// Poisson:
///
/// `Σ_{j<y} ln(1 + jφ) + y ln μ − y ln(1 + φμ) − ln(1 + φμ)/φ − ln y!`.
fn negative_binomial(y: f64, mean: f64, overdispersion: f64) -> f64 {
    if mean == 0.0 {
        return if y == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    let phi = overdispersion;
    let first_terms = if y < SUM_TERMS || phi < SMALL_OVERDISPERSION {
        (0..y as u64)
            .map(|j| libm::log1p(j as f64 * phi))
            .sum::<f64>()
    } else {
        let r = 1.0 / phi;
        ln_gamma(y + r) - ln_gamma(r) - y * libm::log(r)
    };
    let a = phi * mean;
    let size_term = if phi == 0.0 {
        mean
    } else {
        libm::log1p(a) / phi
    };
    first_terms + y * libm::log(mean) - x_log1p_y(y, a) - size_term - ln_gamma(y + 1.0)
}

/// Below this `φμ`, `ln(1 + φμ) − φμ/(1 + φμ)` is read from its series.
const SMALL_PRODUCT: f64 = 1e-4;
/// The counts summed for the information of a dispersion stop once a term
/// falls below this share of the sum, above the mean.
const NEGLIGIBLE: f64 = 1e-17;

/// `(ln(1 + a) − a/(1 + a)) / φ²` with `a = φμ`, accurate as `φ → 0`, where
/// it tends to `μ²/2`.
fn dispersion_bracket(mean: f64, phi: f64) -> f64 {
    let a = phi * mean;
    if a < SMALL_PRODUCT {
        mean * mean * (0.5 - 2.0 * a / 3.0 + 0.75 * a * a)
    } else {
        (libm::log1p(a) - a / (1.0 + a)) / (phi * phi)
    }
}

/// The derivative of the log probability of the count `y` with respect to
/// the overdispersion `φ > 0` of a negative binomial of mean `mean`, given
/// `Σ_{j<y} j/(1 + jφ)`:
///
/// `Σ_{j<y} j/(1 + jφ) − yμ/(1 + φμ) + (ln(1 + φμ) − φμ/(1 + φμ))/φ²`.
fn dispersion_score(y: f64, mean: f64, phi: f64, first_terms: f64) -> f64 {
    first_terms - y * mean / (1.0 + phi * mean) + dispersion_bracket(mean, phi)
}

/// `Σ_{j<y} j/(1 + jφ)`: the sum of its terms below [`SUM_TERMS`] or
/// [`SMALL_OVERDISPERSION`], otherwise `y/φ − (ψ(y + 1/φ) − ψ(1/φ))/φ²`.
fn dispersion_first_terms(y: f64, phi: f64) -> f64 {
    if y < SUM_TERMS || phi < SMALL_OVERDISPERSION {
        (0..y as u64)
            .map(|j| j as f64 / (1.0 + j as f64 * phi))
            .sum()
    } else {
        let r = 1.0 / phi;
        y / phi - (digamma(y + r) - digamma(r)) / (phi * phi)
    }
}

/// The Fisher information of the overdispersion `φ` of a negative binomial
/// of mean `mean`: `Σ_y p(y) s(y)²`, a sum of positive terms over the
/// counts, from 0 until the terms above the mean are negligible.
fn dispersion_information(mean: f64, phi: f64) -> f64 {
    if mean <= 0.0 {
        return 0.0;
    }
    let r = 1.0 / phi;
    // `ln p(y)`, from `p(0) = (1 + φμ)^(−1/φ)` and
    // `p(y + 1) / p(y) = (y + r)/(y + 1) · φμ/(1 + φμ)`.
    let mut log_probability = -libm::log1p(phi * mean) / phi;
    let log_ratio = libm::log(phi * mean) - libm::log1p(phi * mean);
    let mut first_terms = 0.0;
    let mut total = 0.0;
    let mut y = 0.0;
    loop {
        let score = dispersion_score(y, mean, phi, first_terms);
        let term = libm::exp(log_probability) * score * score;
        total += term;
        if y > mean && term <= NEGLIGIBLE * total && libm::exp(log_probability) <= NEGLIGIBLE {
            return total;
        }
        first_terms += y / (1.0 + y * phi);
        log_probability += libm::log((y + r) / (y + 1.0)) + log_ratio;
        y += 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_counts_sum_the_dispersion_score_by_digamma() {
        let (y, phi) = (1500.0, 0.01);
        let summed: f64 = (0..1500).map(|j| j as f64 / (1.0 + j as f64 * phi)).sum();
        let by_digamma = dispersion_first_terms(y, phi);
        assert!(
            (summed - by_digamma).abs() < 1e-9 * summed,
            "{summed} {by_digamma}"
        );
    }

    #[test]
    fn no_trials_and_no_mean_carry_no_information() {
        let binomial = Noise::Binomial { trials: 0 };
        assert_eq!(binomial.information(0.0), vec![0.0]);
        assert_eq!(binomial.scores(0.0, 0.0), vec![0.0]);
        assert_eq!(dispersion_information(0.0, 0.5), 0.0);
    }

    #[test]
    fn the_dispersion_bracket_joins_its_series() {
        let (mean, phi) = (3.0, SMALL_PRODUCT / 3.0);
        let series = dispersion_bracket(mean, phi * 0.999_999);
        let direct = dispersion_bracket(mean, phi * 1.000_001);
        assert!((series - direct).abs() < 1e-6 * direct, "{series} {direct}");
    }

    #[test]
    fn the_negative_binomial_of_no_overdispersion_is_the_poisson() {
        let poisson = Noise::Poisson.log_density(4.0, 2.5);
        let limit = Noise::NegativeBinomial {
            overdispersion: 1e-12,
        }
        .log_density(4.0, 2.5);
        assert!((poisson - limit).abs() < 1e-9);
    }

    #[test]
    fn large_counts_read_the_gamma_function() {
        // Both forms of the first terms agree where both are accurate.
        let y = 1500.0;
        let phi = 0.01;
        let summed: f64 = (0..1500).map(|j| libm::log1p(j as f64 * phi)).sum();
        let r = 1.0 / phi;
        let gamma = ln_gamma(y + r) - ln_gamma(r) - y * libm::log(r);
        assert!((summed - gamma).abs() < 1e-8 * summed.abs());
    }

    #[test]
    fn a_mean_of_zero_gives_mass_only_to_zero() {
        assert_eq!(Noise::Poisson.log_density(0.0, 0.0), 0.0);
        assert_eq!(Noise::Poisson.log_density(1.0, 0.0), f64::NEG_INFINITY);
        assert_eq!(Noise::Binomial { trials: 0 }.log_density(0.0, 0.0), 0.0);
        assert_eq!(Noise::Binomial { trials: 3 }.log_density(3.0, 3.0), 0.0);
    }
}
