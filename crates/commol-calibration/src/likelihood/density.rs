//! The log density of each observation model at an observed value, given its
//! predicted mean and its noise.
//!
//! Every elementary function comes from `libm`, so a value is the same on
//! every platform.

use std::f64::consts::PI;

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
                -0.5 * libm::log(2.0 * PI) - libm::log(sd) - 0.5 * z * z
            }
            Self::LogNormal { sigma, center } => {
                let z = (libm::log(y) - log_location(mean, sigma, center)) / sigma;
                -libm::log(y) - libm::log(sigma) - 0.5 * libm::log(2.0 * PI) - 0.5 * z * z
            }
            Self::StudentT { df, scale } => {
                let z = (y - mean) / scale;
                libm::lgamma(0.5 * (df + 1.0))
                    - libm::lgamma(0.5 * df)
                    - 0.5 * libm::log(df * PI)
                    - libm::log(scale)
                    - 0.5 * (df + 1.0) * libm::log1p(z * z / df)
            }
            Self::Poisson => negative_binomial(y, mean, 0.0),
            Self::NegativeBinomial { overdispersion } => negative_binomial(y, mean, overdispersion),
            Self::Binomial { trials } => {
                let n = trials as f64;
                if n == 0.0 {
                    return 0.0;
                }
                let p = mean / n;
                libm::lgamma(n + 1.0) - libm::lgamma(y + 1.0) - libm::lgamma(n - y + 1.0)
                    + x_log(y, p)
                    + x_log1p(n - y, -p)
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

/// `x · ln y`, with `0 · ln 0 = 0`.
fn x_log(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * libm::log(y)
    }
}

/// `x · ln(1 + y)`, with `0 · ln 0 = 0`.
fn x_log1p(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * libm::log1p(y)
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
        libm::lgamma(y + r) - libm::lgamma(r) - y * libm::log(r)
    };
    let a = phi * mean;
    let size_term = if phi == 0.0 {
        mean
    } else {
        libm::log1p(a) / phi
    };
    first_terms + y * libm::log(mean) - x_log1p(y, a) - size_term - libm::lgamma(y + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let gamma = libm::lgamma(y + r) - libm::lgamma(r) - y * libm::log(r);
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
