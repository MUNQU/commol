//! The one family of distributions of commol: priors, and later the inputs
//! of a sensitivity analysis and the noise parameters of an observation
//! model.
//!
//! A [`Distribution`] is a [`Family`] truncated to the bounds of a
//! parameter. `Uniform`, `LogUniform` and `Beta` live on the bounds; the
//! others are truncated to them, and `LogNormal` and `Gamma` also to
//! `(0, ∞)`.
//!
//! A draw inverts the distribution function of one uniform number of the
//! [`Generator`], so a seed gives the same draws on every platform: the
//! special functions come from [`crate::special`], on `libm`.

use std::fmt;

use commol_random::Generator;

use crate::special::{
    beta_i, gamma_p, gamma_q, ln_beta, ln_gamma, normal_cdf, normal_density, normal_quantile,
    normal_quantile_sf, normal_sf,
};

/// Iterations of the inversion of a distribution function by safeguarded
/// Newton steps.
const INVERSION_STEPS: usize = 200;

/// A family of distributions and its parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family {
    /// Uniform on the bounds.
    Uniform,
    /// Uniform in the logarithm, on the bounds: density `∝ 1/x`.
    LogUniform,
    /// Beta of shapes `a` and `b`, scaled to the bounds.
    Beta {
        /// The first shape, above 0.
        a: f64,
        /// The second shape, above 0.
        b: f64,
    },
    /// Normal of mean `mean` and standard deviation `sd`.
    Normal {
        /// The mean.
        mean: f64,
        /// The standard deviation, above 0.
        sd: f64,
    },
    /// Log-normal: `log x` is normal of mean `mu` and standard deviation
    /// `sigma`.
    LogNormal {
        /// The mean of `log x`.
        mu: f64,
        /// The standard deviation of `log x`, above 0.
        sigma: f64,
    },
    /// Gamma of shape `shape` and rate `rate`: mean `shape / rate`.
    Gamma {
        /// The shape, above 0.
        shape: f64,
        /// The rate, above 0.
        rate: f64,
    },
}

/// The kinds of [`Family`], in the order messages list them.
const KINDS: [&str; 6] = [
    "Uniform",
    "LogUniform",
    "Beta",
    "Normal",
    "LogNormal",
    "Gamma",
];

impl Family {
    /// The family of a kind and its parameters, in the order of the
    /// variant's fields: `"uniform"`, `"log_uniform"`, `"beta"`, `"normal"`,
    /// `"log_normal"` or `"gamma"`.
    ///
    /// # Errors
    ///
    /// Returns a message when the kind is unknown or the number of
    /// parameters is wrong.
    pub fn from_name(kind: &str, parameters: &[f64]) -> Result<Self, String> {
        let family = match (kind, parameters) {
            ("uniform", []) => Self::Uniform,
            ("log_uniform", []) => Self::LogUniform,
            ("beta", &[a, b]) => Self::Beta { a, b },
            ("normal", &[mean, sd]) => Self::Normal { mean, sd },
            ("log_normal", &[mu, sigma]) => Self::LogNormal { mu, sigma },
            ("gamma", &[shape, rate]) => Self::Gamma { shape, rate },
            _ => {
                return Err(format!(
                    "Unknown distribution '{kind}' with {} parameters",
                    parameters.len()
                ));
            }
        };
        Ok(family)
    }

    /// The kind, as the Python class names it, such as `"Beta"`.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Uniform => KINDS[0],
            Self::LogUniform => KINDS[1],
            Self::Beta { .. } => KINDS[2],
            Self::Normal { .. } => KINDS[3],
            Self::LogNormal { .. } => KINDS[4],
            Self::Gamma { .. } => KINDS[5],
        }
    }

    /// Why bounds `[lower, upper]` do not fit the kind of this family, or
    /// `None` when they fit.
    pub fn bound_problem(&self, lower: f64, upper: f64) -> Option<&'static str> {
        kind_bound_problem(self.kind(), lower, upper)
    }

    fn check_parameters(&self) -> Result<(), String> {
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!(
                    "{} needs a finite {name} above 0, got {value}",
                    self.kind()
                ))
            }
        };
        let finite = |name: &str, value: f64| {
            if value.is_finite() {
                Ok(())
            } else {
                Err(format!(
                    "{} needs a finite {name}, got {value}",
                    self.kind()
                ))
            }
        };
        match *self {
            Self::Uniform | Self::LogUniform => Ok(()),
            Self::Beta { a, b } => positive("a", a).and(positive("b", b)),
            Self::Normal { mean, sd } => finite("mean", mean).and(positive("sd", sd)),
            Self::LogNormal { mu, sigma } => finite("mu", mu).and(positive("sigma", sigma)),
            Self::Gamma { shape, rate } => positive("shape", shape).and(positive("rate", rate)),
        }
    }
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uniform => f.write_str("Uniform()"),
            Self::LogUniform => f.write_str("LogUniform()"),
            Self::Beta { a, b } => write!(f, "Beta(a={a}, b={b})"),
            Self::Normal { mean, sd } => write!(f, "Normal(mean={mean}, sd={sd})"),
            Self::LogNormal { mu, sigma } => write!(f, "LogNormal(mu={mu}, sigma={sigma})"),
            Self::Gamma { shape, rate } => write!(f, "Gamma(shape={shape}, rate={rate})"),
        }
    }
}

fn kind_bound_problem(kind: &str, lower: f64, upper: f64) -> Option<&'static str> {
    match kind {
        "Uniform" | "Beta" if !(lower.is_finite() && upper.is_finite()) => {
            Some("needs finite bounds")
        }
        "LogUniform" if !(lower.is_finite() && upper.is_finite()) => Some("needs finite bounds"),
        "LogUniform" if lower <= 0.0 => Some("needs a lower bound above 0"),
        "LogNormal" | "Gamma" if upper <= 0.0 => Some("needs an upper bound above 0"),
        _ => None,
    }
}

/// The kinds of distribution whose bounds `[lower, upper]` fit, in order.
pub fn kinds_that_fit(lower: f64, upper: f64) -> Vec<&'static str> {
    KINDS
        .into_iter()
        .filter(|kind| kind_bound_problem(kind, lower, upper).is_none())
        .collect()
}

/// A [`Family`] truncated to `[lower, upper]`.
///
/// The truncated distribution function is `(F(x) − F(l)) / (F(u) − F(l))`,
/// or, when the bounds lie in the upper half of the family, the same with
/// the survival function `S = 1 − F`, which keeps its accuracy in that tail.
#[derive(Debug, Clone, PartialEq)]
pub struct Distribution {
    family: Family,
    lower: f64,
    upper: f64,
    /// The lower end of the support inside the bounds.
    low: f64,
    /// Whether the truncation reads the survival function.
    upper_tail: bool,
    /// `F(low)`, or `S(low)` in the upper tail.
    at_low: f64,
    /// `F(upper) − F(low)`, or `S(low) − S(upper)`: the mass of the bounds.
    mass: f64,
}

impl Distribution {
    /// `family` truncated to `[lower, upper]`.
    ///
    /// # Errors
    ///
    /// Returns a message when the bounds are not ordered, a parameter of the
    /// family is not valid, the bounds do not fit the family
    /// ([`Family::bound_problem`]), or the family has no mass between them.
    pub fn new(family: Family, lower: f64, upper: f64) -> Result<Self, String> {
        if lower.is_nan() || upper.is_nan() || lower >= upper {
            return Err(format!(
                "A distribution needs bounds with lower < upper, got [{lower:?}, {upper:?}]"
            ));
        }
        family.check_parameters()?;
        if let Some(problem) = family.bound_problem(lower, upper) {
            return Err(format!("{family} {problem}, got [{lower:?}, {upper:?}]"));
        }
        let low = match family {
            Family::LogNormal { .. } | Family::Gamma { .. } => lower.max(0.0),
            _ => lower,
        };
        let mut distribution = Self {
            family,
            lower,
            upper,
            low,
            upper_tail: false,
            at_low: 0.0,
            mass: 1.0,
        };
        let cdf_low = distribution.base_cdf(low);
        distribution.upper_tail = cdf_low > 0.5;
        let (at_low, at_upper) = if distribution.upper_tail {
            (distribution.base_sf(low), distribution.base_sf(upper))
        } else {
            (cdf_low, distribution.base_cdf(upper))
        };
        let mass = (at_upper - at_low).abs();
        if mass.is_nan() || mass <= 0.0 {
            return Err(format!(
                "{family} has no mass between the bounds [{lower:?}, {upper:?}]"
            ));
        }
        distribution.at_low = at_low;
        distribution.mass = mass;
        Ok(distribution)
    }

    /// The family.
    pub fn family(&self) -> Family {
        self.family
    }

    /// The bounds `(lower, upper)` the family is truncated to.
    pub fn bounds(&self) -> (f64, f64) {
        (self.lower, self.upper)
    }

    /// The untruncated distribution function; `Uniform`, `LogUniform` and
    /// `Beta` live on the bounds.
    fn base_cdf(&self, x: f64) -> f64 {
        let (lower, upper) = (self.lower, self.upper);
        match self.family {
            Family::Uniform => ((x - lower) / (upper - lower)).clamp(0.0, 1.0),
            Family::LogUniform => (libm::log(x / lower) / libm::log(upper / lower)).clamp(0.0, 1.0),
            Family::Beta { a, b } => beta_i(a, b, (x - lower) / (upper - lower)),
            Family::Normal { mean, sd } => normal_cdf((x - mean) / sd),
            Family::LogNormal { mu, sigma } => {
                if x <= 0.0 {
                    0.0
                } else {
                    normal_cdf((libm::log(x) - mu) / sigma)
                }
            }
            Family::Gamma { shape, rate } => gamma_p(shape, rate * x),
        }
    }

    /// The untruncated survival function `1 − F(x)`.
    fn base_sf(&self, x: f64) -> f64 {
        match self.family {
            Family::Normal { mean, sd } => normal_sf((x - mean) / sd),
            Family::LogNormal { mu, sigma } => {
                if x <= 0.0 {
                    1.0
                } else {
                    normal_sf((libm::log(x) - mu) / sigma)
                }
            }
            Family::Gamma { shape, rate } => gamma_q(shape, rate * x),
            _ => 1.0 - self.base_cdf(x),
        }
    }

    /// The untruncated log density at `x`, inside the support.
    fn base_log_density(&self, x: f64) -> f64 {
        let (lower, upper) = (self.lower, self.upper);
        match self.family {
            Family::Uniform => -libm::log(upper - lower),
            Family::LogUniform => -libm::log(x) - libm::log(libm::log(upper / lower)),
            Family::Beta { a, b } => {
                let width = upper - lower;
                let t = (x - lower) / width;
                x_log_y(a - 1.0, t) + x_log_y(b - 1.0, 1.0 - t) - ln_beta(a, b) - libm::log(width)
            }
            Family::Normal { mean, sd } => {
                let z = (x - mean) / sd;
                -0.5 * z * z - libm::log(sd) - 0.5 * libm::log(2.0 * std::f64::consts::PI)
            }
            Family::LogNormal { mu, sigma } => {
                if x <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                let z = (libm::log(x) - mu) / sigma;
                -0.5 * z * z - libm::log(x * sigma) - 0.5 * libm::log(2.0 * std::f64::consts::PI)
            }
            Family::Gamma { shape, rate } => {
                if x < 0.0 || (x == 0.0 && shape > 1.0) {
                    return f64::NEG_INFINITY;
                }
                shape * libm::log(rate) + x_log_y(shape - 1.0, x) - rate * x - ln_gamma(shape)
            }
        }
    }

    /// The log density of the truncated distribution: `−∞` outside the
    /// bounds.
    pub fn log_density(&self, x: f64) -> f64 {
        if !(x >= self.low && x <= self.upper) {
            return f64::NEG_INFINITY;
        }
        self.base_log_density(x) - libm::log(self.mass)
    }

    /// The distribution function of the truncated distribution: 0 at and
    /// below the lower bound, 1 at and above the upper.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= self.low {
            return 0.0;
        }
        if x >= self.upper {
            return 1.0;
        }
        let p = if self.upper_tail {
            (self.at_low - self.base_sf(x)) / self.mass
        } else {
            (self.base_cdf(x) - self.at_low) / self.mass
        };
        p.clamp(0.0, 1.0)
    }

    /// The quantile of the truncated distribution at `p`, in `[0, 1]`: the
    /// lower bound at 0, the upper at 1.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `p` is not in `[0, 1]`.
    pub fn quantile(&self, p: f64) -> f64 {
        debug_assert!((0.0..=1.0).contains(&p), "p = {p}");
        if p <= 0.0 {
            return self.low;
        }
        if p >= 1.0 {
            return self.upper;
        }
        let (lower, upper) = (self.lower, self.upper);
        let x = match self.family {
            Family::Uniform => lower + p * (upper - lower),
            Family::LogUniform => lower * libm::pow(upper / lower, p),
            Family::Normal { mean, sd } => mean + sd * self.standard_quantile(p),
            Family::LogNormal { mu, sigma } => libm::exp(mu + sigma * self.standard_quantile(p)),
            Family::Beta { .. } | Family::Gamma { .. } => self.invert(p),
        };
        x.clamp(self.low, self.upper)
    }

    /// The standard normal quantile of the truncated level `p`, for the
    /// normal and log-normal families.
    fn standard_quantile(&self, p: f64) -> f64 {
        if self.upper_tail {
            normal_quantile_sf(self.at_low - p * self.mass)
        } else {
            normal_quantile(self.at_low + p * self.mass)
        }
    }

    /// The `x` with `cdf(x) = p`, by Newton steps kept inside a bracket that
    /// bisection narrows.
    fn invert(&self, p: f64) -> f64 {
        let mut low = self.low;
        let mut high = self.upper;
        if high.is_infinite() {
            let scale = match self.family {
                Family::Gamma { shape, rate } => (shape / rate).max(1.0 / rate),
                _ => 1.0,
            };
            high = low.max(0.0) + scale;
            while self.cdf(high) < p {
                low = high;
                high = 2.0 * high + scale;
            }
        }
        let mut x = 0.5 * (low + high);
        for _ in 0..INVERSION_STEPS {
            let error = self.cdf(x) - p;
            if error == 0.0 {
                return x;
            }
            if error > 0.0 {
                high = x;
            } else {
                low = x;
            }
            let density = self.log_density(x).exp();
            let newton = x - error / density;
            let next = if density > 0.0 && newton > low && newton < high {
                newton
            } else {
                0.5 * (low + high)
            };
            if (next - x).abs() <= 4.0 * f64::EPSILON * x.abs() || next == low || next == high {
                return next;
            }
            x = next;
        }
        x
    }

    /// The mean of the truncated distribution.
    pub fn mean(&self) -> f64 {
        self.moments().0
    }

    /// The variance of the truncated distribution.
    pub fn variance(&self) -> f64 {
        self.moments().1
    }

    /// The mean and the variance, in closed form.
    fn moments(&self) -> (f64, f64) {
        let (lower, upper) = (self.lower, self.upper);
        let width = upper - lower;
        match self.family {
            Family::Uniform => (0.5 * (lower + upper), width * width / 12.0),
            Family::LogUniform => {
                let log_ratio = libm::log(upper / lower);
                let mean = width / log_ratio;
                let second = (upper * upper - lower * lower) / (2.0 * log_ratio);
                (mean, second - mean * mean)
            }
            Family::Beta { a, b } => {
                let total = a + b;
                (
                    lower + width * a / total,
                    width * width * a * b / (total * total * (total + 1.0)),
                )
            }
            Family::Normal { mean, sd } => {
                let alpha = (self.low - mean) / sd;
                let beta = (upper - mean) / sd;
                let (pa, pb) = (normal_density(alpha), normal_density(beta));
                let shift = (pa - pb) / self.mass;
                let edge = |z: f64, p: f64| if z.is_infinite() { 0.0 } else { z * p };
                let spread = (edge(alpha, pa) - edge(beta, pb)) / self.mass;
                (mean + sd * shift, sd * sd * (1.0 + spread - shift * shift))
            }
            Family::LogNormal { mu, sigma } => {
                let alpha = if self.low > 0.0 {
                    (libm::log(self.low) - mu) / sigma
                } else {
                    f64::NEG_INFINITY
                };
                let beta = (libm::log(upper) - mu) / sigma;
                let moment = |k: f64| {
                    let between = self.normal_between(alpha - k * sigma, beta - k * sigma);
                    libm::exp(k * mu + 0.5 * k * k * sigma * sigma) * between / self.mass
                };
                let mean = moment(1.0);
                (mean, moment(2.0) - mean * mean)
            }
            Family::Gamma { shape, rate } => {
                let (low, high) = (rate * self.low, rate * upper);
                let moment = |k: f64| {
                    let between = if self.upper_tail {
                        gamma_q(shape + k, low) - gamma_q(shape + k, high)
                    } else {
                        gamma_p(shape + k, high) - gamma_p(shape + k, low)
                    };
                    libm::exp(ln_gamma(shape + k) - ln_gamma(shape) - k * libm::log(rate)) * between
                        / self.mass
                };
                let mean = moment(1.0);
                (mean, moment(2.0) - mean * mean)
            }
        }
    }

    /// `Φ(b) − Φ(a)`, by the survival function in the upper tail.
    fn normal_between(&self, a: f64, b: f64) -> f64 {
        if self.upper_tail {
            normal_sf(a) - normal_sf(b)
        } else {
            normal_cdf(b) - normal_cdf(a)
        }
    }

    /// One draw: the quantile of a uniform number in `(0, 1)`.
    pub fn sample(&self, generator: &mut Generator) -> f64 {
        self.quantile(open_uniform(generator))
    }
}

/// `c · ln y`, 0 when `c` is 0, also at `y = 0`.
fn x_log_y(c: f64, y: f64) -> f64 {
    if c == 0.0 { 0.0 } else { c * libm::log(y) }
}

/// A uniform number strictly between 0 and 1: the 53 high bits of the
/// generator, at the middle of their interval.
pub(crate) fn open_uniform(generator: &mut Generator) -> f64 {
    ((generator.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_family_is_read_from_its_name() {
        assert_eq!(
            Family::from_name("gamma", &[2.0, 3.0]),
            Ok(Family::Gamma {
                shape: 2.0,
                rate: 3.0
            })
        );
        assert!(Family::from_name("normal", &[1.0]).is_err());
        assert_eq!(
            Family::Beta { a: 2.0, b: 0.5 }.to_string(),
            "Beta(a=2, b=0.5)"
        );
    }

    #[test]
    fn the_kinds_that_fit_the_bounds_are_listed_in_order() {
        assert_eq!(
            kinds_that_fit(0.0, f64::INFINITY),
            ["Normal", "LogNormal", "Gamma"]
        );
        assert_eq!(
            kinds_that_fit(0.0, 1.0),
            ["Uniform", "Beta", "Normal", "LogNormal", "Gamma"]
        );
        assert_eq!(kinds_that_fit(-2.0, -1.0), ["Uniform", "Beta", "Normal"]);
    }

    #[test]
    fn a_normal_far_in_its_upper_tail_keeps_its_accuracy() {
        // 30 standard deviations up, the truncated normal is nearly an
        // exponential of rate 30 above its lower bound.
        let d = Distribution::new(Family::Normal { mean: 0.0, sd: 1.0 }, 30.0, 31.0).unwrap();
        assert!(
            (d.mean() - (30.0 + 1.0 / 30.0)).abs() < 1e-3,
            "{}",
            d.mean()
        );
        assert!((d.quantile(0.5) - (30.0 + 2f64.ln() / 30.0)).abs() < 1e-3);
        assert!((d.cdf(d.quantile(0.3)) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn a_family_with_no_mass_between_the_bounds_is_rejected() {
        let error =
            Distribution::new(Family::Normal { mean: 0.0, sd: 1.0 }, 100.0, 101.0).unwrap_err();
        assert!(error.contains("no mass"), "{error}");
    }
}
