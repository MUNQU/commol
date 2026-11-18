//! The density of a sum of independent observations, one per part of the
//! window of an observation ([`crate::ObservedDataPoint::parts`]), each
//! scattered by the model of its series around the mean `μₜ` of its part,
//! with `M = Σ μₜ` and `K` parts.
//!
//! | Model | Density of the sum |
//! |---|---|
//! | `Poisson` | `Poisson(M)`, exact |
//! | `Normal` | `N(M, K · sd²)`, exact |
//! | `NegativeBinomial`, [`Sum::Exact`] | the distribution of the sum (`Recursion`) |
//! | `NegativeBinomial`, [`Sum::Moments`] | `NB(M, φ · Σμₜ²/M²)`, the mean and the variance of the sum; the share `Σμₜ²/M²` is 1 when `M = 0` |
//!
//! The other models read no sum, and [`crate::Likelihood`] refuses their
//! parts. One part is the observation of its window, which [`Noise`] reads
//! alone, so every function here gives what [`Noise`] gives for one part.
//!
//! The scores and the information are over `(μ₁, …, μ_K, noise arguments…)`:
//! the derivatives of the density each reading uses. A reading through a
//! model of `M` chains the scores and the information of that model with the
//! derivatives of its arguments over the parts.

use serde::{Deserialize, Serialize};

use super::density::{NEGLIGIBLE, Noise, dispersion_bracket};

/// How a negative binomial reads the sum of the parts of a window:
/// `"exact"` or `"moments"` in the Python API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sum {
    /// The exact distribution of the sum of independent negative binomials
    /// (`Recursion`): `O(K·y + y²)` per evaluation at a total `y`.
    Exact,
    /// A negative binomial with the mean and the variance of the sum.
    Moments,
}

/// A value of the recursion above this is scaled down, with the values
/// before it.
const LARGE: f64 = 1e150;
/// The standard deviations above the mean that the tilt of the information
/// centres the recursion on.
const TILT_DEVIATIONS: f64 = 10.0;

/// How a model reads a sum of parts: through a model of the total, with the
/// derivatives of its arguments over the parts, or by the exact recursion of
/// a negative binomial of overdispersion `φ`.
enum Reading {
    /// The model of the total at its mean, and the Jacobian of its
    /// arguments `(M, noise arguments…)` over `(μ₁, …, μ_K, noise
    /// arguments…)`, row-major.
    Total {
        noise: Noise,
        mean: f64,
        jacobian: Vec<f64>,
    },
    /// The recursion of a negative binomial.
    Exact { overdispersion: f64 },
}

impl Noise {
    /// How far the means of `parts` lie outside the means this model allows,
    /// summed, or `None` when each is inside them ([`Noise::mean_outside`]).
    pub fn sum_mean_outside(&self, parts: &[f64]) -> Option<f64> {
        parts
            .iter()
            .filter_map(|&mean| self.mean_outside(mean))
            .reduce(|sum, distance| sum + distance)
    }

    /// The log density of the sum `y` of parts of means `parts`, each of
    /// which [`Noise::mean_outside`] allows.
    ///
    /// # Panics
    ///
    /// Panics for more than one part of a model that reads no sum, or of a
    /// negative binomial without a [`Sum`].
    pub fn sum_log_density(&self, y: f64, parts: &[f64]) -> f64 {
        if let [mean] = parts {
            return self.log_density(y, *mean);
        }
        match self.reading(parts) {
            Reading::Total { noise, mean, .. } => noise.log_density(y, mean),
            Reading::Exact { overdispersion } => {
                let last = count(y);
                let mut recursion = Recursion::new(parts, overdispersion, y, false);
                recursion.extend_to(last);
                recursion.log_probability(last)
            }
        }
    }

    /// The derivatives of the log density of the sum `y` over
    /// `(μ₁, …, μ_K, noise arguments…)`.
    ///
    /// # Panics
    ///
    /// As [`Self::sum_log_density`].
    pub fn sum_scores(&self, y: f64, parts: &[f64]) -> Vec<f64> {
        if let [mean] = parts {
            return self.scores(y, *mean);
        }
        match self.reading(parts) {
            Reading::Total {
                noise,
                mean,
                jacobian,
            } => {
                let scores = noise.scores(y, mean);
                let columns = jacobian.len() / scores.len();
                (0..columns)
                    .map(|c| {
                        scores
                            .iter()
                            .enumerate()
                            .map(|(r, score)| jacobian[r * columns + c] * score)
                            .sum()
                    })
                    .collect()
            }
            Reading::Exact { overdispersion } => {
                let last = count(y);
                let mut recursion = Recursion::new(parts, overdispersion, y, true);
                recursion.extend_to(last);
                let mut accumulated = vec![0.0; parts.len()];
                for n in 0..last {
                    recursion.accumulate(n, &mut accumulated);
                }
                recursion.scores(last, &accumulated)
            }
        }
    }

    /// The expected Fisher information of the sum over
    /// `(μ₁, …, μ_K, noise arguments…)`, row-major: that of the model of the
    /// total chained with the derivatives of its arguments, or for
    /// [`Sum::Exact`] `Σ_y p(y) s(y) s(y)ᵀ`, summed over the counts from 0
    /// until the terms above the mean are negligible.
    ///
    /// # Panics
    ///
    /// As [`Self::sum_log_density`].
    pub fn sum_information(&self, parts: &[f64]) -> Vec<f64> {
        if let [mean] = parts {
            return self.information(*mean);
        }
        match self.reading(parts) {
            Reading::Total {
                noise,
                mean,
                jacobian,
            } => {
                let information = noise.information(mean);
                let rows = information.len().isqrt();
                let columns = jacobian.len() / rows;
                let a = |r: usize, c: usize| jacobian[r * columns + c];
                let mut chained = vec![0.0; columns * columns];
                for c in 0..columns {
                    for d in 0..columns {
                        chained[c * columns + d] = (0..rows)
                            .flat_map(|r| (0..rows).map(move |s| (r, s)))
                            .map(|(r, s)| a(r, c) * information[r * rows + s] * a(s, d))
                            .sum();
                    }
                }
                chained
            }
            Reading::Exact { overdispersion } => exact_information(parts, overdispersion),
        }
    }

    /// How this model reads the sum of `parts`, more than one.
    fn reading(&self, parts: &[f64]) -> Reading {
        let k = parts.len();
        let mean: f64 = parts.iter().sum();
        let ones = || std::iter::repeat_n(1.0, k);
        match *self {
            Self::Poisson => Reading::Total {
                noise: Self::Poisson,
                mean,
                jacobian: ones().collect(),
            },
            Self::Normal { sd } => {
                // N(M, K sd²) is the normal of the sd √K sd.
                let root = (k as f64).sqrt();
                let mut jacobian: Vec<f64> = ones().chain([0.0]).collect();
                jacobian.extend(std::iter::repeat_n(0.0, k).chain([root]));
                Reading::Total {
                    noise: Self::Normal { sd: root * sd },
                    mean,
                    jacobian,
                }
            }
            Self::NegativeBinomial {
                overdispersion,
                sum: Some(Sum::Exact),
            } => Reading::Exact { overdispersion },
            // With no mean both readings are the mass at 0, and the scores of
            // the exact one are finite there.
            Self::NegativeBinomial {
                overdispersion,
                sum: Some(Sum::Moments),
            } if mean <= 0.0 => Reading::Exact { overdispersion },
            Self::NegativeBinomial {
                overdispersion,
                sum: Some(Sum::Moments),
            } => {
                // NB(M, φ s), s = Σμₜ²/M², and ∂s/∂μₜ = 2(μₜM − Σμ²)/M³.
                let squares: f64 = parts.iter().map(|mu| mu * mu).sum();
                let cube = mean * mean * mean;
                let share = squares / (mean * mean);
                let slopes: Vec<f64> = parts
                    .iter()
                    .map(|mu| 2.0 * (mu * mean - squares) / cube)
                    .collect();
                let mut jacobian: Vec<f64> = ones().chain([0.0]).collect();
                jacobian.extend(slopes.iter().map(|slope| overdispersion * slope));
                jacobian.push(share);
                Reading::Total {
                    noise: Self::NegativeBinomial {
                        overdispersion: overdispersion * share,
                        sum: None,
                    },
                    mean,
                    jacobian,
                }
            }
            // coverage: invariant, the likelihood refuses the parts of these models
            _ => unreachable!("the likelihood refuses a sum of parts of a {self:?} model"),
        }
    }
}

/// The count `y`, a whole number of at least 0, as an index.
fn count(y: f64) -> usize {
    // An observed count is a whole number of at least 0.
    y as usize
}

/// The exact information of a sum of negative binomials of overdispersion
/// `phi` and means `parts` over `(μ₁, …, μ_K, φ)`.
fn exact_information(parts: &[f64], phi: f64) -> Vec<f64> {
    let mean: f64 = parts.iter().sum();
    let variance: f64 = parts.iter().map(|mu| mu * (1.0 + phi * mu)).sum();
    let size = parts.len() + 1;
    let mut recursion = Recursion::new(
        parts,
        phi,
        mean + TILT_DEVIATIONS * variance.sqrt() + TILT_DEVIATIONS,
        true,
    );
    let mut accumulated = vec![0.0; parts.len()];
    let mut information = vec![0.0; size * size];
    let mut total = 0.0;
    let mut n = 0usize;
    loop {
        if n > 0 {
            let factor = recursion.extend_to(n);
            accumulated.iter_mut().for_each(|value| *value *= factor);
        }
        let probability = libm::exp(recursion.log_probability(n));
        // A count of no mass adds nothing: its scores can be infinite (a
        // count above 0 of parts of mean 0).
        let mut term = 0.0;
        if probability > 0.0 {
            let scores = recursion.scores(n, &accumulated);
            for (a, first) in scores.iter().enumerate() {
                for (b, second) in scores.iter().enumerate() {
                    information[a * size + b] += probability * first * second;
                }
                term += probability * first * first;
            }
        }
        total += term;
        if n as f64 > mean && term <= NEGLIGIBLE * total && probability <= NEGLIGIBLE {
            return information;
        }
        recursion.accumulate(n, &mut accumulated);
        n += 1;
    }
}

/// Panjer's recursion for the sum of independent negative binomials of
/// means `μₜ` and one overdispersion `φ`, a compound Poisson of logarithmic
/// jumps (Panjer 1981; Adelson 1966):
/// `p(0) = Πₜ (1 + φμₜ)^(−1/φ)` and `p(n) = (1/n) Σ_{j=1..n} hⱼ p(n − j)`,
/// `hⱼ = Σₜ cₜ qₜ^(j−1)`, `cₜ = μₜ/(1 + φμₜ)`, `qₜ = φμₜ/(1 + φμₜ)`.
///
/// The terms are positive and nothing divides by `φ`:
/// `log p(0) = −Σₜ μₜ · ln(1 + φμₜ)/(φμₜ)`, with the ratio 1 at `φμₜ = 0`,
/// so `φ = 0` is the Poisson of `M`. It holds `vₙ = p(n) ρⁿ / (p(0) e^L)`:
/// the tilt `ρ` centres the terms on a count (the observed one, or a count
/// above the tail the information sums), so they grow up to it; `L` scales
/// them down whenever one exceeds [`LARGE`], and `log p(0)` stays apart, so
/// a `p(0)` below the smallest number is no zero.
///
/// With the derivative in `φ`, it also holds `wₙ`, `∂p(n)/∂φ` on the scale
/// of `vₙ`, from the derivative of the recursion:
/// `p′(n) = (1/n) Σⱼ (hⱼ′ p(n − j) + hⱼ p′(n − j))`, `p′(0) = p(0) Σₜ B(μₜ)`
/// with `B` the bracket of the dispersion score, and
/// `∂(cₜ qₜ^(j−1))/∂φ = cₜ² ((j − 1)(1 − qₜ) qₜ^(j−2) − qₜ^(j−1))`. The
/// derivative in `μₜ` comes from `∂p(n)/∂μₜ = (gₜ(n − 1) − p(n))/(1 + φμₜ)`,
/// `gₜ(n) = (1 − qₜ) p(n) + qₜ gₜ(n − 1)` ([`Self::accumulate`]).
struct Recursion {
    c: Vec<f64>,
    q: Vec<f64>,
    rho: f64,
    log_rho: f64,
    /// `(ρ qₜ)^(j−1)` of the next jump `j`.
    powers: Vec<f64>,
    /// `(ρ qₜ)^(j−2)` of the next jump `j`.
    previous: Vec<f64>,
    /// `ρʲ hⱼ`, jump `j` at index `j − 1`.
    jumps: Vec<f64>,
    /// `ρʲ hⱼ′`, when the recursion has the derivative in `φ`.
    jump_slopes: Option<Vec<f64>>,
    values: Vec<f64>,
    slopes: Vec<f64>,
    log_scale: f64,
}

impl Recursion {
    /// The recursion of `parts` and `phi` tilted to the count `target`,
    /// with the derivative in `φ` when `with_slopes`, holding `p(0)`.
    fn new(parts: &[f64], phi: f64, target: f64, with_slopes: bool) -> Self {
        let products: Vec<f64> = parts.iter().map(|mu| phi * mu).collect();
        let c: Vec<f64> = parts
            .iter()
            .zip(&products)
            .map(|(mu, a)| mu / (1.0 + a))
            .collect();
        let q: Vec<f64> = products.iter().map(|a| a / (1.0 + a)).collect();
        let log_zero: f64 = -parts
            .iter()
            .zip(&products)
            .map(|(&mu, &a)| {
                if a == 0.0 {
                    mu
                } else {
                    mu * libm::log1p(a) / a
                }
            })
            .sum::<f64>();
        let rho = tilt(parts, phi, &q, target);
        let slope_of_zero: f64 = parts.iter().map(|&mu| dispersion_bracket(mu, phi)).sum();
        Self {
            powers: vec![1.0; parts.len()],
            previous: vec![0.0; parts.len()],
            c,
            q,
            rho,
            log_rho: libm::log(rho),
            jumps: Vec::new(),
            jump_slopes: with_slopes.then(Vec::new),
            values: vec![1.0],
            slopes: if with_slopes {
                vec![slope_of_zero]
            } else {
                Vec::new()
            },
            log_scale: log_zero,
        }
    }

    /// Compute the terms up to the count `last`; returns the factor that
    /// scaled the terms before, 1 when none was scaled.
    fn extend_to(&mut self, last: usize) -> f64 {
        let mut factor = 1.0;
        while self.values.len() <= last {
            factor *= self.push();
        }
        factor
    }

    /// The next jump `ρʲ hⱼ` and its slope.
    fn push_jump(&mut self) {
        let j = self.jumps.len() + 1;
        let rho = self.rho;
        let mut jump = 0.0;
        let mut slope = 0.0;
        for t in 0..self.c.len() {
            let (c, q) = (self.c[t], self.q[t]);
            let power = self.powers[t];
            jump += c * rho * power;
            if self.jump_slopes.is_some() {
                // A count of jumps is far below 2^53.
                let earlier = (j - 1) as f64 * (1.0 - q) * rho * self.previous[t];
                slope += c * c * rho * (earlier - power);
            }
            self.previous[t] = power;
            self.powers[t] = power * rho * q;
        }
        self.jumps.push(jump);
        if let Some(slopes) = &mut self.jump_slopes {
            slopes.push(slope);
        }
    }

    /// The term of the next count; returns the factor that scaled the
    /// terms, 1 when none was.
    fn push(&mut self) -> f64 {
        self.push_jump();
        let n = self.values.len();
        let inverse = 1.0 / n as f64;
        let value: f64 = (1..=n)
            .map(|j| self.jumps[j - 1] * self.values[n - j])
            .sum::<f64>()
            * inverse;
        self.values.push(value);
        if let Some(jump_slopes) = &self.jump_slopes {
            let slope: f64 = (1..=n)
                .map(|j| {
                    jump_slopes[j - 1] * self.values[n - j] + self.jumps[j - 1] * self.slopes[n - j]
                })
                .sum::<f64>()
                * inverse;
            self.slopes.push(slope);
        }
        if value > LARGE {
            let factor = 1.0 / value;
            self.values.iter_mut().for_each(|v| *v *= factor);
            self.slopes.iter_mut().for_each(|w| *w *= factor);
            self.log_scale -= libm::log(factor);
            factor
        } else {
            1.0
        }
    }

    /// `log p(n)` of a computed count `n`; `−∞` where it has no mass.
    fn log_probability(&self, n: usize) -> f64 {
        self.log_scale + libm::log(self.values[n]) - n as f64 * self.log_rho
    }

    /// Move `accumulated` from `gₜ(n − 1)` to `gₜ(n)` of each part, on the
    /// scale of the terms: `(1 − qₜ) vₙ + ρ qₜ gₜ(n − 1)`.
    fn accumulate(&self, n: usize, accumulated: &mut [f64]) {
        let value = self.values[n];
        for (g, &q) in accumulated.iter_mut().zip(&self.q) {
            *g = (1.0 - q) * value + self.rho * q * *g;
        }
    }

    /// The scores `(∂/∂μ₁, …, ∂/∂μ_K, ∂/∂φ)` of `log p(n)`, given
    /// `accumulated`, the `gₜ(n − 1)` of [`Self::accumulate`].
    fn scores(&self, n: usize, accumulated: &[f64]) -> Vec<f64> {
        let value = self.values[n];
        let mut scores: Vec<f64> = accumulated
            .iter()
            .zip(&self.q)
            .map(|(&g, &q)| (self.rho * g - value) * (1.0 - q) / value)
            .collect();
        scores.push(self.slopes[n] / value);
        scores
    }
}

/// The tilt `ρ` whose tilted sum has the mean `target`:
/// `Σₜ ρμₜ / (1 + φμₜ(1 − ρ)) = target`, below `1 / max qₜ`, where the
/// tilted jumps stop converging; 1 for a target or a mean of 0.
fn tilt(parts: &[f64], phi: f64, q: &[f64], target: f64) -> f64 {
    let mean: f64 = parts.iter().sum();
    if target <= 0.0 || mean <= 0.0 {
        return 1.0;
    }
    let largest = q.iter().copied().fold(0.0, f64::max);
    if largest == 0.0 {
        return target / mean;
    }
    let tilted = |rho: f64| -> f64 {
        parts
            .iter()
            .map(|&mu| rho * mu / (1.0 + phi * mu * (1.0 - rho)))
            .sum()
    };
    let (mut low, mut high) = (0.0, 1.0 / largest);
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if middle <= low || middle >= high {
            break;
        }
        if tilted(middle) < target {
            low = middle;
        } else {
            high = middle;
        }
    }
    low.max(f64::MIN_POSITIVE)
}
