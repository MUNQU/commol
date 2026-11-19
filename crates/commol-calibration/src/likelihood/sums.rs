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
//! The tails of a sum are those of the same reading: of the model of the
//! total, or the cumulative sums of the recursion.
//!
//! The exact reading computes at most [`MAX_COUNTS`] counts, and fails with
//! [`UnendedSum`] when its sum needs more.
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

use super::density::{
    NEGLIGIBLE, Noise, SMALL_OVERDISPERSION, Tails, dispersion_bracket, negative_binomial,
};

/// How a negative binomial reads the sum of the parts of a window:
/// `"exact"` or `"moments"` in the Python API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sum {
    /// The exact distribution of the sum of independent negative binomials
    /// (`Recursion`): `O(K·y)` per evaluation at a total `y`, more for the
    /// tails and the information, which reach further, over at most
    /// [`MAX_COUNTS`] counts.
    Exact,
    /// A negative binomial with the mean and the variance of the sum.
    Moments,
}

/// The most counts, `0` to `MAX_COUNTS − 1`, that the exact reading of a
/// sum computes. A sum that needs more fails with [`UnendedSum`]: the
/// observed total is at the cap or above, or its tail or its information
/// has not reached a negligible term below it. Why this count:
/// `docs/codebase/decisions/sums-of-parts.md` (An exact sum ends at a fixed
/// count).
pub const MAX_COUNTS: usize = 1_000_000;

/// The exact reading of a sum of negative binomials needs more than
/// [`MAX_COUNTS`] counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnendedSum {
    /// The mean `M` of the sum, the sum of the means of its parts.
    pub mean: f64,
    /// The overdispersion `φ` of the parts.
    pub overdispersion: f64,
}

impl UnendedSum {
    /// The message of an observation of `label` at `at`, of the observed
    /// total `value` in `parts` parts, whose sum does not end.
    pub fn message(&self, label: &str, at: f64, value: f64, parts: usize) -> String {
        format!(
            "Observation '{label}' at {at} has the total {value} in {parts} parts; its \
             NegativeBinomial model with sum=\"exact\", the mean {} of the total and the \
             overdispersion {}, does not end within {MAX_COUNTS} counts. Read the sum with \
             sum=\"moments\", whose cost does not grow with the total",
            self.mean, self.overdispersion
        )
    }
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
    /// # Errors
    ///
    /// [`UnendedSum`] for [`Sum::Exact`] when `y` is not below
    /// [`MAX_COUNTS`].
    ///
    /// # Panics
    ///
    /// Panics for more than one part of a model that reads no sum, or of a
    /// negative binomial without a [`Sum`].
    pub fn sum_log_density(&self, y: f64, parts: &[f64]) -> Result<f64, UnendedSum> {
        if let [mean] = parts {
            return Ok(self.log_density(y, *mean));
        }
        match self.reading(parts) {
            Reading::Total { noise, mean, .. } => Ok(noise.log_density(y, mean)),
            Reading::Exact { overdispersion } => {
                let last = count(y);
                let mut recursion = Recursion::new(parts, overdispersion, y, false);
                recursion.extend_to(last)?;
                Ok(recursion.log_probability(last))
            }
        }
    }

    /// The tails of the sum `y` of parts of means `parts`, each of which
    /// [`Noise::mean_outside`] allows: those of the model of the total, or
    /// for [`Sum::Exact`] the sums of the recursion below and above `y`.
    ///
    /// # Errors
    ///
    /// As [`Self::sum_log_density`], and [`UnendedSum`] when the sum above
    /// `y` has not reached a negligible term within [`MAX_COUNTS`].
    ///
    /// # Panics
    ///
    /// As [`Self::sum_log_density`].
    pub fn sum_tails(&self, y: f64, parts: &[f64]) -> Result<Tails, UnendedSum> {
        if let [mean] = parts {
            return Ok(self.tails(y, *mean));
        }
        match self.reading(parts) {
            Reading::Total { noise, mean, .. } => Ok(noise.tails(y, mean)),
            Reading::Exact { overdispersion } => exact_tails(parts, overdispersion, y),
        }
    }

    /// The derivatives of the log density of the sum `y` over
    /// `(μ₁, …, μ_K, noise arguments…)`.
    ///
    /// # Errors
    ///
    /// As [`Self::sum_log_density`].
    ///
    /// # Panics
    ///
    /// As [`Self::sum_log_density`].
    pub fn sum_scores(&self, y: f64, parts: &[f64]) -> Result<Vec<f64>, UnendedSum> {
        if let [mean] = parts {
            return Ok(self.scores(y, *mean));
        }
        match self.reading(parts) {
            Reading::Total {
                noise,
                mean,
                jacobian,
            } => {
                let scores = noise.scores(y, mean);
                let columns = jacobian.len() / scores.len();
                Ok((0..columns)
                    .map(|c| {
                        scores
                            .iter()
                            .enumerate()
                            .map(|(r, score)| jacobian[r * columns + c] * score)
                            .sum()
                    })
                    .collect())
            }
            Reading::Exact { overdispersion } => {
                let last = count(y);
                let mut recursion = Recursion::new(parts, overdispersion, y, true);
                recursion.extend_to(last)?;
                let mut accumulated = vec![0.0; parts.len()];
                for n in 0..last {
                    recursion.accumulate(n, &mut accumulated);
                }
                Ok(recursion.scores(last, &accumulated))
            }
        }
    }

    /// The expected Fisher information of the sum over
    /// `(μ₁, …, μ_K, noise arguments…)`, row-major: that of the model of the
    /// total chained with the derivatives of its arguments, or for
    /// [`Sum::Exact`] `Σ_y p(y) s(y) s(y)ᵀ`, summed over the counts from 0
    /// until the terms above the mean are negligible.
    ///
    /// # Errors
    ///
    /// [`UnendedSum`] for [`Sum::Exact`] when the terms are not negligible
    /// within [`MAX_COUNTS`].
    ///
    /// # Panics
    ///
    /// As [`Self::sum_log_density`].
    pub fn sum_information(&self, parts: &[f64]) -> Result<Vec<f64>, UnendedSum> {
        if let [mean] = parts {
            return Ok(self.information(*mean));
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
                Ok(chained)
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

/// The smallest `P(Y > y)` of a sum read as the complement of the counts up
/// to `y`: it keeps the relative accuracy of `f64` to about `1e-14`, and
/// spares the sum above for all but the upper tail.
const COMPLEMENT_TAIL: f64 = 0.01;

/// The count `y`, a whole number of at least 0, as an index.
fn count(y: f64) -> usize {
    // An observed count is a whole number of at least 0.
    y as usize
}

/// The tails at the count `y` of a sum of negative binomials of
/// overdispersion `phi` and means `parts`: `P(Y < y)` sums the recursion up
/// to `y`; `P(Y > y)` is its complement while that is at least
/// [`COMPLEMENT_TAIL`], and otherwise the sum of the counts above `y` until
/// a term is negligible, so it keeps its relative accuracy in the upper
/// tail. The sum above costs `O(K·n)` in the last count `n` it reaches,
/// which a heavy tail (`φ μ` large) puts far beyond `y`; it fails at
/// [`MAX_COUNTS`], at once when [`ends_within_counts`] shows it cannot end.
fn exact_tails(parts: &[f64], phi: f64, y: f64) -> Result<Tails, UnendedSum> {
    let last = count(y);
    let mut recursion = Recursion::new(parts, phi, y, false);
    recursion.extend_to(last)?;
    let probability = |recursion: &Recursion, n: usize| libm::exp(recursion.log_probability(n));
    let below = (0..last).fold(0.0, |sum, n| sum + probability(&recursion, n));
    let at = probability(&recursion, last);
    let complement = (1.0 - below) - at;
    if complement >= COMPLEMENT_TAIL {
        return Ok(Tails {
            below,
            at,
            above: complement,
        });
    }
    if !ends_within_counts(parts, phi) {
        return Err(recursion.unended());
    }
    let mean: f64 = parts.iter().sum();
    let mut above = 0.0;
    let mut n = last;
    loop {
        n += 1;
        recursion.extend_to(n)?;
        let term = probability(&recursion, n);
        above += term;
        if n as f64 > mean && term <= NEGLIGIBLE * above {
            return Ok(Tails { below, at, above });
        }
    }
}

/// Whether the sum of negative binomials of overdispersion `phi` and means
/// `parts` can reach, within [`MAX_COUNTS`], a count `n > M` of mass at most
/// [`NEGLIGIBLE`], where the upper tail and the information may stop. It
/// cannot when `M ≥ MAX_COUNTS − 1`, or when a lower bound of `p(n)` at
/// `n = MAX_COUNTS − 1` is above [`NEGLIGIBLE`]:
/// `p(n) ≥ maxₜ pₜ(n) Π_{s≠t} pₛ(0)`, the mass of the sum where one part
/// takes the whole count, with `pₜ` the negative binomial of the part `t`.
/// Each `pₜ` falls above its mode, which is at most `μₜ ≤ M`, so the bound
/// falls on `(M, MAX_COUNTS − 1]` and holds the mass of every count there
/// above [`NEGLIGIBLE`]. Below [`SMALL_OVERDISPERSION`], where `pₜ(n)` sums
/// one term per count, it takes no bound: the sum is about `Poisson(M)`,
/// which ends a few sd above `M`, or at worst runs to the cap.
fn ends_within_counts(parts: &[f64], phi: f64) -> bool {
    let last = (MAX_COUNTS - 1) as f64;
    let mean: f64 = parts.iter().sum();
    if mean >= last {
        return false;
    }
    if phi < SMALL_OVERDISPERSION {
        return true;
    }
    let zeros: Vec<f64> = parts
        .iter()
        .map(|&mu| negative_binomial(0.0, mu, phi))
        .collect();
    let none: f64 = zeros.iter().sum();
    let lower = parts
        .iter()
        .zip(&zeros)
        .map(|(&mu, &zero)| negative_binomial(last, mu, phi) - zero)
        .fold(f64::NEG_INFINITY, f64::max);
    none + lower <= libm::log(NEGLIGIBLE)
}

/// The exact information of a sum of negative binomials of overdispersion
/// `phi` and means `parts` over `(μ₁, …, μ_K, φ)`, `O(K²)` per count; it fails at
/// [`MAX_COUNTS`], at once when [`ends_within_counts`] shows it cannot end.
fn exact_information(parts: &[f64], phi: f64) -> Result<Vec<f64>, UnendedSum> {
    let mean: f64 = parts.iter().sum();
    if !ends_within_counts(parts, phi) {
        return Err(UnendedSum {
            mean,
            overdispersion: phi,
        });
    }
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
            let factor = recursion.extend_to(n)?;
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
            return Ok(information);
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
/// The jumps of a part are geometric, so its share of the sum over the jumps
/// is a state, updated once per count: `p(n) = (1/n) Σₜ cₜ Aₜ(n − 1)`,
/// `Aₜ(m) = Σ_{i≤m} qₜ^(m−i) p(i) = p(m) + qₜ Aₜ(m − 1)`. A count costs
/// `O(K)` and the counts up to `y` cost `O(K·y)`, with no jump cut: every
/// term is still a sum of positive terms.
///
/// The terms are positive and nothing divides by `φ`:
/// `log p(0) = −Σₜ μₜ · ln(1 + φμₜ)/(φμₜ)`, with the ratio 1 at `φμₜ = 0`,
/// so `φ = 0` is the Poisson of `M`. It holds `vₙ = p(n) ρⁿ / (p(0) e^L)`:
/// the tilt `ρ` centres the terms on a count (the observed one, or a count
/// above the tail the information sums), so they grow up to it; `L` scales
/// them down whenever one exceeds [`LARGE`], and `log p(0)` stays apart, so
/// a `p(0)` below the smallest number is no zero. On that scale
/// `Aₜ(m) = vₘ + ρqₜ Aₜ(m − 1)` and `vₙ = (ρ/n) Σₜ cₜ Aₜ(n − 1)`.
///
/// With the derivative in `φ`, it also holds `wₙ`, `∂p(n)/∂φ` on the scale
/// of `vₙ`, from the derivative of the recursion:
/// `p′(n) = (1/n) Σⱼ (hⱼ′ p(n − j) + hⱼ p′(n − j))`, `p′(0) = p(0) Σₜ B(μₜ)`
/// with `B` the bracket of the dispersion score, and
/// `∂(cₜ qₜ^(j−1))/∂φ = cₜ² ((j − 1)(1 − qₜ) qₜ^(j−2) − qₜ^(j−1))`. Its sums
/// over the jumps are states as well: `Wₜ`, the `Aₜ` of the `wₘ`, and
/// `Dₜ(m) = Σ_{i≤m} (m − i) (ρqₜ)^(m−i−1) vᵢ = ρqₜ Dₜ(m − 1) + Aₜ(m − 1)`, so
/// `n wₙ = ρ Σₜ (cₜ Wₜ(n − 1) + cₜ² ((1 − qₜ) ρ Dₜ(n − 1) − Aₜ(n − 1)))`.
/// The derivative in `μₜ` comes from
/// `∂p(n)/∂μₜ = (gₜ(n − 1) − p(n))/(1 + φμₜ)`,
/// `gₜ(n) = (1 − qₜ) p(n) + qₜ gₜ(n − 1)` ([`Self::accumulate`]).
struct Recursion {
    /// `M`, the sum of the means of the parts.
    mean: f64,
    /// The overdispersion `φ` of the parts.
    phi: f64,
    c: Vec<f64>,
    q: Vec<f64>,
    rho: f64,
    log_rho: f64,
    /// `Aₜ` of the last count of each part.
    sums: Vec<f64>,
    /// `Wₜ` of the last count, when the recursion has the derivative in `φ`.
    slope_sums: Vec<f64>,
    /// `Dₜ` of the last count, when the recursion has the derivative in `φ`.
    weighted_sums: Vec<f64>,
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
        let k = parts.len();
        let (slopes, slope_sums, weighted_sums) = if with_slopes {
            let slope_of_zero: f64 = parts.iter().map(|&mu| dispersion_bracket(mu, phi)).sum();
            (vec![slope_of_zero], vec![slope_of_zero; k], vec![0.0; k])
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        Self {
            mean: parts.iter().sum(),
            phi,
            c,
            q,
            rho,
            log_rho: libm::log(rho),
            sums: vec![1.0; k],
            slope_sums,
            weighted_sums,
            values: vec![1.0],
            slopes,
            log_scale: log_zero,
        }
    }

    /// Compute the terms up to the count `last`; returns the factor that
    /// scaled the terms before, 1 when none was scaled.
    ///
    /// # Errors
    ///
    /// [`UnendedSum`], before any term, when `last` is not below
    /// [`MAX_COUNTS`].
    fn extend_to(&mut self, last: usize) -> Result<f64, UnendedSum> {
        if last >= MAX_COUNTS {
            return Err(self.unended());
        }
        let mut factor = 1.0;
        while self.values.len() <= last {
            factor *= self.push();
        }
        Ok(factor)
    }

    /// The error of this sum when it needs more than [`MAX_COUNTS`] counts.
    fn unended(&self) -> UnendedSum {
        UnendedSum {
            mean: self.mean,
            overdispersion: self.phi,
        }
    }

    /// The term of the next count, and the states of the parts at it;
    /// returns the factor that scaled the terms, 1 when none was.
    fn push(&mut self) -> f64 {
        let n = self.values.len();
        let inverse = 1.0 / n as f64;
        let rho = self.rho;
        let value = rho
            * self
                .c
                .iter()
                .zip(&self.sums)
                .map(|(c, sum)| c * sum)
                .sum::<f64>()
            * inverse;
        self.values.push(value);
        if !self.slopes.is_empty() {
            let slope = rho
                * (0..self.c.len())
                    .map(|t| {
                        let c = self.c[t];
                        let jumps = (1.0 - self.q[t]) * rho * self.weighted_sums[t] - self.sums[t];
                        c * self.slope_sums[t] + c * c * jumps
                    })
                    .sum::<f64>()
                * inverse;
            self.slopes.push(slope);
            for t in 0..self.c.len() {
                let ratio = rho * self.q[t];
                self.weighted_sums[t] = ratio * self.weighted_sums[t] + self.sums[t];
                self.slope_sums[t] = slope + ratio * self.slope_sums[t];
            }
        }
        for (sum, &q) in self.sums.iter_mut().zip(&self.q) {
            *sum = value + rho * q * *sum;
        }
        if value > LARGE {
            let factor = 1.0 / value;
            for state in [
                &mut self.values,
                &mut self.slopes,
                &mut self.sums,
                &mut self.slope_sums,
                &mut self.weighted_sums,
            ] {
                state.iter_mut().for_each(|v| *v *= factor);
            }
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
