//! Samplers of the distributions the stochastic engines and the inference
//! methods draw from.
//!
//! Each sampler reads only [`Generator`], so a seed gives the same values on
//! every platform. The algorithms are exact and their choice depends only on
//! the arguments:
//!
//! | Sampler | Algorithm |
//! |---|---|
//! | [`exponential`] | Inversion |
//! | [`poisson`] | Multiplication of uniforms below a mean of 10; PTRS above (Hörmann 1993) |
//! | [`binomial`] | Inversion below a mean of 10; BTRS above (Hörmann 1993) |
//! | [`multinomial`] | Conditional binomials, category by category |
//! | [`normal`] | Box–Muller (1958), one draw from two uniforms |
//! | [`gamma`] | Marsaglia and Tsang (2000); below a shape of 1, a draw of shape + 1 times `U^(1/shape)` |
//! | [`index`] | The floor of `n` times a uniform |
//!
//! The logarithms, roots and cosines of [`normal`] and [`gamma`] come from
//! `libm`, so their values are the same on every platform, as the special
//! functions of the inference methods are.
//!
//! A change of algorithm changes the values of a seed, so
//! `tests/test_samplers.rs` holds the first values of each sampler for a
//! fixed seed (`tests/reference_values.txt`).

use crate::Generator;

/// Below this mean, [`poisson`] and [`binomial`] use their simple algorithms.
const SMALL_MEAN: f64 = 10.0;

/// A draw of an exponential distribution of rate `rate`: a waiting time.
///
/// Returns `f64::INFINITY` when the rate is 0: the event never happens.
///
/// # Panics
///
/// Panics in debug builds when the rate is negative or not finite.
pub fn exponential(generator: &mut Generator, rate: f64) -> f64 {
    debug_assert!(rate.is_finite() && rate >= 0.0, "rate {rate}");
    if rate == 0.0 {
        return f64::INFINITY;
    }
    -generator.uniform_positive().ln() / rate
}

/// A draw of a Poisson distribution of mean `mean`.
///
/// # Panics
///
/// Panics in debug builds when the mean is negative or not finite.
pub fn poisson(generator: &mut Generator, mean: f64) -> u64 {
    debug_assert!(mean.is_finite() && mean >= 0.0, "mean {mean}");
    if mean == 0.0 {
        return 0;
    }
    if mean < SMALL_MEAN {
        return poisson_multiplication(generator, mean);
    }
    poisson_ptrs(generator, mean)
}

fn poisson_multiplication(generator: &mut Generator, mean: f64) -> u64 {
    let limit = (-mean).exp();
    let mut count = 0;
    let mut product = generator.uniform();
    while product > limit {
        count += 1;
        product *= generator.uniform();
    }
    count
}

/// The transformed rejection with squeeze of Hörmann (1993), "The
/// transformed rejection method for generating Poisson random variables".
fn poisson_ptrs(generator: &mut Generator, mean: f64) -> u64 {
    let sqrt_mean = mean.sqrt();
    let log_mean = mean.ln();
    let b = 0.931 + 2.53 * sqrt_mean;
    let a = -0.059 + 0.024_83 * b;
    let inverse_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let v_r = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = generator.uniform() - 0.5;
        let v = generator.uniform();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + mean + 0.43).floor();
        if us >= 0.07 && v <= v_r {
            return k as u64;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        if v.ln() + inverse_alpha.ln() - (a / (us * us) + b).ln()
            <= -mean + k * log_mean - ln_factorial(k)
        {
            return k as u64;
        }
    }
}

/// A draw of a binomial distribution of `trials` trials of probability
/// `probability`: how many of `trials` units move.
///
/// # Panics
///
/// Panics in debug builds when the probability is outside `[0, 1]`.
pub fn binomial(generator: &mut Generator, trials: u64, probability: f64) -> u64 {
    debug_assert!(
        (0.0..=1.0).contains(&probability),
        "probability {probability}"
    );
    if trials == 0 || probability == 0.0 {
        return 0;
    }
    if probability == 1.0 {
        return trials;
    }
    if probability > 0.5 {
        return trials - binomial(generator, trials, 1.0 - probability);
    }
    if trials as f64 * probability < SMALL_MEAN {
        return binomial_inversion(generator, trials, probability);
    }
    binomial_btrs(generator, trials, probability)
}

/// Sequential search of the cumulative distribution, from 0.
fn binomial_inversion(generator: &mut Generator, trials: u64, probability: f64) -> u64 {
    let n = trials as f64;
    let q = 1.0 - probability;
    let q_n = (n * q.ln()).exp();
    let mean = n * probability;
    let bound = n.min(mean + 10.0 * (mean * q + 1.0).sqrt());
    let mut count = 0.0;
    let mut mass = q_n;
    let mut u = generator.uniform();
    while u > mass {
        count += 1.0;
        if count > bound {
            count = 0.0;
            mass = q_n;
            u = generator.uniform();
        } else {
            u -= mass;
            mass *= (n - count + 1.0) * probability / (count * q);
        }
    }
    count as u64
}

/// The transformed rejection with squeeze of Hörmann (1993), "The generation
/// of binomial random variates"; `probability <= 0.5` and a mean of at least
/// 10.
fn binomial_btrs(generator: &mut Generator, trials: u64, probability: f64) -> u64 {
    let n = trials as f64;
    let q = 1.0 - probability;
    let spq = (n * probability * q).sqrt();
    let b = 1.15 + 2.53 * spq;
    let a = -0.0873 + 0.0248 * b + 0.01 * probability;
    let c = n * probability + 0.5;
    let v_r = 0.92 - 4.2 / b;
    let alpha = (2.83 + 5.1 / b) * spq;
    let log_ratio = (probability / q).ln();
    let mode = ((n + 1.0) * probability).floor();
    let h = ln_factorial(mode) + ln_factorial(n - mode);
    loop {
        let u = generator.uniform() - 0.5;
        let v = generator.uniform();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + c).floor();
        if k < 0.0 || k > n {
            continue;
        }
        if us >= 0.07 && v <= v_r {
            return k as u64;
        }
        let v = (v * alpha / (a / (us * us) + b)).ln();
        if v <= h - ln_factorial(k) - ln_factorial(n - k) + (k - mode) * log_ratio {
            return k as u64;
        }
    }
}

/// A draw of a multinomial distribution: how `trials` units split among
/// categories of probabilities `probabilities`, written into `counts`.
///
/// The probabilities need not sum to 1: they are weights, and the units that
/// no category takes are left out. Each category draws a binomial of the
/// units left, at its probability among the mass left.
///
/// # Panics
///
/// Panics when `counts` and `probabilities` differ in length, and in debug
/// builds when a probability is negative or their sum is above 1.
pub fn multinomial(
    generator: &mut Generator,
    trials: u64,
    probabilities: &[f64],
    counts: &mut [u64],
) {
    assert_eq!(probabilities.len(), counts.len());
    debug_assert!(probabilities.iter().all(|p| *p >= 0.0));
    let mut left = trials;
    let mut mass = 1.0;
    for (probability, count) in probabilities.iter().zip(counts.iter_mut()) {
        *count = if left == 0 || mass <= 0.0 {
            0
        } else {
            binomial(generator, left, (probability / mass).clamp(0.0, 1.0))
        };
        left -= *count;
        mass -= probability;
    }
}

/// A draw of the standard normal distribution: `√(−2 ln U₁) cos(2π U₂)`
/// of two uniforms strictly between 0 and 1 (Box and Muller 1958).
pub fn normal(generator: &mut Generator) -> f64 {
    let radius = libm::sqrt(-2.0 * libm::log(generator.open_uniform()));
    let angle = 2.0 * std::f64::consts::PI * generator.open_uniform();
    radius * libm::cos(angle)
}

/// A draw of a gamma distribution of shape `shape` and rate `rate`: mean
/// `shape / rate`.
///
/// Marsaglia and Tsang (2000), "A simple method for generating gamma
/// variables". Below a shape of 1, a draw of shape `shape + 1` times
/// `U^(1/shape)`, with `U` uniform.
///
/// # Panics
///
/// Panics in debug builds when the shape or the rate is not positive and
/// finite.
pub fn gamma(generator: &mut Generator, shape: f64, rate: f64) -> f64 {
    debug_assert!(shape.is_finite() && shape > 0.0, "shape {shape}");
    debug_assert!(rate.is_finite() && rate > 0.0, "rate {rate}");
    if shape < 1.0 {
        let draw = gamma_squeeze(generator, shape + 1.0);
        return draw * libm::pow(generator.open_uniform(), 1.0 / shape) / rate;
    }
    gamma_squeeze(generator, shape) / rate
}

/// The rejection with squeeze of Marsaglia and Tsang for a shape of at least
/// 1, at rate 1.
fn gamma_squeeze(generator: &mut Generator, shape: f64) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / libm::sqrt(9.0 * d);
    loop {
        let x = normal(generator);
        let root = 1.0 + c * x;
        if root <= 0.0 {
            continue;
        }
        let v = root * root * root;
        let u = generator.open_uniform();
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 || libm::log(u) < 0.5 * x2 + d * (1.0 - v + libm::log(v)) {
            return d * v;
        }
    }
}

/// A draw of a whole number below `n`, each with probability `1/n`: the
/// floor of `n` times [`Generator::uniform`]. The 53 bits of the uniform
/// leave each index within a few multiples of `2⁻⁵³` of `1/n`.
///
/// # Panics
///
/// Panics in debug builds when `n` is 0.
pub fn index(generator: &mut Generator, n: usize) -> usize {
    debug_assert!(n > 0, "an index below 0");
    ((generator.uniform() * n as f64) as usize).min(n - 1)
}

/// `ln(k!)` for a whole number `k`: a table below 10, the Stirling series of
/// `ln Γ(k + 1)` from there (absolute error below `1e-12`).
fn ln_factorial(k: f64) -> f64 {
    const TABLE: [f64; 10] = [
        0.0,
        0.0,
        std::f64::consts::LN_2,
        1.791_759_469_228_055,
        3.178_053_830_347_945_8,
        4.787_491_742_782_046,
        6.579_251_212_010_101,
        8.525_161_361_065_415,
        10.604_602_902_745_25,
        12.801_827_480_081_469,
    ];
    if k < 10.0 {
        return TABLE[k as usize];
    }
    let x = k + 1.0;
    let x2 = x * x;
    let x3 = x * x2;
    (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * x)
        - 1.0 / (360.0 * x3)
        + 1.0 / (1260.0 * x3 * x2)
        - 1.0 / (1680.0 * x3 * x2 * x2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_factorial_matches_the_product() {
        let mut exact = 0.0_f64;
        for k in 1..=40 {
            exact += (k as f64).ln();
            assert!((ln_factorial(k as f64) - exact).abs() < 1e-11, "{k}");
        }
    }

    #[test]
    fn an_inversion_past_its_bound_draws_again() {
        // The largest uniform, 1 − 2⁻⁵³, is above the mass of every count up
        // to the bound of a mean of 1, mean + 10 standard deviations.
        let mut forced = Generator::from_state([0, 0, 0, u64::MAX]);
        let mut next = forced.clone();
        assert_eq!(next.uniform(), 1.0 - f64::EPSILON / 2.0);

        let draw = binomial(&mut forced, 1_000, 0.001);

        assert_eq!(draw, binomial(&mut next, 1_000, 0.001));
    }
}
