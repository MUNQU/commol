//! Samplers of the distributions the stochastic engines draw from.
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
//!
//! A change of algorithm changes the values of a seed, so the tests of this
//! module hold the first values of each sampler for a fixed seed.

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

    const SEED: u64 = 20_261_016;
    const DRAWS: usize = 40_000;

    fn mean_and_variance(values: &[f64]) -> (f64, f64) {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, variance)
    }

    /// The mean and the variance of many draws match the distribution
    /// within five standard errors.
    fn assert_moments(values: &[f64], mean: f64, variance: f64) {
        let (sample_mean, sample_variance) = mean_and_variance(values);
        let n = values.len() as f64;
        let mean_error = (variance / n).sqrt();
        assert!(
            (sample_mean - mean).abs() <= 5.0 * mean_error + 1e-12,
            "mean {sample_mean}, expected {mean}"
        );
        assert!(
            (sample_variance - variance).abs() <= 0.05 * variance + 1e-12,
            "variance {sample_variance}, expected {variance}"
        );
    }

    #[test]
    fn ln_factorial_matches_the_product() {
        let mut exact = 0.0_f64;
        for k in 1..=40 {
            exact += (k as f64).ln();
            assert!((ln_factorial(k as f64) - exact).abs() < 1e-11, "{k}");
        }
    }

    #[test]
    fn the_first_values_of_each_sampler_are_fixed() {
        let mut generator = Generator::from_seed(SEED);
        let exponentials: Vec<f64> = (0..3).map(|_| exponential(&mut generator, 2.0)).collect();
        let small_poisson: Vec<u64> = (0..5).map(|_| poisson(&mut generator, 3.5)).collect();
        let large_poisson: Vec<u64> = (0..5).map(|_| poisson(&mut generator, 250.0)).collect();
        let small_binomial: Vec<u64> = (0..5).map(|_| binomial(&mut generator, 20, 0.3)).collect();
        let large_binomial: Vec<u64> = (0..5)
            .map(|_| binomial(&mut generator, 10_000, 0.2))
            .collect();
        let mut counts = [0; 3];
        multinomial(&mut generator, 1_000, &[0.2, 0.5, 0.3], &mut counts);

        let reference = include_str!("../tests/reference_values.txt");
        let rendered = format!(
            "{exponentials:?}\n{small_poisson:?}\n{large_poisson:?}\n{small_binomial:?}\n\
             {large_binomial:?}\n{counts:?}\n"
        );
        assert_eq!(rendered, reference);
    }

    #[test]
    fn exponential_draws_have_the_moments_of_their_rate() {
        let mut generator = Generator::from_seed(SEED);
        let values: Vec<f64> = (0..DRAWS)
            .map(|_| exponential(&mut generator, 4.0))
            .collect();
        assert_moments(&values, 0.25, 0.0625);
        assert_eq!(exponential(&mut generator, 0.0), f64::INFINITY);
    }

    #[test]
    fn poisson_draws_have_the_moments_of_their_mean() {
        for mean in [0.5, 3.0, 9.9, 10.0, 37.5, 1_200.0] {
            let mut generator = Generator::from_seed(SEED);
            let values: Vec<f64> = (0..DRAWS)
                .map(|_| poisson(&mut generator, mean) as f64)
                .collect();
            assert_moments(&values, mean, mean);
        }
    }

    #[test]
    fn binomial_draws_have_the_moments_of_their_trials() {
        for (trials, probability) in [
            (10, 0.3),
            (100, 0.05),
            (100, 0.5),
            (1_000, 0.2),
            (1_000_000, 0.003),
            (500, 0.9),
        ] {
            let mut generator = Generator::from_seed(SEED);
            let values: Vec<f64> = (0..DRAWS)
                .map(|_| binomial(&mut generator, trials, probability) as f64)
                .collect();
            let n = trials as f64;
            assert_moments(
                &values,
                n * probability,
                n * probability * (1.0 - probability),
            );
            assert!(values.iter().all(|value| *value <= n));
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

    #[test]
    fn binomial_edges_are_exact() {
        let mut generator = Generator::from_seed(SEED);
        assert_eq!(binomial(&mut generator, 0, 0.5), 0);
        assert_eq!(binomial(&mut generator, 50, 0.0), 0);
        assert_eq!(binomial(&mut generator, 50, 1.0), 50);
        assert_eq!(poisson(&mut generator, 0.0), 0);
    }

    #[test]
    fn multinomial_draws_split_every_unit() {
        let mut generator = Generator::from_seed(SEED);
        let probabilities = [0.1, 0.6, 0.3];
        let mut sums = [0.0; 3];
        for _ in 0..DRAWS {
            let mut counts = [0; 3];
            multinomial(&mut generator, 200, &probabilities, &mut counts);
            assert_eq!(counts.iter().sum::<u64>(), 200);
            for (sum, count) in sums.iter_mut().zip(counts) {
                *sum += count as f64;
            }
        }
        for (sum, probability) in sums.iter().zip(probabilities) {
            let mean = sum / DRAWS as f64;
            assert!((mean - 200.0 * probability).abs() < 0.2, "{mean}");
        }
    }

    #[test]
    fn multinomial_leaves_out_the_units_no_category_takes() {
        let mut generator = Generator::from_seed(SEED);
        let mut total = 0;
        for _ in 0..DRAWS {
            let mut counts = [0; 2];
            multinomial(&mut generator, 100, &[0.2, 0.3], &mut counts);
            total += counts.iter().sum::<u64>();
        }
        let mean = total as f64 / DRAWS as f64;
        assert!((mean - 50.0).abs() < 0.2, "{mean}");
    }
}
