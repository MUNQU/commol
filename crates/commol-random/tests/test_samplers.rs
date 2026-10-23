//! The samplers of `commol-random` through their public functions: the
//! moments of many draws, and the first values of each for a fixed seed.

use commol_random::Generator;
use commol_random::sample::{binomial, exponential, gamma, index, multinomial, normal, poisson};

const SEED: u64 = 20_261_016;
const DRAWS: usize = 40_000;

fn mean_and_variance(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, variance)
}

/// The mean and the variance of many draws match the distribution: the mean
/// within five standard errors, the variance within 5 %.
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

fn draws(count: usize, mut draw: impl FnMut(&mut Generator) -> f64) -> Vec<f64> {
    let mut generator = Generator::from_seed(SEED);
    (0..count).map(|_| draw(&mut generator)).collect()
}

#[test]
fn the_new_samplers_have_the_moments_of_their_distribution() {
    const COUNT: usize = 100_000;
    assert_moments(&draws(COUNT, normal), 0.0, 1.0);
    // Shape 2.5 and rate 0.5: mean 5, variance 10.
    assert_moments(&draws(COUNT, |g| gamma(g, 2.5, 0.5)), 5.0, 10.0);
    // A shape below 1 takes the other branch: mean 0.25, variance 0.125.
    assert_moments(&draws(COUNT, |g| gamma(g, 0.5, 2.0)), 0.25, 0.125);
    // An index below 7 is uniform on 0..7: mean 3, variance (7² − 1) / 12.
    let indices = draws(COUNT, |g| index(g, 7) as f64);
    assert_moments(&indices, 3.0, 4.0);
    assert!(indices.iter().all(|&i| i < 7.0));
    let uniforms = draws(COUNT, Generator::open_uniform);
    assert_moments(&uniforms, 0.5, 1.0 / 12.0);
    assert!(uniforms.iter().all(|&u| u > 0.0 && u < 1.0));
}

#[test]
fn the_first_values_of_the_new_samplers_are_fixed() {
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
    let open_uniforms: Vec<f64> = (0..3).map(|_| generator.open_uniform()).collect();
    let normals: Vec<f64> = (0..3).map(|_| normal(&mut generator)).collect();
    let gammas: Vec<f64> = (0..3).map(|_| gamma(&mut generator, 2.5, 0.5)).collect();
    let small_gammas: Vec<f64> = (0..3).map(|_| gamma(&mut generator, 0.5, 2.0)).collect();
    let indices: Vec<usize> = (0..5).map(|_| index(&mut generator, 7)).collect();

    let reference = include_str!("reference_values.txt");
    let rendered = format!(
        "{exponentials:?}\n{small_poisson:?}\n{large_poisson:?}\n{small_binomial:?}\n\
         {large_binomial:?}\n{counts:?}\n{open_uniforms:?}\n{normals:?}\n{gammas:?}\n\
         {small_gammas:?}\n{indices:?}\n"
    );
    assert_eq!(rendered, reference);
}

#[test]
fn exponential_draws_have_the_moments_of_their_rate() {
    assert_moments(&draws(DRAWS, |g| exponential(g, 4.0)), 0.25, 0.0625);
    assert_eq!(
        exponential(&mut Generator::from_seed(SEED), 0.0),
        f64::INFINITY
    );
}

#[test]
fn poisson_draws_have_the_moments_of_their_mean() {
    for mean in [0.5, 3.0, 9.9, 10.0, 37.5, 1_200.0] {
        assert_moments(&draws(DRAWS, |g| poisson(g, mean) as f64), mean, mean);
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
        let values = draws(DRAWS, |g| binomial(g, trials, probability) as f64);
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
