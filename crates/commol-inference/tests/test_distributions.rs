//! The distributions of priors: their densities, distribution functions,
//! quantiles and moments against closed forms, and their draws against a
//! stored reference.

use commol_inference::distribution::{Distribution, Family};
use commol_random::Generator;

const E: f64 = std::f64::consts::E;
const PI: f64 = std::f64::consts::PI;

/// Φ(1) and Φ⁻¹(0.975) of the standard normal distribution.
const PHI_1: f64 = 0.841_344_746_068_542_9;
const Z_975: f64 = 1.959_963_984_540_054;

fn distribution(family: Family, lower: f64, upper: f64) -> Distribution {
    Distribution::new(family, lower, upper).unwrap()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{what}: {actual} against {expected}"
    );
}

/// `∫ f(x) dx` over `[a, b]` by Simpson's rule with `n` (even) intervals.
fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let inner: f64 = (1..n)
        .map(|i| {
            let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
            weight * f(a + i as f64 * h)
        })
        .sum();
    (f(a) + f(b) + inner) * h / 3.0
}

/// The mass, mean and variance of a density over `[a, b]`, by Simpson's
/// rule.
fn integrated_moments(d: &Distribution, a: f64, b: f64) -> (f64, f64, f64) {
    let density = |x: f64| d.log_density(x).exp();
    let n = 200_000;
    let mass = simpson(density, a, b, n);
    let mean = simpson(|x| x * density(x), a, b, n);
    let second = simpson(|x| x * x * density(x), a, b, n);
    (mass, mean, second - mean * mean)
}

#[test]
fn distributions_match_their_closed_forms() {
    // Uniform on [2, 6].
    let uniform = distribution(Family::Uniform, 2.0, 6.0);
    assert_close(uniform.cdf(3.0), 0.25, 1e-14, "uniform cdf");
    assert_close(uniform.quantile(0.5), 4.0, 1e-14, "uniform quantile");
    assert_close(
        uniform.log_density(3.0),
        -(4f64.ln()),
        1e-14,
        "uniform density",
    );
    assert_close(uniform.mean(), 4.0, 1e-14, "uniform mean");
    assert_close(uniform.variance(), 16.0 / 12.0, 1e-14, "uniform variance");

    // Log-uniform on [1, e²]: log x is uniform on [0, 2].
    let log_uniform = distribution(Family::LogUniform, 1.0, E * E);
    assert_close(log_uniform.cdf(E), 0.5, 1e-14, "log-uniform cdf");
    assert_close(
        log_uniform.quantile(0.25),
        E.sqrt(),
        1e-14,
        "log-uniform quantile",
    );
    let mean = (E * E - 1.0) / 2.0;
    assert_close(log_uniform.mean(), mean, 1e-13, "log-uniform mean");
    let second = (E.powi(4) - 1.0) / 4.0;
    assert_close(
        log_uniform.variance(),
        second - mean * mean,
        1e-13,
        "log-uniform variance",
    );

    // The standard normal distribution, untruncated.
    let normal = distribution(
        Family::Normal { mean: 0.0, sd: 1.0 },
        f64::NEG_INFINITY,
        f64::INFINITY,
    );
    assert_close(normal.cdf(1.0), PHI_1, 1e-15, "normal cdf");
    assert_close(normal.quantile(0.975), Z_975, 1e-14, "normal quantile");
    assert_close(
        normal.log_density(1.0),
        -0.5 - 0.5 * (2.0 * PI).ln(),
        1e-14,
        "normal density",
    );
    assert_close(normal.mean(), 0.0, 1e-14, "normal mean");
    assert_close(normal.variance(), 1.0, 1e-14, "normal variance");

    // The half-normal distribution: a standard normal truncated to [0, ∞).
    let half = distribution(Family::Normal { mean: 0.0, sd: 1.0 }, 0.0, f64::INFINITY);
    assert_close(half.cdf(1.0), 2.0 * PHI_1 - 1.0, 1e-14, "half-normal cdf");
    assert_close(
        half.quantile(2.0 * PHI_1 - 1.0),
        1.0,
        1e-12,
        "half-normal quantile",
    );
    assert_close(half.mean(), (2.0 / PI).sqrt(), 1e-14, "half-normal mean");
    assert_close(
        half.variance(),
        1.0 - 2.0 / PI,
        1e-14,
        "half-normal variance",
    );

    // The log-normal distribution of log x ~ N(0, 1).
    let log_normal = distribution(
        Family::LogNormal {
            mu: 0.0,
            sigma: 1.0,
        },
        0.0,
        f64::INFINITY,
    );
    assert_close(log_normal.cdf(1.0), 0.5, 1e-15, "log-normal cdf");
    assert_close(log_normal.quantile(PHI_1), E, 1e-13, "log-normal quantile");
    assert_close(log_normal.mean(), E.sqrt(), 1e-14, "log-normal mean");
    assert_close(
        log_normal.variance(),
        (E - 1.0) * E,
        1e-13,
        "log-normal variance",
    );

    // Beta(2, 3): I_x(2, 3) = Σ_{j=2}^{4} C(4, j) x^j (1 − x)^{4−j}.
    let beta = distribution(Family::Beta { a: 2.0, b: 3.0 }, 0.0, 1.0);
    let x: f64 = 0.3;
    let cdf = 6.0 * x.powi(2) * (1.0 - x).powi(2) + 4.0 * x.powi(3) * (1.0 - x) + x.powi(4);
    assert_close(beta.cdf(x), cdf, 1e-14, "beta cdf");
    assert_close(beta.quantile(cdf), x, 1e-12, "beta quantile");
    assert_close(beta.mean(), 0.4, 1e-14, "beta mean");
    assert_close(beta.variance(), 0.04, 1e-14, "beta variance");
    // On [10, 20] it is the same distribution, scaled.
    let scaled = distribution(Family::Beta { a: 2.0, b: 3.0 }, 10.0, 20.0);
    assert_close(scaled.cdf(13.0), cdf, 1e-14, "scaled beta cdf");
    assert_close(scaled.mean(), 14.0, 1e-14, "scaled beta mean");
    assert_close(scaled.variance(), 4.0, 1e-13, "scaled beta variance");

    // Gamma(3, rate 2): P(3, 2x) = 1 − e^{−2x}(1 + 2x + 2x²).
    let gamma = distribution(
        Family::Gamma {
            shape: 3.0,
            rate: 2.0,
        },
        0.0,
        f64::INFINITY,
    );
    let cdf = 1.0 - (-2.0_f64).exp() * (1.0 + 2.0 + 2.0);
    assert_close(gamma.cdf(1.0), cdf, 1e-14, "gamma cdf");
    assert_close(gamma.quantile(cdf), 1.0, 1e-12, "gamma quantile");
    assert_close(gamma.mean(), 1.5, 1e-14, "gamma mean");
    assert_close(gamma.variance(), 0.75, 1e-13, "gamma variance");

    // Gamma(1, rate 1) on [0, 2]: an exponential truncated at 2.
    let truncated = distribution(
        Family::Gamma {
            shape: 1.0,
            rate: 1.0,
        },
        0.0,
        2.0,
    );
    let mass = 1.0 - (-2.0_f64).exp();
    assert_close(
        truncated.cdf(1.0),
        (1.0 - (-1.0_f64).exp()) / mass,
        1e-14,
        "cdf",
    );
    assert_close(
        truncated.mean(),
        1.0 - 2.0 * (-2.0_f64).exp() / mass,
        1e-14,
        "truncated exponential mean",
    );
}

#[test]
fn truncated_distributions_hold_their_mass_inside_the_bounds() {
    let cases = [
        (Family::Normal { mean: 1.0, sd: 2.0 }, -1.0, 4.0),
        (
            Family::LogNormal {
                mu: 0.5,
                sigma: 0.8,
            },
            0.5,
            6.0,
        ),
        (Family::Beta { a: 1.7, b: 2.5 }, -1.0, 3.0),
        (
            Family::Gamma {
                shape: 2.5,
                rate: 1.5,
            },
            0.2,
            5.0,
        ),
        (Family::LogUniform, 0.01, 10.0),
    ];
    for (family, lower, upper) in cases {
        let d = distribution(family, lower, upper);
        let what = format!("{family:?} on [{lower}, {upper}]");
        assert_eq!(d.log_density(lower - 1e-9), f64::NEG_INFINITY, "{what}");
        assert_eq!(d.log_density(upper + 1e-9), f64::NEG_INFINITY, "{what}");
        assert_eq!(d.cdf(lower), 0.0, "{what}");
        assert_eq!(d.cdf(upper), 1.0, "{what}");
        assert_eq!(d.quantile(0.0), lower, "{what}");
        assert_eq!(d.quantile(1.0), upper, "{what}");
        let (mass, mean, variance) = integrated_moments(&d, lower, upper);
        assert_close(mass, 1.0, 1e-4, &format!("{what} mass"));
        assert_close(d.mean(), mean, 1e-4, &format!("{what} mean"));
        assert_close(d.variance(), variance, 1e-3, &format!("{what} variance"));
        for p in [0.01, 0.2, 0.5, 0.8, 0.99] {
            assert_close(d.cdf(d.quantile(p)), p, 1e-10, &format!("{what} at {p}"));
        }
    }
}

#[test]
fn a_distribution_needs_parameters_and_bounds_that_fit_it() {
    let error = |family, lower, upper| Distribution::new(family, lower, upper).unwrap_err();
    assert!(error(Family::Uniform, 0.0, f64::INFINITY).contains("finite bounds"));
    assert!(error(Family::Beta { a: 1.0, b: 1.0 }, f64::NEG_INFINITY, 1.0).contains("finite"));
    assert!(error(Family::LogUniform, 0.0, 1.0).contains("above 0"));
    assert!(error(Family::Normal { mean: 0.0, sd: 0.0 }, 0.0, 1.0).contains("sd"));
    assert!(
        error(
            Family::Gamma {
                shape: -1.0,
                rate: 1.0
            },
            0.0,
            1.0
        )
        .contains("shape")
    );
    assert!(
        error(
            Family::LogNormal {
                mu: 0.0,
                sigma: 1.0
            },
            -2.0,
            0.0
        )
        .contains("above 0")
    );
    // A bound below 0 leaves the positive part of a log-normal or a gamma.
    assert!(
        Distribution::new(
            Family::LogNormal {
                mu: 0.0,
                sigma: 1.0
            },
            -2.0,
            3.0
        )
        .is_ok()
    );
}

#[test]
fn draws_of_a_seed_equal_the_reference() {
    let cases = [
        (Family::Uniform, 2.0, 6.0),
        (Family::LogUniform, 0.01, 10.0),
        (
            Family::Normal { mean: 1.0, sd: 2.0 },
            f64::NEG_INFINITY,
            f64::INFINITY,
        ),
        (
            Family::LogNormal {
                mu: 0.5,
                sigma: 0.8,
            },
            0.0,
            f64::INFINITY,
        ),
        (Family::Beta { a: 0.7, b: 2.5 }, -1.0, 3.0),
        (
            Family::Gamma {
                shape: 2.5,
                rate: 1.5,
            },
            0.0,
            f64::INFINITY,
        ),
    ];
    let mut generator = Generator::from_seed(2024);
    let draws: Vec<Vec<f64>> = cases
        .iter()
        .map(|&(family, lower, upper)| {
            let d = distribution(family, lower, upper);
            (0..3).map(|_| d.sample(&mut generator)).collect()
        })
        .collect();
    let reference: [[f64; 3]; 6] = REFERENCE;
    for (case, (drawn, expected)) in draws.iter().zip(reference).enumerate() {
        for (value, expected) in drawn.iter().zip(expected) {
            assert_close(*value, expected, 1e-13, &format!("case {case}"));
        }
    }
}

/// The first three draws of each case of `draws_of_a_seed_equal_the_reference`.
const REFERENCE: [[f64; 3]; 6] = [
    [
        4.097_726_165_840_605,
        3.179_622_357_867_493,
        2.974_204_996_424_141,
    ],
    [
        0.184_063_976_254_971_06,
        1.756_186_537_258_092,
        2.660_091_051_414_15,
    ],
    [
        5.048_464_291_804_56,
        1.841_116_690_493_844_7,
        -1.621_051_180_170_447,
    ],
    [
        1.297_715_505_841_080_8,
        1.979_441_748_984_425_6,
        3.395_113_629_270_184_7,
    ],
    [
        -0.281_340_988_577_469_25,
        -0.371_267_204_917_134_6,
        -0.635_019_129_265_543_7,
    ],
    [
        2.826_629_431_560_361_3,
        1.133_532_313_116_699_7,
        0.924_991_416_647_284_9,
    ],
];

#[test]
fn draws_follow_their_distribution() {
    let d = distribution(
        Family::Gamma {
            shape: 2.5,
            rate: 1.5,
        },
        0.5,
        4.0,
    );
    let mut generator = Generator::from_seed(9);
    let n = 20_000;
    let draws: Vec<f64> = (0..n).map(|_| d.sample(&mut generator)).collect();
    assert!(draws.iter().all(|x| (0.5..=4.0).contains(x)));
    let mean = draws.iter().sum::<f64>() / n as f64;
    let error = 4.0 * (d.variance() / n as f64).sqrt();
    assert!(
        (mean - d.mean()).abs() < error,
        "{mean} against {}",
        d.mean()
    );
    let below = draws.iter().filter(|&&x| x <= d.quantile(0.3)).count() as f64 / n as f64;
    assert!(
        (below - 0.3).abs() < 4.0 * (0.3 * 0.7 / n as f64).sqrt(),
        "{below}"
    );
}
