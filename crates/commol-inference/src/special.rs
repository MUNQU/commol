//! The special functions of the distributions: the normal distribution
//! function and its inverse, and the regularized incomplete gamma and beta
//! functions.
//!
//! Every elementary function comes from `libm`, a port of musl's, so a value
//! is the same on every platform.

use std::f64::consts::{PI, SQRT_2};

/// Relative accuracy the series and continued fractions stop at.
const ACCURACY: f64 = 1e-16;
/// Terms of a series or continued fraction before it stops.
const MAX_TERMS: usize = 10_000;
/// The smallest denominator of the modified Lentz method.
const TINY: f64 = 1e-300;

/// `x`, or `TINY` when `x` is nearer 0: the guard of a denominator of the
/// modified Lentz method.
fn away_from_zero(x: f64) -> f64 {
    if x.abs() < TINY { TINY } else { x }
}

/// `ln Γ(x)` for `x > 0`.
pub(crate) fn ln_gamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// `ln B(a, b)`.
pub(crate) fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// The standard normal density `φ(z)`; 0 at `±∞`.
pub(crate) fn normal_density(z: f64) -> f64 {
    if z.is_infinite() {
        0.0
    } else {
        libm::exp(-0.5 * z * z) / (2.0 * PI).sqrt()
    }
}

/// The standard normal distribution function `Φ(z)`.
pub(crate) fn normal_cdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z / SQRT_2)
}

/// The standard normal survival function `1 − Φ(z)`, accurate in the upper
/// tail.
pub(crate) fn normal_sf(z: f64) -> f64 {
    0.5 * libm::erfc(z / SQRT_2)
}

/// The standard normal quantile `Φ⁻¹(p)`: Acklam's rational approximation,
/// refined by one step of Halley's method on `erfc`.
pub(crate) fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.024_25;
    let tail = |q: f64| {
        let q = libm::sqrt(-2.0 * libm::log(q));
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    let x = if p < LOW {
        tail(p)
    } else if p > 1.0 - LOW {
        -tail(1.0 - p)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    // `Φ(x) − p`; in the upper half as `(1 − p) − (1 − Φ(x))`, which keeps
    // its relative accuracy there.
    let error = if x > 0.0 {
        (1.0 - p) - normal_sf(x)
    } else {
        normal_cdf(x) - p
    };
    let u = error * (2.0 * PI).sqrt() * libm::exp(0.5 * x * x);
    x - u / (1.0 + 0.5 * x * u)
}

/// The standard normal quantile of the upper tail: the `z` with
/// `1 − Φ(z) = q`.
pub(crate) fn normal_quantile_sf(q: f64) -> f64 {
    -normal_quantile(q)
}

/// The regularized lower incomplete gamma function `P(a, x)`, `a > 0`.
pub(crate) fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x.is_infinite() {
        1.0
    } else if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_continued_fraction(a, x)
    }
}

/// The regularized upper incomplete gamma function `Q(a, x) = 1 − P(a, x)`,
/// accurate where `P` is near 1.
pub(crate) fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        1.0
    } else if x.is_infinite() {
        0.0
    } else if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_continued_fraction(a, x)
    }
}

/// `P(a, x)` by its series, for `x < a + 1`.
fn gamma_series(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut denominator = a;
    for _ in 0..MAX_TERMS {
        denominator += 1.0;
        term *= x / denominator;
        sum += term;
        if term.abs() < sum.abs() * ACCURACY {
            break;
        }
    }
    sum * libm::exp(-x + a * libm::log(x) - ln_gamma(a))
}

/// `Q(a, x)` by its continued fraction (modified Lentz), for `x ≥ a + 1`.
fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..MAX_TERMS {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        d = away_from_zero(d);
        c = b + an / c;
        c = away_from_zero(c);
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < ACCURACY {
            break;
        }
    }
    libm::exp(-x + a * libm::log(x) - ln_gamma(a)) * h
}

/// The regularized incomplete beta function `I_x(a, b)`, `a, b > 0`.
pub(crate) fn beta_i(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = libm::exp(a * libm::log(x) + b * libm::log1p(-x) - ln_beta(a, b));
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// The continued fraction of `I_x(a, b)` (modified Lentz).
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    d = away_from_zero(d);
    d = 1.0 / d;
    let mut h = d;
    for m in 1..MAX_TERMS {
        let m = m as f64;
        let m2 = 2.0 * m;
        let even = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + even * d;
        d = away_from_zero(d);
        c = 1.0 + even / c;
        c = away_from_zero(c);
        d = 1.0 / d;
        h *= d * c;
        let odd = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + odd * d;
        d = away_from_zero(d);
        c = 1.0 + odd / c;
        c = away_from_zero(c);
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < ACCURACY {
            break;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance * b.abs().max(1e-300)
    }

    #[test]
    fn the_normal_quantile_inverts_the_distribution_function_in_both_tails() {
        for p in [1e-300, 1e-12, 1e-3, 0.02, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-12] {
            let z = normal_quantile(p);
            let back = if z > 0.0 {
                1.0 - normal_sf(z)
            } else {
                normal_cdf(z)
            };
            assert!(close(back, p, 1e-13), "{p}: {back}");
        }
        assert!(close(normal_sf(normal_quantile_sf(1e-200)), 1e-200, 1e-12));
        assert_eq!(normal_quantile(0.5), 0.0);
    }

    #[test]
    fn the_functions_reach_their_limits_at_the_ends() {
        assert_eq!(normal_density(f64::INFINITY), 0.0);
        assert_eq!(normal_quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(normal_quantile(1.0), f64::INFINITY);
        assert_eq!((gamma_p(2.0, 0.0), gamma_q(2.0, 0.0)), (0.0, 1.0));
        assert_eq!(
            (gamma_p(2.0, f64::INFINITY), gamma_q(2.0, f64::INFINITY)),
            (1.0, 0.0)
        );
        assert_eq!((beta_i(2.0, 3.0, 0.0), beta_i(2.0, 3.0, 1.0)), (0.0, 1.0));
    }

    #[test]
    fn integer_shapes_give_the_poisson_sums() {
        // P(k, x) = 1 − e^{−x} Σ_{j<k} x^j / j!.
        for (k, x) in [(1.0, 0.5), (3.0, 2.0), (5.0, 12.0), (2.0, 40.0)] {
            let mut term = 1.0;
            let mut sum = 0.0;
            for j in 0..k as usize {
                if j > 0 {
                    term *= x / j as f64;
                }
                sum += term;
            }
            let q = libm::exp(-x) * sum;
            assert!(close(gamma_q(k, x), q, 1e-13), "{k} {x}");
            assert!(close(gamma_p(k, x), 1.0 - q, 1e-13), "{k} {x}");
        }
    }

    #[test]
    fn integer_parameters_give_the_binomial_sums() {
        // I_x(a, b) = Σ_{j=a}^{a+b−1} C(n, j) x^j (1 − x)^{n−j}, n = a + b − 1.
        let choose = |n: u32, j: u32| -> f64 {
            (1..=j)
                .map(|i| f64::from(n - j + i) / f64::from(i))
                .product()
        };
        for (a, b, x) in [(1u32, 1u32, 0.3_f64), (2, 3, 0.7), (5, 2, 0.1), (4, 4, 0.5)] {
            let n = a + b - 1;
            let expected: f64 = (a..=n)
                .map(|j| choose(n, j) * x.powi(j as i32) * (1.0 - x).powi((n - j) as i32))
                .sum();
            let actual = beta_i(f64::from(a), f64::from(b), x);
            assert!(close(actual, expected, 1e-13), "{a} {b} {x}: {actual}");
        }
    }
}
