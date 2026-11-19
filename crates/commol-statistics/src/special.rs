//! The special functions of the distributions: the normal distribution
//! function and its inverse, and the regularized incomplete gamma and beta
//! functions, each accurate in its own tail.
//!
//! The factor `xᵃ e⁻ˣ / Γ(a)` of the incomplete gamma and
//! `xᵃ (1 − x)ᵇ / B(a, b)` of the incomplete beta are read from Stirling's
//! series once a shape reaches 10 (Temme 1979; Didonato and
//! Morris 1992), so no term of their logarithm is larger than the logarithm
//! itself.
//!
//! Every elementary function comes from `libm`, a port of musl's, so a value
//! is the same on every platform.

use std::f64::consts::{PI, SQRT_2};

use crate::numeric::{ln_beta, ln_gamma};

/// Relative accuracy the series and continued fractions stop at.
const ACCURACY: f64 = 1e-16;
/// Terms of a series or continued fraction before it stops, at the least.
const MAX_TERMS: usize = 10_000;
/// Terms of the gamma series per `√a`: near `x = a` it needs about `8 √a`
/// of them, the continued fraction fewer.
const TERMS_PER_ROOT: f64 = 16.0;

/// The terms the series and the continued fraction of the incomplete gamma
/// of shape `a` may take before they stop.
fn gamma_terms(a: f64) -> usize {
    MAX_TERMS.max((TERMS_PER_ROOT * a.sqrt()) as usize)
}
/// The smallest denominator of the modified Lentz method.
const TINY: f64 = 1e-300;

/// From this shape on, the factor of an incomplete gamma or beta function
/// reads Stirling's series of `ln Γ`, whose terms after the last kept one are
/// below `1e-16` there.
const LARGE_SHAPE: f64 = 10.0;

/// `δ(x) = ln Γ(x) − ((x − ½) ln x − x + ½ ln 2π)`, Stirling's correction,
/// for `x ≥ LARGE_SHAPE`.
fn stirling_correction(x: f64) -> f64 {
    let inverse = 1.0 / (x * x);
    (1.0 / 12.0
        - inverse
            * (1.0 / 360.0
                - inverse
                    * (1.0 / 1260.0
                        - inverse
                            * (1.0 / 1680.0
                                - inverse
                                    * (1.0 / 1188.0
                                        - inverse * (691.0 / 360_360.0 - inverse / 156.0))))))
        / x
}

/// Below this `|t|`, `t − ln(1 + t)` is read from its series.
const SMALL_ARGUMENT: f64 = 0.1;

/// `t − ln(1 + t)` for `t > −1`, accurate near 0, where it is about `t²/2`.
fn log1p_gap(t: f64) -> f64 {
    if t.abs() < SMALL_ARGUMENT {
        // Σ_{k≥2} (−1)ᵏ tᵏ / k, to the precision of `f64`.
        let mut power = t * t;
        let mut sum = 0.0;
        let mut k = 2.0;
        loop {
            let term = power / k;
            sum += term;
            if term.abs() <= 1e-17 * sum.abs() {
                return sum;
            }
            power *= -t;
            k += 1.0;
        }
    } else {
        t - libm::log1p(t)
    }
}

/// `ln Γ(a) − ln Γ(a + b)` for `a ≥ LARGE_SHAPE`, from Stirling's series:
/// `−(a − ½) ln(1 + b/a) − b ln(a + b) + b + δ(a) − δ(a + b)`.
fn ln_gamma_ratio(a: f64, b: f64) -> f64 {
    -(a - 0.5) * libm::log1p(b / a) - b * libm::log(a + b) + b + stirling_correction(a)
        - stirling_correction(a + b)
}

/// `ln x` given `x` and `y = 1 − x`: `ln(1 − y)` where `x` is near 1.
fn ln_of(x: f64, y: f64) -> f64 {
    if x > 0.5 {
        libm::log1p(-y)
    } else {
        libm::log(x)
    }
}

/// `xᵃ e⁻ˣ / Γ(a)`, the factor of the incomplete gamma function, for
/// `a > 0` and `x > 0`: `x` times the Poisson probability of `a − 1` at
/// the mean `x`.
pub fn gamma_factor(a: f64, x: f64) -> f64 {
    if a < LARGE_SHAPE {
        libm::exp(a * libm::log(x) - x - ln_gamma(a))
    } else {
        // (x/a)ᵃ e^{a − x} √(a/2π) e^{−δ(a)}.
        let t = (x - a) / a;
        libm::exp(-a * log1p_gap(t) + 0.5 * libm::log(a / (2.0 * PI)) - stirling_correction(a))
    }
}

/// `xᵃ yᵇ / B(a, b)` with `y = 1 − x`, the factor of the incomplete beta
/// function, for `a, b > 0` and `x, y` in `(0, 1)`: the probabilities of the
/// binomial and the negative binomial are such factors.
pub fn beta_factor(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let exponent = match (a >= LARGE_SHAPE, b >= LARGE_SHAPE) {
        (true, true) => {
            // λ = (a + b) x − a = b x − a y; a ln((a + b) x / a) = a ln(1 + λ/a).
            let lambda = b * x - a * y;
            let total = a + b;
            -a * log1p_gap(lambda / a) - b * log1p_gap(-lambda / b)
                + 0.5 * libm::log(a * b / (total * 2.0 * PI))
                - stirling_correction(a)
                - stirling_correction(b)
                + stirling_correction(total)
        }
        (true, false) => a * ln_of(x, y) + b * ln_of(y, x) - ln_gamma(b) - ln_gamma_ratio(a, b),
        (false, true) => a * ln_of(x, y) + b * ln_of(y, x) - ln_gamma(a) - ln_gamma_ratio(b, a),
        (false, false) => a * ln_of(x, y) + b * ln_of(y, x) - ln_beta(a, b),
    };
    libm::exp(exponent)
}

/// `x`, or `TINY` when `x` is nearer 0: the guard of a denominator of the
/// modified Lentz method.
fn away_from_zero(x: f64) -> f64 {
    if x.abs() < TINY { TINY } else { x }
}

/// The standard normal density `φ(z)`; 0 at `±∞`.
pub fn normal_density(z: f64) -> f64 {
    if z.is_infinite() {
        0.0
    } else {
        libm::exp(-0.5 * z * z) / (2.0 * PI).sqrt()
    }
}

/// The standard normal distribution function `Φ(z)`.
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z / SQRT_2)
}

/// The standard normal survival function `1 − Φ(z)`, accurate in the upper
/// tail.
pub fn normal_sf(z: f64) -> f64 {
    0.5 * libm::erfc(z / SQRT_2)
}

/// The standard normal quantile `Φ⁻¹(p)`: Acklam's rational approximation,
/// refined by one step of Halley's method on `erfc`.
pub fn normal_quantile(p: f64) -> f64 {
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
pub fn normal_quantile_sf(q: f64) -> f64 {
    -normal_quantile(q)
}

/// Bisection steps of a quantile: each halves its bracket.
const BISECTION_STEPS: usize = 200;

/// The point `x ≥ 0` where the nondecreasing `cdf` reaches `level`: the
/// bracket `[0, start]`, doubled until `cdf` reaches `level` at its top,
/// then bisection to the precision of `f64`.
fn bisect(cdf: impl Fn(f64) -> f64, level: f64, start: f64) -> f64 {
    let mut low = 0.0;
    let mut high = start;
    while cdf(high) < level {
        low = high;
        high *= 2.0;
    }
    for _ in 0..BISECTION_STEPS {
        let middle = 0.5 * (low + high);
        if middle <= low || middle >= high {
            break;
        }
        if cdf(middle) < level {
            low = middle;
        } else {
            high = middle;
        }
    }
    0.5 * (low + high)
}

/// The quantile of the Student t with `df > 0` degrees of freedom at
/// `p` in `(0, 1)`, by bisection on its upper tail
/// `1 − F(t) = ½ I_{ν/(ν+t²)}(ν/2, ½)` for `t ≥ 0`, and symmetry.
pub fn student_t_quantile(p: f64, df: f64) -> f64 {
    if p < 0.5 {
        return -student_t_quantile(1.0 - p, df);
    }
    let upper_tail = |t: f64| 0.5 * beta_i(0.5 * df, 0.5, df / (df + t * t));
    bisect(|t| -upper_tail(t), -(1.0 - p), 1.0)
}

/// The regularized lower incomplete gamma function `P(a, x)`, `a > 0`.
pub fn gamma_p(a: f64, x: f64) -> f64 {
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
pub fn gamma_q(a: f64, x: f64) -> f64 {
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
    for _ in 0..gamma_terms(a) {
        denominator += 1.0;
        term *= x / denominator;
        sum += term;
        if term.abs() < sum.abs() * ACCURACY {
            break;
        }
    }
    sum * gamma_factor(a, x)
}

/// `Q(a, x)` by its continued fraction (modified Lentz), for `x ≥ a + 1`.
fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..gamma_terms(a) {
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
    gamma_factor(a, x) * h
}

/// The regularized incomplete beta function `I_x(a, b)`, `a, b > 0`.
pub fn beta_i(a: f64, b: f64, x: f64) -> f64 {
    beta_tails(a, b, x, 1.0 - x).0
}

/// The two tails of the regularized incomplete beta function,
/// `(I_x(a, b), I_y(b, a))` with `y = 1 − x`, which they sum to 1, for
/// `a, b > 0`: given both `x` and `y`, so that each tail keeps its relative
/// accuracy where the other is near 1. The continued fraction reads the
/// side of `x` below `(a + 1)/(a + b + 2)`, where it converges, and the
/// other tail is its complement.
pub fn beta_tails(a: f64, b: f64, x: f64, y: f64) -> (f64, f64) {
    if x <= 0.0 {
        return (0.0, 1.0);
    }
    if y <= 0.0 {
        return (1.0, 0.0);
    }
    let front = beta_factor(a, b, x, y);
    if x < (a + 1.0) / (a + b + 2.0) {
        let lower = front * beta_continued_fraction(a, b, x) / a;
        (lower, 1.0 - lower)
    } else {
        let upper = front * beta_continued_fraction(b, a, y) / b;
        (1.0 - upper, upper)
    }
}

/// The quantile at `level`, in `(0, 1)`, of the F distribution with 1 and
/// `nu` degrees of freedom: the square of the quantile `(1 + level) / 2`
/// of Student's t with `nu` degrees of freedom. Bisection on its
/// distribution function `I_{x/(x+ν)}(1/2, ν/2)`, to the precision of
/// `f64`.
pub fn f_one_quantile(nu: f64, level: f64) -> f64 {
    bisect(|x| beta_i(0.5, nu / 2.0, x / (x + nu)), level, 1.0)
}

/// The quantile at `level`, in `(0, 1)`, of the chi-square distribution
/// with `nu > 0` degrees of freedom: bisection on its distribution function
/// `P(ν/2, x/2)`, to the precision of `f64`.
pub fn chi_square_quantile(nu: f64, level: f64) -> f64 {
    bisect(|x| gamma_p(nu / 2.0, x / 2.0), level, nu.max(1.0))
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
    fn student_t_quantiles_have_their_closed_forms() {
        // One degree of freedom is the Cauchy, `tan(π(p − ½))`; two give
        // `(2p − 1) / √(2p(1 − p))`.
        for p in [0.05, 0.5, 0.9, 0.975] {
            let cauchy = (std::f64::consts::PI * (p - 0.5)).tan();
            assert!((student_t_quantile(p, 1.0) - cauchy).abs() < 1e-12, "{p}");
            let two = (2.0 * p - 1.0) / (2.0 * p * (1.0 - p)).sqrt();
            assert!((student_t_quantile(p, 2.0) - two).abs() < 1e-12, "{p}");
        }
        assert!((student_t_quantile(0.95, 1e6) - normal_quantile(0.95)).abs() < 1e-5);
    }

    #[test]
    fn chi_square_quantiles_have_their_closed_forms() {
        // With 2 degrees of freedom the distribution function is
        // `1 − e^{−x/2}`; the medians of 1, 10 and 10⁶ degrees of freedom.
        for level in [0.05_f64, 0.5, 0.9] {
            let two = -2.0 * (1.0 - level).ln();
            assert!(
                close(chi_square_quantile(2.0, level), two, 1e-12),
                "{level}"
            );
        }
        assert!(close(
            chi_square_quantile(1.0, 0.5),
            0.454_936_423_119_572,
            1e-12
        ));
        assert!(close(
            chi_square_quantile(10.0, 0.5),
            9.341_817_765_591_97,
            1e-12
        ));
        assert!(close(
            chi_square_quantile(1e6, 0.5),
            999_999.333_333_412_3,
            1e-12
        ));
    }

    #[test]
    fn the_f_quantile_is_the_square_of_the_t_quantile() {
        // The quantiles 0.975 of Student's t with 8 degrees of freedom and
        // 0.95 with 30.
        let t8 = 2.306004135204166;
        let t30 = 1.6972608865939573;
        assert!(close(f_one_quantile(8.0, 0.95), t8 * t8, 1e-12));
        assert!(close(f_one_quantile(30.0, 0.9), t30 * t30, 1e-12));
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
    fn stirling_reads_join_the_direct_ones_at_the_large_shape() {
        // Below and above LARGE_SHAPE, the two forms of each factor agree.
        for (a, x) in [(10.0, 3.0), (10.0, 10.0), (10.0, 25.0)] {
            let direct = libm::exp(a * libm::log(x) - x - ln_gamma(a));
            assert!(close(gamma_factor(a, x), direct, 1e-14), "{a} {x}");
        }
        for (a, b, x) in [(10.0, 12.0, 0.4), (10.0, 3.0, 0.8), (2.0, 10.0, 0.1)] {
            let direct = libm::exp(a * libm::log(x) + b * libm::log1p(-x) - ln_beta(a, b));
            assert!(
                close(beta_factor(a, b, x, 1.0 - x), direct, 1e-14),
                "{a} {b} {x}"
            );
        }
        // The series joins the direct form at its edge, and tends to t²/2.
        for t in [-0.099_999, 0.099_999] {
            assert!(close(log1p_gap(t), t - libm::log1p(t), 1e-13), "{t}");
        }
        let t = 1e-9;
        assert!(close(log1p_gap(t), t * t / 2.0 - t * t * t / 3.0, 1e-15));
    }

    #[test]
    fn the_beta_tails_sum_to_one_and_meet_their_ends() {
        let (lower, upper) = beta_tails(3.0, 4.0, 0.2, 0.8);
        assert!(close(lower + upper, 1.0, 1e-15));
        assert_eq!(beta_tails(3.0, 4.0, 0.0, 1.0), (0.0, 1.0));
        assert_eq!(beta_tails(3.0, 4.0, 1.0, 0.0), (1.0, 0.0));
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
