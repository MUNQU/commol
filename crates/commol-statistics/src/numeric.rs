//! The numerical helpers and log densities that the calibration and the
//! inference methods share, one copy of each.
//!
//! A log density takes the squared standardized distance of its point and
//! the log of its scale, so that one function serves one dimension and
//! several: in `d` dimensions with the covariance (or the scale matrix)
//! `L Lᵀ`, the squared distance is `|L⁻¹(x − c)|²` and the log scale is
//! `ln |L| = Σ ln Lᵢᵢ`; in one dimension they are `((x − c)/s)²` and `ln s`.

use std::f64::consts::PI;

/// `log Σ exp(xᵢ)`, from the largest term, so that no term overflows:
/// `+∞` when a term is `+∞`, and `−∞` when every term is `−∞` or there is
/// none.
pub fn log_sum_exp(terms: &[f64]) -> f64 {
    let largest = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if largest.is_infinite() {
        return largest;
    }
    largest + libm::log(terms.iter().map(|&x| libm::exp(x - largest)).sum::<f64>())
}

/// `x · ln y`, with `0 · ln y = 0` also at `y = 0`.
pub fn x_log_y(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * libm::log(y) }
}

/// `x · ln(1 + y)`, with `0 · ln(1 + y) = 0` also at `y = −1`.
pub fn x_log1p_y(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * libm::log1p(y) }
}

/// `ln Γ(x)` for `x > 0`.
pub fn ln_gamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)`.
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Where the asymptotic series of digamma and trigamma start: their next
/// terms are below `1e-14` there.
const ASYMPTOTIC: f64 = 12.0;

/// The digamma function `ψ(x) = d ln Γ(x)/dx` for `x > 0`: the recurrence
/// `ψ(x) = ψ(x + 1) − 1/x` up to 12, then its asymptotic series.
pub fn digamma(mut x: f64) -> f64 {
    let mut shift = 0.0;
    while x < ASYMPTOTIC {
        shift -= 1.0 / x;
        x += 1.0;
    }
    let inverse = 1.0 / (x * x);
    shift + libm::log(x)
        - 0.5 / x
        - inverse
            * (1.0 / 12.0
                - inverse
                    * (1.0 / 120.0
                        - inverse * (1.0 / 252.0 - inverse * (1.0 / 240.0 - inverse / 132.0))))
}

/// The trigamma function `ψ'(x)` for `x > 0`: the recurrence
/// `ψ'(x) = ψ'(x + 1) + 1/x²` up to 12, then its asymptotic
/// series.
pub fn trigamma(mut x: f64) -> f64 {
    let mut shift = 0.0;
    while x < ASYMPTOTIC {
        shift += 1.0 / (x * x);
        x += 1.0;
    }
    let inverse = 1.0 / (x * x);
    shift
        + 1.0 / x
        + 0.5 * inverse
        + inverse / x
            * (1.0 / 6.0
                - inverse
                    * (1.0 / 30.0
                        - inverse * (1.0 / 42.0 - inverse * (1.0 / 30.0 - inverse * 5.0 / 66.0))))
}

/// The log density of a normal in `dimensions` dimensions at the squared
/// standardized distance `squared` from its mean, with the log scale
/// `log_scale` (`ln σ` in one dimension):
/// `−½ d ln 2π − log_scale − ½ squared`.
pub fn normal_log_density(squared: f64, dimensions: usize, log_scale: f64) -> f64 {
    -0.5 * dimensions as f64 * libm::log(2.0 * PI) - log_scale - 0.5 * squared
}

/// The log density at `x > 0` of a log-normal whose `ln X` is a normal of
/// mean `location` and standard deviation `sigma`: the normal density of
/// `ln x`, minus `ln x`.
pub fn log_normal_log_density(x: f64, location: f64, sigma: f64) -> f64 {
    let log_x = libm::log(x);
    let z = (log_x - location) / sigma;
    normal_log_density(z * z, 1, libm::log(sigma)) - log_x
}

/// The log density of a Student t of `df` degrees of freedom in
/// `dimensions` dimensions at the squared standardized distance `squared`
/// from its centre, with the log scale `log_scale` (`ln s` in one
/// dimension):
/// `ln Γ((ν + d)/2) − ln Γ(ν/2) − ½ d ln(νπ) − log_scale − ½ (ν + d) ln(1 + squared/ν)`.
pub fn student_t_log_density(squared: f64, df: f64, dimensions: usize, log_scale: f64) -> f64 {
    let d = dimensions as f64;
    ln_gamma(0.5 * (df + d))
        - ln_gamma(0.5 * df)
        - 0.5 * d * libm::log(df * PI)
        - log_scale
        - 0.5 * (df + d) * libm::log1p(squared / df)
}
