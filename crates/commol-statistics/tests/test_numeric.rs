//! The numerical helpers and log densities of `commol-statistics` through
//! their public functions: their limits and their closed forms.

use std::f64::consts::{EULER_GAMMA, LN_2, PI};

use commol_statistics::numeric::{
    digamma, ln_beta, ln_gamma, log_normal_log_density, log_sum_exp, normal_log_density,
    student_t_log_density, trigamma, x_log_y, x_log1p_y,
};

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * b.abs().max(1.0)
}

#[test]
fn log_sum_exp_has_its_limits() {
    assert_eq!(log_sum_exp(&[1.0, f64::INFINITY]), f64::INFINITY);
    assert_eq!(
        log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
        f64::NEG_INFINITY
    );
    assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
}

#[test]
fn log_sum_exp_keeps_the_digits_of_large_terms() {
    // log(e¹⁰⁰⁰ + e¹⁰⁰⁰) = 1000 + ln 2, where exp overflows.
    let value = log_sum_exp(&[1000.0, 1000.0]);
    assert!(close(value, 1000.0 + LN_2, 1e-15), "{value}");
    assert!(close(log_sum_exp(&[0.0, f64::NEG_INFINITY]), 0.0, 1e-15));
}

#[test]
fn x_log_y_is_zero_at_zero() {
    assert_eq!(x_log_y(0.0, 0.0), 0.0);
    assert_eq!(x_log1p_y(0.0, -1.0), 0.0);
    assert!(close(x_log_y(2.0, 8.0), 6.0 * LN_2, 1e-15));
    assert!(close(x_log1p_y(3.0, 1.0), 3.0 * LN_2, 1e-15));
}

#[test]
fn the_special_functions_have_their_values() {
    // Γ(5) = 24 and B(2, 3) = 1/12.
    assert!(close(ln_gamma(5.0), 24f64.ln(), 1e-15));
    assert!(close(ln_beta(2.0, 3.0), -(12f64.ln()), 1e-14));
    assert!(close(digamma(1.0), -EULER_GAMMA, 1e-14));
    assert!(close(digamma(0.5), -EULER_GAMMA - 2.0 * LN_2, 1e-14));
    assert!(close(trigamma(1.0), PI * PI / 6.0, 1e-14));
    assert!(close(trigamma(0.5), PI * PI / 2.0, 1e-13));
}

#[test]
fn the_log_densities_have_their_closed_forms() {
    // A normal of standard deviation 2, one standard deviation from its mean.
    let normal = (1.0 / (2.0 * (2.0 * PI).sqrt()) * (-0.5f64).exp()).ln();
    assert!(close(normal_log_density(1.0, 1, 2f64.ln()), normal, 1e-15));
    // Two independent standard normals at the distance 1 from their mean.
    let two = 2.0 * ((2.0 * PI).sqrt().recip()).ln() - 0.5;
    assert!(close(normal_log_density(1.0, 2, 0.0), two, 1e-15));
    // log Y ~ N(0, 1) at y = e: the normal density at 1, over e.
    let log_normal = (-0.5 - 0.5 * (2.0 * PI).ln()) - 1.0;
    assert!(close(
        log_normal_log_density(std::f64::consts::E, 0.0, 1.0),
        log_normal,
        1e-15
    ));
    // A t of 3 degrees of freedom and scale 2 at one scale from its centre:
    // Γ(2)/(Γ(3/2)√(3π)·2)·(1 + 1/3)⁻².
    let t =
        (1.0 / (libm::tgamma(1.5) * (3.0 * PI).sqrt() * 2.0) * (1.0f64 + 1.0 / 3.0).powi(-2)).ln();
    assert!(close(
        student_t_log_density(1.0, 3.0, 1, 2f64.ln()),
        t,
        1e-14
    ));
}
