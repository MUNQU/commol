//! The conformal quantile, the scores and the working level of adaptive
//! conformal inference.

use commol_inference::conformal::{AdaptiveLevel, conformal_quantile};

#[test]
fn conformal_quantile_is_the_ceiling_order_statistic() {
    // ⌈(19 + 1) · 0.9⌉ = 18: the 18th smallest of 1, …, 19, given in any order.
    let scores: Vec<f64> = (1..=19).rev().map(f64::from).collect();

    assert_eq!(conformal_quantile(&scores, 0.9), 18.0);
}

#[test]
fn adaptive_level_follows_the_update_rule() {
    // α = 0.1 and γ = 0.05: a miss moves α_t by γ (α − 1), a cover by γ α.
    let mut level = AdaptiveLevel::new(0.9, 0.05).unwrap();
    assert!((level.alpha() - 0.1).abs() < 1e-12);

    level.update(false);
    assert!((level.alpha() - (0.1 + 0.05 * (0.1 - 1.0))).abs() < 1e-12);

    level.update(true);
    assert!((level.alpha() - (0.1 + 0.05 * (0.1 - 1.0) + 0.05 * 0.1)).abs() < 1e-12);
    assert!((level.level() - (1.0 - level.alpha())).abs() < 1e-12);
}
