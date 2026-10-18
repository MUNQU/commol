//! The conformal quantile, the scores and the working level of adaptive
//! conformal inference.

use commol_inference::conformal::conformal_quantile;

#[test]
fn conformal_quantile_is_the_ceiling_order_statistic() {
    // ⌈(19 + 1) · 0.9⌉ = 18: the 18th smallest of 1, …, 19, given in any order.
    let scores: Vec<f64> = (1..=19).rev().map(f64::from).collect();

    assert_eq!(conformal_quantile(&scores, 0.9), 18.0);
}
