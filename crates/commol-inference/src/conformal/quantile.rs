//! The conformal quantile: an order statistic of the scores, with the rank
//! that makes an interval cover at its level.
//!
//! Of `n` values, the upper quantile at level `1 − α` is the
//! `⌈(n + 1)(1 − α)⌉`-th smallest, `+∞` when that rank is above `n`; the
//! lower quantile at `α` is the `⌊α (n + 1)⌋`-th smallest, `−∞` when that
//! rank is 0 (Barber et al. 2021). So fewer than `1/α − 1` values give an
//! unbounded interval.

use std::collections::HashMap;

use crate::conformal::score::Pooling;

/// Ranks within this distance of an integer are that integer, so that a
/// level such as 0.9 does not move a rank by its rounding error.
const RANK_TOLERANCE: f64 = 1e-9;

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted
}

/// The `⌈(n + 1) level⌉`-th smallest of `values`: `+∞` when the rank is
/// above `n`, and `−∞` when it is 0 or less.
pub fn upper_quantile(values: &[f64], level: f64) -> f64 {
    let n = values.len();
    let rank = ((n + 1) as f64 * level - RANK_TOLERANCE).ceil();
    if rank > n as f64 {
        f64::INFINITY
    } else if rank < 1.0 {
        f64::NEG_INFINITY
    } else {
        sorted(values)[rank as usize - 1]
    }
}

/// The `⌊α (n + 1)⌋`-th smallest of `values`, `−∞` when the rank is 0.
pub fn lower_quantile(values: &[f64], alpha: f64) -> f64 {
    let n = values.len();
    let rank = ((n + 1) as f64 * alpha + RANK_TOLERANCE).floor();
    if rank < 1.0 {
        f64::NEG_INFINITY
    } else {
        sorted(values)[(rank as usize).min(n) - 1]
    }
}

/// The conformal quantile of the scores at `level`: [`upper_quantile`],
/// and 0 when the rank is 0 or less, where an interval is the point
/// prediction.
pub fn conformal_quantile(scores: &[f64], level: f64) -> f64 {
    upper_quantile(scores, level).max(0.0)
}

/// The conformal quantile of each group of scores: one per series with
/// [`Pooling::Series`], one for all with [`Pooling::All`]. The group of a
/// score is the group of its label.
pub fn group_quantiles(
    labels: &[String],
    scores: &[f64],
    by: Pooling,
    level: f64,
) -> HashMap<String, f64> {
    let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
    for (label, &score) in labels.iter().zip(scores) {
        groups
            .entry(by.group(label).to_string())
            .or_default()
            .push(score);
    }
    groups
        .into_iter()
        .map(|(group, scores)| {
            let quantile = conformal_quantile(&scores, level);
            (group, quantile)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_values_give_infinite_quantiles() {
        // 1/0.1 − 1 = 9 values are needed at level 0.9.
        let eight: Vec<f64> = (1..=8).map(f64::from).collect();
        let nine: Vec<f64> = (1..=9).map(f64::from).collect();
        assert_eq!(upper_quantile(&eight, 0.9), f64::INFINITY);
        assert_eq!(upper_quantile(&nine, 0.9), 9.0);
        assert_eq!(lower_quantile(&eight, 0.1), f64::NEG_INFINITY);
        assert_eq!(lower_quantile(&nine, 0.1), 1.0);
    }

    #[test]
    fn a_level_of_zero_or_less_gives_a_point_interval() {
        assert_eq!(conformal_quantile(&[1.0, 2.0], 0.0), 0.0);
        assert_eq!(conformal_quantile(&[1.0, 2.0], -0.5), 0.0);
    }

    #[test]
    fn pooled_scores_share_one_quantile() {
        let labels: Vec<String> = ["a", "a", "b"].iter().map(|s| s.to_string()).collect();
        let scores = [1.0, 2.0, 3.0];
        let series = group_quantiles(&labels, &scores, Pooling::Series, 0.5);
        assert_eq!(series["a"], 2.0);
        assert_eq!(series["b"], 3.0);
        let all = group_quantiles(&labels, &scores, Pooling::All, 0.5);
        assert_eq!(all[""], 2.0);
    }
}
