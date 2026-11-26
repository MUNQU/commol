//! The tests and bands of the statistics of a set against a reference: the
//! statistics of replicate sets under independence, and the result an
//! analysis returns.

use commol_statistics::statistics;

use super::grid::Layout;
use super::sets::{Summary, pairs};
use super::{
    DrawPValues, ProblemDependence, SeriesCorrelation, SeriesDependence, SeriesStatistics,
};

/// The test of one series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SeriesTest {
    /// The p-value of `Q̃`.
    pub p_value: Option<f64>,
    /// The central interval of `τ`, an undefined one `−∞`; None when no
    /// replicate has a `τ`.
    pub band: Option<(f64, f64)>,
}

/// The tests of a set.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Test {
    /// Each series with statistics, None for the others.
    pub series: Vec<Option<SeriesTest>>,
    /// The p-value of the information factor; None when neither the set
    /// nor any replicate has one, or when that of the set is NaN.
    pub problem: Option<f64>,
}

/// A statistic, `−∞` when it is undefined, so that it reads below every
/// defined one.
fn or_below(value: Option<f64>) -> f64 {
    value.unwrap_or(f64::NEG_INFINITY)
}

/// The relative distance within which a replicate statistic equals the
/// observed one: `|x_b − x| ≤ TIE · (1 + |x|)` counts as `x_b ≥ x`. A
/// permutation that reverses a series, or that moves only values whose
/// products are equal, gives the observed `Q̃` with its sums in another
/// order, so it differs from it only by rounding.
const TIE: f64 = 1e-10;

/// The Monte Carlo p-value `(1 + #{x_b ≥ x}) / (B + 1)`, a tie within
/// [`TIE`] counted as `x_b ≥ x`; `−∞`, an undefined statistic, reads below
/// every replicate. None when `observed` is NaN or `+∞`, the statistic of
/// no comparison.
fn monte_carlo(observed: f64, replicates: &[f64]) -> Option<f64> {
    if observed.is_nan() || observed == f64::INFINITY {
        return None;
    }
    let floor = observed - TIE * (1.0 + observed.abs());
    let above = replicates.iter().filter(|&&x| x >= floor).count();
    Some((1 + above) as f64 / (replicates.len() + 1) as f64)
}

/// The `Q̃` of each series in each replicate of a reference, as `[b][s]`;
/// None for a series with no `Q̃`.
pub(crate) type Portmanteaus = Vec<Vec<Option<f64>>>;

/// The replicates of `Q̃` of a test.
pub(crate) enum PortmanteauLaw {
    /// Sets of their own, as the permutations of the values.
    Sets(Portmanteaus),
    /// The `Q̃` of the replicates of the test.
    Replicates,
}

/// The central interval of `confidence_level` of `values`, with equal
/// weights; a bound among the values at `−∞` is `−∞`.
fn band(values: &[f64], confidence_level: f64) -> (f64, f64) {
    let weights = vec![1.0; values.len()];
    let levels = statistics::interval_levels(confidence_level);
    let q = statistics::quantiles(values, &weights, &levels).expect("no value is NaN");
    let bound = |x: f64| if x.is_nan() { f64::NEG_INFINITY } else { x };
    (bound(q[0]), bound(q[1]))
}

/// Test `observed` against its reference: `Q̃` of each series by its Monte
/// Carlo p-value over the replicates of `law`; the information factor by
/// its Monte Carlo p-value over `replicates`, and the band of `τ` by their
/// quantiles. An undefined statistic reads below every defined one; one
/// that no replicate defines either, as of a constant series, has no test.
/// A statistic that is NaN, of values whose products pass the largest
/// number, has no p-value.
///
/// # Panics
///
/// Panics when `replicates` or the sets of `law` are empty.
pub(crate) fn test(
    observed: &Summary,
    replicates: &[Summary],
    law: &PortmanteauLaw,
    confidence_level: f64,
) -> Test {
    let portmanteau = |r: &Summary, s: usize| r.series[s].as_ref().and_then(|x| x.portmanteau);
    let sets = match law {
        PortmanteauLaw::Sets(sets) => sets.len(),
        PortmanteauLaw::Replicates => replicates.len(),
    };
    assert!(
        !replicates.is_empty() && sets > 0,
        "a test needs a replicate"
    );
    let series = observed
        .series
        .iter()
        .enumerate()
        .map(|(s, summary)| {
            let summary = summary.as_ref()?;
            let p_value = summary.portmanteau.and_then(|q| {
                let replicates: Vec<f64> = match law {
                    PortmanteauLaw::Sets(sets) => sets.iter().map(|r| or_below(r[s])).collect(),
                    PortmanteauLaw::Replicates => replicates
                        .iter()
                        .map(|r| or_below(portmanteau(r, s)))
                        .collect(),
                };
                monte_carlo(q, &replicates)
            });
            let taus: Vec<f64> = replicates
                .iter()
                .map(|r| or_below(r.series[s].as_ref().and_then(|x| x.tau.clone().ok())))
                .collect();
            let defined = taus.iter().any(|&tau| tau > f64::NEG_INFINITY);
            Some(SeriesTest {
                p_value,
                band: defined.then(|| band(&taus, confidence_level)),
            })
        })
        .collect();
    let factors: Vec<f64> = replicates
        .iter()
        .map(|r| or_below(r.problem.information_factor))
        .collect();
    let observed = observed.problem.information_factor;
    let defined = observed.is_some() || factors.iter().any(|&f| f > f64::NEG_INFINITY);
    Test {
        series,
        problem: defined
            .then(|| monte_carlo(or_below(observed), &factors))
            .flatten(),
    }
}

/// The spread of the p-values of the draws, with `weights`.
pub(crate) fn spread(p_values: &[f64], weights: &[f64], confidence_level: f64) -> DrawPValues {
    let median = statistics::quantiles(p_values, weights, &[0.5]).expect("no p-value is NaN")[0];
    let level = 1.0 - confidence_level;
    let total: f64 = weights.iter().sum();
    let below: f64 = p_values
        .iter()
        .zip(weights)
        .filter(|(p, _)| **p <= level)
        .map(|(_, w)| w)
        .sum();
    DrawPValues {
        median,
        share_below: below / total,
    }
}

/// How the result tests its statistics.
pub(crate) enum Tests {
    /// One draw: its tests.
    One(Test),
    /// Many draws: the spread of the p-values of `Q̃` of each series, and of
    /// the information factor.
    Draws {
        /// Each series with a portmanteau, None for the others.
        series: Vec<Option<DrawPValues>>,
        /// The information factor, None when a draw has no test of it.
        problem: Option<DrawPValues>,
    },
}

/// The result of an analysis of the statistics `summary` of the series of
/// `layout` at the lags `1 … max_lag`.
pub(crate) fn result(
    layout: &Layout,
    max_lag: usize,
    summary: &Summary,
    tests: &Tests,
    draws: usize,
) -> SeriesDependence {
    let series = layout
        .series
        .iter()
        .enumerate()
        .map(|(s, series)| {
            let mut statistics = SeriesStatistics {
                label: series.label.clone(),
                values: series.members.len(),
                reason: series.reason.clone(),
                max_lag,
                correlations: Vec::new(),
                pairs: Vec::new(),
                few_pairs: Vec::new(),
                portmanteau: None,
                portmanteau_reason: series.reason.clone(),
                degrees_of_freedom: 0,
                p_value: None,
                draw_p_values: None,
                tau: None,
                tau_reason: series.reason.clone(),
                tau_band: None,
                cut_lag: None,
            };
            if let Some(summary) = &summary.series[s] {
                statistics.correlations.clone_from(&summary.correlations);
                statistics.pairs.clone_from(&summary.pairs);
                statistics.few_pairs.clone_from(&summary.few_pairs);
                statistics.portmanteau = summary.portmanteau;
                statistics.portmanteau_reason = summary.portmanteau_reason.map(String::from);
                statistics.degrees_of_freedom = summary.degrees_of_freedom;
                statistics.tau = summary.tau.clone().ok();
                statistics.tau_reason = summary.tau.clone().err();
                statistics.cut_lag = summary.cut_lag;
                match tests {
                    Tests::One(test) => {
                        let test = test.series[s].expect("a series with statistics is tested");
                        statistics.p_value = test.p_value;
                        statistics.tau_band = test.band;
                    }
                    Tests::Draws { series, .. } => statistics.draw_p_values = series[s],
                }
            }
            statistics
        })
        .collect();
    let correlations = pairs(layout)
        .into_iter()
        .zip(&summary.problem.correlations)
        .map(|((i, j), &correlation)| SeriesCorrelation {
            first: layout.series[i].label.clone(),
            second: layout.series[j].label.clone(),
            correlation,
        })
        .collect();
    let problem = &summary.problem;
    let (p_value, draw_p_values) = match tests {
        Tests::One(test) => (test.problem, None),
        Tests::Draws { problem, .. } => (None, *problem),
    };
    SeriesDependence {
        series,
        correlations,
        problem: ProblemDependence {
            variance_factor: problem.variance_ratio,
            tau: problem.tau.clone().ok(),
            tau_reason: problem.tau.clone().err(),
            information_factor: problem.information_factor,
            p_value,
            draw_p_values,
        },
        draws,
        failed: Vec::new(),
        not_converged: 0,
    }
}

#[cfg(test)]
mod tests {
    use commol_statistics::dependence::{centered, robust_portmanteau};

    use super::*;

    #[test]
    fn a_replicate_within_the_tie_reaches_the_observed_statistic() {
        let values = [0.3, -1.2, 2.5, 0.7, -0.4, 1.9, -2.2, 0.1, 1.4, -0.8];
        let q = |values: Vec<f64>| -> f64 {
            let series: Vec<Option<f64>> = values.into_iter().map(Some).collect();
            robust_portmanteau(&centered(&series), 3).statistic
        };
        let observed = q(values.to_vec());
        // The identity and the reversal are permutations of the same Q̃.
        let reversed = q(values.iter().rev().copied().collect());
        assert_eq!(monte_carlo(observed, &[observed, reversed]), Some(1.0));
        // A rounding below the observed value is a tie; a real difference is
        // not.
        let rounded = observed * (1.0 - 1e-13);
        assert_eq!(monte_carlo(observed, &[rounded]), Some(1.0));
        assert_eq!(monte_carlo(observed, &[observed * (1.0 - 1e-6)]), Some(0.5));
    }

    #[test]
    fn a_statistic_that_is_not_a_number_has_no_p_value() {
        assert_eq!(monte_carlo(f64::NAN, &[1.0, 2.0]), None);
        assert_eq!(monte_carlo(f64::INFINITY, &[1.0, 2.0]), None);
        // An undefined statistic reads below every replicate.
        assert_eq!(monte_carlo(f64::NEG_INFINITY, &[1.0, 2.0]), Some(1.0));
    }
}
