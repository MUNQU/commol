//! The tests and bands of the statistics of a set against a reference: the
//! statistics of replicate sets under independence, and the result an
//! analysis returns.

use commol_statistics::special::gamma_q;
use commol_statistics::statistics;

use super::grid::Layout;
use super::sets::{Summary, pairs};
use super::{
    DrawPValues, ProblemDependence, SeriesCorrelation, SeriesDependence, SeriesStatistics,
};

/// The law of the portmanteau under the reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Portmanteau {
    /// `χ²` of its degrees of freedom: the law of `Q̃` of one set of
    /// independent values.
    ChiSquare,
    /// The portmanteau of the replicates: the law of a mean of sets, or of
    /// the residuals of refits.
    Replicates,
}

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
    /// nor any replicate has one.
    pub problem: Option<f64>,
}

/// A statistic, `−∞` when it is undefined, so that it reads below every
/// defined one.
fn or_below(value: Option<f64>) -> f64 {
    value.unwrap_or(f64::NEG_INFINITY)
}

/// The Monte Carlo p-value `(1 + #{x_b ≥ x}) / (B + 1)`.
fn monte_carlo(observed: f64, replicates: &[f64]) -> f64 {
    let above = replicates.iter().filter(|&&x| x >= observed).count();
    (1 + above) as f64 / (replicates.len() + 1) as f64
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

/// Test `observed` against `replicates`: `Q̃` by `portmanteau`, the band of
/// `τ` and the information factor by the replicates. An undefined statistic
/// reads below every defined one; one that no replicate defines either, as
/// of a constant series, has no test.
///
/// # Panics
///
/// Panics when `replicates` is empty.
pub(crate) fn test(
    observed: &Summary,
    replicates: &[Summary],
    portmanteau: Portmanteau,
    confidence_level: f64,
) -> Test {
    assert!(!replicates.is_empty(), "a test needs a replicate");
    let series = observed
        .series
        .iter()
        .enumerate()
        .map(|(s, summary)| {
            let summary = summary.as_ref()?;
            let p_value = summary.portmanteau.map(|q| match portmanteau {
                Portmanteau::ChiSquare => gamma_q(summary.degrees_of_freedom as f64 / 2.0, q / 2.0),
                Portmanteau::Replicates => {
                    let law: Vec<f64> = replicates
                        .iter()
                        .map(|r| or_below(r.series[s].as_ref().and_then(|x| x.portmanteau)))
                        .collect();
                    monte_carlo(q, &law)
                }
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
        problem: defined.then(|| monte_carlo(or_below(observed), &factors)),
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
/// `layout`, with `lags[s]` lags for series `s`.
pub(crate) fn result(
    layout: &Layout,
    lags: &[usize],
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
                max_lag: lags[s],
                correlations: Vec::new(),
                pairs: Vec::new(),
                portmanteau: None,
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
                statistics.portmanteau = summary.portmanteau;
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
