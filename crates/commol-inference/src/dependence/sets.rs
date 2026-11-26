//! The statistics of one set of values, one per observation, and their mean
//! over several sets with weights: several randomizations of the quantile
//! residuals, and several draws of a sample.

use commol_statistics::dependence::{
    Autocovariance, Portmanteau, Time, autocovariance, centered, initial_monotone_time,
    robust_portmanteau,
};

use super::grid::Layout;

/// The statistics of one series in one set of values.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SeriesSet {
    /// The autocovariance at lags `1 … max_lag`, None at a lag with no pair.
    pub lags: Vec<Option<Autocovariance>>,
    /// The variance of the values.
    pub variance: f64,
    /// The robust portmanteau at lags `1 … max_lag`.
    pub portmanteau: Portmanteau,
    /// The integrated time, or why there is none.
    pub time: Result<Time, &'static str>,
}

/// The statistics of the series together in one set of values.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProblemSet {
    /// The correlation at lag 0 of each pair of series with statistics, in
    /// the order of [`pairs`], over their shared positions; None with fewer
    /// than two shared positions or a constant series among them.
    pub correlations: Vec<Option<f64>>,
    /// `Σ_t (Σ_s z_st)² / Σ_st z_st²` over the standardized series `z`:
    /// `Var(Σz) / ΣVar(z)`; None with no series that varies.
    pub variance_ratio: Option<f64>,
    /// The integrated time of `Σz` on the grid of the shared positions, or
    /// why there is none.
    pub sum_time: Result<Time, String>,
}

/// The statistics of one set of values.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SetStatistics {
    /// Each series with statistics, None for the others.
    pub series: Vec<Option<SeriesSet>>,
    /// The series together.
    pub problem: ProblemSet,
}

/// Each pair `(i, j)`, `i < j`, of the series of `layout` with statistics.
pub(crate) fn pairs(layout: &Layout) -> Vec<(usize, usize)> {
    let valid: Vec<usize> = (0..layout.series.len())
        .filter(|&s| layout.series[s].reason.is_none())
        .collect();
    let mut found = Vec::new();
    for (k, &i) in valid.iter().enumerate() {
        for &j in &valid[k + 1..] {
            found.push((i, j));
        }
    }
    found
}

/// The Pearson correlation of the pairs `(x, y)`; None with fewer than two
/// or with one side constant.
fn correlation(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;
    let (mx, my) = pairs
        .iter()
        .fold((0.0, 0.0), |(x, y), (a, b)| (x + a / n, y + b / n));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in pairs {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    (sxx > 0.0 && syy > 0.0).then(|| sxy / libm::sqrt(sxx * syy))
}

/// The statistics of `values`, one per observation, on the series of
/// `layout`, at the lags `1 … max_lag` of each.
pub(crate) fn statistics(layout: &Layout, max_lag: usize, values: &[f64]) -> SetStatistics {
    let mut standardized: Vec<Vec<Option<f64>>> = Vec::with_capacity(layout.series.len());
    let series = layout
        .series
        .iter()
        .map(|series| {
            if series.reason.is_some() {
                standardized.push(Vec::new());
                return None;
            }
            let gridded = centered(&series.gridded(values));
            let variance = autocovariance(&gridded, 0).map_or(0.0, |gamma| gamma.value);
            let scale = if variance > 0.0 {
                1.0 / libm::sqrt(variance)
            } else {
                0.0
            };
            standardized.push(
                series
                    .points
                    .iter()
                    .map(|&point| gridded[point].map(|e| e * scale))
                    .collect(),
            );
            Some(SeriesSet {
                lags: (1..=max_lag)
                    .map(|lag| autocovariance(&gridded, lag))
                    .collect(),
                variance,
                portmanteau: robust_portmanteau(&gridded, max_lag),
                time: initial_monotone_time(&gridded),
            })
        })
        .collect();
    SetStatistics {
        series,
        problem: problem(layout, &standardized),
    }
}

/// The statistics of the series together, from the standardized value of
/// each member of each series with statistics.
fn problem(layout: &Layout, standardized: &[Vec<Option<f64>>]) -> ProblemSet {
    let count = layout.shared_positions();
    let at_shared = |s: usize| -> Vec<Option<f64>> {
        let mut values = vec![None; count];
        for (&position, &value) in layout.shared[s].iter().zip(&standardized[s]) {
            values[position] = value;
        }
        values
    };
    let shared: Vec<Vec<Option<f64>>> = (0..layout.series.len())
        .map(|s| {
            if layout.series[s].reason.is_none() {
                at_shared(s)
            } else {
                Vec::new()
            }
        })
        .collect();
    let correlations = pairs(layout)
        .into_iter()
        .map(|(i, j)| {
            let both: Vec<(f64, f64)> = shared[i]
                .iter()
                .zip(&shared[j])
                .filter_map(|(a, b)| Some(((*a)?, (*b)?)))
                .collect();
            correlation(&both)
        })
        .collect();
    let mut sum = vec![None; count];
    let mut squares = 0.0;
    for values in &shared {
        for (total, value) in sum.iter_mut().zip(values) {
            if let Some(z) = value {
                *total = Some(total.unwrap_or(0.0) + z);
                squares += z * z;
            }
        }
    }
    let variance_ratio = (squares > 0.0)
        .then(|| sum.iter().flatten().map(|total| total * total).sum::<f64>() / squares);
    let sum_time = match &layout.shared_grid {
        Ok((points, length)) => {
            let mut gridded = vec![None; *length];
            for (&point, &total) in points.iter().zip(&sum) {
                gridded[point] = total;
            }
            initial_monotone_time(&centered(&gridded)).map_err(String::from)
        }
        Err(reason) => Err(reason.clone()),
    };
    ProblemSet {
        correlations,
        variance_ratio,
        sum_time,
    }
}

/// The statistics of one series averaged over several sets.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SeriesSummary {
    /// The mean autocorrelation at lags `1 … max_lag`, over the sets whose
    /// series varies; None at a lag with no pair.
    pub correlations: Vec<Option<f64>>,
    /// The pairs of each lag.
    pub pairs: Vec<usize>,
    /// Each lag left out of the portmanteau for its few pairs, as `(lag,
    /// pairs)`.
    pub few_pairs: Vec<(usize, usize)>,
    /// The mean robust portmanteau, None with no lag in its sum.
    pub portmanteau: Option<f64>,
    /// Why there is no portmanteau.
    pub portmanteau_reason: Option<&'static str>,
    /// The lags in the sum of the portmanteau.
    pub degrees_of_freedom: usize,
    /// `τ`, the ratio of the mean long-run variance to the mean variance,
    /// or why there is none.
    pub tau: Result<f64, String>,
    /// The largest lag the sum of `τ` reads in any set.
    pub cut_lag: Option<usize>,
}

/// The statistics of the series together, averaged over several sets.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProblemSummary {
    /// The mean correlation of each pair of series.
    pub correlations: Vec<Option<f64>>,
    /// The mean `Var(Σz) / ΣVar(z)`.
    pub variance_ratio: Option<f64>,
    /// `τ` of `Σz`, as the `τ` of a series.
    pub tau: Result<f64, String>,
    /// The product of the two: `LRV(Σz) / ΣVar(z)`, the factor by which the
    /// dependence multiplies the information.
    pub information_factor: Option<f64>,
}

/// The statistics of several sets of values, averaged with weights.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Summary {
    /// Each series with statistics, None for the others.
    pub series: Vec<Option<SeriesSummary>>,
    /// The series together.
    pub problem: ProblemSummary,
}

/// The weighted mean of the values that are Some; None when none is.
fn mean_of(values: impl Iterator<Item = (f64, Option<f64>)>) -> Option<f64> {
    let (sum, total) = values.fold((0.0, 0.0), |(sum, total), (w, value)| match value {
        Some(v) => (sum + w * v, total + w),
        None => (sum, total),
    });
    (total > 0.0).then(|| sum / total)
}

/// `τ` of the times of several sets with weights: the ratio of the mean
/// long-run variance to the mean variance; the reason of the first set that
/// has no time, or the reason of a ratio not above 0.
fn tau_of<'a, E: ToString + 'a>(
    times: impl Iterator<Item = (f64, &'a Result<Time, E>)>,
) -> (Result<f64, String>, Option<usize>) {
    let (mut long_run, mut variance, mut cut) = (0.0, 0.0, 0);
    for (w, time) in times {
        match time {
            Ok(time) => {
                long_run += w * time.long_run_variance;
                variance += w * time.variance;
                cut = cut.max(time.cut_lag);
            }
            Err(reason) => return (Err(reason.to_string()), None),
        }
    }
    let tau = long_run / variance;
    if tau > 0.0 {
        (Ok(tau), Some(cut))
    } else {
        (
            Err(format!(
                "its estimate {tau:.3} is not above 0: the correlation at the first lags is \
                 negative"
            )),
            Some(cut),
        )
    }
}

/// The weighted mean of the statistics of `sets`.
///
/// # Panics
///
/// Panics when `sets` is empty.
pub(crate) fn summarize(sets: &[(f64, &SetStatistics)]) -> Summary {
    let first = sets.first().expect("at least one set").1;
    let series = (0..first.series.len())
        .map(|s| {
            let head = first.series[s].as_ref()?;
            let of = |set: &SetStatistics| -> SeriesSet {
                set.series[s].clone().expect("every set has one layout")
            };
            let members: Vec<(f64, SeriesSet)> =
                sets.iter().map(|(w, set)| (*w, of(set))).collect();
            let correlations = (0..head.lags.len())
                .map(|k| {
                    mean_of(members.iter().map(|(w, series)| {
                        let value = (series.variance > 0.0)
                            .then_some(series.lags[k])
                            .flatten()
                            .map(|gamma| gamma.value / series.variance);
                        (*w, value)
                    }))
                })
                .collect();
            let degrees_of_freedom = members
                .iter()
                .map(|(_, series)| series.portmanteau.lags)
                .max()
                .unwrap_or(0);
            let portmanteau = (degrees_of_freedom > 0).then(|| {
                mean_of(
                    members
                        .iter()
                        .map(|(w, series)| (*w, Some(series.portmanteau.statistic))),
                )
                .expect("a positive weight")
            });
            let (tau, cut_lag) = tau_of(members.iter().map(|(w, series)| (*w, &series.time)));
            // The pairs of each lag come from the layout, the same in every
            // set, so the first set gives the lags left out.
            Some(SeriesSummary {
                correlations,
                pairs: head
                    .lags
                    .iter()
                    .map(|lag| lag.map_or(0, |gamma| gamma.pairs))
                    .collect(),
                few_pairs: head.portmanteau.few_pairs.clone(),
                portmanteau,
                portmanteau_reason: (degrees_of_freedom == 0)
                    .then_some(head.portmanteau.reason)
                    .flatten(),
                degrees_of_freedom,
                tau,
                cut_lag,
            })
        })
        .collect();
    let correlations = (0..first.problem.correlations.len())
        .map(|k| {
            mean_of(
                sets.iter()
                    .map(|(w, set)| (*w, set.problem.correlations[k])),
            )
        })
        .collect();
    let variance_ratio = mean_of(sets.iter().map(|(w, set)| (*w, set.problem.variance_ratio)));
    let (tau, _) = tau_of(sets.iter().map(|(w, set)| (*w, &set.problem.sum_time)));
    let information_factor = match (variance_ratio, &tau) {
        (Some(ratio), Ok(tau)) => Some(ratio * tau),
        _ => None,
    };
    Summary {
        series,
        problem: ProblemSummary {
            correlations,
            variance_ratio,
            tau,
            information_factor,
        },
    }
}
