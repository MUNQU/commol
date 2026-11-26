//! The dependence of any values, one per observation: the statistics of one
//! set, `Q̃` against permutations of the values, and a wild band of `τ` and
//! of the information factor.

use commol_calibration::ObservedDataPoint;
use commol_core::CONFIDENCE_LEVEL;
use commol_random::{Generator, Stream, label_seed, replicate_seed, sample};
use commol_statistics::dependence::{centered, robust_portmanteau};
use rayon::prelude::*;

use super::SeriesDependence;
use super::grid::Layout;
use super::law::{PortmanteauLaw, Portmanteaus, Tests, result, test};
use super::sets::{SetStatistics, Summary, statistics, summarize};
use crate::error::MethodError;
use crate::ranges::{MAX_LAG, REPLICATES};
use crate::resampling::Multipliers;

/// `values` with the values of each series of `layout` that has statistics
/// centered on their mean and multiplied by a sign ±1 each, drawn with
/// `generator`: a set of independent values with the scale of each value.
pub(crate) fn wild(layout: &Layout, values: &[f64], generator: &mut Generator) -> Vec<f64> {
    let mut flipped = values.to_vec();
    for series in layout
        .series
        .iter()
        .filter(|series| series.reason.is_none())
    {
        let n = series.members.len() as f64;
        let mean: f64 = series.members.iter().map(|&i| values[i]).sum::<f64>() / n;
        for &member in &series.members {
            flipped[member] = Multipliers::Rademacher.draw(generator) * (values[member] - mean);
        }
    }
    flipped
}

/// The summary of one set of values.
pub(crate) fn one(set: &SetStatistics) -> Summary {
    summarize(&[(1.0, set)])
}

/// The summaries of `replicates` wild sets of `values` ([`wild`]) at the lags
/// `1 … max_lag`: set `b` draws its signs from `replicate_seed(seed, b)`.
pub(crate) fn wild_band(
    layout: &Layout,
    max_lag: usize,
    values: &[f64],
    replicates: usize,
    seed: u64,
) -> Vec<Summary> {
    (0..replicates as u64)
        .into_par_iter()
        .map(|b| {
            let mut generator = Generator::from_seed(replicate_seed(seed, b));
            one(&statistics(
                layout,
                max_lag,
                &wild(layout, values, &mut generator),
            ))
        })
        .collect()
}

/// The `Q̃` at the lags `1 … max_lag` of each series of `layout` in
/// `replicates` permutations of `values`, as `[b][s]`: in set `b`, the
/// values of each series with statistics take its positions in a random
/// order, drawn from `label_seed(replicate_seed(seed, b), label)`. Under
/// independent values of one law the order of a series is uniform, so these
/// sets are its exact law. A series draws from its own label, so its sets
/// do not change with the other series of `layout` or their order. A series
/// with no statistics, or no lag in the sum of `Q̃`, is None. The order does
/// not read `max_lag`, so one seed gives the same sets at every `max_lag`,
/// and more replicates keep the first ones.
pub(crate) fn permuted_portmanteaus(
    layout: &Layout,
    max_lag: usize,
    values: &[f64],
    replicates: usize,
    seed: u64,
) -> Portmanteaus {
    (0..replicates as u64)
        .into_par_iter()
        .map(|b| {
            let seed = replicate_seed(seed, b);
            let mut permuted = values.to_vec();
            layout
                .series
                .iter()
                .map(|series| {
                    if series.reason.is_some() {
                        return None;
                    }
                    let mut generator = Generator::from_seed(label_seed(seed, &series.label));
                    let members = &series.members;
                    for i in (1..members.len()).rev() {
                        let j = sample::index(&mut generator, i + 1);
                        permuted.swap(members[i], members[j]);
                    }
                    let portmanteau =
                        robust_portmanteau(&centered(&series.gridded(&permuted)), max_lag);
                    (portmanteau.lags > 0).then_some(portmanteau.statistic)
                })
                .collect()
        })
        .collect()
}

/// Check that every value of `values` is finite, one per observation.
fn check_values(observations: &[ObservedDataPoint], values: &[f64]) -> Result<(), MethodError> {
    if values.len() != observations.len() {
        return Err(MethodError::Invalid(format!(
            "series_dependence reads one value per observation: {} values for {} observations. \
             For the residuals of several draws or randomizations, use residual_dependence",
            values.len(),
            observations.len()
        )));
    }
    if let Some((value, obs)) = values
        .iter()
        .zip(observations)
        .find(|(value, _)| !value.is_finite())
    {
        return Err(MethodError::Invalid(format!(
            "The value of observation '{}' at {} is {value}; series_dependence reads finite \
             values",
            obs.label, obs.at
        )));
    }
    Ok(())
}

/// The dependence of `values`, one per observation of `observations`: the
/// statistics of each series at the lags `1 … max_lag` and of the series
/// together; the p-value of `Q̃` against `replicates` permutations of the
/// values of each series over its positions, whose order comes from
/// the stream [`Stream::DependencePermutations`] of `seed` and the label of
/// the series, so it does not change with the other series; and the band of
/// `τ` at `confidence_level` and the p-value of the information factor from
/// `replicates` wild sets of the values, whose signs come from the stream
/// [`Stream::DependenceSigns`] of `seed`.
///
/// # Errors
///
/// [`MethodError::Invalid`] when a setting is outside its range, or the
/// values are not one finite value per observation.
pub fn series_dependence(
    observations: &[ObservedDataPoint],
    values: &[f64],
    max_lag: usize,
    replicates: usize,
    confidence_level: f64,
    seed: u64,
) -> Result<SeriesDependence, MethodError> {
    MAX_LAG.check(max_lag).map_err(MethodError::Invalid)?;
    REPLICATES.check(replicates).map_err(MethodError::Invalid)?;
    CONFIDENCE_LEVEL
        .check(confidence_level)
        .map_err(MethodError::Invalid)?;
    check_values(observations, values)?;
    let layout = Layout::new(observations);
    let observed = one(&statistics(&layout, max_lag, values));
    let band = wild_band(
        &layout,
        max_lag,
        values,
        replicates,
        Stream::DependenceSigns.seed(seed),
    );
    let portmanteaus = permuted_portmanteaus(
        &layout,
        max_lag,
        values,
        replicates,
        Stream::DependencePermutations.seed(seed),
    );
    let law = PortmanteauLaw::Sets(portmanteaus);
    let tests = Tests::One(test(&observed, &band, &law, confidence_level));
    Ok(result(&layout, max_lag, &observed, &tests, 1))
}
