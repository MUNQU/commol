//! The dependence of any values, one per observation: the statistics of one
//! set, `Q̃` against `χ²` and a wild band of `τ` and of the information
//! factor.

use commol_calibration::ObservedDataPoint;
use commol_core::CONFIDENCE_LEVEL;
use commol_random::{Generator, Stream, replicate_seed};
use rayon::prelude::*;

use super::SeriesDependence;
use super::grid::Layout;
use super::law::{Portmanteau, Tests, result, test};
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
/// together, the p-value of `Q̃` against `χ²(h)`, and the band of `τ` at
/// `confidence_level` and the p-value of the information factor from
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
    let tests = Tests::One(test(
        &observed,
        &band,
        Portmanteau::ChiSquare,
        confidence_level,
    ));
    Ok(result(&layout, max_lag, &observed, &tests, 1))
}
