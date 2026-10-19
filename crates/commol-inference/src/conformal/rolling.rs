//! Rolling-origin conformal: intervals ahead of the data, one quantile per
//! horizon.
//!
//! Each origin fits the observations at or before it (the later ones get
//! weight 0) and scores the later ones. Horizon `k` is the `k`-th
//! observation of a series after the origin: a count, since positions can be
//! irregular. The intervals are around the fit of every observation, at
//! positions after the data, where the `k`-th position is horizon `k`.

use std::collections::HashMap;

use commol_calibration::ObservedDataPoint;
use commol_core::SimulationEngine;
use rayon::prelude::*;

use crate::conformal::ConformalError;
use crate::conformal::intervals::{ConformalRun, PredictionIntervals};
use crate::conformal::predict::{set_predictions, targets};
use crate::conformal::quantile::conformal_quantile;
use crate::conformal::score::{ConformalSettings, NonconformityScore, scales};
use crate::resampling::runner::every_replicate_failed;
use crate::resampling::{Perturbation, ReplicateRunner};

/// The indices of the observations of each series after `origin`, in
/// position order, by label in order of first appearance.
fn later_by_series(observed: &[ObservedDataPoint], origin: f64) -> Vec<Vec<usize>> {
    let mut labels: Vec<&str> = Vec::new();
    let mut series: Vec<Vec<usize>> = Vec::new();
    for (index, obs) in observed.iter().enumerate() {
        let slot = match labels.iter().position(|label| *label == obs.label) {
            Some(slot) => slot,
            None => {
                labels.push(&obs.label);
                series.push(Vec::new());
                series.len() - 1
            }
        };
        if obs.at > origin {
            series[slot].push(index);
        }
    }
    for indices in &mut series {
        indices.sort_by(|&a, &b| observed[a].at.total_cmp(&observed[b].at));
    }
    series
}

fn check<E: SimulationEngine + Send + Sync>(
    runner: &ReplicateRunner<'_, E>,
    origins: &[f64],
    positions: &[f64],
    settings: ConformalSettings,
) -> Result<(), ConformalError> {
    let invalid = |message: String| Err(ConformalError::Invalid(message));
    if settings.score == NonconformityScore::Spread {
        return invalid(
            "RollingConformal fits once per origin, so its predictions have no spread for \
             score=\"spread\"; use \"absolute\" or \"normalized\""
                .to_string(),
        );
    }
    let observed = runner.problem().observed_data();
    let last = observed
        .iter()
        .map(|obs| obs.at)
        .fold(f64::NEG_INFINITY, f64::max);
    if let Some(position) = positions.iter().find(|&&p| p <= last || p.is_nan()) {
        return invalid(format!(
            "RollingConformal gives intervals ahead of the data: position {position} is not \
             after the last observation, at {last}"
        ));
    }
    for &origin in origins {
        let before = observed.iter().any(|obs| obs.at <= origin);
        let after = observed.iter().any(|obs| obs.at > origin);
        if !(before && after) {
            return invalid(format!(
                "The origin {origin} needs observations at or before it, to fit, and after it, \
                 to score"
            ));
        }
    }
    Ok(())
}

/// Rolling-origin conformal intervals of `runner`'s problem at `positions`,
/// for each series of the problem, around the fit of `runner`.
///
/// When the scores of a horizon are exchangeable across origins, the
/// interval at that horizon covers a new observation with probability at
/// least `settings.level`.
///
/// # Errors
///
/// Returns an error when the score is `"spread"`, a position is not after
/// the data, an origin has no observation before or after it, every fit
/// fails, or a point cannot be predicted.
pub fn rolling_conformal<E>(
    runner: &ReplicateRunner<'_, E>,
    origins: &[f64],
    positions: &[f64],
    settings: ConformalSettings,
) -> Result<ConformalRun, ConformalError>
where
    E: SimulationEngine + Send + Sync,
{
    check(runner, origins, positions, settings)?;
    let problem = runner.problem();
    let observed = problem.observed_data();
    let outcomes: Vec<_> = origins
        .par_iter()
        .enumerate()
        .map(|(index, &origin)| {
            let weights = observed
                .iter()
                .map(|obs| if obs.at <= origin { obs.weight } else { 0.0 })
                .collect();
            runner.fit_one(index, &Perturbation::Weights(weights))
        })
        .collect();
    let mut fits = Vec::new();
    let mut fitted_origins = Vec::new();
    let mut failed = Vec::new();
    for (index, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            Ok(result) => {
                fits.push(problem.applied_parameter_values(&result.best_parameters));
                fitted_origins.push(origins[index]);
            }
            Err(reason) => failed.push((index, reason)),
        }
    }
    if fits.is_empty() {
        return Err(ConformalError::Failed(every_replicate_failed(&failed)));
    }
    let predictions = set_predictions(problem, &fits, observed)?;
    let observed_scales =
        scales(settings.score, observed, observed, None).map_err(ConformalError::Invalid)?;
    let mut by_horizon: HashMap<(String, usize), Vec<f64>> = HashMap::new();
    let mut scores = Vec::new();
    for (predicted, &origin) in predictions.iter().zip(&fitted_origins) {
        for series in later_by_series(observed, origin) {
            for (step, &i) in series.iter().enumerate() {
                let obs = &observed[i];
                let score = settings
                    .score
                    .of(obs.value, predicted[i], observed_scales[i]);
                by_horizon
                    .entry((settings.by.group(&obs.label).to_string(), step + 1))
                    .or_default()
                    .push(score);
                scores.push((obs.label.clone(), score));
            }
        }
    }
    let mut sorted = positions.to_vec();
    sorted.sort_by(f64::total_cmp);
    let targets = targets(observed, positions);
    let centre = set_predictions(problem, &[runner.fit().to_vec()], &targets)?.remove(0);
    let target_scales =
        scales(settings.score, observed, &targets, None).map_err(ConformalError::Invalid)?;
    let intervals = PredictionIntervals::symmetric(
        settings.level,
        targets.clone(),
        centre,
        &target_scales,
        |index| {
            let target = &targets[index];
            let horizon = sorted.partition_point(|&p| p < target.at) + 1;
            by_horizon
                .get(&(settings.by.group(&target.label).to_string(), horizon))
                .map(|scores| conformal_quantile(scores, settings.level))
        },
    );
    Ok(ConformalRun {
        intervals,
        scores,
        failed,
    })
}
