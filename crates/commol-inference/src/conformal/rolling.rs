//! Rolling-origin conformal: intervals ahead of the data, one quantile per
//! horizon.
//!
//! Each origin fits the observations at or before it (the later ones get
//! weight 0) and scores the later ones. Horizon `k` is the `k`-th
//! observation of a series after the origin: a count, since positions can be
//! irregular. The intervals are around the fit of every observation, at
//! positions after the data, where the `k`-th position is horizon `k`.

use std::collections::HashMap;

use commol_calibration::SeriesIndex;
use commol_core::SimulationEngine;

use crate::conformal::intervals::{ConformalRun, PredictionIntervals};
use crate::conformal::predict::{set_predictions, targets};
use crate::conformal::quantile::conformal_quantile;
use crate::conformal::score::{ConformalSettings, scales};
use crate::error::MethodError;
use crate::resampling::{FitUnit, Perturbation, ReplicateRunner};

fn check<E: SimulationEngine + Send + Sync>(
    runner: &ReplicateRunner<'_, E>,
    origins: &[f64],
    positions: &[f64],
    settings: ConformalSettings,
) -> Result<(), MethodError> {
    let invalid = |message: String| Err(MethodError::Invalid(message));
    settings
        .score
        .check_rolling()
        .map_err(MethodError::Invalid)?;
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
/// least `settings.confidence_level`.
///
/// # Errors
///
/// Returns an error when the score is `"spread"`, a position is not after
/// the data, an origin has no observation before or after it, every fit
/// fails ([`FitUnit::require`]), or a point cannot be predicted.
pub fn rolling_conformal<E>(
    runner: &ReplicateRunner<'_, E>,
    origins: &[f64],
    positions: &[f64],
    settings: ConformalSettings,
) -> Result<ConformalRun, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    check(runner, origins, positions, settings)?;
    let problem = runner.problem();
    let observed = problem.observed_data();
    let perturbations: Vec<Perturbation> = origins
        .iter()
        .map(|&origin| {
            Perturbation::Weights(
                observed
                    .iter()
                    .map(|obs| if obs.at <= origin { obs.weight } else { 0.0 })
                    .collect(),
            )
        })
        .collect();
    let refits = runner.fit_all(&perturbations);
    refits.require(FitUnit::Origin, 1)?;
    let (fitted_origins, fits): (Vec<f64>, Vec<Vec<f64>>) = refits
        .fits
        .iter()
        .map(|(index, result)| {
            (
                origins[*index],
                problem.applied_parameter_values(&result.best_parameters),
            )
        })
        .unzip();
    let predictions = set_predictions(problem, &fits, observed)?;
    let observed_scales = scales(settings.score, observed, observed, None)?;
    let series = SeriesIndex::new(observed);
    let mut by_horizon: HashMap<(String, usize), Vec<f64>> = HashMap::new();
    let mut scores = Vec::new();
    for (predicted, &origin) in predictions.iter().zip(&fitted_origins) {
        for members in series.indices() {
            let later = members.iter().filter(|&&i| observed[i].at > origin);
            for (step, &i) in later.enumerate() {
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
    let target_scales = scales(settings.score, observed, &targets, None)?;
    let intervals = PredictionIntervals::symmetric(
        settings.confidence_level,
        targets.clone(),
        centre,
        &target_scales,
        |index| {
            let target = &targets[index];
            let horizon = sorted.partition_point(|&p| p < target.at) + 1;
            by_horizon
                .get(&(settings.by.group(&target.label).to_string(), horizon))
                .map(|scores| conformal_quantile(scores, settings.confidence_level))
        },
    );
    Ok(ConformalRun {
        intervals,
        scores,
        failed: refits.failed,
        not_converged: refits.not_converged,
    })
}
