//! Jackknife+ with blocks (Barber et al. 2021): one refit per block left
//! out, and intervals from the left-out residuals and predictions.
//!
//! Observation `i` of block `b` has the residual `Rᵢ = s(yᵢ, μ̂₋ᵦ(xᵢ))`
//! of the fit `μ̂₋ᵦ` that left its block out. At a target `x` on the scale
//! `σ(x)`, the interval is
//! `[q⁻_α{μ̂₋ᵦ₍ᵢ₎(x) − Rᵢ σ(x)}, q⁺_{1−α}{μ̂₋ᵦ₍ᵢ₎(x) + Rᵢ σ(x)}]` over the
//! observations of the target's group ([`lower_quantile`],
//! [`upper_quantile`]). With blocks of more than one observation it is CV+,
//! with the blocks as folds.

use commol_core::SimulationEngine;
use rayon::prelude::*;

use crate::conformal::ConformalError;
use crate::conformal::intervals::{ConformalRun, PredictionIntervals};
use crate::conformal::predict::{ensemble_spread, set_predictions, targets};
use crate::conformal::quantile::{lower_quantile, upper_quantile};
use crate::conformal::score::{ConformalSettings, NonconformityScore, scales};
use crate::resampling::runner::every_replicate_failed;
use crate::resampling::{BlockScheme, Perturbation, ReplicateRunner};
use crate::statistics;

/// The coverage jackknife+ guarantees under exchangeable blocks, with `n`
/// scored observations in `k` blocks and `α = 1 − confidence_level`: the bound of
/// K-fold CV+ (Barber et al. 2021, Theorem 4),
/// `1 − 2α − min{2(1 − 1/K)/(n/K + 1), (1 − K/n)/(K + 1)}`, which is
/// `1 − 2α` when each block holds one observation (`K = n`). The theorem
/// assumes blocks of `n/K` observations each.
pub fn cv_plus_bound(observations: usize, blocks: usize, confidence_level: f64) -> f64 {
    let n = observations as f64;
    let k = blocks as f64;
    let alpha = 1.0 - confidence_level;
    let correction = (2.0 * (1.0 - 1.0 / k) / (n / k + 1.0)).min((1.0 - k / n) / (k + 1.0));
    1.0 - 2.0 * alpha - correction.max(0.0)
}

/// The intervals of jackknife+ and what its guarantee counts.
#[derive(Debug, Clone)]
pub struct JackknifePlusRun {
    /// The intervals, the residuals and the blocks whose refit failed.
    pub run: ConformalRun,
    /// The blocks whose refit succeeded, `K`.
    pub blocks: usize,
    /// The observations scored, `n`.
    pub observations: usize,
    /// The guaranteed coverage, [`cv_plus_bound`] of `n` and `K`.
    pub bound: f64,
}

/// Jackknife+ intervals of `runner`'s problem at `positions`, for each
/// series of the problem, with the blocks of `scheme`.
///
/// # Errors
///
/// Returns an error when `scheme` has fewer than two blocks, fewer than two
/// refits succeed, a point cannot be predicted, or the score has no scale
/// at a point.
pub fn jackknife_plus<E>(
    runner: &ReplicateRunner<'_, E>,
    scheme: &BlockScheme,
    positions: &[f64],
    settings: ConformalSettings,
) -> Result<JackknifePlusRun, ConformalError>
where
    E: SimulationEngine + Send + Sync,
{
    if scheme.count() < 2 {
        return Err(ConformalError::Invalid(format!(
            "block_length {} makes {} block; jackknife+ needs at least two",
            scheme.length(),
            scheme.count()
        )));
    }
    let problem = runner.problem();
    let observed = problem.observed_data();
    let outcomes: Vec<_> = scheme
        .blocks()
        .par_iter()
        .enumerate()
        .map(|(block, members)| {
            let mut weights: Vec<f64> = observed.iter().map(|obs| obs.weight).collect();
            for &i in members {
                weights[i] = 0.0;
            }
            runner.fit_one(block, &Perturbation::Weights(weights))
        })
        .collect();
    let mut fits = Vec::new();
    let mut fit_of_block = vec![None; scheme.count()];
    let mut failed = Vec::new();
    for (block, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            Ok(result) => {
                fit_of_block[block] = Some(fits.len());
                fits.push(problem.applied_parameter_values(&result.best_parameters));
            }
            Err(reason) => failed.push((block, reason)),
        }
    }
    if fits.len() < 2 {
        return Err(ConformalError::Failed(format!(
            "jackknife+ needs two refits; {}",
            every_replicate_failed(&failed)
        )));
    }
    let targets = targets(observed, positions);
    let points: Vec<_> = observed.iter().chain(&targets).cloned().collect();
    let predictions = set_predictions(problem, &fits, &points)?;
    let spread =
        (settings.score == NonconformityScore::Spread).then(|| ensemble_spread(&predictions));
    let scales = scales(settings.score, observed, &points, spread.as_deref())
        .map_err(ConformalError::Invalid)?;
    // The scored observations: each with its residual and the refit that
    // left it out.
    let block_of = scheme.block_of();
    let residuals: Vec<(usize, usize, f64)> = observed
        .iter()
        .enumerate()
        .filter_map(|(i, obs)| {
            let fit = fit_of_block[block_of[i]]?;
            Some((
                i,
                fit,
                settings.score.of(obs.value, predictions[fit][i], scales[i]),
            ))
        })
        .collect();
    let alpha = 1.0 - settings.confidence_level;
    let offset = observed.len();
    let mut intervals = PredictionIntervals {
        confidence_level: settings.confidence_level,
        targets: targets.clone(),
        prediction: Vec::new(),
        lower: Vec::new(),
        upper: Vec::new(),
    };
    for (t, target) in targets.iter().enumerate() {
        let point = offset + t;
        let group = settings.by.group(&target.label);
        let members: Vec<(f64, f64)> = residuals
            .iter()
            .filter(|(i, _, _)| settings.by.group(&observed[*i].label) == group)
            .map(|&(_, fit, residual)| (predictions[fit][point], residual * scales[point]))
            .collect();
        let below: Vec<f64> = members.iter().map(|(p, r)| p - r).collect();
        let above: Vec<f64> = members.iter().map(|(p, r)| p + r).collect();
        let centres: Vec<f64> = members.iter().map(|(p, _)| *p).collect();
        let weights = vec![1.0; centres.len()];
        intervals.prediction.push(if centres.is_empty() {
            f64::NAN
        } else {
            statistics::quantiles(&centres, &weights, &[0.5])
                .expect("the rows of a run are finite, so are its predictions")[0]
        });
        intervals.lower.push(lower_quantile(&below, alpha));
        intervals
            .upper
            .push(upper_quantile(&above, settings.confidence_level));
    }
    let scored = residuals.len();
    Ok(JackknifePlusRun {
        run: ConformalRun {
            intervals,
            scores: residuals
                .into_iter()
                .map(|(i, _, residual)| (observed[i].label.clone(), residual))
                .collect(),
            failed,
        },
        blocks: fits.len(),
        observations: scored,
        bound: cv_plus_bound(scored, fits.len(), settings.confidence_level),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_observation_per_block_gives_one_minus_twice_alpha() {
        assert!((cv_plus_bound(30, 30, 0.9) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn long_blocks_take_the_smaller_correction_of_theorem_four() {
        // n = 30, K = 6: min(2 · 5/6 / 6, (4/5) / 7) = 4/35.
        assert!((cv_plus_bound(30, 6, 0.9) - (0.8 - 4.0 / 35.0)).abs() < 1e-12);
    }
}
