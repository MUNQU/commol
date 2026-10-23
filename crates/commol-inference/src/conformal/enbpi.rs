//! Ensemble batch prediction intervals (Xu and Xie 2021).
//!
//! Each replicate draws as many blocks as there are, with replacement, and
//! fits with each observation's weight multiplied by the count of its block,
//! so the blocks it did not draw get weight 0. The leave-one-out prediction
//! of an observation is the mean of the replicates that left its block out;
//! its score is the score of that prediction. The point prediction at a
//! target is the mean, over every observation, of the leave-one-out
//! predictions there, and the conformal quantile of the scores sets the
//! width. No fit beyond the replicates is needed.

use commol_calibration::ObservedDataPoint;
use commol_core::SimulationEngine;
use commol_random::Generator;
use commol_random::sample::index;

use crate::conformal::intervals::{ConformalRun, PredictionIntervals, quantile_lookup};
use crate::conformal::predict::{ensemble_mean, ensemble_spread, set_predictions, targets};
use crate::conformal::quantile::group_quantiles;
use crate::conformal::score::{ConformalSettings, NonconformityScore, scales};
use crate::error::MethodError;
use crate::resampling::{BlockScheme, FitUnit, Perturbation, ReplicateRunner};
use crate::streams::Stream;

/// Blocks drawn with replacement: one multinomial count per block, of as
/// many draws as there are blocks, multiplies the weight of every
/// observation of the block.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockMultinomial {
    weights: Vec<f64>,
    block_of: Vec<usize>,
    blocks: usize,
}

impl BlockMultinomial {
    /// The draws of the blocks of `scheme` over `observations`.
    pub fn new(observations: &[ObservedDataPoint], scheme: &BlockScheme) -> Self {
        Self {
            weights: observations.iter().map(|obs| obs.weight).collect(),
            block_of: scheme.block_of(),
            blocks: scheme.count(),
        }
    }

    /// The count of each block in one replicate.
    pub fn counts(&self, generator: &mut Generator) -> Vec<usize> {
        let mut counts = vec![0; self.blocks];
        for _ in 0..self.blocks {
            counts[index(generator, self.blocks)] += 1;
        }
        counts
    }

    /// The weights of a replicate with block counts `counts`.
    pub fn weights(&self, counts: &[usize]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(&self.block_of)
            .map(|(weight, &block)| weight * counts[block] as f64)
            .collect()
    }
}

/// EnbPI intervals of `runner`'s problem at `positions`, for each series of
/// the problem, from `replicates` replicates of the blocks of `scheme`.
///
/// Replicate `b` draws its blocks from
/// `Stream::EnbpiBlocks.replicate(runner.seed(), b)` ([`Stream`]).
///
/// # Errors
///
/// Returns an error when a block is left out by no replicate that fitted,
/// every replicate fails ([`FitUnit::require`]), a point cannot be
/// predicted, or the score has no scale at a point.
pub fn enbpi<E>(
    runner: &ReplicateRunner<'_, E>,
    scheme: &BlockScheme,
    replicates: usize,
    positions: &[f64],
    settings: ConformalSettings,
) -> Result<ConformalRun, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    let problem = runner.problem();
    let observed = problem.observed_data();
    let resampler = BlockMultinomial::new(observed, scheme);
    let counts: Vec<Vec<usize>> = (0..replicates)
        .map(|index| {
            let mut generator =
                Generator::from_seed(Stream::EnbpiBlocks.replicate(runner.seed(), index as u64));
            resampler.counts(&mut generator)
        })
        .collect();
    let perturbations: Vec<Perturbation> = counts
        .iter()
        .map(|counts| Perturbation::Weights(resampler.weights(counts)))
        .collect();
    let refits = runner.fit_all(&perturbations);
    refits.require(FitUnit::Replicate, 1)?;
    let (counts_of_fit, fits): (Vec<&Vec<usize>>, Vec<Vec<f64>>) = refits
        .fits
        .iter()
        .map(|(index, result)| {
            (
                &counts[*index],
                problem.applied_parameter_values(&result.best_parameters),
            )
        })
        .unzip();
    let spans = scheme.spans();
    let left_out: Vec<Vec<usize>> = (0..scheme.count())
        .map(|block| {
            let members: Vec<usize> = (0..fits.len())
                .filter(|&f| counts_of_fit[f][block] == 0)
                .collect();
            if members.is_empty() {
                let (first, last) = spans[block];
                Err(MethodError::Invalid(format!(
                    "Block [{first}, {last}] is left out by none of the {} replicates that \
                     fitted, so its observations have no leave-one-out prediction; ask for more \
                     replicates",
                    fits.len()
                )))
            } else {
                Ok(members)
            }
        })
        .collect::<Result<_, _>>()?;
    let targets = targets(observed, positions);
    let points: Vec<_> = observed.iter().chain(&targets).cloned().collect();
    let predictions = set_predictions(problem, &fits, &points)?;
    // The leave-one-out prediction of each block at every point.
    let leave_one_out: Vec<Vec<f64>> = left_out
        .iter()
        .map(|members| {
            let rows: Vec<&Vec<f64>> = members.iter().map(|&f| &predictions[f]).collect();
            ensemble_mean(&rows, points.len())
        })
        .collect();
    let spread =
        (settings.score == NonconformityScore::Spread).then(|| ensemble_spread(&predictions));
    let scales = scales(settings.score, observed, &points, spread.as_deref())?;
    let block_of = scheme.block_of();
    let labels: Vec<String> = observed.iter().map(|obs| obs.label.clone()).collect();
    let scores: Vec<f64> = observed
        .iter()
        .enumerate()
        .map(|(i, obs)| {
            settings
                .score
                .of(obs.value, leave_one_out[block_of[i]][i], scales[i])
        })
        .collect();
    let offset = observed.len();
    let centre: Vec<f64> = (0..targets.len())
        .map(|t| {
            block_of
                .iter()
                .map(|&block| leave_one_out[block][offset + t])
                .sum::<f64>()
                / offset as f64
        })
        .collect();
    let quantiles = group_quantiles(&labels, &scores, settings.by, settings.confidence_level);
    let intervals = PredictionIntervals::symmetric(
        settings.confidence_level,
        targets.clone(),
        centre,
        &scales[offset..],
        quantile_lookup(&targets, &quantiles, settings.by),
    );
    Ok(ConformalRun {
        intervals,
        scores: labels.into_iter().zip(scores).collect(),
        failed: refits.failed,
        not_converged: refits.not_converged,
    })
}
