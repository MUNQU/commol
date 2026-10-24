//! Importance sampling: draws of a proposal on the unbounded scale, each
//! weighted by the target over the proposal and smoothed by PSIS
//! ([`crate::psis`]).
//!
//! The weight of a draw `u` is `p(θ) · |dθ/du| / q(u)`, with `θ` the draw
//! mapped back: the density of the target on the unbounded scale adds the log
//! Jacobian of [`Transform`], as the moves of the SMC kernel do.

use commol_calibration::Workspace;
use commol_random::Generator;
use commol_statistics::statistics::effective_size;
use rayon::prelude::*;

use crate::psis::smooth;
use crate::smc::TemperedTarget;
use crate::target::Target;
use crate::transform::Transform;
use commol_random::Stream;

/// The port of a proposal: a distribution on the unbounded scale that draws
/// and gives its log density.
pub trait Proposal: Sync {
    /// One draw on the unbounded scale.
    fn draw(&self, generator: &mut Generator) -> Vec<f64>;

    /// The log density at a point of the unbounded scale.
    fn log_density(&self, unbounded: &[f64]) -> f64;
}

/// The last level of a [`TemperedTarget`] as a [`Target`]: the log prior plus
/// the log kernel of the value one run keeps, `−∞` outside the support. A
/// stochastic model runs one replicate, from [`Stream::PosteriorReplicate`]
/// of the seed, at every point: the density is a function of the point, with
/// common random numbers for all points.
pub struct TemperedPosterior<'a, T: TemperedTarget> {
    target: &'a T,
    replicate: u64,
}

impl<'a, T: TemperedTarget> TemperedPosterior<'a, T> {
    /// The target at the last level of `target`, with the replicate of
    /// `seed`.
    pub fn new(target: &'a T, seed: u64) -> Self {
        Self {
            target,
            replicate: Stream::PosteriorReplicate.seed(seed),
        }
    }
}

impl<T: TemperedTarget> Target for TemperedPosterior<'_, T> {
    fn dimension(&self) -> usize {
        self.target.bounds().len()
    }

    fn log_density(&self, values: &[f64], workspace: &mut Workspace) -> Result<f64, String> {
        let log_prior = self.target.log_prior(values);
        let kept =
            self.target
                .evaluate(values, &mut Generator::from_seed(self.replicate), workspace)?;
        Ok(match kept {
            Some(value) => log_prior + self.target.log_kernel(value, self.target.last_level()),
            None => f64::NEG_INFINITY,
        })
    }
}

/// The draws of an importance sampler and their smoothed weights.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportanceRun {
    /// One row per draw, on the scale of the parameters.
    pub values: Vec<Vec<f64>>,
    /// The smoothed log weights, normalized so that the weights sum to 1.
    pub log_weights: Vec<f64>,
    /// The log weights before smoothing, of any scale.
    pub raw_log_weights: Vec<f64>,
    /// The Pareto shape `k̂` of the tail of the weights.
    pub k_hat: f64,
    /// Kish's effective size of the smoothed weights.
    pub effective_size: f64,
    /// Whether `k̂ ≤ 0.7`.
    pub reliable: bool,
}

/// Draw `draws` points of `proposal`, map them back with `transform`, and
/// weight each by `target` over `proposal`, smoothed by PSIS. Draw `i` comes
/// from its own stream of `seed`, so a run repeats whatever the threads.
///
/// # Errors
///
/// Returns a message when the target fails, or PSIS refuses the weights
/// ([`smooth`]).
pub fn importance_sample<T, P>(
    target: &T,
    proposal: &P,
    transform: &Transform,
    draws: usize,
    seed: u64,
) -> Result<ImportanceRun, String>
where
    T: Target + Sync,
    P: Proposal,
{
    let points: Vec<(Vec<f64>, f64)> = (0..draws)
        .into_par_iter()
        .map_init(Workspace::default, |workspace, i| {
            let mut generator =
                Generator::from_seed(Stream::ImportanceDraws.replicate(seed, i as u64));
            let unbounded = proposal.draw(&mut generator);
            let values = transform.from_unbounded(&unbounded);
            // A target of `−∞` gives a weight of 0 whatever the rest.
            let log_weight = target.log_density(&values, workspace)?
                + transform.log_jacobian(&values)
                - proposal.log_density(&unbounded);
            Ok((values, log_weight))
        })
        .collect::<Result<_, String>>()?;
    let (values, raw_log_weights): (Vec<Vec<f64>>, Vec<f64>) = points.into_iter().unzip();
    let smoothed = smooth(&raw_log_weights)?;
    let weights: Vec<f64> = smoothed.log_weights.iter().map(|w| w.exp()).collect();
    Ok(ImportanceRun {
        values,
        effective_size: effective_size(&weights),
        reliable: smoothed.reliable(),
        k_hat: smoothed.k_hat,
        log_weights: smoothed.log_weights,
        raw_log_weights,
    })
}
