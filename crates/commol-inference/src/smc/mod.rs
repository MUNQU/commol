//! Sequential Monte Carlo: one sampler of a sequence of distributions, from
//! the prior to a target, the targets that need no noise model, and the
//! posterior of observation models.
//!
//! | Module | Holds |
//! |---|---|
//! | [`sampler`] | [`SmcSampler`]: the particles, the choice of the next level, resampling, the moves and the log evidence; [`abc_rejection`] |
//! | [`kernel`] | The [`Kernel`] port and [`RandomWalk`], random-walk Metropolis on the unbounded scale |
//! | [`gibbs`] | [`GibbsTarget`]: the Gibbs posterior of the loss, and the calibrations of `η` |
//! | [`abc`] | [`AbcTarget`]: approximate Bayesian computation on a distance |
//! | [`likelihood`] | [`LikelihoodTarget`]: the posterior of observation models, and its evidence |
//!
//! A target is a [`TemperedTarget`]: a prior, a model run that gives the
//! value a particle keeps, and a kernel that is a pure function of that
//! value and the level. The sampler never knows which target it runs, and
//! reweighting runs no model.

pub mod abc;
pub mod gibbs;
pub mod kernel;
pub mod likelihood;
pub mod sampler;

use commol_calibration::Workspace;
use commol_random::Generator;

pub use abc::{AbcTarget, Distance, SeriesMetric};
pub use gibbs::{
    CoverageCalibration, GibbsRun, GibbsTarget, bootstrap_coverage_eta, gibbs_posterior,
    information_matching_eta,
};
pub use kernel::{Kernel, RandomWalk};
pub use likelihood::{LikelihoodTarget, bayesian_posterior};
pub use sampler::{SmcRun, SmcSampler, SmcState, SmcStep, Stop, abc_rejection, next_level};

/// The port of a target of the sampler: the distribution at each level is
/// `π_ℓ(θ) ∝ prior(θ) · exp(log_kernel(v(θ), ℓ))`, where `v(θ)` is the value
/// [`Self::evaluate`] gives, and is 0 where it gives `None`.
///
/// Levels run from [`Self::first_level`], the prior, to
/// [`Self::last_level`], the target, and the conditional effective sample
/// size of a move from one level to the next falls as the next one nears the
/// last: the sampler chooses each level by bisection on it.
pub trait TemperedTarget: Sync {
    /// The bounds of each parameter.
    fn bounds(&self) -> &[(f64, f64)];

    /// The log density of the prior: `−∞` outside its support.
    fn log_prior(&self, values: &[f64]) -> f64;

    /// One draw of the prior.
    fn sample_prior(&self, stream: &mut Generator) -> Vec<f64>;

    /// Run the model once at `values` and give the value the particle
    /// keeps, or `None` outside the support (a violation of the problem). A
    /// stochastic model draws its seed from `stream`.
    ///
    /// # Errors
    ///
    /// Returns a message when the model fails, a failure no point avoids.
    fn evaluate(
        &self,
        values: &[f64],
        stream: &mut Generator,
        workspace: &mut Workspace,
    ) -> Result<Option<f64>, String>;

    /// The log kernel of a kept `value` at `level`; a pure function.
    fn log_kernel(&self, value: f64, level: f64) -> f64;

    /// The level of the prior, given the finite values the draws of the
    /// prior keep: every one of them has a kernel of 0 there.
    fn first_level(&self, kept: &[f64]) -> f64;

    /// The level of the target.
    fn last_level(&self) -> f64;
}
