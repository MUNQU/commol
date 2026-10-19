//! The Bayesian posterior of the observation models of a problem,
//! `π(θ) ∝ prior(θ) · L(θ)`, and its evidence.
//!
//! The level is a temperature from 0 (the prior) to 1 (the posterior): the
//! particle keeps its log-likelihood `v`, and the log kernel at level `ℓ` is
//! `ℓ · v`. The log evidence of the run is the log marginal likelihood.

use commol_calibration::{CalibrationProblem, Workspace};
use commol_core::SimulationEngine;
use commol_random::Generator;

use super::TemperedTarget;
use super::gibbs::reached_the_posterior;
use super::kernel::Kernel;
use super::sampler::{SmcRun, SmcSampler};
use crate::priors::Priors;
use crate::target::discrepancy_in_support;

/// The posterior of the observation models of a problem with priors.
pub struct LikelihoodTarget<'a, E: SimulationEngine> {
    problem: &'a CalibrationProblem<E>,
    priors: &'a Priors,
    bounds: Vec<(f64, f64)>,
}

impl<'a, E: SimulationEngine> LikelihoodTarget<'a, E> {
    /// The posterior of `problem` with `priors`.
    ///
    /// # Errors
    ///
    /// Returns a message when the problem has a loss instead of observation
    /// models.
    pub fn new(problem: &'a CalibrationProblem<E>, priors: &'a Priors) -> Result<Self, String> {
        if problem.discrepancy().likelihood().is_none() {
            return Err(
                "BayesianPosterior(...) needs observation models; the problem has a loss. Use \
                 GibbsPosterior(...) for the posterior of a loss"
                    .to_string(),
            );
        }
        Ok(Self {
            problem,
            priors,
            bounds: problem.parameter_bounds(),
        })
    }
}

impl<E: SimulationEngine + Sync> TemperedTarget for LikelihoodTarget<'_, E> {
    fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    fn log_prior(&self, values: &[f64]) -> f64 {
        self.priors.log_density(values)
    }

    fn sample_prior(&self, stream: &mut Generator) -> Vec<f64> {
        self.priors.sample(stream)
    }

    fn evaluate(
        &self,
        values: &[f64],
        _stream: &mut Generator,
        workspace: &mut Workspace,
    ) -> Result<Option<f64>, String> {
        Ok(discrepancy_in_support(self.problem, values, workspace)?
            .map(|negative_log_likelihood| -negative_log_likelihood))
    }

    fn log_kernel(&self, value: f64, level: f64) -> f64 {
        level * value
    }

    fn first_level(&self, _kept: &[f64]) -> f64 {
        0.0
    }

    fn last_level(&self) -> f64 {
        1.0
    }
}

/// Sample the posterior of the observation models of `problem` with
/// `priors`.
///
/// # Errors
///
/// Returns a message when the problem has a loss, the sampler fails, or no
/// level after one below 1 keeps the effective sample size.
pub fn bayesian_posterior<E, K>(
    problem: &CalibrationProblem<E>,
    priors: &Priors,
    sampler: &mut SmcSampler<K>,
    seed: u64,
) -> Result<SmcRun, String>
where
    E: SimulationEngine + Sync,
    K: Kernel,
{
    let target = LikelihoodTarget::new(problem, priors)?;
    reached_the_posterior(sampler.run(&target, seed)?, "The posterior")
}
