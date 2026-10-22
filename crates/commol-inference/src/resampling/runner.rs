//! The replicate runner: fits a copy of the problem with the data of each
//! replicate, in parallel, in one loop that every method of several fits
//! shares.

use commol_calibration::{
    CalibrationProblem, CalibrationResult, OptimizationConfig, optimize_from, start_points,
};
use commol_core::SimulationEngine;
use commol_random::{Generator, replicate_seed};
use rayon::prelude::*;

use crate::resampling::resamplers::{Perturbation, Resampler};
use crate::sample::ParameterSample;

/// What a method fits once per perturbation: a replicate of the data, a
/// block left out, or an origin. Its messages name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitUnit {
    /// A replicate of the data (a bootstrap, EnbPI).
    Replicate,
    /// A block left out (the jackknife, jackknife+).
    Block,
    /// The origin of a rolling fit.
    Origin,
}

impl FitUnit {
    /// The name of one, in lower case.
    pub fn name(self) -> &'static str {
        match self {
            Self::Replicate => "replicate",
            Self::Block => "block",
            Self::Origin => "origin",
        }
    }

    /// The entry of the fit `index` that failed for `reason`:
    /// `"Block 2 failed: <reason>"`.
    pub fn failed(self, index: usize, reason: &str) -> String {
        let name = self.name();
        let capital = name[..1].to_uppercase() + &name[1..];
        format!("{capital} {index} failed: {reason}")
    }

    /// The entries of `failed`, the index and reason of each fit that
    /// failed ([`FitUnit::failed`]).
    pub fn entries(self, failed: &[(usize, String)]) -> Vec<String> {
        failed
            .iter()
            .map(|(index, reason)| self.failed(*index, reason))
            .collect()
    }

    /// Check that `fitted` of `fitted + failed.len()` fits are at least
    /// `needed`.
    ///
    /// # Errors
    ///
    /// Returns a message that says how many fitted, of how many, how many
    /// the method needs, and the first three reasons; it says that every fit
    /// failed only when none fitted.
    pub fn require(
        self,
        fitted: usize,
        failed: &[(usize, String)],
        needed: usize,
    ) -> Result<(), String> {
        if fitted >= needed {
            return Ok(());
        }
        let name = self.name();
        let total = fitted + failed.len();
        let reasons: Vec<String> = failed
            .iter()
            .take(3)
            .map(|(index, reason)| format!("{name} {index}: {reason}"))
            .collect();
        let count = format!(
            "{fitted} of {total} {name}s fitted, and the method needs {needed}; the first \
             reasons: {}",
            reasons.join("; ")
        );
        Err(if fitted == 0 {
            format!("Every {name} failed: {count}")
        } else {
            format!("Only {count}")
        })
    }
}

/// The fits of the replicates of a method, in the order of their indices.
#[derive(Debug, Clone)]
pub struct ReplicateFits {
    /// The index and the best fit of each replicate that succeeded.
    pub fits: Vec<(usize, CalibrationResult)>,
    /// The index and reason of each replicate that failed.
    pub failed: Vec<(usize, String)>,
    /// The number of fits kept although their optimizer did not converge.
    pub not_converged: usize,
}

impl ReplicateFits {
    /// Check that at least `needed` replicates fitted ([`FitUnit::require`]).
    ///
    /// # Errors
    ///
    /// Returns the message of [`FitUnit::require`], which names `unit`.
    pub fn require(&self, unit: FitUnit, needed: usize) -> Result<(), String> {
        unit.require(self.fits.len(), &self.failed, needed)
    }
}

/// Fits replicates of a problem from its fit.
///
/// Replicate `b` draws its data and its starts from the seed
/// `replicate_seed(seed, b)`. It starts from the fit and, with more than one
/// start, from rows of a Latin hypercube over the bounds
/// ([`commol_calibration::start_points`]), and keeps the end of lowest loss.
pub struct ReplicateRunner<'a, E: SimulationEngine> {
    problem: &'a CalibrationProblem<E>,
    config: &'a OptimizationConfig,
    fit: Vec<f64>,
    starts: usize,
    seed: u64,
    values: Vec<f64>,
    weights: Vec<f64>,
}

impl<'a, E> ReplicateRunner<'a, E>
where
    E: SimulationEngine + Send + Sync,
{
    /// A runner of `problem` from the parameter values `fit`, with
    /// `starts` starts per replicate.
    ///
    /// # Errors
    ///
    /// Returns a message when `starts` is 0 or `fit` has the wrong length.
    pub fn new(
        problem: &'a CalibrationProblem<E>,
        config: &'a OptimizationConfig,
        fit: &[f64],
        starts: usize,
        seed: u64,
    ) -> Result<Self, String> {
        if starts == 0 {
            return Err("A replicate needs at least one start".to_string());
        }
        if fit.len() != problem.num_parameters() {
            return Err(format!(
                "The fit has {} values for {} parameters",
                fit.len(),
                problem.num_parameters()
            ));
        }
        Ok(Self {
            problem,
            config,
            fit: problem.applied_parameter_values(fit),
            starts,
            seed,
            values: problem.observed_data().iter().map(|o| o.value).collect(),
            weights: problem.observed_data().iter().map(|o| o.weight).collect(),
        })
    }

    /// The problem.
    pub fn problem(&self) -> &CalibrationProblem<E> {
        self.problem
    }

    /// The fit every replicate starts from.
    pub fn fit(&self) -> &[f64] {
        &self.fit
    }

    /// The seed of the replicates: replicate `b` draws from
    /// `replicate_seed(seed, b)`.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The optimizer of every fit.
    pub fn config(&self) -> &OptimizationConfig {
        self.config
    }

    /// The problem of a replicate: a copy of the problem with the data of
    /// `perturbation`.
    ///
    /// # Errors
    ///
    /// Returns the reason when the data are not valid.
    pub fn replicate_problem(
        &self,
        perturbation: &Perturbation,
    ) -> Result<CalibrationProblem<E>, String> {
        match perturbation {
            Perturbation::Weights(weights) => self.problem.with_data(&self.values, weights),
            Perturbation::Values(values) => self.problem.with_data(values, &self.weights),
        }
    }

    /// Fit replicate `index` with the data of `perturbation`.
    ///
    /// # Errors
    ///
    /// Returns the reason when the data are not valid or every start fails.
    pub fn fit_one(
        &self,
        index: usize,
        perturbation: &Perturbation,
    ) -> Result<CalibrationResult, String> {
        let problem = self.replicate_problem(perturbation)?;
        let seed = replicate_seed(self.seed, index as u64);
        let mut starts = start_points(problem.parameters(), self.starts, replicate_seed(seed, 1));
        starts[0] = self.fit.clone();
        let mut best: Option<CalibrationResult> = None;
        let mut first_error = None;
        for (start_index, start) in starts.iter().enumerate() {
            let start_seed = replicate_seed(seed, 2 + start_index as u64);
            match optimize_from(&problem, self.config, start, Some(start_seed)) {
                Ok(result) => {
                    if best
                        .as_ref()
                        .is_none_or(|kept| result.final_loss < kept.final_loss)
                    {
                        best = Some(result);
                    }
                }
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        best.ok_or_else(|| first_error.unwrap_or_default())
    }

    /// The perturbations of `replicates` replicates of `resampler`:
    /// replicate `b` draws from `replicate_seed(seed, b)`.
    pub fn perturbations(&self, resampler: &dyn Resampler, replicates: usize) -> Vec<Perturbation> {
        (0..replicates)
            .into_par_iter()
            .map(|index| {
                let mut generator = Generator::from_seed(replicate_seed(self.seed, index as u64));
                resampler.replicate(&mut generator)
            })
            .collect()
    }

    /// Fit one replicate per perturbation, in parallel: replicate `b` has
    /// the data of `perturbations[b]` ([`ReplicateRunner::fit_one`]).
    pub fn fit_all(&self, perturbations: &[Perturbation]) -> ReplicateFits {
        let outcomes: Vec<_> = perturbations
            .par_iter()
            .enumerate()
            .map(|(index, perturbation)| self.fit_one(index, perturbation))
            .collect();
        let mut fits = Vec::new();
        let mut failed = Vec::new();
        let mut not_converged = 0;
        for (index, outcome) in outcomes.into_iter().enumerate() {
            match outcome {
                Ok(result) => {
                    not_converged += usize::from(!result.converged);
                    fits.push((index, result));
                }
                Err(reason) => failed.push((index, reason)),
            }
        }
        ReplicateFits {
            fits,
            failed,
            not_converged,
        }
    }
}

/// The draws of a bootstrap and what happened to its replicates.
#[derive(Debug, Clone)]
pub struct BootstrapRun {
    /// One draw per replicate that did not fail, with equal weights; the
    /// group of a draw is its replicate.
    pub sample: ParameterSample,
    /// The index and reason of each replicate that failed.
    pub failed: Vec<(usize, String)>,
    /// The number of replicates kept although their optimizer did not
    /// converge.
    pub not_converged: usize,
    /// The number of synthetic observed values below zero, over every
    /// replicate.
    pub negative_values: usize,
}

/// Run `replicates` replicates of `resampler` with `runner` and gather
/// them into a sample.
///
/// # Errors
///
/// Returns the message of [`FitUnit::require`] when every replicate
/// fails.
pub fn bootstrap<E>(
    runner: &ReplicateRunner<'_, E>,
    resampler: &dyn Resampler,
    replicates: usize,
) -> Result<BootstrapRun, String>
where
    E: SimulationEngine + Send + Sync,
{
    let perturbations = runner.perturbations(resampler, replicates);
    let negative_values = perturbations
        .iter()
        .map(|perturbation| match perturbation {
            Perturbation::Values(values) => values.iter().filter(|&&v| v < 0.0).count(),
            Perturbation::Weights(_) => 0,
        })
        .sum();
    let fits = runner.fit_all(&perturbations);
    fits.require(FitUnit::Replicate, 1)?;
    let problem = runner.problem();
    let (groups, values): (Vec<usize>, Vec<Vec<f64>>) = fits
        .fits
        .iter()
        .map(|(index, fit)| {
            (
                *index,
                problem.applied_parameter_values(&fit.best_parameters),
            )
        })
        .unzip();
    let log_weights = vec![0.0; values.len()];
    let sample = ParameterSample::new(
        problem.parameter_names(),
        problem.parameter_bounds(),
        values,
        &log_weights,
        Some(groups),
    )?;
    Ok(BootstrapRun {
        sample,
        failed: fits.failed,
        not_converged: fits.not_converged,
        negative_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entry_names_the_unit_and_the_index_of_its_fit() {
        let failed = [(2, "no weight".to_string())];
        assert_eq!(
            FitUnit::Block.entries(&failed),
            ["Block 2 failed: no weight"]
        );
        assert_eq!(
            FitUnit::Origin.failed(0, "x"),
            "Origin 0 failed: x".to_string()
        );
        assert_eq!(
            FitUnit::Replicate.failed(1, "x"),
            "Replicate 1 failed: x".to_string()
        );
    }

    #[test]
    fn enough_fits_pass_the_check() {
        let failed = [(0, "x".to_string())];
        assert_eq!(FitUnit::Block.require(2, &failed, 2), Ok(()));
        assert!(FitUnit::Block.require(1, &failed, 2).is_err());
    }
}
