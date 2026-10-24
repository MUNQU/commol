//! The replicate runner: fits a copy of the problem with the data of each
//! replicate, in parallel, in one loop that every method of several fits
//! shares.

use commol_calibration::{
    CalibrationProblem, CalibrationResult, OptimizationConfig, optimize_starts, start_points,
};
use commol_core::SimulationEngine;
use commol_random::Generator;
use rayon::prelude::*;

use crate::error::MethodError;
use crate::resampling::resamplers::{Perturbation, Resampler};
use crate::sample::ParameterSample;
use commol_random::Stream;

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
    /// [`MethodError::Failed`], whose message says how many fitted, of how
    /// many, how many the method needs, and the first three reasons; it says
    /// that every fit failed only when none fitted.
    pub fn require(
        self,
        fitted: usize,
        failed: &[(usize, String)],
        needed: usize,
    ) -> Result<(), MethodError> {
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
        Err(MethodError::Failed(if fitted == 0 {
            format!("Every {name} failed: {count}")
        } else {
            format!("Only {count}")
        }))
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
    pub fn require(&self, unit: FitUnit, needed: usize) -> Result<(), MethodError> {
        unit.require(self.fits.len(), &self.failed, needed)
    }
}

/// Fits replicates of a problem from its fit.
///
/// Replicate `b` draws its data from `Stream::ReplicateData.replicate(seed, b)`
/// ([`Stream`]), and is the calibration of its data seeded with
/// `Stream::ReplicateFits.replicate(seed, b)`: it starts from the fit and,
/// with more than one start, from rows of a Latin hypercube over the bounds
/// ([`commol_calibration::start_points`]), and keeps the best start
/// ([`commol_calibration::optimize_starts`]).
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
    /// [`MethodError::Invalid`] when `starts` is 0 or `fit` has the wrong
    /// length.
    pub fn new(
        problem: &'a CalibrationProblem<E>,
        config: &'a OptimizationConfig,
        fit: &[f64],
        starts: usize,
        seed: u64,
    ) -> Result<Self, MethodError> {
        commol_calibration::ranges::STARTS
            .check(starts)
            .map_err(MethodError::Invalid)?;
        if fit.len() != problem.num_parameters() {
            return Err(MethodError::Invalid(format!(
                "The fit has {} values for {} parameters",
                fit.len(),
                problem.num_parameters()
            )));
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

    /// The seed of the method, which its streams come from ([`Stream`]).
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
    /// [`MethodError::Invalid`] when the data are not valid.
    pub fn replicate_problem(
        &self,
        perturbation: &Perturbation,
    ) -> Result<CalibrationProblem<E>, MethodError> {
        match perturbation {
            Perturbation::Weights(weights) => self.problem.with_data(&self.values, weights),
            Perturbation::Values(values) => self.problem.with_data(values, &self.weights),
        }
        .map_err(MethodError::Invalid)
    }

    /// Fit replicate `index` with the data of `perturbation`.
    ///
    /// # Errors
    ///
    /// As [`Self::replicate_problem`], and [`MethodError::Failed`] with the
    /// reason of each start when every start fails.
    pub fn fit_one(
        &self,
        index: usize,
        perturbation: &Perturbation,
    ) -> Result<CalibrationResult, MethodError> {
        let problem = self.replicate_problem(perturbation)?;
        let seed = Stream::ReplicateFits.replicate(self.seed, index as u64);
        let mut points = start_points(problem.parameters(), self.starts, seed);
        points[0] = self.fit.clone();
        optimize_starts(&problem, self.config, points, seed)
            .map(|starts| starts.into_best())
            .map_err(MethodError::Failed)
    }

    /// The perturbations of `replicates` replicates of `resampler`:
    /// replicate `b` draws from `Stream::ReplicateData.replicate(seed, b)`.
    pub fn perturbations(&self, resampler: &dyn Resampler, replicates: usize) -> Vec<Perturbation> {
        (0..replicates)
            .into_par_iter()
            .map(|index| {
                let mut generator =
                    Generator::from_seed(Stream::ReplicateData.replicate(self.seed, index as u64));
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
                Err(reason) => failed.push((index, reason.to_string())),
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
/// [`MethodError::Failed`] of [`FitUnit::require`] when every replicate
/// fails; [`MethodError::Invalid`] when `replicates` is outside its range
/// ([`crate::ranges::REPLICATES`]) or the problem names a parameter twice,
/// so its fits are no sample.
pub fn bootstrap<E>(
    runner: &ReplicateRunner<'_, E>,
    resampler: &dyn Resampler,
    replicates: usize,
) -> Result<BootstrapRun, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    crate::ranges::REPLICATES
        .check(replicates)
        .map_err(MethodError::Invalid)?;
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
    )
    .map_err(MethodError::Invalid)?;
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
