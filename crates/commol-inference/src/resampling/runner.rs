//! The replicate runner: fits a copy of the problem with the data of each
//! replicate, in parallel. It knows no resampler.

use commol_calibration::{
    CalibrationProblem, CalibrationResult, OptimizationConfig, optimize_from, start_points,
};
use commol_core::SimulationEngine;
use commol_random::{Generator, replicate_seed};
use rayon::prelude::*;

use crate::resampling::resamplers::{Perturbation, Resampler};
use crate::sample::ParameterSample;

/// The fit of one replicate, or why it failed.
#[derive(Debug, Clone)]
pub struct ReplicateOutcome {
    /// The index of the replicate.
    pub index: usize,
    /// The number of synthetic observed values below zero.
    pub negative_values: usize,
    /// The best fit of the replicate's starts, or the reason it failed.
    pub result: Result<CalibrationResult, String>,
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

    /// Fit `replicates` replicates of `resampler` in parallel, in order.
    pub fn run(&self, resampler: &dyn Resampler, replicates: usize) -> Vec<ReplicateOutcome> {
        (0..replicates)
            .into_par_iter()
            .map(|index| {
                let mut generator = Generator::from_seed(replicate_seed(self.seed, index as u64));
                let perturbation = resampler.replicate(&mut generator);
                let negative_values = match &perturbation {
                    Perturbation::Values(values) => values.iter().filter(|&&v| v < 0.0).count(),
                    Perturbation::Weights(_) => 0,
                };
                ReplicateOutcome {
                    index,
                    negative_values,
                    result: self.fit_one(index, &perturbation),
                }
            })
            .collect()
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
/// Returns the first reasons when every replicate fails.
pub fn bootstrap<E>(
    runner: &ReplicateRunner<'_, E>,
    resampler: &dyn Resampler,
    replicates: usize,
) -> Result<BootstrapRun, String>
where
    E: SimulationEngine + Send + Sync,
{
    let outcomes = runner.run(resampler, replicates);
    let problem = runner.problem();
    let mut values = Vec::new();
    let mut groups = Vec::new();
    let mut failed = Vec::new();
    let mut not_converged = 0;
    let mut negative_values = 0;
    for outcome in outcomes {
        negative_values += outcome.negative_values;
        match outcome.result {
            Ok(result) => {
                not_converged += usize::from(!result.converged);
                values.push(problem.applied_parameter_values(&result.best_parameters));
                groups.push(outcome.index);
            }
            Err(reason) => failed.push((outcome.index, reason)),
        }
    }
    if values.is_empty() {
        return Err(every_replicate_failed(&failed));
    }
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
        failed,
        not_converged,
        negative_values,
    })
}

pub(crate) fn every_replicate_failed(failed: &[(usize, String)]) -> String {
    let reasons: Vec<String> = failed
        .iter()
        .take(3)
        .map(|(index, reason)| format!("replicate {index}: {reason}"))
        .collect();
    format!(
        "Every replicate failed ({}); the first reasons: {}",
        failed.len(),
        reasons.join("; ")
    )
}
