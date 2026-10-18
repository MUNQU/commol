//! The Gibbs posterior of the loss of a problem,
//! `π(θ) ∝ prior(θ) · exp(−η · loss(θ))`, and the two calibrations of `η`.
//!
//! The level is a temperature from 0 (the prior) to 1 (the posterior): the
//! log kernel of a kept loss `v` at level `ℓ` is `−ℓ · η · v`. A
//! stochastic engine scores the loss of the problem's replicates, with
//! common random numbers, so the loss is a function of the parameters.

use commol_calibration::{CalibrationProblem, LossConfig, Workspace};
use commol_core::SimulationEngine;
use commol_random::{Generator, replicate_seed};
use nalgebra::DMatrix;

use super::TemperedTarget;
use super::kernel::Kernel;
use super::sampler::{SmcRun, SmcSampler, SmcState, Stop};
use crate::priors::Priors;
use crate::resampling::{BayesianBootstrap, BlockScheme, Perturbation, Resampler};
use crate::sensitivity::{Design, FiniteDifferences, SensitivitySource};
use crate::statistics::{interval_levels, normalized_weights, quantiles};
use crate::target::discrepancy_in_support;

/// Growth of the largest `η` of a calibration by coverage, while its
/// intervals still cover too often.
const ETA_GROWTH: f64 = 4.0;
/// Growths before a calibration by coverage gives up.
const MAX_GROWTHS: usize = 30;
/// Bisection steps of the calibration by coverage, on `log η`.
const BISECTION_STEPS: usize = 100;

/// The Gibbs posterior of a problem with priors and a scale `η`.
pub struct GibbsTarget<'a, E: SimulationEngine> {
    problem: &'a CalibrationProblem<E>,
    priors: &'a Priors,
    eta: f64,
    bounds: Vec<(f64, f64)>,
}

impl<'a, E: SimulationEngine> GibbsTarget<'a, E> {
    /// The Gibbs posterior of `problem` with `priors` and `eta`.
    ///
    /// # Errors
    ///
    /// Returns a message when `eta` is not finite and above 0.
    pub fn new(
        problem: &'a CalibrationProblem<E>,
        priors: &'a Priors,
        eta: f64,
    ) -> Result<Self, String> {
        if !(eta.is_finite() && eta > 0.0) {
            return Err(format!("eta must be a finite number above 0, got {eta}"));
        }
        Ok(Self {
            problem,
            priors,
            eta,
            bounds: problem.parameter_bounds(),
        })
    }
}

impl<E: SimulationEngine + Sync> TemperedTarget for GibbsTarget<'_, E> {
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
        discrepancy_in_support(self.problem, values, workspace)
    }

    fn log_kernel(&self, value: f64, level: f64) -> f64 {
        -level * self.eta * value
    }

    fn first_level(&self, _kept: &[f64]) -> f64 {
        0.0
    }

    fn last_level(&self) -> f64 {
        1.0
    }
}

/// A Gibbs posterior and the `η` it ran with.
#[derive(Debug, Clone, PartialEq)]
pub struct GibbsRun {
    /// The run.
    pub run: SmcRun,
    /// The `η` of the posterior.
    pub eta: f64,
}

/// Sample the Gibbs posterior of `problem` with `priors` and `eta`.
///
/// # Errors
///
/// Returns a message when `eta` is not valid, the sampler fails, or no
/// level after one below 1 keeps the effective sample size.
pub fn gibbs_posterior<E, K>(
    problem: &CalibrationProblem<E>,
    priors: &Priors,
    eta: f64,
    sampler: &mut SmcSampler<K>,
    seed: u64,
) -> Result<GibbsRun, String>
where
    E: SimulationEngine + Sync,
    K: Kernel,
{
    let target = GibbsTarget::new(problem, priors, eta)?;
    let run = sampler.run(&target, seed)?;
    if let Some(Stop::NoNextLevel { level }) = run.stopped_at {
        return Err(no_next_level(level));
    }
    Ok(GibbsRun { run, eta })
}

fn no_next_level(level: f64) -> String {
    format!(
        "The Gibbs posterior stopped at the temperature {level}: no higher one keeps the \
         effective sample size; use more particles"
    )
}

/// `η` by information matching (Holmes and Walker 2017; Lyddon, Holmes and
/// Walker 2019): `η = tr(J I⁻¹ J) / tr(J)` at the fit `θ̂`, with
/// `J = Σ ∇²lᵢ` the curvature of the loss terms (Gauss–Newton,
/// `2 wᵢ nᵢ² ∇fᵢ ∇fᵢᵀ`) and `I = Σ ∇lᵢ ∇lᵢᵀ` their variance, where
/// `lᵢ = wᵢ (nᵢ rᵢ)²` is the term of observation `i` with weight `wᵢ`,
/// normalization `nᵢ` and residual `rᵢ`. The derivatives `∇fᵢ` come from
/// [`FiniteDifferences`] on the observations.
///
/// # Errors
///
/// Returns a message when the loss is not the sum of squares, the engine is
/// stochastic, the derivatives fail at the fit, or `I` is singular.
pub fn information_matching_eta<E>(
    problem: &CalibrationProblem<E>,
    fit: &[f64],
) -> Result<f64, String>
where
    E: SimulationEngine + Send + Sync,
{
    let discrepancy = problem.discrepancy();
    if discrepancy.loss() != LossConfig::SumSquaredError {
        return Err(format!(
            "EtaCalibration.information_matching() needs loss_function=\"sse\", a sum of smooth \
             terms; the loss is {:?}. Use EtaCalibration.bootstrap_coverage(...) or a number",
            discrepancy.loss()
        ));
    }
    if problem.forward().replicates().is_some() {
        return Err(
            "EtaCalibration.information_matching() needs a deterministic engine: the runs of \
             a stochastic engine have no derivatives. Use EtaCalibration.bootstrap_coverage(...) \
             or a number"
                .to_string(),
        );
    }
    let source =
        FiniteDifferences::new(problem, Design::Observations).map_err(|error| error.to_string())?;
    let sensitivities = source.jacobian(fit).map_err(|error| error.to_string())?;
    let jacobian = &sensitivities.jacobian;
    let d = fit.len();
    let mut curvature = DMatrix::<f64>::zeros(d, d);
    let mut variance = DMatrix::<f64>::zeros(d, d);
    for i in 0..jacobian.nrows() {
        let weight = discrepancy.weights()[i];
        let normalization = discrepancy.normalization()[i];
        let residual = discrepancy.observed()[i] - sensitivities.predictions[i];
        let gradient = jacobian.row(i).transpose();
        let outer = &gradient * gradient.transpose();
        let scale = weight * normalization * normalization;
        curvature += &outer * (2.0 * scale);
        variance += &outer * (4.0 * scale * scale * residual * residual);
    }
    let inverse = variance.clone().try_inverse().ok_or_else(|| {
        "The variance of the loss terms at the fit is singular: information matching needs \
         more observations than parameters and residuals that are not all 0"
            .to_string()
    })?;
    let eta = (&curvature * inverse * &curvature).trace() / curvature.trace();
    if eta.is_finite() && eta > 0.0 {
        Ok(eta)
    } else {
        Err(format!(
            "Information matching gives eta = {eta}; the loss has no curvature at the fit"
        ))
    }
}

/// The outcome of [`bootstrap_coverage_eta`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageCalibration {
    /// The calibrated `η`.
    pub eta: f64,
    /// The coverage of the fit by the intervals at that `η`.
    pub coverage: f64,
    /// The index and reason of each replicate whose posterior failed.
    pub failed: Vec<(usize, String)>,
}

/// The particles of one replicate at one `η`: the values, normalized log
/// weights and kept losses.
struct Snapshot {
    eta: f64,
    values: Vec<Vec<f64>>,
    log_weights: Vec<f64>,
    kept: Vec<Option<f64>>,
}

impl Snapshot {
    fn of(state: &SmcState, eta_max: f64) -> Self {
        let (values, log_weights, kept) = state.particles();
        Self {
            eta: state.level() * eta_max,
            values: values.to_vec(),
            log_weights: log_weights.to_vec(),
            kept: kept.to_vec(),
        }
    }

    /// Whether the central interval of `levels` of each parameter at `eta`
    /// (at least this snapshot's) holds the fit: the count of parameters it
    /// holds.
    fn covered(&self, eta: f64, fit: &[f64], levels: &[f64; 2]) -> usize {
        let log_weights: Vec<f64> = self
            .log_weights
            .iter()
            .zip(&self.kept)
            .map(|(w, kept)| match kept {
                Some(loss) => w - (eta - self.eta) * loss,
                None => f64::NEG_INFINITY,
            })
            .collect();
        let weights = normalized_weights(&log_weights);
        (0..fit.len())
            .filter(|&j| {
                let column: Vec<f64> = self.values.iter().map(|v| v[j]).collect();
                let interval = quantiles(&column, &weights, levels);
                interval[0] <= fit[j] && fit[j] <= interval[1]
            })
            .count()
    }
}

/// `η` by bootstrap coverage (Syring and Martin 2019): the `η` at which the
/// central intervals of `confidence_level` of the Gibbs posteriors of
/// Bayesian-bootstrap data sets hold the fit `θ̂` of the data at that rate,
/// over the replicates and the parameters.
///
/// Replicate `b` reweights the observations by blocks of `blocks`, from the
/// seed `replicate_seed(seed, b)`, and runs one tempered sampler to a large
/// `η`; the posterior at any smaller `η` reweights the particles of the step
/// below it, so the bisection on `log η` runs no model. The largest `η`
/// starts at `n / loss(θ̂)` and grows four times while its intervals still
/// cover too often.
///
/// # Errors
///
/// Returns a message when `confidence_level` is not strictly between 0 and
/// 1, the loss at the fit is 0, every replicate fails, or the coverage does
/// not cross the level.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_coverage_eta<E, K>(
    problem: &CalibrationProblem<E>,
    priors: &Priors,
    fit: &[f64],
    sampler: &mut SmcSampler<K>,
    blocks: &BlockScheme,
    replicates: usize,
    confidence_level: f64,
    seed: u64,
) -> Result<CoverageCalibration, String>
where
    E: SimulationEngine + Send + Sync,
    K: Kernel,
{
    let levels = interval_levels(confidence_level)?;
    let mut workspace = Workspace::default();
    let loss = discrepancy_in_support(problem, fit, &mut workspace)?
        .ok_or_else(|| "The fit is outside the support of the problem".to_string())?;
    if loss <= 0.0 {
        return Err(
            "The loss at the fit is 0, so bootstrap coverage has no scale for eta; give eta \
             as a number"
                .to_string(),
        );
    }
    let observed = problem.observed_data();
    let values: Vec<f64> = observed.iter().map(|obs| obs.value).collect();
    let resampler = BayesianBootstrap::new(observed, blocks);
    let mut eta_max = observed.len() as f64 / loss;

    let mut problems = Vec::new();
    let mut failed = Vec::new();
    for b in 0..replicates {
        let mut generator = Generator::from_seed(replicate_seed(seed, b as u64));
        let Perturbation::Weights(weights) = resampler.replicate(&mut generator) else {
            unreachable!("the Bayesian bootstrap reweights");
        };
        match problem.with_data(&values, &weights) {
            Ok(replicate) => problems.push((b, replicate)),
            Err(reason) => failed.push((b, reason)),
        }
    }

    // One run per replicate, with a snapshot of the particles at each step.
    let mut runs: Vec<(usize, SmcState, Vec<Snapshot>)> = Vec::new();
    for (b, replicate) in &problems {
        let target = GibbsTarget::new(replicate, priors, eta_max)?;
        let start = sampler.start(&target, replicate_seed(seed, (replicates + *b) as u64));
        let outcome = start.and_then(|mut state| {
            let mut snapshots = vec![Snapshot::of(&state, eta_max)];
            sampler.advance(&mut state, &target, |s| {
                snapshots.push(Snapshot::of(s, eta_max))
            })?;
            Ok((state, snapshots))
        });
        match outcome {
            Ok((state, snapshots)) => runs.push((*b, state, snapshots)),
            Err(reason) => failed.push((*b, reason)),
        }
    }
    if runs.is_empty() {
        return Err(crate::resampling::runner::every_replicate_failed(&failed));
    }

    let coverage = |runs: &[(usize, SmcState, Vec<Snapshot>)], eta: f64| -> f64 {
        let covered: usize = runs
            .iter()
            .map(|(_, _, snapshots)| {
                let below = snapshots
                    .iter()
                    .rev()
                    .find(|snapshot| snapshot.eta <= eta)
                    .unwrap_or(&snapshots[0]);
                below.covered(eta, fit, &levels)
            })
            .sum();
        covered as f64 / (runs.len() * fit.len()) as f64
    };

    let mut growths = 0;
    while coverage(&runs, eta_max) >= confidence_level {
        growths += 1;
        if growths > MAX_GROWTHS {
            return Err(format!(
                "The intervals still cover the fit at eta = {eta_max}; bootstrap coverage \
                 cannot reach confidence_level = {confidence_level}"
            ));
        }
        let grown = eta_max * ETA_GROWTH;
        for (b, state, snapshots) in &mut runs {
            let replicate = &problems
                .iter()
                .find(|(index, _)| index == b)
                .expect("kept")
                .1;
            let target = GibbsTarget::new(replicate, priors, grown)?;
            state.rescale_level(1.0 / ETA_GROWTH);
            sampler.advance(state, &target, |s| snapshots.push(Snapshot::of(s, grown)))?;
        }
        eta_max = grown;
    }

    // Bisection on log η between a scale far below and the largest.
    let mut high = eta_max;
    let mut low = eta_max * f64::EPSILON;
    if coverage(&runs, low) < confidence_level {
        return Err(format!(
            "Even the prior intervals cover the fit less often than confidence_level = \
             {confidence_level}; widen the priors"
        ));
    }
    for _ in 0..BISECTION_STEPS {
        let middle = (low * high).sqrt();
        if middle <= low || middle >= high {
            break;
        }
        if coverage(&runs, middle) >= confidence_level {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok(CoverageCalibration {
        eta: low,
        coverage: coverage(&runs, low),
        failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stop_below_the_posterior_says_why() {
        assert!(no_next_level(0.3).contains("temperature 0.3"));
    }
}
