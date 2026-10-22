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
use crate::local::{CurvatureKind, LocalError, block_meat, blocks_at, curvature_at};
use crate::priors::Priors;
use crate::resampling::{BayesianBootstrap, BlockScheme};
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
    let run = reached_the_posterior(sampler.run(&target, seed)?, "The Gibbs posterior")?;
    Ok(GibbsRun { run, eta })
}

/// `run`, when it reached the temperature 1; `posterior` names it in the
/// message.
pub(crate) fn reached_the_posterior(run: SmcRun, posterior: &str) -> Result<SmcRun, String> {
    match run.stopped_at {
        Some(Stop::NoNextLevel { level }) | Some(Stop::LowAcceptance { level, .. }) => {
            Err(format!(
                "{posterior} stopped at the temperature {level}: no higher one keeps the \
                 effective sample size; use more particles"
            ))
        }
        None => Ok(run),
    }
}

/// `η` by information matching (Holmes and Walker 2017; Lyddon, Holmes and
/// Walker 2019): `η = tr(J I⁻¹ J) / tr(J)` at the fit `θ̂`, on the scale of
/// the parameters. `J` is the Gauss–Newton [`crate::local::Curvature`] of the objective,
/// and `I` the meat of the sandwich ([`block_meat`]): the outer products of
/// the gradients of the terms summed over blocks of `block_length`
/// positions, or of the automatic length for `None`, so `I` holds under
/// dependence inside blocks. The terms are those of the sum of squares or of
/// `−log L`; with a correct observation model `I ≈ J` and `η ≈ 1`.
///
/// # Errors
///
/// [`LocalError::Refused`] when the loss is not the sum of squares, the
/// engine is stochastic, the fit is on a bound, or the blocks are not more
/// than the parameters; [`LocalError::Failed`] when the derivatives fail at
/// the fit or `I` is singular.
pub fn information_matching_eta<E>(
    problem: &CalibrationProblem<E>,
    fit: &[f64],
    block_length: Option<usize>,
) -> Result<f64, LocalError>
where
    E: SimulationEngine + Send + Sync,
{
    let name = match problem.discrepancy().loss() {
        Some(LossConfig::SumSquaredError) | None => None,
        Some(LossConfig::RootMeanSquaredError) => Some("the loss is \"rmse\""),
        Some(LossConfig::MeanAbsoluteError) => Some("the loss is \"mae\""),
    };
    if let Some(name) = name {
        return Err(LocalError::Refused(format!(
            "InformationMatchingEta(...) needs loss_function=\"sse\" or \
             observation models, sums of smooth terms; {name}. Use \
             BootstrapCoverageEta(...) or a number"
        )));
    }
    if problem.forward().replicates().is_some() {
        return Err(LocalError::Refused(
            "InformationMatchingEta(...) needs a deterministic engine: the runs \
             of a stochastic engine have no derivatives. Use \
             BootstrapCoverageEta(...) or a number"
                .to_string(),
        ));
    }
    let curvature = curvature_at(problem, fit, CurvatureKind::GaussNewton)?;
    let (blocks, _) = blocks_at(problem, &curvature, block_length)?;
    let meat = block_meat(&curvature, &blocks)?;
    // Back to the scale of the parameters: `X_θ = D⁻¹ X_u D⁻¹`.
    let derivatives = curvature.derivatives();
    let d = fit.len();
    let natural = |m: &DMatrix<f64>| {
        DMatrix::from_fn(d, d, |j, k| m[(j, k)] / (derivatives[j] * derivatives[k]))
    };
    let information = natural(curvature.matrix());
    let variance = natural(&meat);
    let inverse = variance.try_inverse().ok_or_else(|| {
        LocalError::Failed(
            "The variance of the scores of the terms at the fit is singular: information \
             matching needs every parameter read by the data and scores that are not all 0"
                .to_string(),
        )
    })?;
    // With `I` positive definite, some gradient is not 0, so `tr(J) > 0`
    // and `η > 0`.
    Ok((&information * inverse * &information).trace() / information.trace())
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
    /// (at least this snapshot's) holds the fit.
    fn covers(&self, eta: f64, fit: &[f64], levels: &[f64; 2]) -> Vec<bool> {
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
            .map(|j| {
                let column: Vec<f64> = self.values.iter().map(|v| v[j]).collect();
                let interval = quantiles(&column, &weights, levels)
                    .expect("the values of a sample are finite");
                interval[0] <= fit[j] && fit[j] <= interval[1]
            })
            .collect()
    }
}

/// The posteriors of one replicate at every `η` up to the largest: each
/// reweights the snapshot of the step below it.
fn covers_at(snapshots: &[Snapshot], eta: f64, fit: &[f64], levels: &[f64; 2]) -> Vec<bool> {
    snapshots
        .iter()
        .rev()
        .find(|snapshot| snapshot.eta <= eta)
        .unwrap_or(&snapshots[0])
        .covers(eta, fit, levels)
}

/// The `η` at which the interval of parameter `j` stops holding the fit,
/// by bisection on `log η` in `[low, high]`: 0 when it does not hold it at
/// `low`, `∞` when it still does at `high`.
fn threshold(
    snapshots: &[Snapshot],
    j: usize,
    fit: &[f64],
    levels: &[f64; 2],
    (mut low, mut high): (f64, f64),
) -> f64 {
    let covers = |eta: f64| covers_at(snapshots, eta, fit, levels)[j];
    if !covers(low) {
        return 0.0;
    }
    if covers(high) {
        return f64::INFINITY;
    }
    for _ in 0..BISECTION_STEPS {
        let middle = (low * high).sqrt();
        if middle <= low || middle >= high {
            break;
        }
        if covers(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

/// The `η` whose share of thresholds above it is `confidence_level`: the
/// order statistics of the thresholds `t₁ ≤ … ≤ t_M` sit at the positions
/// `i / (M + 1)`, which their distribution function has in expectation, and
/// `log η` interpolates between the two around `1 − confidence_level`.
fn interpolated_eta(thresholds: &mut [f64], confidence_level: f64) -> Result<f64, String> {
    thresholds.sort_by(f64::total_cmp);
    let m = thresholds.len();
    let position = (1.0 - confidence_level) * (m + 1) as f64;
    if position < 1.0 {
        let needed = (m..)
            .find(|&count| (1.0 - confidence_level) * (count + 1) as f64 >= 1.0)
            .unwrap_or(m);
        return Err(format!(
            "Bootstrap coverage at confidence_level = {confidence_level} needs at least \
             {needed} replicates times parameters, got {m}; use more replicates"
        ));
    }
    let below = position.floor() as usize;
    let lower = thresholds[below - 1];
    if lower <= 0.0 {
        return Err(format!(
            "Even the prior intervals cover the fit less often than confidence_level = \
             {confidence_level}; widen the priors"
        ));
    }
    let Some(&upper) = thresholds.get(below).filter(|upper| upper.is_finite()) else {
        return Ok(lower);
    };
    let fraction = position - below as f64;
    Ok((lower.ln() + fraction * (upper.ln() - lower.ln())).exp())
}

/// `η` by bootstrap coverage (Syring and Martin 2019): the `η` at which the
/// central intervals of `confidence_level` of the Gibbs posteriors of
/// Bayesian-bootstrap data sets hold the fit `θ̂` of the data at that rate,
/// over the replicates and the parameters.
///
/// Replicate `b` reweights the observations by blocks of `blocks`, from the
/// seed `replicate_seed(seed, b)`, and runs one tempered sampler to a large
/// `η`; the posterior at any smaller `η` reweights the particles of the step
/// below it, so no search below runs a model. The largest `η` starts at
/// `1 / loss(θ̂)`, where the kernel of the fit is `e⁻¹`, and grows four times,
/// each sampler going on from its particles, while its intervals still cover
/// too often. Each replicate and parameter has a threshold, the `η` at which its
/// interval stops holding the fit (bisection on `log η`), and `η` is the
/// interpolated quantile of the thresholds, whose expected coverage is the
/// level: the largest `η` that covers at the level would be an order
/// statistic, biased to cover less.
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
    let mut eta_max = 1.0 / loss;

    let problems = (0..replicates)
        .map(|b| {
            let mut generator = Generator::from_seed(replicate_seed(seed, b as u64));
            let weights = resampler.draw_weights(&mut generator);
            problem
                .with_data(&values, &weights)
                .map(|replicate| (b, replicate))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut failed = Vec::new();

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
    crate::resampling::FitUnit::Replicate.require(runs.len(), &failed, 1)?;

    let coverage = |runs: &[(usize, SmcState, Vec<Snapshot>)], eta: f64| -> f64 {
        let covered: usize = runs
            .iter()
            .map(|(_, _, snapshots)| {
                covers_at(snapshots, eta, fit, &levels)
                    .into_iter()
                    .filter(|&c| c)
                    .count()
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

    let bracket = (eta_max * f64::EPSILON, eta_max);
    let mut thresholds: Vec<f64> = runs
        .iter()
        .flat_map(|(_, _, snapshots)| {
            (0..fit.len()).map(|j| threshold(snapshots, j, fit, &levels, bracket))
        })
        .collect();
    let eta = interpolated_eta(&mut thresholds, confidence_level)?;
    Ok(CoverageCalibration {
        eta,
        coverage: coverage(&runs, eta),
        failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_interpolates_the_thresholds_at_their_expected_positions() {
        // Nine thresholds sit at 0.1, …, 0.9: the level 0.8 is the second.
        let mut thresholds: Vec<f64> = (1..=9).rev().map(f64::from).collect();
        let eta = interpolated_eta(&mut thresholds, 0.8).unwrap();
        assert!((eta - 2.0).abs() < 1e-12, "{eta}");
        // Between the first two, on the scale of log η.
        let mut thresholds = vec![1.0, 4.0, f64::INFINITY];
        let eta = interpolated_eta(&mut thresholds, 0.625).unwrap();
        assert!((eta - 2.0).abs() < 1e-12, "{eta}");
        let mut thresholds = vec![1.0, f64::INFINITY];
        assert_eq!(interpolated_eta(&mut thresholds, 0.5), Ok(1.0));
    }

    #[test]
    fn too_few_thresholds_or_none_above_zero_have_no_eta() {
        let mut few = vec![1.0, 2.0];
        let error = interpolated_eta(&mut few, 0.9).unwrap_err();
        assert!(
            error.contains("needs at least") && error.contains("got 2"),
            "{error}"
        );
        let mut zeros = vec![0.0; 9];
        assert!(
            interpolated_eta(&mut zeros, 0.8)
                .unwrap_err()
                .contains("widen the priors")
        );
    }

    fn run(stopped_at: Option<Stop>) -> SmcRun {
        SmcRun {
            values: vec![vec![0.5]],
            log_weights: vec![0.0],
            kept: vec![Some(1.0)],
            steps: Vec::new(),
            log_evidence: 0.0,
            evaluations: 1,
            stopped_at,
        }
    }

    #[test]
    fn a_stop_below_the_posterior_says_why() {
        let error = reached_the_posterior(
            run(Some(Stop::NoNextLevel { level: 0.3 })),
            "The Gibbs posterior",
        )
        .unwrap_err();
        assert!(error.contains("temperature 0.3"), "{error}");
        assert!(reached_the_posterior(run(None), "The Gibbs posterior").is_ok());
    }

    fn snapshot(values: &[f64], kept: &[Option<f64>]) -> Snapshot {
        Snapshot {
            eta: 0.0,
            values: values.iter().map(|&v| vec![v]).collect(),
            log_weights: vec![0.0; values.len()],
            kept: kept.to_vec(),
        }
    }

    #[test]
    fn a_particle_outside_the_support_has_no_weight_in_an_interval() {
        // The particle at 9 is outside the support: the interval of the
        // others, [0, 1], does not hold 5.
        let snapshots = [snapshot(&[0.0, 1.0, 9.0], &[Some(0.0), Some(0.0), None])];
        let levels = [0.0, 1.0];
        assert_eq!(covers_at(&snapshots, 1.0, &[5.0], &levels), vec![false]);
        assert_eq!(covers_at(&snapshots, 1.0, &[0.5], &levels), vec![true]);
    }

    #[test]
    fn a_threshold_is_zero_or_infinite_outside_its_bracket() {
        let levels = [0.0, 1.0];
        // The fit at 5 is outside every interval: no η covers it.
        let never = [snapshot(&[0.0, 1.0], &[Some(1.0), Some(1.0)])];
        assert_eq!(threshold(&never, 0, &[5.0], &levels, (1e-3, 1e3)), 0.0);
        // Equal losses: every η gives the same interval, which holds 0.5.
        assert_eq!(
            threshold(&never, 0, &[0.5], &levels, (1e-3, 1e3)),
            f64::INFINITY
        );
        // The particle at 1 keeps a loss of 1, so a large η leaves the one
        // at 0, and the interval stops holding 1 between the two ends.
        let narrowing = [snapshot(&[0.0, 1.0], &[Some(0.0), Some(1.0)])];
        let eta = threshold(&narrowing, 0, &[1.0], &levels, (1e-3, 1e3));
        assert!(eta > 1e-3 && eta < 1e3, "{eta}");
    }
}
