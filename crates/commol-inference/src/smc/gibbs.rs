//! The Gibbs posterior of the loss of a problem,
//! `π(θ) ∝ prior(θ) · exp(−η · loss(θ))`, and the two calibrations of `η`.
//!
//! The level is a temperature from 0 (the prior) to 1 (the posterior): the
//! log kernel of a kept loss `v` at level `ℓ` is `−ℓ · η · v`. A
//! stochastic engine scores the loss of the problem's replicates, with
//! common random numbers, so the loss is a function of the parameters.

use commol_calibration::{CalibrationProblem, LossConfig, Workspace};
use commol_core::{CONFIDENCE_LEVEL, SimulationEngine};
use commol_random::Generator;
use commol_statistics::statistics::{interval_levels, normalized_weights, quantiles};
use nalgebra::DMatrix;

use super::TemperedTarget;
use super::kernel::Kernel;
use super::sampler::{SmcRun, SmcSampler, SmcState, Stop};
use crate::error::MethodError;
use crate::local::{CurvatureKind, block_meat, blocks_at, curvature_at};
use crate::priors::Priors;
use crate::resampling::{BayesianBootstrap, BlockScheme, Estimates};
use crate::target::discrepancy_in_support;
use commol_random::Stream;

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
    /// [`MethodError::Invalid`] when `eta` is outside its range
    /// ([`crate::ranges::ETA`]).
    pub fn new(
        problem: &'a CalibrationProblem<E>,
        priors: &'a Priors,
        eta: f64,
    ) -> Result<Self, MethodError> {
        crate::ranges::ETA
            .check(eta)
            .map_err(MethodError::Invalid)?;
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
/// [`MethodError::Invalid`] when `eta` is not valid; as
/// [`SmcSampler::run`]; [`MethodError::Failed`] when the sampler stops
/// before the temperature 1.
pub fn gibbs_posterior<E, K>(
    problem: &CalibrationProblem<E>,
    priors: &Priors,
    eta: f64,
    sampler: &mut SmcSampler<K>,
    seed: u64,
) -> Result<GibbsRun, MethodError>
where
    E: SimulationEngine + Sync,
    K: Kernel,
{
    let target = GibbsTarget::new(problem, priors, eta)?;
    let run = reached_the_posterior(sampler.run(&target, seed)?, "The Gibbs posterior")?;
    Ok(GibbsRun { run, eta })
}

/// `run`, when it reached the temperature 1; `posterior` names it in the
/// message of [`MethodError::Failed`] otherwise.
pub(crate) fn reached_the_posterior(run: SmcRun, posterior: &str) -> Result<SmcRun, MethodError> {
    match run.stopped_at {
        Some(Stop::NoNextLevel { level }) | Some(Stop::LowAcceptance { level, .. }) => {
            Err(MethodError::Failed(format!(
                "{posterior} stopped at the temperature {level}: no higher one keeps the \
                 effective sample size; use more particles"
            )))
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
/// `−log L`; with a correct observation model `I ≈ J` and `η ≈ 1`. With the
/// automatic length, the outcome keeps the estimate of each series
/// ([`BlockScheme::estimates`]).
///
/// # Errors
///
/// [`MethodError::Invalid`] when the loss is not the sum of squares, the
/// engine is stochastic, the fit is on a bound, or the blocks are not more
/// than the parameters; [`MethodError::Failed`] when the derivatives fail at
/// the fit or `I` is singular.
pub fn information_matching_eta<E>(
    problem: &CalibrationProblem<E>,
    fit: &[f64],
    block_length: Option<usize>,
) -> Result<InformationCalibration, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    let name = match problem.discrepancy().loss() {
        Some(LossConfig::SumSquaredError) | None => None,
        Some(LossConfig::RootMeanSquaredError) => Some("the loss is \"rmse\""),
        Some(LossConfig::MeanAbsoluteError) => Some("the loss is \"mae\""),
    };
    if let Some(name) = name {
        return Err(MethodError::Invalid(format!(
            "InformationMatchingEta(...) needs loss_function=\"sse\" or \
             observation models, sums of smooth terms; {name}. Use \
             BootstrapCoverageEta(...) or a number"
        )));
    }
    if problem.forward().replicates().is_some() {
        return Err(MethodError::Invalid(
            "InformationMatchingEta(...) needs a deterministic engine: the runs \
             of a stochastic engine have no derivatives. Use \
             BootstrapCoverageEta(...) or a number"
                .to_string(),
        ));
    }
    let curvature = curvature_at(problem, fit, CurvatureKind::GaussNewton)?;
    let blocks = blocks_at(problem, &curvature, block_length)?;
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
        MethodError::Failed(
            "The variance of the scores of the terms at the fit is singular: information \
             matching needs every parameter read by the data and scores that are not all 0"
                .to_string(),
        )
    })?;
    // With `I` positive definite, some gradient is not 0, so `tr(J) > 0`
    // and `η > 0`.
    Ok(InformationCalibration {
        eta: (&information * inverse * &information).trace() / information.trace(),
        estimates: blocks.estimates().map(<[_]>::to_vec),
    })
}

/// The outcome of [`information_matching_eta`].
#[derive(Debug, Clone, PartialEq)]
pub struct InformationCalibration {
    /// The calibrated `η`.
    pub eta: f64,
    /// With the automatic length, the estimate of each series by label.
    pub estimates: Option<Estimates>,
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

/// Check that `count` thresholds have an `η` at `confidence_level`: the
/// level `1 − confidence_level` is at least the first position of the one
/// quantile with equal weights, `1 / (2 count)`, so that no level falls below
/// the smallest threshold.
fn check_threshold_count(count: usize, confidence_level: f64) -> Result<(), MethodError> {
    let reaches = |count: usize| (1.0 - confidence_level) * (2 * count) as f64 >= 1.0;
    if reaches(count) {
        return Ok(());
    }
    let needed = (count..).find(|&more| reaches(more)).unwrap_or(count);
    Err(MethodError::Invalid(format!(
        "Bootstrap coverage at confidence_level = {confidence_level} needs at least \
         {needed} replicates times parameters, got {count}; use more replicates"
    )))
}

/// The `η` whose share of thresholds above it is `confidence_level`:
/// `exp` of the one quantile (equal weights) of the log thresholds at
/// `1 − confidence_level`.
///
/// An infinite threshold, an interval that holds the fit at every `η` of the
/// bracket, counts as the largest finite one: the bracket grows until fewer
/// than `confidence_level` of the thresholds are infinite, so above the
/// largest finite threshold the coverage is below the level, and `η` is at
/// most that threshold.
fn interpolated_eta(thresholds: &[f64], confidence_level: f64) -> Result<f64, MethodError> {
    check_threshold_count(thresholds.len(), confidence_level)?;
    let largest = thresholds
        .iter()
        .copied()
        .filter(|t| t.is_finite())
        .fold(0.0, f64::max);
    let logs: Vec<f64> = thresholds
        .iter()
        .map(|&t| libm::log(t.min(largest)))
        .collect();
    let weights = vec![1.0; logs.len()];
    let log_eta =
        quantiles(&logs, &weights, &[1.0 - confidence_level]).expect("a threshold is not NaN")[0];
    if !log_eta.is_finite() {
        return Err(MethodError::Invalid(format!(
            "Even the prior intervals cover the fit less often than confidence_level = \
             {confidence_level}; widen the priors"
        )));
    }
    Ok(libm::exp(log_eta))
}

/// `η` by bootstrap coverage (Syring and Martin 2019): the `η` at which the
/// central intervals of `confidence_level` of the Gibbs posteriors of
/// Bayesian-bootstrap data sets hold the fit `θ̂` of the data at that rate,
/// over the replicates and the parameters.
///
/// Replicate `b` reweights the observations by blocks of `blocks`, from the
/// seed `Stream::EtaWeights.replicate(seed, b)`, and runs one tempered
/// sampler, seeded with `Stream::EtaSamplers.replicate(seed, b)`, to a large
/// `η`; the posterior at any smaller `η` reweights the particles of the step
/// below it, so no search below runs a model. The largest `η` starts at
/// `1 / loss(θ̂)`, where the kernel of the fit is `e⁻¹`, and grows four times,
/// each sampler going on from its particles, while its intervals still cover
/// too often. Each replicate and parameter has a threshold, the `η` at which its
/// interval stops holding the fit (bisection on `log η`), and `η` is the one
/// quantile of the log thresholds at `1 − confidence_level`, which
/// interpolates between two order statistics: the largest `η` that covers at
/// the level would be an order statistic, biased to cover less.
///
/// # Errors
///
/// [`MethodError::Invalid`] when `confidence_level` is not strictly between
/// 0 and 1, the replicates times the parameters are too few for it, the fit
/// is outside the support or its loss is 0, or even the intervals of the
/// priors cover the fit too rarely; [`MethodError::Failed`] when the model
/// fails at the fit, every replicate fails, or the coverage does not cross
/// the level.
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
) -> Result<CoverageCalibration, MethodError>
where
    E: SimulationEngine + Send + Sync,
    K: Kernel,
{
    crate::ranges::REPLICATES
        .check(replicates)
        .and(CONFIDENCE_LEVEL.check(confidence_level))
        .map_err(MethodError::Invalid)?;
    let levels = interval_levels(confidence_level);
    check_threshold_count(replicates * fit.len(), confidence_level)?;
    let mut workspace = Workspace::default();
    let loss = discrepancy_in_support(problem, fit, &mut workspace)
        .map_err(MethodError::Failed)?
        .ok_or_else(|| {
            MethodError::Invalid("The fit is outside the support of the problem".to_string())
        })?;
    if loss <= 0.0 {
        return Err(MethodError::Invalid(
            "The loss at the fit is 0, so bootstrap coverage has no scale for eta; give eta \
             as a number"
                .to_string(),
        ));
    }
    let observed = problem.observed_data();
    let values: Vec<f64> = observed.iter().map(|obs| obs.value).collect();
    let resampler = BayesianBootstrap::new(observed, blocks);
    let mut eta_max = 1.0 / loss;

    let problems = (0..replicates)
        .map(|b| {
            let mut generator = Generator::from_seed(Stream::EtaWeights.replicate(seed, b as u64));
            let weights = resampler.draw_weights(&mut generator);
            problem
                .with_data(&values, &weights)
                .map(|replicate| (b, replicate))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(MethodError::Invalid)?;
    let mut failed = Vec::new();

    // One run per replicate, with a snapshot of the particles at each step.
    let mut runs: Vec<(usize, SmcState, Vec<Snapshot>)> = Vec::new();
    for (b, replicate) in &problems {
        let target = GibbsTarget::new(replicate, priors, eta_max)?;
        let start = sampler.start(&target, Stream::EtaSamplers.replicate(seed, *b as u64));
        let outcome = start.and_then(|mut state| {
            let mut snapshots = vec![Snapshot::of(&state, eta_max)];
            sampler.advance(&mut state, &target, |s| {
                snapshots.push(Snapshot::of(s, eta_max))
            })?;
            Ok((state, snapshots))
        });
        match outcome {
            Ok((state, snapshots)) => runs.push((*b, state, snapshots)),
            Err(reason) => failed.push((*b, reason.to_string())),
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
            return Err(MethodError::Failed(format!(
                "The intervals still cover the fit at eta = {eta_max}; bootstrap coverage \
                 cannot reach confidence_level = {confidence_level}"
            )));
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
    let thresholds: Vec<f64> = runs
        .iter()
        .flat_map(|(_, _, snapshots)| {
            (0..fit.len()).map(|j| threshold(snapshots, j, fit, &levels, bracket))
        })
        .collect();
    let eta = interpolated_eta(&thresholds, confidence_level)?;
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
    fn eta_is_the_one_quantile_of_the_log_thresholds() {
        // Nine thresholds 1, …, 9 at the level 0.2: Hazen's positions
        // (i − ½)/9 put it 0.3 of the way from ln 2 to ln 3.
        let thresholds: Vec<f64> = (1..=9).rev().map(f64::from).collect();
        let eta = interpolated_eta(&thresholds, 0.8).unwrap();
        let logs: Vec<f64> = thresholds.iter().map(|t| t.ln()).collect();
        let quantile = quantiles(&logs, &[1.0; 9], &[0.2]).unwrap()[0];
        assert!((eta - quantile.exp()).abs() < 1e-12, "{eta}");
        assert!((eta - 2.0 * 1.5f64.powf(0.3)).abs() < 1e-12, "{eta}");
    }

    #[test]
    fn an_infinite_threshold_leaves_eta_at_the_largest_finite_one() {
        // The level reads only finite thresholds: half way from ln 4 to ln 9.
        let eta = interpolated_eta(&[1.0, 4.0, 9.0, f64::INFINITY], 0.5).unwrap();
        assert!((eta - 6.0).abs() < 1e-12, "{eta}");
        // The level lies between 4 and the infinite thresholds: every η above
        // 4 covers less than 0.6, so η is 4.
        let eta = interpolated_eta(&[f64::INFINITY, 1.0, 4.0, f64::INFINITY], 0.6).unwrap();
        assert!((eta - 4.0).abs() < 1e-12, "{eta}");
    }

    #[test]
    fn too_few_thresholds_or_none_above_zero_have_no_eta() {
        // The level 0.25 is the first position of two thresholds, 1/4.
        assert!(interpolated_eta(&[1.0, 2.0], 0.75).is_ok());
        let error = interpolated_eta(&[1.0], 0.75).unwrap_err().to_string();
        assert!(
            error.contains("needs at least 2") && error.contains("got 1"),
            "{error}"
        );
        // The level 1/8 needs four thresholds, whose first position is 1/8.
        assert!(interpolated_eta(&[1.0, 2.0, 3.0, 4.0], 0.875).is_ok());
        let error = interpolated_eta(&[1.0, 2.0, 3.0], 0.875)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("needs at least 4") && error.contains("got 3"),
            "{error}"
        );
        assert!(
            interpolated_eta(&[0.0; 9], 0.8)
                .unwrap_err()
                .to_string()
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
        .unwrap_err()
        .to_string();
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
