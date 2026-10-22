//! Python bindings of the distributions, the priors and the sequential Monte
//! Carlo methods of `commol-inference`.
//!
//! A prior crosses as `(parameter, kind, parameters)`; `None` in place of
//! the list is `Priors.uniform_on_bounds()`. Each runner takes the problem
//! built once by a `Calibrator` and returns the run as plain values; Python
//! builds the result.

use crate::parallelism::parallel;
use commol_inference::resampling::{
    BlockDrawing, BlockScheme, FitUnit, ResidualScale, Residuals, automatic_block_length,
};
use commol_inference::smc::{self, AbcTarget, Distance, RandomWalk, SmcRun, SmcSampler, Stop};
use commol_inference::{Distribution, Family, Priors};
use commol_random::{Generator, replicate_seed};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::inference::{PyCalibrationProblem, method_error};

/// A prior as Python passes it: the parameter, the kind and its parameters.
pub(crate) type PriorArgs = (String, String, Vec<f64>);

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// The priors of `names` with `bounds`; `None` is uniform on the bounds.
pub(crate) fn priors(
    names: &[String],
    bounds: &[(f64, f64)],
    given: Option<Vec<PriorArgs>>,
) -> PyResult<Priors> {
    match given {
        None => Priors::uniform_on_bounds(names, bounds),
        Some(given) => {
            let families = given
                .into_iter()
                .map(|(name, kind, parameters)| {
                    Family::from_name(&kind, &parameters).map(|family| (name, family))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(value_error)?;
            Priors::new(names, bounds, &families)
        }
    }
    .map_err(value_error)
}

/// A distribution truncated to the bounds of a parameter
///
/// Args:
///     kind: "uniform", "log_uniform", "beta", "normal", "log_normal" or
///         "gamma"
///     parameters: The parameters of the kind, in order
///     lower: The lower bound
///     upper: The upper bound
///
/// Raises:
///     ValueError: If a parameter is not valid or the bounds do not fit the
///         kind
#[pyclass(name = "Distribution", frozen)]
pub struct PyDistribution {
    inner: Distribution,
}

#[pymethods]
impl PyDistribution {
    #[new]
    fn new(kind: &str, parameters: Vec<f64>, lower: f64, upper: f64) -> PyResult<Self> {
        let family = Family::from_name(kind, &parameters).map_err(value_error)?;
        Distribution::new(family, lower, upper)
            .map(|inner| Self { inner })
            .map_err(value_error)
    }

    /// The log density at `x`: -inf outside the bounds
    fn log_density(&self, x: f64) -> f64 {
        self.inner.log_density(x)
    }

    /// The distribution function at `x`
    fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }

    /// The quantile at `p`, in [0, 1]
    ///
    /// Raises:
    ///     ValueError: If `p` is not in [0, 1]
    fn quantile(&self, p: f64) -> PyResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(value_error(format!(
                "A quantile level is between 0 and 1, got {p}"
            )));
        }
        Ok(self.inner.quantile(p))
    }

    /// `count` draws from the generator seeded with `seed`
    fn sample(&self, count: usize, seed: u64) -> Vec<f64> {
        let mut generator = Generator::from_seed(replicate_seed(seed, 0));
        (0..count)
            .map(|_| self.inner.sample(&mut generator))
            .collect()
    }
}

/// A run of the sequential Monte Carlo sampler, as the runners return it
#[pyclass(name = "SMCRun", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PySmcRun {
    /// One row per particle, in problem order
    values: Vec<Vec<f64>>,
    /// The normalized log weight of each particle
    log_weights: Vec<f64>,
    /// The loss or distance each particle keeps, or None outside the support
    kept: Vec<Option<f64>>,
    /// The level of each step
    levels: Vec<f64>,
    /// Kish's effective size after the reweighting of each step
    effective_sizes: Vec<f64>,
    /// Whether each step resampled
    resampled: Vec<bool>,
    /// The share of the first move's proposals accepted, per step
    acceptance_rates: Vec<f64>,
    /// How many times each particle moved, per step
    move_counts: Vec<usize>,
    /// The log of the product of the mean incremental weights
    log_evidence: f64,
    /// The number of model runs
    evaluations: usize,
    /// Why the run stopped before its last level, or None
    stop_reason: Option<String>,
}

impl PySmcRun {
    fn of(run: SmcRun) -> Self {
        let stop_reason = run.stopped_at.as_ref().map(|stop| match stop {
            Stop::LowAcceptance {
                level,
                acceptance_rate,
            } => format!(
                "The first move at the tolerance {level} accepted {acceptance_rate:.3} of its \
                 proposals, below min_acceptance; the sample is that of the tolerance before"
            ),
            Stop::NoNextLevel { level } => format!(
                "No tolerance below {level} keeps target_cess of the particles: too few of \
                 their distances lie below it"
            ),
        });
        Self {
            levels: run.steps.iter().map(|s| s.level).collect(),
            effective_sizes: run.steps.iter().map(|s| s.effective_size).collect(),
            resampled: run.steps.iter().map(|s| s.resampled).collect(),
            acceptance_rates: run.steps.iter().map(|s| s.acceptance_rate).collect(),
            move_counts: run.steps.iter().map(|s| s.moves).collect(),
            values: run.values,
            log_weights: run.log_weights,
            kept: run.kept,
            log_evidence: run.log_evidence,
            evaluations: run.evaluations,
            stop_reason,
        }
    }
}

/// The `η` of bootstrap coverage
#[pyclass(name = "EtaCoverage", frozen, get_all)]
pub struct PyEtaCoverage {
    /// The calibrated `η`
    eta: f64,
    /// The coverage of the fit by the intervals at `η`
    coverage: f64,
    /// Each replicate that failed: "Replicate <index> failed: <reason>"
    failed: Vec<String>,
}

/// The sampler of the settings of a method
fn sampler(
    particles: usize,
    resample_threshold: f64,
    target_cess: f64,
    stay_probability: f64,
) -> PyResult<SmcSampler<RandomWalk>> {
    SmcSampler::new(
        particles,
        RandomWalk::new(),
        resample_threshold,
        target_cess,
        stay_probability,
    )
    .map_err(method_error)
}

pub(crate) fn problem_priors(
    problem: &PyCalibrationProblem,
    given: Option<Vec<PriorArgs>>,
) -> PyResult<Priors> {
    priors(
        &problem.inner.parameter_names(),
        &problem.inner.parameter_bounds(),
        given,
    )
}

/// Sample the Gibbs posterior with a given `η`
///
/// Args:
///     problem: The calibration problem
///     priors: The priors, or None for uniform on the bounds
///     eta: The scale of the loss, above 0
///     particles: Number of particles
///     resample_threshold: Resample below this share of effective size
///     target_cess: The conditional effective sample size of each step
///     stay_probability: The probability that a particle never moves in a
///         step
///     seed: Seed of every random number
///
/// Raises:
///     ValueError: If a prior or a setting is not valid
///     RuntimeError: If a run fails or the sampler stops before its target
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gibbs_posterior(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    priors: Option<Vec<PriorArgs>>,
    eta: f64,
    particles: usize,
    resample_threshold: f64,
    target_cess: f64,
    stay_probability: f64,
    seed: u64,
) -> PyResult<PySmcRun> {
    let priors = problem_priors(problem, priors)?;
    let mut sampler = sampler(particles, resample_threshold, target_cess, stay_probability)?;
    let run = parallel(py, || {
        smc::gibbs_posterior(&problem.inner, &priors, eta, &mut sampler, seed)
    })?
    .map_err(method_error)?;
    Ok(PySmcRun::of(run.run))
}

/// Sample the posterior of the observation models of the problem
///
/// Args:
///     problem: The calibration problem, with observation models
///     priors: The priors, or None for uniform on the bounds
///     particles: Number of particles
///     resample_threshold: Resample below this share of effective size
///     target_cess: The conditional effective sample size of each step
///     stay_probability: The probability that a particle never moves in a
///         step
///     seed: Seed of every random number
///
/// Raises:
///     ValueError: If the problem has a loss, or a prior or a setting is not
///         valid
///     RuntimeError: If a run fails or the sampler stops before its target
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bayesian_posterior(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    priors: Option<Vec<PriorArgs>>,
    particles: usize,
    resample_threshold: f64,
    target_cess: f64,
    stay_probability: f64,
    seed: u64,
) -> PyResult<PySmcRun> {
    let priors = problem_priors(problem, priors)?;
    let mut sampler = sampler(particles, resample_threshold, target_cess, stay_probability)?;
    let run = parallel(py, || {
        smc::bayesian_posterior(&problem.inner, &priors, &mut sampler, seed)
    })?
    .map_err(method_error)?;
    Ok(PySmcRun::of(run))
}

/// The `η` of information matching at the fit
///
/// Args:
///     problem: The calibration problem
///     fit: The fitted values, in problem order
///     block_length: Positions per block of the scores, or None for the
///         automatic length
///
/// Raises:
///     ValueError: If the loss is not the sum of squares, the engine is
///         stochastic, the fit is on a bound, or the blocks are not more
///         than the parameters
///     RuntimeError: If the derivatives fail at the fit or the variance of
///         the scores is singular
#[pyfunction]
fn information_matching_eta(
    problem: &PyCalibrationProblem,
    fit: Vec<f64>,
    block_length: Option<usize>,
) -> PyResult<f64> {
    smc::information_matching_eta(&problem.inner, &fit, block_length).map_err(method_error)
}

/// The `η` of bootstrap coverage
///
/// Args:
///     problem: The calibration problem
///     priors: The priors, or None for uniform on the bounds
///     fit: The fitted values, in problem order
///     replicates: Number of Bayesian-bootstrap data sets
///     block_length: Positions per block, or None for the automatic length
///     confidence_level: The level of the intervals
///     particles: Number of particles of each replicate
///     resample_threshold: Resample below this share of effective size
///     target_cess: The conditional effective sample size of each step
///     stay_probability: The probability that a particle never moves in a
///         step
///     seed: Seed of every random number
///
/// Raises:
///     ValueError: If a prior or a setting is not valid, the replicates
///         are too few for the level, the fit is outside the support, or the
///         loss at the fit is 0
///     RuntimeError: If the run at the fit fails, every replicate fails, or
///         the coverage does not cross the level
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bootstrap_coverage_eta(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    priors: Option<Vec<PriorArgs>>,
    fit: Vec<f64>,
    replicates: usize,
    block_length: Option<usize>,
    confidence_level: f64,
    particles: usize,
    resample_threshold: f64,
    target_cess: f64,
    stay_probability: f64,
    seed: u64,
) -> PyResult<PyEtaCoverage> {
    let priors = problem_priors(problem, priors)?;
    let mut sampler = sampler(particles, resample_threshold, target_cess, stay_probability)?;
    let observed = problem.inner.observed_data();
    let length = match block_length {
        Some(length) => length,
        None => {
            let residuals = Residuals::new(&problem.inner, &fit, ResidualScale::Additive)
                .map_err(method_error)?;
            automatic_block_length(
                observed,
                residuals.residuals(),
                BlockDrawing::NonOverlapping,
            )
            .length
        }
    };
    let blocks = BlockScheme::new(observed, length).map_err(method_error)?;
    let calibration = parallel(py, || {
        smc::bootstrap_coverage_eta(
            &problem.inner,
            &priors,
            &fit,
            &mut sampler,
            &blocks,
            replicates,
            confidence_level,
            seed,
        )
    })?
    .map_err(method_error)?;
    Ok(PyEtaCoverage {
        eta: calibration.eta,
        coverage: calibration.coverage,
        failed: FitUnit::Replicate.entries(&calibration.failed),
    })
}

/// Sample ABC by sequential Monte Carlo
///
/// Args:
///     problem: The calibration problem
///     priors: The priors, or None for uniform on the bounds
///     metric: The metric of each series, "rmse" or "mae"; None for the
///         problem's loss
///     particles: Number of particles
///     target_cess: The weighted share of particles each tolerance keeps
///     min_acceptance: Stop when the first move of a step accepts less
///     stay_probability: The probability that a particle never moves in a
///         step
///     seed: Seed of every random number
///
/// Returns:
///     The run and whether each distance read one fresh replicate of a
///     stochastic engine
///
/// Raises:
///     ValueError: If a prior or a setting is not valid
///     RuntimeError: If a run fails or the sampler stops before its target
#[pyfunction]
#[pyo3(signature = (problem, priors, metric, particles, target_cess, min_acceptance, stay_probability, seed))]
#[allow(clippy::too_many_arguments)]
fn abc_smc(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    priors: Option<Vec<PriorArgs>>,
    metric: Option<&str>,
    particles: usize,
    target_cess: f64,
    min_acceptance: f64,
    stay_probability: f64,
    seed: u64,
) -> PyResult<(PySmcRun, bool)> {
    let priors = problem_priors(problem, priors)?;
    let distance = Distance::from_metric(metric).map_err(method_error)?;
    // With weights of 0 or 1, a particle of weight 0 cannot move: every
    // step resamples.
    let mut sampler = sampler(particles, 1.0, target_cess, stay_probability)?
        .with_min_acceptance(min_acceptance)
        .map_err(method_error)?;
    let target = AbcTarget::new(&problem.inner, &priors, distance).map_err(method_error)?;
    let run = parallel(py, || sampler.run(&target, seed))?.map_err(method_error)?;
    Ok((PySmcRun::of(run), target.stochastic()))
}

/// ABC rejection: keep the closest share of draws of the prior
///
/// Args:
///     problem: The calibration problem
///     priors: The priors, or None for uniform on the bounds
///     metric: The metric of each series, "rmse" or "mae"; None for the
///         problem's loss
///     draws: Number of draws
///     quantile: The share of draws kept
///     seed: Seed of every random number
///
/// Returns:
///     The run and whether each distance read one fresh replicate of a
///     stochastic engine
///
/// Raises:
///     ValueError: If a prior or a setting is not valid
///     RuntimeError: If a run fails or the sampler stops before its target
#[pyfunction]
#[pyo3(signature = (problem, priors, metric, draws, quantile, seed))]
fn abc_rejection(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    priors: Option<Vec<PriorArgs>>,
    metric: Option<&str>,
    draws: usize,
    quantile: f64,
    seed: u64,
) -> PyResult<(PySmcRun, bool)> {
    let priors = problem_priors(problem, priors)?;
    let distance = Distance::from_metric(metric).map_err(method_error)?;
    let target = AbcTarget::new(&problem.inner, &priors, distance).map_err(method_error)?;
    let run = parallel(py, || smc::abc_rejection(&target, draws, quantile, seed))?
        .map_err(method_error)?;
    Ok((PySmcRun::of(run), target.stochastic()))
}

/// Register the classes and functions of this module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistribution>()?;
    m.add_class::<PySmcRun>()?;
    m.add_class::<PyEtaCoverage>()?;
    m.add_function(wrap_pyfunction!(gibbs_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(information_matching_eta, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_coverage_eta, m)?)?;
    m.add_function(wrap_pyfunction!(abc_smc, m)?)?;
    m.add_function(wrap_pyfunction!(abc_rejection, m)?)?;
    Ok(())
}
