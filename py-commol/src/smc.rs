//! Python bindings of the distributions, the priors and the sequential Monte
//! Carlo methods of `commol-inference`.
//!
//! Each runner takes the problem built once by a `Calibrator` and the JSON
//! of its method ([`commol_inference::settings::smc`]), runs every step of
//! the method, the fit and the calibration of `η` too, and returns the run
//! as plain values; Python builds the result.

use crate::parallelism::parallel;
use commol_inference::resampling::{BlockDrawing, BlockScheme, FitUnit};
use commol_inference::settings::PriorsSettings;
use commol_inference::settings::smc::{
    AbcRejectionSettings, AbcSmcSettings, BayesianPosteriorSettings, Eta, EtaCalibration,
    GibbsPosteriorSettings,
};
use commol_inference::smc::{self, AbcTarget, RandomWalk, SmcRun, SmcSampler, Stop};
use commol_inference::{Distribution, Family, Priors};
use commol_random::Generator;
use commol_random::Stream;
use pyo3::prelude::*;

use crate::calibration::PyStarts;
use crate::inference::{PyCalibrationProblem, best_values, method_error, value_error};

/// The priors of `settings` for the parameters of `problem`.
pub(crate) fn problem_priors(
    problem: &PyCalibrationProblem,
    settings: &PriorsSettings,
) -> PyResult<Priors> {
    settings
        .priors(
            &problem.inner.parameter_names(),
            &problem.inner.parameter_bounds(),
        )
        .map_err(value_error)
}

/// A distribution truncated to the bounds of a parameter
///
/// Args:
///     distribution: The JSON of a distribution of `commol`, by its `kind`
///     lower: The lower bound
///     upper: The upper bound
///
/// Raises:
///     ValueError: If the JSON is not a distribution, a parameter is not
///         valid or the bounds do not fit the kind
#[pyclass(name = "Distribution", frozen)]
pub struct PyDistribution {
    inner: Distribution,
}

#[pymethods]
impl PyDistribution {
    #[new]
    fn new(distribution: &str, lower: f64, upper: f64) -> PyResult<Self> {
        let family: Family = crate::json::read(distribution)?;
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
        commol_statistics::statistics::check_levels(&[p]).map_err(value_error)?;
        Ok(self.inner.quantile(p))
    }

    /// `count` draws from the stream of draws of a distribution of `seed`
    fn sample(&self, count: i64, seed: i128) -> PyResult<Vec<f64>> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
        let count: usize = crate::counts::count(&commol_inference::ranges::COUNT, count)?;
        let mut generator = Generator::from_seed(Stream::DistributionDraws.seed(seed));
        Ok((0..count)
            .map(|_| self.inner.sample(&mut generator))
            .collect())
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

/// The `η` of a Gibbs posterior and, when it was calibrated, how
#[pyclass(name = "EtaCalibration", frozen, get_all)]
pub struct PyEtaCalibration {
    /// The `η` the posterior used
    eta: f64,
    /// The coverage of the fit by the intervals at `η`, for bootstrap
    /// coverage
    coverage: Option<f64>,
    /// Each replicate that failed: "Replicate <index> failed: <reason>"
    failed: Vec<String>,
    /// The positions per block of a calibration
    block_length: Option<usize>,
    /// With the automatic length, the estimate of each series by label
    block_length_estimates: Option<Vec<(String, f64)>>,
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

/// The sampler of a Gibbs posterior.
fn gibbs_sampler(settings: &GibbsPosteriorSettings) -> PyResult<SmcSampler<RandomWalk>> {
    sampler(
        settings.particles,
        settings.resample_threshold,
        settings.target_cess,
        settings.stay_probability,
    )
}

/// The `η` of `calibration`, from a fit of the problem.
fn calibrated_eta(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &GibbsPosteriorSettings,
    priors: &Priors,
    calibration: &EtaCalibration,
) -> PyResult<(PyStarts, PyEtaCalibration)> {
    match calibration {
        EtaCalibration::InformationMatching(calibration) => {
            let starts = problem.fit(py, calibration.starts)?;
            let found = smc::information_matching_eta(
                &problem.inner,
                &best_values(&starts),
                calibration.block_length.0,
            )
            .map_err(method_error)?;
            Ok((
                PyStarts { inner: starts },
                PyEtaCalibration {
                    eta: found.eta,
                    coverage: None,
                    failed: Vec::new(),
                    block_length: Some(found.block_length),
                    block_length_estimates: found.estimates,
                },
            ))
        }
        EtaCalibration::BootstrapCoverage(calibration) => {
            let starts = problem.fit(py, calibration.starts)?;
            let fit = best_values(&starts);
            let blocks = BlockScheme::at_fit(
                &problem.inner,
                &fit,
                calibration.block_length.0,
                BlockDrawing::NonOverlapping,
            )
            .map_err(method_error)?;
            let mut sampler = gibbs_sampler(settings)?;
            let found = parallel(py, || {
                smc::bootstrap_coverage_eta(
                    &problem.inner,
                    priors,
                    &fit,
                    &mut sampler,
                    &blocks,
                    calibration.replicates,
                    calibration.confidence_level,
                    problem.seed,
                )
            })?
            .map_err(method_error)?;
            Ok((
                PyStarts { inner: starts },
                PyEtaCalibration {
                    eta: found.eta,
                    coverage: Some(found.coverage),
                    failed: FitUnit::Replicate.entries(&found.failed),
                    block_length: Some(blocks.length()),
                    block_length_estimates: blocks.estimates().map(<[_]>::to_vec),
                },
            ))
        }
    }
}

/// Sample a Gibbs posterior, after a fit and the calibration of its `η`
/// when the settings calibrate it
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `GibbsPosterior`
///
/// Returns:
///     The starts of the fit, or None for a given `η`; the run; and the `η`
///     with its calibration
///
/// Raises:
///     ValueError: If a prior or a setting is not valid, or the problem
///         does not fit the calibration of `η`
///     RuntimeError: If the fit or a run fails, the calibration of `η`
///         fails, or the sampler stops before its target
#[pyfunction]
fn gibbs_posterior(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(Option<PyStarts>, PySmcRun, PyEtaCalibration)> {
    let settings: GibbsPosteriorSettings = crate::json::read(settings)?;
    let priors = problem_priors(problem, &settings.priors)?;
    let (starts, eta) = match &settings.eta {
        Eta::Value(eta) => (
            None,
            PyEtaCalibration {
                eta: *eta,
                coverage: None,
                failed: Vec::new(),
                block_length: None,
                block_length_estimates: None,
            },
        ),
        Eta::Calibrated(calibration) => {
            let (starts, eta) = calibrated_eta(py, problem, &settings, &priors, calibration)?;
            (Some(starts), eta)
        }
    };
    let mut sampler = gibbs_sampler(&settings)?;
    let run = parallel(py, || {
        smc::gibbs_posterior(&problem.inner, &priors, eta.eta, &mut sampler, problem.seed)
    })?
    .map_err(method_error)?;
    Ok((starts, PySmcRun::of(run.run), eta))
}

/// Sample the posterior of the observation models of the problem
///
/// Args:
///     problem: The calibration problem, with observation models
///     settings: The JSON of a `BayesianPosterior`
///
/// Raises:
///     ValueError: If the problem has a loss, or a prior or a setting is not
///         valid
///     RuntimeError: If a run fails or the sampler stops before its target
#[pyfunction]
fn bayesian_posterior(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<PySmcRun> {
    let settings: BayesianPosteriorSettings = crate::json::read(settings)?;
    let priors = problem_priors(problem, &settings.priors)?;
    let mut sampler = sampler(
        settings.particles,
        settings.resample_threshold,
        settings.target_cess,
        settings.stay_probability,
    )?;
    let run = parallel(py, || {
        smc::bayesian_posterior(&problem.inner, &priors, &mut sampler, problem.seed)
    })?
    .map_err(method_error)?;
    Ok(PySmcRun::of(run))
}

/// Sample ABC by sequential Monte Carlo
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of an `ABCSMC`
///
/// Returns:
///     The run and whether each distance read one fresh replicate of a
///     stochastic engine
///
/// Raises:
///     ValueError: If a prior or a setting is not valid
///     RuntimeError: If a run fails or the sampler stops before its target
#[pyfunction]
fn abc_smc(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(PySmcRun, bool)> {
    let settings: AbcSmcSettings = crate::json::read(settings)?;
    let priors = problem_priors(problem, &settings.priors)?;
    // With weights of 0 or 1, a particle of weight 0 cannot move: every
    // step resamples.
    let mut sampler = sampler(
        settings.particles,
        1.0,
        settings.target_cess,
        settings.stay_probability,
    )?
    .with_min_acceptance(settings.min_acceptance)
    .map_err(method_error)?;
    let target = AbcTarget::new(&problem.inner, &priors, settings.distance.distance())
        .map_err(method_error)?;
    let run = parallel(py, || sampler.run(&target, problem.seed))?.map_err(method_error)?;
    Ok((PySmcRun::of(run), target.stochastic()))
}

/// ABC rejection: keep the closest share of draws of the prior
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of an `ABCRejection`
///
/// Returns:
///     The run and whether each distance read one fresh replicate of a
///     stochastic engine
///
/// Raises:
///     ValueError: If a prior or a setting is not valid
///     RuntimeError: If a run fails
#[pyfunction]
fn abc_rejection(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(PySmcRun, bool)> {
    let settings: AbcRejectionSettings = crate::json::read(settings)?;
    let priors = problem_priors(problem, &settings.priors)?;
    let target = AbcTarget::new(&problem.inner, &priors, settings.distance.distance())
        .map_err(method_error)?;
    let run = parallel(py, || {
        smc::abc_rejection(&target, settings.draws, settings.quantile, problem.seed)
    })?
    .map_err(method_error)?;
    Ok((PySmcRun::of(run), target.stochastic()))
}

/// Register the classes and functions of this module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistribution>()?;
    m.add_class::<PySmcRun>()?;
    m.add_class::<PyEtaCalibration>()?;
    m.add_function(wrap_pyfunction!(gibbs_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(abc_smc, m)?)?;
    m.add_function(wrap_pyfunction!(abc_rejection, m)?)?;
    Ok(())
}
