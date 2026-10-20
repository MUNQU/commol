//! Python bindings of the profile likelihood of `commol-inference`.
//!
//! [`trace_profiles`] takes the problem built once by a `Calibrator`, the fit and
//! the threshold, traces every profile in Rust and returns them as plain
//! values; Python builds the result.

use commol_inference::profiles::{
    Profile, ProfileEnd, ProfileTracer, ProfiledQuantity, beale, bootstrap_thresholds, chi_square,
};
use commol_inference::resampling::ReplicateRunner;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::calibration::{PyObservedDataPoint, PyOptimizationConfig};
use crate::inference::{PyCalibrationProblem, ResamplerSpec, value_error};

/// The bootstrap that calibrates a profile threshold
///
/// Args:
///     method: "bayesian", "parametric", "residual" or "wild"
///     replicates: Number of replicates, at least 1
///     starts: Starts per replicate, the fit first
///     block_length: Positions per block, or None for the automatic length
///     residual_scale: "additive" or "multiplicative", for "residual"
///     scheme: "moving" or "stationary", for "residual"
///     multipliers: "rademacher" or "mammen", for "wild"
///
/// Raises:
///     ValueError: If the method is unknown or a setting it needs is
///         missing or unknown
#[pyclass(name = "BootstrapSettings", frozen)]
pub struct PyBootstrapSettings {
    spec: ResamplerSpec,
    replicates: usize,
    starts: usize,
}

#[pymethods]
impl PyBootstrapSettings {
    #[new]
    #[pyo3(signature = (method, replicates, starts, block_length=None, residual_scale=None, scheme=None, multipliers=None))]
    fn new(
        method: &str,
        replicates: usize,
        starts: usize,
        block_length: Option<usize>,
        residual_scale: Option<&str>,
        scheme: Option<&str>,
        multipliers: Option<&str>,
    ) -> PyResult<Self> {
        fn needed<'s>(method: &str, setting: Option<&'s str>, name: &str) -> PyResult<&'s str> {
            setting.ok_or_else(|| value_error(format!("A {method} bootstrap needs {name}")))
        }
        let spec = match method {
            "bayesian" => ResamplerSpec::Bayesian { block_length },
            "parametric" => ResamplerSpec::Parametric,
            "residual" => ResamplerSpec::residual(
                block_length,
                needed(method, residual_scale, "residual_scale")?,
                needed(method, scheme, "scheme")?,
            )?,
            "wild" => {
                ResamplerSpec::wild(block_length, needed(method, multipliers, "multipliers")?)?
            }
            _ => {
                return Err(value_error(format!(
                    "Unknown bootstrap '{method}'; the bootstraps are 'bayesian', \
                     'parametric', 'residual' and 'wild'"
                )));
            }
        };
        Ok(Self {
            spec,
            replicates,
            starts,
        })
    }
}

/// One profile, as the runner returns it
#[pyclass(name = "Profile", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyProfile {
    /// The value of the quantity at the fit
    center: f64,
    /// The threshold of the statistic
    threshold: f64,
    /// The value of each point, increasing
    values: Vec<f64>,
    /// The statistic of each point
    statistics: Vec<f64>,
    /// Every parameter of each point, in problem order
    parameters: Vec<Vec<f64>>,
    /// The multiplier of each point of a prediction profile
    multipliers: Vec<Option<f64>>,
    /// Whether the refit of each point converged
    converged: Vec<bool>,
    /// Why the side below and the side above end
    ends: (String, String),
    /// Where the statistic crosses the threshold below and above
    interval: (Option<f64>, Option<f64>),
    /// Evaluations of the objective
    evaluations: usize,
    /// The rise of each kept replicate, with a bootstrap threshold
    rises: Option<Vec<f64>>,
    /// "identifiable", "practically_non_identifiable" or
    /// "structurally_non_identifiable"; None when the threshold is below the
    /// tolerance
    verdict: Option<String>,
    /// The lowest statistic
    lowest: f64,
}

fn end_name(end: ProfileEnd) -> String {
    match end {
        ProfileEnd::Crossed => "crossed",
        ProfileEnd::Bound => "bound",
        ProfileEnd::MaxSteps => "max_steps",
        ProfileEnd::Unreachable => "unreachable",
    }
    .to_string()
}

impl PyProfile {
    fn new(profile: Profile, rises: Option<Vec<f64>>, tolerance: f64) -> Self {
        let [lower, upper] = profile.interval();
        Self {
            verdict: profile
                .verdict(tolerance)
                .map(|verdict| verdict.name().to_string()),
            lowest: profile.lowest(),
            center: profile.center,
            threshold: profile.threshold,
            values: profile.points.iter().map(|p| p.value).collect(),
            statistics: profile.points.iter().map(|p| p.statistic).collect(),
            multipliers: profile.points.iter().map(|p| p.multiplier).collect(),
            converged: profile.points.iter().map(|p| p.converged).collect(),
            parameters: profile.points.into_iter().map(|p| p.parameters).collect(),
            ends: (end_name(profile.ends[0]), end_name(profile.ends[1])),
            interval: (lower, upper),
            evaluations: profile.evaluations,
            rises,
        }
    }
}

/// The profiles of a problem, as the runner returns them
#[pyclass(name = "ProfileRun", frozen, get_all)]
pub struct PyProfileRun {
    /// One profile per quantity: the parameters, then the predictions
    profiles: Vec<PyProfile>,
    /// The objective at the fit
    fit_loss: f64,
    /// The numerical tolerance of a profile value
    tolerance: f64,
    /// The replicates kept by a bootstrap threshold
    replicates: Option<Vec<usize>>,
    /// The index and reason of each replicate that failed
    failed: Vec<(usize, String)>,
}

/// Trace the profile of each parameter and prediction from the fit
///
/// Args:
///     problem: The calibration problem
///     config: The problem's optimizer
///     fit: The fitted values, in problem order
///     parameters: The indices of the parameters to profile
///     predictions: The predictions to profile
///     threshold: "chi_square", "beale" or "bootstrap"
///     level: The level of the threshold
///     max_steps: The most points on each side of a profile
///     seed: The seed of the bootstrap and of a particle swarm
///     bootstrap: The bootstrap of a "bootstrap" threshold
///
/// Raises:
///     ValueError: If the threshold does not fit the problem, a setting is
///         not valid, or the fit is not feasible
///     RuntimeError: If a refit fails, or every replicate of the bootstrap
///         fails
#[pyfunction]
#[pyo3(signature = (problem, config, fit, parameters, predictions, threshold, level, max_steps, seed, bootstrap=None))]
#[allow(clippy::too_many_arguments)]
fn trace_profiles(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    config: &PyOptimizationConfig,
    fit: Vec<f64>,
    parameters: Vec<usize>,
    predictions: Vec<PyObservedDataPoint>,
    threshold: &str,
    level: f64,
    max_steps: usize,
    seed: u64,
    bootstrap: Option<PyRef<'_, PyBootstrapSettings>>,
) -> PyResult<PyProfileRun> {
    let problem_ref = &problem.inner;
    let tracer = ProfileTracer::new(problem_ref, &config.inner, &fit, seed).map_err(value_error)?;
    let quantities: Vec<ProfiledQuantity> = parameters
        .into_iter()
        .map(ProfiledQuantity::Parameter)
        .chain(
            predictions
                .into_iter()
                .map(|spec| ProfiledQuantity::Prediction(spec.inner)),
        )
        .collect();
    let mut rises = None;
    let mut replicates = None;
    let mut failed = Vec::new();
    let thresholds = match (threshold, bootstrap) {
        ("chi_square", None) => {
            vec![chi_square(problem_ref, level).map_err(value_error)?; quantities.len()]
        }
        ("beale", None) => vec![
            beale(problem_ref, tracer.fit_loss(), level).map_err(value_error)?;
            quantities.len()
        ],
        ("bootstrap", Some(settings)) => {
            let centers = quantities
                .iter()
                .map(|quantity| tracer.center(quantity))
                .collect::<Result<Vec<f64>, String>>()
                .map_err(value_error)?;
            let runner =
                ReplicateRunner::new(problem_ref, &config.inner, &fit, settings.starts, seed)
                    .map_err(value_error)?;
            let (resampler, _) = problem.resampler(&settings.spec, runner.fit())?;
            let count = settings.replicates;
            let run = py
                .detach(|| {
                    bootstrap_thresholds(
                        &runner,
                        resampler.as_ref(),
                        count,
                        &quantities,
                        &centers,
                        level,
                    )
                })
                .map_err(PyRuntimeError::new_err)?;
            rises = Some(run.rises);
            replicates = Some(run.replicates);
            failed = run.failed;
            run.thresholds
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "The threshold '{threshold}' takes a bootstrap only when it is 'bootstrap'"
            )));
        }
    };
    let traced = py.detach(|| tracer.trace_all(&quantities, &thresholds, max_steps));
    let mut profiles = Vec::with_capacity(traced.len());
    for (index, profile) in traced.into_iter().enumerate() {
        let profile = profile.map_err(PyRuntimeError::new_err)?;
        let rises = rises
            .as_ref()
            .map(|rises: &Vec<Vec<f64>>| rises[index].clone());
        profiles.push(PyProfile::new(profile, rises, tracer.tolerance()));
    }
    Ok(PyProfileRun {
        profiles,
        fit_loss: tracer.fit_loss(),
        tolerance: tracer.tolerance(),
        replicates,
        failed,
    })
}

/// Register the profile bindings with the inference module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBootstrapSettings>()?;
    m.add_class::<PyProfile>()?;
    m.add_class::<PyProfileRun>()?;
    m.add_function(wrap_pyfunction!(trace_profiles, m)?)?;
    Ok(())
}
