//! Python bindings of the local approximations of `commol-inference`: the
//! Laplace approximation, the delta method, the sandwich and the Laplace
//! mixture.
//!
//! Each runner takes the problem built once by a `Calibrator` and the fit,
//! and returns plain values; Python builds the result.

use crate::parallelism::parallel;
use commol_inference::importance::TemperedPosterior;
use commol_inference::local::{self, CurvatureKind, Gaussian, MixtureRun};
use commol_inference::sensitivity::Matrix;
use commol_inference::smc::{GibbsTarget, LikelihoodTarget};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::inference::{PyCalibrationProblem, method_error};
use crate::smc::{PriorArgs, problem_priors};

fn curvature_kind(name: &str) -> PyResult<CurvatureKind> {
    CurvatureKind::from_name(name).map_err(method_error)
}

fn rows(matrix: &Matrix) -> Vec<Vec<f64>> {
    matrix
        .row_iter()
        .map(|row| row.iter().copied().collect())
        .collect()
}

/// A local approximation, as the runners return it
#[pyclass(name = "LocalRun", frozen, get_all)]
pub struct PyLocalRun {
    /// One row per draw, on the scale of the parameters
    values: Vec<Vec<f64>>,
    /// The fit on the unbounded scale
    center: Vec<f64>,
    /// The covariance on the unbounded scale
    covariance: Vec<Vec<f64>>,
    /// The degrees of freedom of a Student t, or None for a normal
    degrees_of_freedom: Option<f64>,
    /// The curvature entries whose error is not small, in words
    notes: Vec<String>,
}

impl PyLocalRun {
    fn of(gaussian: &Gaussian, draws: usize, seed: u64, notes: Vec<String>) -> Self {
        Self {
            values: gaussian.draws(draws, seed),
            center: gaussian.center().to_vec(),
            covariance: rows(gaussian.covariance()),
            degrees_of_freedom: gaussian.degrees_of_freedom(),
            notes,
        }
    }
}

/// The Laplace approximation of the observation models at the fit
///
/// Args:
///     problem: The calibration problem, with observation models
///     fit: The fitted values, in problem order
///     curvature: "gauss_newton" or "hessian"
///     draws: Number of draws
///     seed: Seed of the draws
///
/// Raises:
///     ValueError: If the problem has a loss, a fit is on a bound, the
///         engine is stochastic or the curvature is singular
///     RuntimeError: If the derivatives fail at the fit
#[pyfunction]
fn laplace(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    fit: Vec<f64>,
    curvature: &str,
    #[pyo3(from_py_with = crate::counts::required::draws)] draws: usize,
    seed: u64,
) -> PyResult<PyLocalRun> {
    let kind = curvature_kind(curvature)?;
    let (curvature, gaussian) = parallel(py, || {
        local::laplace_approximation(&problem.inner, &fit, kind)
    })?
    .map_err(method_error)?;
    Ok(PyLocalRun::of(
        &gaussian,
        draws,
        seed,
        curvature.notes().to_vec(),
    ))
}

/// A sandwich: the run, the blocks, the block length and its estimates.
type SandwichTables = (PyLocalRun, usize, usize, Option<Vec<(String, f64)>>);

/// The sandwich covariance at the fit, with the meat summed over blocks
///
/// Args:
///     problem: The calibration problem
///     fit: The fitted values, in problem order
///     curvature: "gauss_newton" or "hessian"
///     block_length: Positions per block, or None for the automatic length
///     draws: Number of draws
///     seed: Seed of the draws
///
/// Returns:
///     The run, the number of blocks, the positions per block and, with the
///     automatic length, the estimate of each series
///
/// Raises:
///     ValueError: If the loss is the absolute error, a fit is on a bound,
///         the engine is stochastic, the block length does not fit the
///         positions, or there are not more blocks than parameters
///     RuntimeError: If the derivatives fail at the fit
#[pyfunction]
fn sandwich(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    fit: Vec<f64>,
    curvature: &str,
    #[pyo3(from_py_with = crate::counts::optional::block_length)] block_length: Option<usize>,
    #[pyo3(from_py_with = crate::counts::required::draws)] draws: usize,
    seed: u64,
) -> PyResult<SandwichTables> {
    let kind = curvature_kind(curvature)?;
    let run = parallel(py, || {
        local::sandwich_approximation(&problem.inner, &fit, kind, block_length)
    })?
    .map_err(method_error)?;
    let local = PyLocalRun::of(&run.gaussian, draws, seed, run.curvature.notes().to_vec());
    Ok((local, run.blocks, run.block_length, run.estimates))
}

/// A band and its outputs: `(outputs, mean, lower, upper)`, each table
/// `[position][output]`, as the band of a predictive.
type BandTables = (Vec<String>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// The band of every output by the delta method
///
/// Args:
///     problem: The calibration problem
///     fit: The fitted values, in problem order
///     covariance: The covariance on the unbounded scale
///     degrees_of_freedom: The degrees of freedom of a Student t, or None
///         for a normal
///     positions: Positions to report at
///     confidence_level: The level of the band
///
/// Returns:
///     The output names, and the prediction at the fit, the lower and the
///     upper end at each position and output, `[position][output]`
///
/// Raises:
///     ValueError: If the covariance is not one row and column per
///         parameter, the level is not strictly between 0 and 1, the engine is
///         stochastic, or a position is one the engine rejects
///     RuntimeError: If the derivatives fail at the fit
#[pyfunction]
fn delta_band(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    fit: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    degrees_of_freedom: Option<f64>,
    positions: Vec<f64>,
    confidence_level: f64,
) -> PyResult<BandTables> {
    let p = fit.len();
    if covariance.len() != p || covariance.iter().any(|row| row.len() != p) {
        return Err(PyValueError::new_err(format!(
            "The covariance needs {p} rows of {p} values, one per parameter"
        )));
    }
    let matrix = Matrix::from_fn(p, p, |j, k| covariance[j][k]);
    let band = parallel(py, || {
        local::delta_band(
            &problem.inner,
            &fit,
            &matrix,
            degrees_of_freedom,
            &positions,
            confidence_level,
        )
    })?
    .map_err(method_error)?;
    Ok((
        band.outputs,
        band.band.mean,
        band.band.lower,
        band.band.upper,
    ))
}

/// A Laplace mixture and its importance sample, as the runner returns it
#[pyclass(name = "MixtureRun", frozen, get_all)]
pub struct PyMixtureRun {
    /// One row per draw, on the scale of the parameters
    values: Vec<Vec<f64>>,
    /// The smoothed log weight of each draw, normalized
    log_weights: Vec<f64>,
    /// The centre of each component, on the scale of the parameters
    centers: Vec<Vec<f64>>,
    /// The weight of each component
    weights: Vec<f64>,
    /// The covariance of each component, on the unbounded scale
    covariances: Vec<Vec<Vec<f64>>>,
    /// The log Laplace evidence of each component
    log_evidences: Vec<f64>,
    /// The Pareto shape of the tail of the weights
    k_hat: f64,
    /// Kish's effective size of the smoothed weights
    effective_size: f64,
    /// Whether k_hat is at most 0.7
    reliable: bool,
    /// Ends that gave no component, and an unreliable sample, in words
    notes: Vec<String>,
}

impl PyMixtureRun {
    fn of(run: MixtureRun) -> Self {
        Self {
            centers: run.components.iter().map(|c| c.center.clone()).collect(),
            weights: run.components.iter().map(|c| c.weight).collect(),
            covariances: run
                .components
                .iter()
                .map(|c| rows(c.gaussian.covariance()))
                .collect(),
            log_evidences: run.components.iter().map(|c| c.log_evidence).collect(),
            values: run.sample.values,
            log_weights: run.sample.log_weights,
            k_hat: run.sample.k_hat,
            effective_size: run.sample.effective_size,
            reliable: run.reliable,
            notes: run.notes,
        }
    }
}

/// A Laplace mixture at the ends of the starts, the proposal of an
/// importance sampler of a posterior
///
/// Args:
///     problem: The calibration problem
///     priors: The priors, or None for uniform on the bounds
///     eta: The scale of the loss of a Gibbs posterior, or None for the
///         posterior of the observation models
///     ends: The end of each start, in problem order
///     draws: Number of draws
///     curvature: "gauss_newton" or "hessian"
///     seed: Seed of the draws
///
/// Raises:
///     ValueError: If a prior or eta is not valid, the target needs
///         observation models the problem lacks, or no end gives a
///         component
///     RuntimeError: If a run, the derivatives or the weights fail
#[pyfunction]
#[pyo3(signature = (problem, priors, eta, ends, draws, curvature, seed))]
#[allow(clippy::too_many_arguments)]
fn laplace_mixture(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    priors: Option<Vec<PriorArgs>>,
    eta: Option<f64>,
    ends: Vec<Vec<f64>>,
    #[pyo3(from_py_with = crate::counts::required::draws)] draws: usize,
    curvature: &str,
    seed: u64,
) -> PyResult<PyMixtureRun> {
    let kind = curvature_kind(curvature)?;
    let priors = problem_priors(problem, priors)?;
    let inner = &problem.inner;
    let run = match eta {
        None => {
            let target = LikelihoodTarget::new(inner, &priors).map_err(method_error)?;
            parallel(py, || {
                local::laplace_mixture(
                    inner,
                    &TemperedPosterior::new(&target, seed),
                    1.0,
                    &ends,
                    draws,
                    kind,
                    seed,
                )
            })?
        }
        Some(eta) => {
            let target = GibbsTarget::new(inner, &priors, eta).map_err(method_error)?;
            parallel(py, || {
                local::laplace_mixture(
                    inner,
                    &TemperedPosterior::new(&target, seed),
                    eta,
                    &ends,
                    draws,
                    kind,
                    seed,
                )
            })?
        }
    }
    .map_err(method_error)?;
    Ok(PyMixtureRun::of(run))
}

/// Register the classes and functions of this module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLocalRun>()?;
    m.add_class::<PyMixtureRun>()?;
    m.add_function(wrap_pyfunction!(laplace, m)?)?;
    m.add_function(wrap_pyfunction!(sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(delta_band, m)?)?;
    m.add_function(wrap_pyfunction!(laplace_mixture, m)?)?;
    Ok(())
}
