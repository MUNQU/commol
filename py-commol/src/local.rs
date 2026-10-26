//! Python bindings of the local approximations of `commol-inference`: the
//! Laplace approximation, the delta method, the sandwich and the Laplace
//! mixture.
//!
//! Each runner takes the problem built once by a `Calibrator` and the JSON
//! of its method ([`commol_inference::settings::local`]), fits the problem,
//! and returns the starts of the fit and plain values; Python builds the
//! result.

use crate::parallelism::parallel;
use commol_inference::importance::TemperedPosterior;
use commol_inference::local::{self, Gaussian, MixtureRun};
use commol_inference::sensitivity::Matrix;
use commol_inference::settings::local::{
    LaplaceMixtureSettings, LaplaceSettings, MixtureTarget, SandwichSettings,
};
use commol_inference::smc::{GibbsTarget, LikelihoodTarget};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::calibration::PyStarts;
use crate::inference::{PyCalibrationProblem, best_values, method_error, start_ends};
use crate::smc::problem_priors;

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

/// Fit the problem, then take the Laplace approximation of the
/// observation models at the fit
///
/// Args:
///     problem: The calibration problem, with observation models
///     settings: The JSON of a `Laplace`
///
/// Returns:
///     The starts of the fit, and the approximation
///
/// Raises:
///     ValueError: If a setting is not valid, the problem has a loss, a fit
///         is on a bound, the engine is stochastic or the curvature is
///         singular
///     RuntimeError: If the fit or the derivatives fail at the fit
#[pyfunction]
fn laplace(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(PyStarts, PyLocalRun)> {
    let settings: LaplaceSettings = crate::json::read(settings)?;
    let starts = problem.fit(py, settings.starts)?;
    let fit = best_values(&starts);
    let (curvature, gaussian) = parallel(py, || {
        local::laplace_approximation(&problem.inner, &fit, settings.curvature)
    })?
    .map_err(method_error)?;
    let run = PyLocalRun::of(
        &gaussian,
        settings.draws,
        problem.seed,
        curvature.notes().to_vec(),
    );
    Ok((PyStarts { inner: starts }, run))
}

/// A sandwich: the starts of the fit, the run, the blocks, the block length
/// and its estimates.
type SandwichTables = (
    PyStarts,
    PyLocalRun,
    usize,
    usize,
    Option<Vec<(String, f64)>>,
);

/// Fit the problem, then take the sandwich covariance at the fit, with the
/// meat summed over blocks
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `Sandwich`
///
/// Returns:
///     The starts of the fit, the run, the number of blocks, the positions
///     per block and, with the automatic length, the estimate of each
///     series
///
/// Raises:
///     ValueError: If a setting is not valid, the loss is the absolute
///         error, a fit is on a bound, the engine is stochastic, the block
///         length does not fit the positions, or there are not more blocks
///         than parameters
///     RuntimeError: If the fit or the derivatives fail at the fit
#[pyfunction]
fn sandwich(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<SandwichTables> {
    let settings: SandwichSettings = crate::json::read(settings)?;
    let starts = problem.fit(py, settings.starts)?;
    let fit = best_values(&starts);
    let run = parallel(py, || {
        local::sandwich_approximation(
            &problem.inner,
            &fit,
            settings.curvature,
            settings.block_length.0,
        )
    })?
    .map_err(method_error)?;
    let local = PyLocalRun::of(
        &run.gaussian,
        settings.draws,
        problem.seed,
        run.curvature.notes().to_vec(),
    );
    Ok((
        PyStarts { inner: starts },
        local,
        run.blocks,
        run.block_length,
        run.estimates,
    ))
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

/// Fit the problem from several starts, then sample a posterior with the
/// Laplace mixture at the ends of the starts as the proposal of an
/// importance sampler
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `LaplaceMixture`
///
/// Returns:
///     The starts of the fit, and the mixture with its sample
///
/// Raises:
///     ValueError: If a prior or a setting is not valid, the target needs
///         observation models the problem lacks, or no end gives a
///         component
///     RuntimeError: If the fit, a run, the derivatives or the weights fail
#[pyfunction]
fn laplace_mixture(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(PyStarts, PyMixtureRun)> {
    let settings: LaplaceMixtureSettings = crate::json::read(settings)?;
    let priors = problem_priors(problem, settings.target.priors())?;
    let starts = problem.fit(py, settings.starts)?;
    let ends = start_ends(&starts);
    let inner = &problem.inner;
    let seed = problem.seed;
    let (draws, curvature) = (settings.draws, settings.curvature);
    let run = match &settings.target {
        MixtureTarget::Likelihood { .. } => {
            let target = LikelihoodTarget::new(inner, &priors).map_err(method_error)?;
            parallel(py, || {
                local::laplace_mixture(
                    inner,
                    &TemperedPosterior::new(&target, seed),
                    1.0,
                    &ends,
                    draws,
                    curvature,
                    seed,
                )
            })?
        }
        MixtureTarget::Gibbs(gibbs) => {
            let target = GibbsTarget::new(inner, &priors, gibbs.eta).map_err(method_error)?;
            parallel(py, || {
                local::laplace_mixture(
                    inner,
                    &TemperedPosterior::new(&target, seed),
                    gibbs.eta,
                    &ends,
                    draws,
                    curvature,
                    seed,
                )
            })?
        }
    }
    .map_err(method_error)?;
    Ok((PyStarts { inner: starts }, PyMixtureRun::of(run)))
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
