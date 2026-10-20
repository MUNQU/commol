//! Python bindings of the local approximations of `commol-inference`: the
//! Laplace approximation and the delta method.
//!
//! Each runner takes the problem built once by a `Calibrator` and the fit,
//! and returns plain values; Python builds the result.

use commol_inference::local::{self, CurvatureKind, Gaussian, LocalError};
use commol_inference::sensitivity::Matrix;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::inference::PyCalibrationProblem;

/// A refusal is a `ValueError`, a failure a `RuntimeError`.
fn local_error(error: LocalError) -> PyErr {
    match error {
        LocalError::Refused(message) => PyValueError::new_err(message),
        LocalError::Failed(message) => PyRuntimeError::new_err(message),
    }
}

fn curvature_kind(name: &str) -> PyResult<CurvatureKind> {
    CurvatureKind::from_name(name).map_err(PyValueError::new_err)
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
    draws: usize,
    seed: u64,
) -> PyResult<PyLocalRun> {
    let kind = curvature_kind(curvature)?;
    let (curvature, gaussian) = py
        .detach(|| local::laplace_approximation(&problem.inner, &fit, kind))
        .map_err(local_error)?;
    Ok(PyLocalRun::of(
        &gaussian,
        draws,
        seed,
        curvature.notes().to_vec(),
    ))
}

/// Register the classes and functions of this module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLocalRun>()?;
    m.add_function(wrap_pyfunction!(laplace, m)?)?;
    Ok(())
}
