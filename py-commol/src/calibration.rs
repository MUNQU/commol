//! Python bindings of the results of a calibration, and of the questions a
//! calibration asks the engine: the values a model has, the state they give
//! and the windows of an observation. Every setting crosses as JSON
//! ([`crate::json`]).

use pyo3::prelude::*;

use commol_calibration::CalibrationParameter;

use crate::engine::PyEngine;

/// One evaluation of a feasible point during an optimization
#[pyclass(name = "CalibrationEvaluation", from_py_object)]
#[derive(Clone)]
pub struct PyCalibrationEvaluation {
    pub inner: commol_calibration::CalibrationEvaluation,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyCalibrationEvaluation {
    /// The parameter values the objective scored, in problem order
    #[getter]
    fn parameters(&self) -> Vec<f64> {
        self.inner.parameters.clone()
    }

    /// The loss at these values
    #[getter]
    fn loss(&self) -> f64 {
        self.inner.loss
    }
}

/// Calibration result
#[pyclass(name = "CalibrationResult", from_py_object)]
#[derive(Clone)]
pub struct PyCalibrationResult {
    pub inner: commol_calibration::CalibrationResult,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyCalibrationResult {
    /// The best value of each parameter, as `(name, value)` pairs in problem
    /// order
    #[getter]
    fn best_parameters(&self) -> Vec<(String, f64)> {
        self.inner
            .parameter_names
            .iter()
            .cloned()
            .zip(self.inner.best_parameters.iter().copied())
            .collect()
    }

    /// The final loss value
    #[getter]
    fn final_loss(&self) -> f64 {
        self.inner.final_loss
    }

    /// The number of iterations performed
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    /// Whether the optimization converged
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Why the optimization stopped
    #[getter]
    fn termination_reason(&self) -> String {
        self.inner.termination_reason.clone()
    }
}

/// Calibration result with every evaluation of a feasible point
#[pyclass(name = "CalibrationResultWithHistory")]
pub struct PyCalibrationResultWithHistory {
    pub inner: commol_calibration::CalibrationResultWithHistory,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyCalibrationResultWithHistory {
    /// The result of the optimization
    #[getter]
    fn result(&self) -> PyCalibrationResult {
        PyCalibrationResult {
            inner: self.inner.result.clone(),
        }
    }

    /// Every evaluation of a feasible point, in the order they were made
    #[getter]
    fn evaluations(&self) -> Vec<PyCalibrationEvaluation> {
        self.inner
            .evaluations
            .iter()
            .map(|e| PyCalibrationEvaluation { inner: e.clone() })
            .collect()
    }
}

/// The outcome of one start of a multi-start optimization
#[pyclass(name = "StartOutcome")]
pub struct PyStartOutcome {
    pub inner: commol_calibration::StartOutcome,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyStartOutcome {
    /// The start point, in problem order
    #[getter]
    fn start(&self) -> Vec<f64> {
        self.inner.start.clone()
    }

    /// The result, or None when the start failed
    #[getter]
    fn result(&self) -> Option<PyCalibrationResult> {
        self.inner
            .result
            .as_ref()
            .ok()
            .map(|r| PyCalibrationResult { inner: r.clone() })
    }

    /// Why the start failed, or None
    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.result.as_ref().err().cloned()
    }
}

/// The starts of a multi-start optimization and the best of them: the
/// start of lowest loss among those that did not fail, the first on a tie
#[pyclass(name = "Starts")]
pub struct PyStarts {
    pub inner: commol_calibration::Starts,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyStarts {
    /// The outcome of each start, in order
    #[getter]
    fn outcomes(&self) -> Vec<PyStartOutcome> {
        self.inner
            .outcomes()
            .iter()
            .map(|outcome| PyStartOutcome {
                inner: outcome.clone(),
            })
            .collect()
    }

    /// The index of the best start
    #[getter]
    fn best(&self) -> usize {
        self.inner.best_index()
    }
}

/// Initial state a compartmental engine starts from once calibration
/// values are applied
///
/// The values are applied exactly as the calibration objective scores them:
/// clamped to their bounds, with the remainder initial condition
/// auto-calculated, and written in parameter order.
///
/// Args:
///     engine: Engine built from a compartmental model, before calibration
///     parameters: The JSON of the calibration parameters, in problem order
///     values: Value of each parameter, parallel to `parameters`
///     initial_condition_remainder: Compartment or bin that takes what the
///         other compartments leave of the amounts the model states, if any
///
/// Returns:
///     A `(values, state, covered)` tuple: the values the engine applied,
///     parallel to `parameters`; the initial amount of each compartment,
///     in engine order; and the indices of the compartments a calibrated bin
///     or compartment value, or the remainder, sets
///
/// Raises:
///     ValueError: If the engine is not compartmental, an initial-condition id
///         cannot be resolved, the number of values is wrong, or the
///         calibrated amounts leave the remainder below zero
#[pyfunction]
#[pyo3(signature = (engine, parameters, values, initial_condition_remainder=None))]
fn calibrated_initial_state(
    engine: PyRef<'_, PyEngine>,
    parameters: &str,
    values: Vec<f64>,
    initial_condition_remainder: Option<String>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let parameters: Vec<CalibrationParameter> = crate::json::read(parameters)?;
    let (configured, applied, plan) = commol_calibration::configured_engine(
        &engine.engine(),
        &parameters,
        initial_condition_remainder.as_deref(),
        &values,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let (state, covered) = crate::engine::compartmental_state(&configured, &plan)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok((applied, state, covered))
}

/// Check that the engine sets every parameter and resolves every value of
/// its model kind, such as an initial condition, and the remainder, and that
/// the model has every value a run needs once they are written
///
/// Args:
///     engine: The simulation engine
///     parameters: The JSON of the calibration parameters, in problem order
///     initial_condition_remainder: The id that takes what the values leave
///         of the sum the model kind keeps (a compartment or bin of a
///         compartmental model), if any
///
/// Raises:
///     ValueError: If the engine does not set a parameter (one the model
///         does not have, or one a formula or a series sets), the kind does
///         not accept a type of value, or a value or the remainder cannot be
///         resolved; for example a category of a stratification that is not
///         binary; or if the model still lacks a value that the
///         calibration does not set
#[pyfunction]
#[pyo3(signature = (engine, parameters, initial_condition_remainder=None))]
fn check_model_values(
    engine: PyRef<'_, PyEngine>,
    parameters: &str,
    initial_condition_remainder: Option<String>,
) -> PyResult<()> {
    let parameters: Vec<CalibrationParameter> = crate::json::read(parameters)?;
    commol_calibration::check_model_values(
        &engine.engine(),
        &parameters,
        initial_condition_remainder.as_deref(),
    )
    .map(|_| ())
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

/// The compartments each initial-condition value sets, and those of the
/// remainder, as the engine resolves them
///
/// Args:
///     engine: Engine built from a compartmental model
///     parameters: The JSON of the initial-condition calibration parameters, in
///         problem order
///     initial_condition_remainder: The id that takes what the values leave,
///         if any
///
/// Returns:
///     A `(values, remainder)` tuple: for each parameter, the indices of the
///     compartments it sets (none for a category split), and the indices of
///     the compartments of the remainder (none without one)
///
/// Raises:
///     ValueError: If the engine is not compartmental, or a value or the
///         remainder cannot be resolved
#[pyfunction]
#[pyo3(signature = (engine, parameters, initial_condition_remainder=None))]
fn initial_condition_coverage(
    engine: PyRef<'_, PyEngine>,
    parameters: &str,
    initial_condition_remainder: Option<String>,
) -> PyResult<(Vec<Vec<usize>>, Vec<usize>)> {
    let parameters: Vec<CalibrationParameter> = crate::json::read(parameters)?;
    commol_calibration::plan_model_values(
        &engine.engine(),
        &parameters,
        initial_condition_remainder.as_deref(),
    )
    .and_then(|plan| crate::engine::compartmental_coverage(&plan))
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

/// The message of a position that results do not hold.
fn no_value(at: f64) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "The results have no value at {at}; run the simulation at that position with \
         Simulation.run_at."
    ))
}

/// The index of each position of `at` in `positions`, found with the
/// tolerance of the axis, as an observation finds its row
///
/// Args:
///     positions: Positions on the axis, ascending
///     at: The positions to find
///
/// Raises:
///     ValueError: If a position of `at` is not one of `positions`; the
///         message names it
#[pyfunction]
fn index_at(positions: Vec<f64>, at: Vec<f64>) -> PyResult<Vec<usize>> {
    at.into_iter()
        .map(|at| commol_calibration::index_at(&positions, at).ok_or_else(|| no_value(at)))
        .collect()
}

/// The start of the window of length `window` that ends at `at`: `at -
/// window`, or 0 within the rounding of the window
///
/// Args:
///     at: The end of the window
///     window: The length of the window
#[pyfunction]
fn window_start(at: f64, window: f64) -> f64 {
    commol_calibration::window_start(at, window)
}

/// The error of a window outside its rule, with the message of the rule.
fn window_error(message: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message)
}

/// Check the length of a window
///
/// Raises:
///     ValueError: If `window` is not a finite, positive length
#[pyfunction]
fn check_window(window: f64) -> PyResult<()> {
    commol_calibration::ranges::WINDOW
        .check(window)
        .map_err(window_error)
}

/// Check the window of length `window` that ends at `at`, and give its
/// start: it starts at 0 or after it
///
/// Raises:
///     ValueError: If `window` is not a finite, positive length, or the
///         window starts before 0
#[pyfunction]
fn check_window_end(at: f64, window: f64) -> PyResult<f64> {
    commol_calibration::check_window_end(at, window).map_err(window_error)
}

/// The number of windows of length `window` in `span`, a whole number when
/// it is one within the tolerance of the axis
///
/// Raises:
///     ValueError: If `window` is not a finite, positive length
#[pyfunction]
fn windows_in(span: f64, window: f64) -> PyResult<f64> {
    commol_calibration::windows_in(span, window).map_err(window_error)
}

/// The positions that bound window `index` of the grid of length `window`,
/// `(index · window, (index + 1) · window)`
///
/// Raises:
///     ValueError: If `window` is not a finite, positive length or `index`
///         is negative
#[pyfunction]
fn window_bounds(index: i64, window: f64) -> PyResult<(f64, f64)> {
    commol_calibration::window_bounds(index, window).map_err(window_error)
}

/// The window of the grid of length `window` that holds the position `at`
///
/// Raises:
///     ValueError: If `window` is not a finite, positive length or `at` is
///         not a position on the axis
#[pyfunction]
fn window_index(at: f64, window: f64) -> PyResult<usize> {
    commol_calibration::window_index(at, window).map_err(window_error)
}

/// The end of each complete window of the grid of length `window` up to
/// `end`
///
/// Raises:
///     ValueError: If `window` is not a finite, positive length or `end` is
///         not finite
#[pyfunction]
fn window_ends(window: f64, end: f64) -> PyResult<Vec<f64>> {
    commol_calibration::window_ends(window, end).map_err(window_error)
}

/// The increment of `series` over the window of length `window` that ends at
/// each of `ends`, as an observation with that window reads it
///
/// Args:
///     series: One value per position
///     positions: Positions on the axis, ascending
///     window: The length of a window
///     ends: The end of each window
///
/// Raises:
///     ValueError: If `window` is not a finite, positive length, a window
///         starts before 0 at no position, or an end or a start is not one
///         of `positions`
#[pyfunction]
fn window_increments(
    series: Vec<f64>,
    positions: Vec<f64>,
    window: f64,
    ends: Vec<f64>,
) -> PyResult<Vec<f64>> {
    use commol_calibration::WindowError;
    commol_calibration::window_increments(&series, &positions, window, &ends).map_err(|error| {
        match error {
            WindowError::Missing(at) => no_value(at),
            WindowError::Length(message) => window_error(message),
            incomplete => window_error(format!("{incomplete}.")),
        }
    })
}

/// Register calibration module with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCalibrationEvaluation>()?;
    m.add_class::<PyCalibrationResultWithHistory>()?;
    m.add_class::<PyCalibrationResult>()?;
    m.add_class::<PyStartOutcome>()?;
    m.add_class::<PyStarts>()?;
    m.add_function(wrap_pyfunction!(calibrated_initial_state, m)?)?;
    m.add_function(wrap_pyfunction!(check_model_values, m)?)?;
    m.add_function(wrap_pyfunction!(initial_condition_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(index_at, m)?)?;
    m.add_function(wrap_pyfunction!(window_start, m)?)?;
    m.add_function(wrap_pyfunction!(window_increments, m)?)?;
    m.add_function(wrap_pyfunction!(check_window, m)?)?;
    m.add_function(wrap_pyfunction!(check_window_end, m)?)?;
    m.add_function(wrap_pyfunction!(windows_in, m)?)?;
    m.add_function(wrap_pyfunction!(window_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(window_index, m)?)?;
    m.add_function(wrap_pyfunction!(window_ends, m)?)?;
    Ok(())
}
