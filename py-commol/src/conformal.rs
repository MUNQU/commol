//! Python bindings of the conformal prediction intervals of
//! `commol-inference`.
//!
//! Each function takes the problem built once by a `Calibrator` and returns
//! the intervals and scores as plain values; Python builds the result.

use commol_inference::conformal::{
    self, ConformalError, ConformalRun, ConformalSettings, NonconformityScore, Pooling, Predictor,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use commol_inference::resampling::ReplicateRunner;

use crate::calibration::{PyObservedDataPoint, PyOptimizationConfig};
use crate::inference::{PyCalibrationProblem, PyParameterSample, predictive_error};

fn conformal_error(error: ConformalError) -> PyErr {
    match error {
        ConformalError::Invalid(message) => PyValueError::new_err(message),
        ConformalError::Prediction(error) => predictive_error(error),
        ConformalError::Failed(message) => PyRuntimeError::new_err(message),
    }
}

fn settings(score: &str, by: &str, level: f64) -> PyResult<ConformalSettings> {
    let score = NonconformityScore::from_name(Some(score)).map_err(PyValueError::new_err)?;
    let by = Pooling::from_name(Some(by)).map_err(PyValueError::new_err)?;
    ConformalSettings::new(score, by, level).map_err(PyValueError::new_err)
}

/// The intervals of a conformal method, as the runner returns them
#[pyclass(name = "ConformalRun", frozen, get_all)]
pub struct PyConformalRun {
    /// The series of each target
    labels: Vec<String>,
    /// The position of each target
    positions: Vec<f64>,
    /// The point prediction at each target
    prediction: Vec<f64>,
    /// The lower end of each interval
    lower: Vec<f64>,
    /// The upper end of each interval
    upper: Vec<f64>,
    /// Each score with the label of its series
    scores: Vec<(String, f64)>,
    /// The index and reason of each refit that failed
    failed: Vec<(usize, String)>,
}

impl From<ConformalRun> for PyConformalRun {
    fn from(run: ConformalRun) -> Self {
        let intervals = run.intervals;
        Self {
            labels: intervals.targets.iter().map(|t| t.label.clone()).collect(),
            positions: intervals.targets.iter().map(|t| t.at).collect(),
            prediction: intervals.prediction,
            lower: intervals.lower,
            upper: intervals.upper,
            scores: run.scores,
            failed: run.failed,
        }
    }
}

/// Check the score and the pooling of a conformal method
///
/// Raises:
///     ValueError: If one is missing or not a valid name, or the pair is
///         rejected
#[pyfunction]
#[pyo3(signature = (score, by))]
fn check_conformal_settings(score: Option<&str>, by: Option<&str>) -> PyResult<()> {
    let score = NonconformityScore::from_name(score).map_err(PyValueError::new_err)?;
    let by = Pooling::from_name(by).map_err(PyValueError::new_err)?;
    ConformalSettings::check_pair(score, by).map_err(PyValueError::new_err)
}

/// Split conformal intervals at `positions`, for each series of
/// `calibration`
///
/// Args:
///     problem: The calibration problem
///     fit: The fitted values, in problem order, or None with a sample
///     sample: The sample whose predictive is the predictor, or None
///     calibration: The held-out observations
///     positions: Where the intervals are
///     score: "absolute", "normalized" or "spread"
///     by: "series" or "all"
///     level: The level of the intervals
///
/// Raises:
///     ValueError: If a setting is not valid, the score has no scale, or a
///         point names an unknown output
///     InfeasibleStateError: If a run leaves the values its model allows
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn split_conformal(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    fit: Option<Vec<f64>>,
    sample: Option<&PyParameterSample>,
    calibration: Vec<PyObservedDataPoint>,
    positions: Vec<f64>,
    score: &str,
    by: &str,
    level: f64,
) -> PyResult<PyConformalRun> {
    let settings = settings(score, by, level)?;
    let calibration: Vec<_> = calibration.into_iter().map(|o| o.inner).collect();
    let predictor = match (&fit, sample) {
        (_, Some(sample)) => Predictor::Sample(&sample.inner),
        (Some(fit), None) => Predictor::Fit(fit),
        (None, None) => {
            return Err(PyValueError::new_err(
                "split conformal needs a fit or a sample",
            ));
        }
    };
    py.detach(|| {
        conformal::split_conformal(
            &problem.inner,
            predictor,
            &calibration,
            &positions,
            settings,
        )
    })
    .map(PyConformalRun::from)
    .map_err(conformal_error)
}

/// Rolling-origin conformal intervals at `positions`, after the data, for
/// each series of the problem
///
/// Args:
///     problem: The calibration problem
///     config: The optimizer
///     fit: The fit of every observation, in problem order
///     origins: The positions of the origins
///     positions: Where the intervals are
///     score: "absolute" or "normalized"
///     by: "series" or "all"
///     level: The level of the intervals
///     starts: Starts per fit, the fit first
///     seed: Seed of the starts
///
/// Raises:
///     ValueError: If a setting is not valid, a position is not after the
///         data, or an origin has no observation before or after it
///     RuntimeError: If every fit fails
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rolling_conformal(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    config: &PyOptimizationConfig,
    fit: Vec<f64>,
    origins: Vec<f64>,
    positions: Vec<f64>,
    score: &str,
    by: &str,
    level: f64,
    starts: usize,
    seed: u64,
) -> PyResult<PyConformalRun> {
    let settings = settings(score, by, level)?;
    let runner = ReplicateRunner::new(&problem.inner, &config.inner, &fit, starts, seed)
        .map_err(PyValueError::new_err)?;
    py.detach(|| conformal::rolling_conformal(&runner, &origins, &positions, settings))
        .map(PyConformalRun::from)
        .map_err(conformal_error)
}

/// Register the conformal functions with the inference module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConformalRun>()?;
    m.add_function(wrap_pyfunction!(check_conformal_settings, m)?)?;
    m.add_function(wrap_pyfunction!(split_conformal, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_conformal, m)?)?;
    Ok(())
}
