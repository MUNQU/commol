//! Python bindings of the conformal prediction intervals of
//! `commol-inference`.
//!
//! Each function takes the problem built once by a `Calibrator` and returns
//! the intervals and scores as plain values; Python builds the result.

use crate::parallelism::parallel;
use commol_inference::conformal::{
    self, ConformalError, ConformalRun, ConformalSettings, NonconformityScore, Pooling, Predictor,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use commol_inference::resampling::{
    BlockDrawing, BlockScheme, ReplicateRunner, ResidualScale, Residuals,
};

use crate::calibration::{PyObservedDataPoint, PyOptimizationConfig};
use crate::inference::{BlockLength, PyCalibrationProblem, PyParameterSample, predictive_error};

fn conformal_error(error: ConformalError) -> PyErr {
    match error {
        ConformalError::Invalid(message) => PyValueError::new_err(message),
        ConformalError::Prediction(error) => predictive_error(error),
        ConformalError::Failed(message) => PyRuntimeError::new_err(message),
    }
}

fn settings(score: &str, by: &str, confidence_level: f64) -> PyResult<ConformalSettings> {
    let score = NonconformityScore::from_name(Some(score)).map_err(PyValueError::new_err)?;
    let by = Pooling::from_name(Some(by)).map_err(PyValueError::new_err)?;
    ConformalSettings::new(score, by, confidence_level).map_err(PyValueError::new_err)
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
    /// The guaranteed coverage, when the method computes one
    bound: Option<f64>,
    /// The number of blocks refitted, for a method with blocks
    blocks: Option<usize>,
    /// The block length used, for a method with blocks
    block_length: Option<usize>,
    /// The estimate of each series, with the automatic length
    block_length_estimates: Option<Vec<(String, f64)>>,
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
            bound: None,
            blocks: None,
            block_length: None,
            block_length_estimates: None,
        }
    }
}

/// The predictor of a method that does not refit: the sample when there is
/// one, otherwise the fit (the runner passes one of them)
fn predictor<'a>(fit: Option<&'a [f64]>, sample: Option<&'a PyParameterSample>) -> Predictor<'a> {
    match sample {
        Some(sample) => Predictor::Sample(&sample.inner),
        None => Predictor::Fit(fit.unwrap_or_default()),
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
///     confidence_level: The level of the intervals
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
    confidence_level: f64,
) -> PyResult<PyConformalRun> {
    let settings = settings(score, by, confidence_level)?;
    let calibration: Vec<_> = calibration.into_iter().map(|o| o.inner).collect();
    let predictor = predictor(fit.as_deref(), sample);
    parallel(py, || {
        conformal::split_conformal(
            &problem.inner,
            predictor,
            &calibration,
            &positions,
            settings,
        )
    })?
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
///     confidence_level: The level of the intervals
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
    confidence_level: f64,
    starts: usize,
    seed: u64,
) -> PyResult<PyConformalRun> {
    let settings = settings(score, by, confidence_level)?;
    let runner = ReplicateRunner::new(&problem.inner, &config.inner, &fit, starts, seed)
        .map_err(PyValueError::new_err)?;
    parallel(py, || {
        conformal::rolling_conformal(&runner, &origins, &positions, settings)
    })?
    .map(PyConformalRun::from)
    .map_err(conformal_error)
}

/// The blocks of `runner`'s problem with `block_length` positions, or the
/// automatic length of the residuals of its fit
fn blocks(
    runner: &ReplicateRunner<'_, crate::engine::Engine>,
    block_length: Option<usize>,
    drawing: BlockDrawing,
) -> PyResult<(BlockScheme, BlockLength)> {
    let problem = runner.problem();
    let residuals = Residuals::new(problem, runner.fit(), ResidualScale::Additive)
        .map_err(PyValueError::new_err)?;
    let chosen = BlockLength::resolve(problem.observed_data(), &residuals, block_length, drawing);
    let scheme =
        BlockScheme::new(problem.observed_data(), chosen.length).map_err(PyValueError::new_err)?;
    Ok((scheme, chosen))
}

/// Jackknife+ intervals at `positions`, for each series of the problem
///
/// Args:
///     problem: The calibration problem
///     config: The optimizer
///     fit: The fit of every observation, in problem order
///     block_length: Positions per block, or None for the automatic length
///     positions: Where the intervals are
///     score: "absolute", "normalized" or "spread"
///     by: "series" or "all"
///     confidence_level: The level of the intervals
///     starts: Starts per refit, the fit first
///     seed: Seed of the starts
///
/// Raises:
///     ValueError: If a setting is not valid or the blocks are fewer than
///         two
///     RuntimeError: If fewer than two refits succeed
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn jackknife_plus(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    config: &PyOptimizationConfig,
    fit: Vec<f64>,
    block_length: Option<usize>,
    positions: Vec<f64>,
    score: &str,
    by: &str,
    confidence_level: f64,
    starts: usize,
    seed: u64,
) -> PyResult<PyConformalRun> {
    let settings = settings(score, by, confidence_level)?;
    let runner = ReplicateRunner::new(&problem.inner, &config.inner, &fit, starts, seed)
        .map_err(PyValueError::new_err)?;
    let (scheme, chosen) = blocks(&runner, block_length, BlockDrawing::NonOverlapping)?;
    let run = parallel(py, || {
        conformal::jackknife_plus(&runner, &scheme, &positions, settings)
    })?
    .map_err(conformal_error)?;
    Ok(PyConformalRun {
        bound: Some(run.bound),
        blocks: Some(run.blocks),
        block_length: Some(chosen.length),
        block_length_estimates: chosen.estimates,
        ..PyConformalRun::from(run.run)
    })
}

/// EnbPI intervals at `positions`, for each series of the problem
///
/// Args:
///     problem: The calibration problem
///     config: The optimizer
///     fit: The fit every replicate starts from, in problem order
///     replicates: Number of replicates
///     block_length: Positions per block, or None for the automatic length
///     positions: Where the intervals are
///     score: "absolute", "normalized" or "spread"
///     by: "series" or "all"
///     confidence_level: The level of the intervals
///     starts: Starts per replicate, the fit first
///     seed: Seed of the replicates
///
/// Raises:
///     ValueError: If a setting is not valid or a block is left out by no
///         replicate
///     RuntimeError: If every replicate fails
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn enbpi(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    config: &PyOptimizationConfig,
    fit: Vec<f64>,
    replicates: usize,
    block_length: Option<usize>,
    positions: Vec<f64>,
    score: &str,
    by: &str,
    confidence_level: f64,
    starts: usize,
    seed: u64,
) -> PyResult<PyConformalRun> {
    let settings = settings(score, by, confidence_level)?;
    let runner = ReplicateRunner::new(&problem.inner, &config.inner, &fit, starts, seed)
        .map_err(PyValueError::new_err)?;
    let (scheme, chosen) = blocks(&runner, block_length, BlockDrawing::NonOverlapping)?;
    let run = parallel(py, || {
        conformal::enbpi(&runner, &scheme, replicates, &positions, settings)
    })?
    .map_err(conformal_error)?;
    Ok(PyConformalRun {
        blocks: Some(scheme.count()),
        block_length: Some(chosen.length),
        block_length_estimates: chosen.estimates,
        ..PyConformalRun::from(run)
    })
}

/// Adaptive conformal intervals at each point of `stream`, in order of
/// position
///
/// Args:
///     problem: The calibration problem
///     fit: The fitted values, in problem order, or None with a sample
///     sample: The sample whose predictive is the predictor, or None
///     stream: The observations that arrive
///     step: The step of the working level
///     score: "absolute", "normalized" or "spread"
///     by: "series" or "all"
///     confidence_level: The target level of the intervals
///
/// Raises:
///     ValueError: If a setting is not valid, the score has no scale, or a
///         point names an unknown output
///     InfeasibleStateError: If a run leaves the values its model allows
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn adaptive_conformal(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    fit: Option<Vec<f64>>,
    sample: Option<&PyParameterSample>,
    stream: Vec<PyObservedDataPoint>,
    step: f64,
    score: &str,
    by: &str,
    confidence_level: f64,
) -> PyResult<PyConformalRun> {
    let settings = settings(score, by, confidence_level)?;
    let stream: Vec<_> = stream.into_iter().map(|o| o.inner).collect();
    let predictor = predictor(fit.as_deref(), sample);
    parallel(py, || {
        conformal::adaptive_conformal(&problem.inner, predictor, &stream, step, settings)
    })?
    .map(PyConformalRun::from)
    .map_err(conformal_error)
}

/// Register the conformal functions with the inference module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConformalRun>()?;
    m.add_function(wrap_pyfunction!(check_conformal_settings, m)?)?;
    m.add_function(wrap_pyfunction!(split_conformal, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_conformal, m)?)?;
    m.add_function(wrap_pyfunction!(jackknife_plus, m)?)?;
    m.add_function(wrap_pyfunction!(enbpi, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_conformal, m)?)?;
    Ok(())
}
