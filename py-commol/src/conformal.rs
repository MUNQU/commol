//! Python bindings of the conformal prediction intervals of
//! `commol-inference`.
//!
//! Each function takes the problem built once by a `Calibrator`, the JSON of
//! the method and, for a predictor that is a sample, its handle; it fits the
//! problem when the method needs a fit and returns the intervals and scores
//! as plain values; Python builds the result.

use crate::parallelism::parallel;
use commol_calibration::observed_data;
use commol_inference::conformal::{self, ConformalRun, Predictor};
use commol_inference::resampling::{BlockDrawing, BlockScheme, FitUnit, ReplicateRunner};
use commol_inference::settings::conformal::{
    AdaptiveConformalSettings, EnbPiSettings, JackknifePlusSettings, RollingConformalSettings,
    ScoreAndPooling, SplitConformalSettings,
};
use pyo3::prelude::*;

use crate::calibration::PyStarts;
use crate::inference::{PyCalibrationProblem, PyParameterSample, best_values, method_error};

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
    /// Each refit that failed: "<Unit> <index> failed: <reason>"
    failed: Vec<String>,
    /// Refits kept although their optimizer did not converge
    not_converged: usize,
    /// The guaranteed coverage, when the method computes one
    bound: Option<f64>,
    /// The number of blocks refitted, for a method with blocks
    blocks: Option<usize>,
    /// The block length used, for a method with blocks
    block_length: Option<usize>,
    /// The estimate of each series, with the automatic length
    block_length_estimates: Option<Vec<(String, f64)>>,
}

impl PyConformalRun {
    /// The run of a method whose refits are of `unit`; a method with no
    /// refits has no failed refit, whatever its unit
    fn of(run: ConformalRun, unit: FitUnit) -> Self {
        let intervals = run.intervals;
        Self {
            labels: intervals.targets.iter().map(|t| t.label.clone()).collect(),
            positions: intervals.targets.iter().map(|t| t.at).collect(),
            prediction: intervals.prediction,
            lower: intervals.lower,
            upper: intervals.upper,
            scores: run.scores,
            failed: unit.entries(&run.failed),
            not_converged: run.not_converged,
            bound: None,
            blocks: None,
            block_length: None,
            block_length_estimates: None,
        }
    }
}

/// The starts of the fit of a method that does not refit, when it has no
/// sample, and its predictor: the sample when there is one, otherwise the
/// fit
fn predictor<'a>(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    starts: usize,
    sample: Option<&'a PyParameterSample>,
    fit: &'a mut Vec<f64>,
) -> PyResult<(Option<PyStarts>, Predictor<'a>)> {
    match sample {
        Some(sample) => Ok((None, Predictor::Sample(&sample.inner))),
        None => {
            let starts = problem.fit(py, starts)?;
            *fit = best_values(&starts);
            Ok((Some(PyStarts { inner: starts }), Predictor::Fit(fit)))
        }
    }
}

/// The fit of a method that refits, and the runner of its refits.
fn refits<'a>(
    py: Python<'_>,
    problem: &'a PyCalibrationProblem,
    starts: usize,
) -> PyResult<(PyStarts, ReplicateRunner<'a, crate::engine::Engine>)> {
    let fitted = problem.fit(py, starts)?;
    let runner = problem.runner(&best_values(&fitted), starts)?;
    Ok((PyStarts { inner: fitted }, runner))
}

/// Check the score and the pooling of a conformal method
///
/// Args:
///     settings: The JSON of the fields `score` and `by` that are given
///
/// Raises:
///     ValueError: If one is missing or not a valid name, or the pair is
///         rejected
#[pyfunction]
fn check_conformal_settings(settings: &str) -> PyResult<()> {
    crate::json::read::<ScoreAndPooling>(settings)?
        .check()
        .map_err(method_error)
}

/// Split conformal intervals at the positions of the settings, for each
/// series of the held-out observations
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `SplitConformal`
///     sample: The sample whose predictive is the predictor, or None for
///         the fit of the problem
///
/// Returns:
///     The starts of the fit, or None with a sample, and the intervals
///
/// Raises:
///     ValueError: If a setting is not valid, the score has no scale, or a
///         point names an unknown output
///     RuntimeError: If the fit fails
///     InfeasibleStateError: If a run leaves the values its model allows
#[pyfunction]
#[pyo3(signature = (problem, settings, sample=None))]
fn split_conformal(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
    sample: Option<&PyParameterSample>,
) -> PyResult<(Option<PyStarts>, PyConformalRun)> {
    let settings: SplitConformalSettings = crate::json::read(settings)?;
    let conformal = settings.conformal().map_err(method_error)?;
    let calibration = observed_data(&settings.calibration_data);
    let mut fit = Vec::new();
    let (starts, predictor) = predictor(py, problem, settings.starts, sample, &mut fit)?;
    let run = parallel(py, || {
        conformal::split_conformal(
            &problem.inner,
            predictor,
            &calibration,
            &settings.positions,
            conformal,
        )
    })?
    .map_err(method_error)?;
    Ok((starts, PyConformalRun::of(run, FitUnit::Replicate)))
}

/// Rolling-origin conformal intervals at the positions of the settings,
/// after the data, for each series of the problem
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `RollingConformal`
///
/// Returns:
///     The starts of the fit of every observation, and the intervals
///
/// Raises:
///     ValueError: If a setting is not valid, a position is not after the
///         data, or an origin has no observation before or after it
///     RuntimeError: If the fit or every refit fails
#[pyfunction]
fn rolling_conformal(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(PyStarts, PyConformalRun)> {
    let settings: RollingConformalSettings = crate::json::read(settings)?;
    let conformal = settings.conformal().map_err(method_error)?;
    let (starts, runner) = refits(py, problem, settings.starts)?;
    let run = parallel(py, || {
        conformal::rolling_conformal(&runner, &settings.origins, &settings.positions, conformal)
    })?
    .map_err(method_error)?;
    Ok((starts, PyConformalRun::of(run, FitUnit::Origin)))
}

/// The blocks of `runner`'s problem with `block_length` positions, or the
/// automatic length of the residuals of its fit
fn blocks(
    runner: &ReplicateRunner<'_, crate::engine::Engine>,
    block_length: Option<usize>,
    drawing: BlockDrawing,
) -> PyResult<BlockScheme> {
    BlockScheme::at_fit(runner.problem(), runner.fit(), block_length, drawing).map_err(method_error)
}

/// Jackknife+ intervals at the positions of the settings, for each series
/// of the problem
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `JackknifePlus`
///
/// Returns:
///     The starts of the fit, and the intervals
///
/// Raises:
///     ValueError: If a setting is not valid or the blocks are fewer than
///         two
///     RuntimeError: If the fit fails or fewer than two refits succeed
#[pyfunction]
fn jackknife_plus(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(PyStarts, PyConformalRun)> {
    let settings: JackknifePlusSettings = crate::json::read(settings)?;
    let conformal = settings.conformal().map_err(method_error)?;
    let (starts, runner) = refits(py, problem, settings.starts)?;
    let scheme = blocks(
        &runner,
        settings.block_length.0,
        BlockDrawing::NonOverlapping,
    )?;
    let run = parallel(py, || {
        conformal::jackknife_plus(&runner, &scheme, &settings.positions, conformal)
    })?
    .map_err(method_error)?;
    let run = PyConformalRun {
        bound: Some(run.bound),
        blocks: Some(run.blocks),
        block_length: Some(scheme.length()),
        block_length_estimates: scheme.estimates().map(<[_]>::to_vec),
        ..PyConformalRun::of(run.run, FitUnit::Block)
    };
    Ok((starts, run))
}

/// EnbPI intervals at the positions of the settings, for each series of the
/// problem
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of an `EnbPI`
///
/// Returns:
///     The starts of the fit, and the intervals
///
/// Raises:
///     ValueError: If a setting is not valid or a block is left out by no
///         replicate
///     RuntimeError: If the fit or every replicate fails
#[pyfunction]
fn enbpi(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(PyStarts, PyConformalRun)> {
    let settings: EnbPiSettings = crate::json::read(settings)?;
    let conformal = settings.conformal().map_err(method_error)?;
    let (starts, runner) = refits(py, problem, settings.starts)?;
    let scheme = blocks(
        &runner,
        settings.block_length.0,
        BlockDrawing::NonOverlapping,
    )?;
    let run = parallel(py, || {
        conformal::enbpi(
            &runner,
            &scheme,
            settings.replicates,
            &settings.positions,
            conformal,
        )
    })?
    .map_err(method_error)?;
    let run = PyConformalRun {
        blocks: Some(scheme.count()),
        block_length: Some(scheme.length()),
        block_length_estimates: scheme.estimates().map(<[_]>::to_vec),
        ..PyConformalRun::of(run, FitUnit::Replicate)
    };
    Ok((starts, run))
}

/// Adaptive conformal intervals at each point of the stream of the
/// settings, in order of position
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of an `AdaptiveConformal`
///     sample: The sample whose predictive is the predictor, or None for
///         the fit of the problem
///
/// Returns:
///     The starts of the fit, or None with a sample, and the intervals
///
/// Raises:
///     ValueError: If a setting is not valid, the score has no scale, or a
///         point names an unknown output
///     RuntimeError: If the fit fails
///     InfeasibleStateError: If a run leaves the values its model allows
#[pyfunction]
#[pyo3(signature = (problem, settings, sample=None))]
fn adaptive_conformal(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
    sample: Option<&PyParameterSample>,
) -> PyResult<(Option<PyStarts>, PyConformalRun)> {
    let settings: AdaptiveConformalSettings = crate::json::read(settings)?;
    let conformal = settings.conformal().map_err(method_error)?;
    let stream = observed_data(&settings.stream);
    let mut fit = Vec::new();
    let (starts, predictor) = predictor(py, problem, settings.starts, sample, &mut fit)?;
    let run = parallel(py, || {
        conformal::adaptive_conformal(&problem.inner, predictor, &stream, settings.step, conformal)
    })?
    .map_err(method_error)?;
    Ok((starts, PyConformalRun::of(run, FitUnit::Replicate)))
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
