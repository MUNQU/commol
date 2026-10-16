//! Python bindings of a built calibration problem and of the inference
//! methods that read it.
//!
//! [`PyCalibrationProblem`] is built once from a `Calibrator` and passed to
//! every call, so Python never converts the problem twice and never
//! transforms a prediction: the observation operator, the replicates and the
//! starts all run here.

use commol_calibration::{
    CalibrationProblem, EvaluationError, StartOutcome, Violation, Workspace, optimize_starts,
    optimize_with_history,
};
use commol_inference::resampling::{
    BayesianBootstrap, BlockDrawing, BlockScheme, Multipliers, ReplicateRunner, Resampler,
    ResidualBlocks, ResidualBootstrap, ResidualNoise, ResidualScale, Residuals, WildBootstrap,
    automatic_block_length, block_jackknife, bootstrap,
};
use commol_inference::{
    NoNoise, ObservationDraws, ObservationNoise, ParameterSample, Predictive, PredictiveError,
    observation_draws, scores, statistics,
};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::calibration::{
    PyCalibrationConstraint, PyCalibrationParameter, PyCalibrationResultWithHistory, PyLossConfig,
    PyObservedDataPoint, PyOptimizationConfig, PyStartOutcome,
};
use crate::engine::{Engine, InfeasibleStateError, PyEngine};

/// A calibration problem, built once
#[pyclass(name = "CalibrationProblem", frozen)]
pub struct PyCalibrationProblem {
    pub(crate) inner: CalibrationProblem<Engine>,
}

/// The Python error of a parameter set that has no prediction:
/// `InfeasibleStateError` when it leaves the values its model allows,
/// `ValueError` for a value that is not finite, `RuntimeError` otherwise.
fn prediction_error(index: usize, error: EvaluationError) -> PyErr {
    match error {
        EvaluationError::Violation(Violation::NonFinite) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Parameter set {index} gives a value that is not finite"
            ))
        }
        EvaluationError::Violation(violation) => InfeasibleStateError::new_err(format!(
            "Parameter set {index} is not feasible: {violation:?}"
        )),
        EvaluationError::Failed(message) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Parameter set {index}: {message}"),
        ),
    }
}

#[pymethods]
impl PyCalibrationProblem {
    /// Build a calibration problem
    ///
    /// Args:
    ///     engine: The simulation engine built from the model
    ///     observed_data: Observations to fit
    ///     parameters: Parameters to calibrate, in problem order
    ///     constraints: Constraints on parameters and outputs
    ///     loss_config: Loss, normalization and replicates
    ///     initial_condition_remainder: Compartment or bin that absorbs the
    ///         difference to the total, if any
    ///
    /// Raises:
    ///     ValueError: If the problem is not valid for the engine
    #[new]
    #[pyo3(signature = (engine, observed_data, parameters, constraints, loss_config, initial_condition_remainder=None))]
    fn new(
        engine: PyRef<'_, PyEngine>,
        observed_data: Vec<PyObservedDataPoint>,
        parameters: Vec<PyCalibrationParameter>,
        constraints: Vec<PyCalibrationConstraint>,
        loss_config: &PyLossConfig,
        initial_condition_remainder: Option<String>,
    ) -> PyResult<Self> {
        let value_error = PyErr::new::<pyo3::exceptions::PyValueError, _>;
        let mut problem = CalibrationProblem::new(
            engine.engine(),
            observed_data.into_iter().map(|d| d.inner).collect(),
            parameters.into_iter().map(|p| p.inner).collect(),
            constraints.into_iter().map(|c| c.inner).collect(),
            loss_config.inner,
        )
        .map_err(value_error)?
        .with_observation_normalization(loss_config.normalize_observations)
        .with_replicates(loss_config.replicates.clone());
        if let Some(remainder) = initial_condition_remainder {
            problem = problem
                .with_initial_condition_remainder(&remainder)
                .map_err(value_error)?;
        }
        Ok(Self { inner: problem })
    }

    /// The parameter names, in problem order
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    /// The number of observations
    #[getter]
    fn observation_count(&self) -> usize {
        self.inner.observation().len()
    }

    /// The number of runs per evaluation: one per replicate, or one
    #[getter]
    fn replicate_count(&self) -> usize {
        self.inner.forward().replicate_count()
    }

    /// Optimize from `starts` start points in parallel
    ///
    /// Start 0 is the initial guess; the others are rows of a Latin
    /// hypercube over the bounds drawn from `seed`.
    ///
    /// Args:
    ///     config: The optimizer
    ///     starts: Number of starts, at least 1
    ///     seed: Seed of the starts
    ///
    /// Returns:
    ///     One StartOutcome per start, in order
    ///
    /// Raises:
    ///     ValueError: If `starts` is 0, or a verbose optimizer has more than
    ///         one start
    fn optimize(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        starts: usize,
        seed: u64,
    ) -> PyResult<Vec<PyStartOutcome>> {
        if starts == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "An optimization needs at least one start",
            ));
        }
        let outcomes = if config.verbose {
            if starts > 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "A verbose optimizer shows one start; set starts=1 or verbose=False",
                ));
            }
            let result = crate::python_observer::optimize_with_python_observer(
                &self.inner,
                &config.inner,
                config.header_interval,
            );
            vec![StartOutcome {
                start: self.inner.initial_parameters(),
                result,
            }]
        } else {
            py.detach(|| optimize_starts(&self.inner, &config.inner, starts, seed))
        };
        Ok(outcomes
            .into_iter()
            .map(|inner| PyStartOutcome { inner })
            .collect())
    }

    /// Optimize from the initial guess and keep every evaluation of a
    /// feasible point
    ///
    /// Raises:
    ///     RuntimeError: If the optimization fails
    fn optimize_with_history(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
    ) -> PyResult<PyCalibrationResultWithHistory> {
        py.detach(|| optimize_with_history(&self.inner, &config.inner))
            .map(|inner| PyCalibrationResultWithHistory { inner })
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// The predicted observations of each parameter set, in parallel
    ///
    /// The observations are read through the observation operator the loss
    /// uses: sums of outputs, windows and factors.
    ///
    /// Args:
    ///     parameter_sets: Values of each set, in problem order
    ///
    /// Returns:
    ///     `[set][replicate][observation]`
    ///
    /// Raises:
    ///     InfeasibleStateError: If a set leaves the values its model allows
    fn predict_observations(
        &self,
        py: Python<'_>,
        parameter_sets: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let n = self.inner.observation().len();
        py.detach(|| {
            parameter_sets
                .par_iter()
                .enumerate()
                .map_init(Workspace::default, |workspace, (index, values)| {
                    self.inner
                        .forward()
                        .run_observed(values, workspace)
                        .map(|predictions| {
                            predictions.chunks(n.max(1)).map(<[f64]>::to_vec).collect()
                        })
                        .map_err(|error| (index, error))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|(index, error)| prediction_error(index, error))
    }

    /// The outputs of each parameter set at `positions`, in parallel
    ///
    /// Args:
    ///     parameter_sets: Values of each set, in problem order
    ///     positions: Positions to report at, sorted
    ///
    /// Returns:
    ///     `[set][replicate][position][output]`
    ///
    /// Raises:
    ///     InfeasibleStateError: If a set leaves the values its model allows
    fn predict_positions(
        &self,
        py: Python<'_>,
        parameter_sets: Vec<Vec<f64>>,
        positions: Vec<f64>,
    ) -> PyResult<Vec<Vec<Vec<Vec<f64>>>>> {
        py.detach(|| {
            parameter_sets
                .par_iter()
                .enumerate()
                .map(|(index, values)| {
                    self.inner
                        .forward()
                        .run_at(values, &positions)
                        .map_err(|error| (index, error))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|(index, error)| prediction_error(index, error))
    }

    /// The run of every draw of `sample` at `positions`, in parallel
    ///
    /// Raises:
    ///     ValueError: If the sample's parameters are not the problem's
    ///     InfeasibleStateError: If a draw leaves the values its model allows
    fn predictive(
        &self,
        py: Python<'_>,
        sample: &PyParameterSample,
        positions: Vec<f64>,
    ) -> PyResult<PyPredictive> {
        py.detach(|| Predictive::new(&self.inner, &sample.inner, &positions))
            .map(|inner| PyPredictive { inner })
            .map_err(predictive_error)
    }

    /// Every draw's value of `observations`, read through the observation
    /// operator, with observation noise drawn from `seed`
    ///
    /// Args:
    ///     sample: The draws
    ///     observations: The observations, which need not be the problem's
    ///     seed: Seed of the noise
    ///     residual_noise: `(fit, block_length, residual_scale)` to add
    ///         blocks of residuals of the fit at `fit`, or None for no noise
    ///
    /// Raises:
    ///     ValueError: If the sample's parameters are not the problem's, an
    ///         observation names an unknown output or factor or a series the
    ///         fit has no residuals of, or the residual noise is not valid
    ///     InfeasibleStateError: If a draw leaves the values its model allows
    #[pyo3(signature = (sample, observations, seed, residual_noise=None))]
    fn observation_draws(
        &self,
        py: Python<'_>,
        sample: &PyParameterSample,
        observations: Vec<PyObservedDataPoint>,
        seed: u64,
        residual_noise: Option<(Vec<f64>, usize, String)>,
    ) -> PyResult<PyObservationDraws> {
        let observations: Vec<_> = observations.into_iter().map(|o| o.inner).collect();
        let noise: Box<dyn ObservationNoise> = match residual_noise {
            None => Box::new(NoNoise),
            Some((fit, length, scale)) => {
                let residuals = Residuals::new(&self.inner, &fit, residual_scale(&scale)?)
                    .map_err(value_error)?;
                Box::new(
                    ResidualNoise::new(self.inner.observed_data(), &residuals, length)
                        .map_err(value_error)?,
                )
            }
        };
        py.detach(|| {
            observation_draws(
                &self.inner,
                &sample.inner,
                &observations,
                noise.as_ref(),
                seed,
            )
        })
        .map(|inner| PyObservationDraws { inner })
        .map_err(predictive_error)
    }

    /// Fit replicates of reweighted or resampled data, from the fit
    ///
    /// Args:
    ///     config: The optimizer
    ///     fit: The fitted values, in problem order
    ///     method: "bayesian", "residual" or "wild"
    ///     replicates: Number of replicates
    ///     block_length: Positions per block, or None for the automatic
    ///         length
    ///     starts: Starts per replicate, the fit first
    ///     seed: Seed of the replicates
    ///     residual_scale: "additive" or "multiplicative", for "residual"
    ///     scheme: "moving" or "stationary", for "residual"
    ///     multipliers: "rademacher" or "mammen", for "wild"
    ///
    /// Raises:
    ///     ValueError: If a setting is not valid, the block length does not
    ///         fit the positions, or the residuals cannot be taken on their
    ///         scale
    ///     RuntimeError: If every replicate fails
    #[pyo3(signature = (config, fit, method, replicates, block_length, starts, seed, residual_scale=None, scheme=None, multipliers=None))]
    #[allow(clippy::too_many_arguments)]
    fn bootstrap(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        fit: Vec<f64>,
        method: &str,
        replicates: usize,
        block_length: Option<usize>,
        starts: usize,
        seed: u64,
        residual_scale: Option<String>,
        scheme: Option<String>,
        multipliers: Option<String>,
    ) -> PyResult<PyBootstrapRun> {
        if replicates == 0 {
            return Err(value_error(
                "A bootstrap needs at least one replicate".to_string(),
            ));
        }
        let observed = self.inner.observed_data();
        let runner = ReplicateRunner::new(&self.inner, &config.inner, &fit, starts, seed)
            .map_err(value_error)?;
        let fit = runner.fit().to_vec();
        let (resampler, chosen): (Box<dyn Resampler>, BlockLength) = match method {
            "bayesian" => {
                let residuals = Residuals::new(&self.inner, &fit, ResidualScale::Additive)
                    .map_err(value_error)?;
                let chosen = BlockLength::resolve(
                    observed,
                    &residuals,
                    block_length,
                    BlockDrawing::NonOverlapping,
                )?;
                let scheme = BlockScheme::new(observed, chosen.length).map_err(value_error)?;
                (Box::new(BayesianBootstrap::new(observed, &scheme)), chosen)
            }
            "residual" => {
                let scale = residual_scale_of(residual_scale.as_deref())?;
                let blocks = match scheme.as_deref() {
                    Some("moving") => ResidualBlocks::Moving,
                    Some("stationary") => ResidualBlocks::Stationary,
                    other => {
                        return Err(value_error(format!(
                            "scheme is {other:?}; it must be \"moving\" or \"stationary\""
                        )));
                    }
                };
                let drawing = match blocks {
                    ResidualBlocks::Moving => BlockDrawing::Moving,
                    ResidualBlocks::Stationary => BlockDrawing::Stationary,
                };
                let residuals = Residuals::new(&self.inner, &fit, scale).map_err(value_error)?;
                let chosen = BlockLength::resolve(observed, &residuals, block_length, drawing)?;
                let block_scheme =
                    BlockScheme::new(observed, chosen.length).map_err(value_error)?;
                (
                    Box::new(ResidualBootstrap::new(
                        observed,
                        residuals,
                        &block_scheme,
                        blocks,
                    )),
                    chosen,
                )
            }
            "wild" => {
                let multipliers = match multipliers.as_deref() {
                    Some("rademacher") => Multipliers::Rademacher,
                    Some("mammen") => Multipliers::Mammen,
                    other => {
                        return Err(value_error(format!(
                            "multipliers is {other:?}; it must be \"rademacher\" or \"mammen\""
                        )));
                    }
                };
                let residuals = Residuals::new(&self.inner, &fit, ResidualScale::Additive)
                    .map_err(value_error)?;
                let chosen = BlockLength::resolve(
                    observed,
                    &residuals,
                    block_length,
                    BlockDrawing::NonOverlapping,
                )?;
                let scheme = BlockScheme::new(observed, chosen.length).map_err(value_error)?;
                (
                    Box::new(WildBootstrap::new(residuals, &scheme, multipliers)),
                    chosen,
                )
            }
            other => {
                return Err(value_error(format!(
                    "Unknown resampling method {other:?}; one of \"bayesian\", \"residual\", \"wild\""
                )));
            }
        };
        let run = py
            .detach(|| bootstrap(&runner, resampler.as_ref(), replicates))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(PyBootstrapRun {
            values: run.sample.values().to_vec(),
            groups: run
                .sample
                .groups()
                .map(<[usize]>::to_vec)
                .unwrap_or_default(),
            failed: run.failed,
            not_converged: run.not_converged,
            negative_values: run.negative_values,
            block_length: chosen.length,
            block_length_estimates: chosen.estimates,
        })
    }

    /// Fit once per block left out, from the fit
    ///
    /// Args:
    ///     config: The optimizer
    ///     fit: The fitted values, in problem order
    ///     block_length: Positions per block, or None for the automatic
    ///         length
    ///     starts: Starts per fit, the fit first
    ///     seed: Seed of the starts
    ///
    /// Raises:
    ///     ValueError: If the block length does not fit the positions or
    ///         makes fewer than two blocks
    ///     RuntimeError: If fewer than two fits succeed
    fn jackknife(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        fit: Vec<f64>,
        block_length: Option<usize>,
        starts: usize,
        seed: u64,
    ) -> PyResult<PyJackknifeRun> {
        let observed = self.inner.observed_data();
        let runner = ReplicateRunner::new(&self.inner, &config.inner, &fit, starts, seed)
            .map_err(value_error)?;
        let residuals = Residuals::new(&self.inner, runner.fit(), ResidualScale::Additive)
            .map_err(value_error)?;
        let chosen = BlockLength::resolve(
            observed,
            &residuals,
            block_length,
            BlockDrawing::NonOverlapping,
        )?;
        let scheme = BlockScheme::new(observed, chosen.length).map_err(value_error)?;
        if scheme.len() < 2 {
            return Err(value_error(format!(
                "block_length {} makes {} block; a jackknife needs at least two",
                scheme.length(),
                scheme.len()
            )));
        }
        let jackknife = py
            .detach(|| block_jackknife(&runner, &scheme))
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        let spans = scheme.spans();
        Ok(PyJackknifeRun {
            spans: jackknife.blocks.iter().map(|&b| spans[b]).collect(),
            estimate: jackknife.statistics.estimate,
            bias: jackknife.statistics.bias,
            standard_error: jackknife.statistics.standard_error,
            pseudo_values: jackknife.statistics.pseudo_values,
            influence: jackknife.statistics.influence,
            failed: jackknife
                .failed
                .into_iter()
                .map(|(block, reason)| (spans[block], reason))
                .collect(),
            not_converged: jackknife.not_converged,
            block_length: chosen.length,
            block_length_estimates: chosen.estimates,
        })
    }
}

/// The block length a method runs with, and the estimates of `"auto"`.
struct BlockLength {
    length: usize,
    estimates: Option<Vec<(String, f64)>>,
}

impl BlockLength {
    fn resolve(
        observed: &[commol_calibration::ObservedDataPoint],
        residuals: &Residuals,
        requested: Option<usize>,
        drawing: BlockDrawing,
    ) -> PyResult<Self> {
        Ok(match requested {
            Some(length) => Self {
                length,
                estimates: None,
            },
            None => {
                let automatic = automatic_block_length(observed, residuals.residuals(), drawing);
                Self {
                    length: automatic.length,
                    estimates: Some(automatic.estimates),
                }
            }
        })
    }
}

fn residual_scale(name: &str) -> PyResult<ResidualScale> {
    residual_scale_of(Some(name))
}

fn residual_scale_of(name: Option<&str>) -> PyResult<ResidualScale> {
    match name {
        Some("additive") => Ok(ResidualScale::Additive),
        Some("multiplicative") => Ok(ResidualScale::Multiplicative),
        other => Err(value_error(format!(
            "residual_scale is {other:?}; it must be \"additive\" or \"multiplicative\""
        ))),
    }
}

/// The replicates of a bootstrap, as the runner returns them
#[pyclass(name = "BootstrapRun", frozen, get_all)]
pub struct PyBootstrapRun {
    /// One row per replicate that did not fail, in problem order
    values: Vec<Vec<f64>>,
    /// The replicate of each row
    groups: Vec<usize>,
    /// The index and reason of each replicate that failed
    failed: Vec<(usize, String)>,
    /// Replicates kept although their optimizer did not converge
    not_converged: usize,
    /// Synthetic observed values below zero, over every replicate
    negative_values: usize,
    /// The block length used
    block_length: usize,
    /// The estimate of each series, with the automatic length
    block_length_estimates: Option<Vec<(String, f64)>>,
}

/// A block jackknife, as the runner returns it
#[pyclass(name = "JackknifeRun", frozen, get_all)]
pub struct PyJackknifeRun {
    /// The first and last position of each block that fitted
    spans: Vec<(f64, f64)>,
    /// The bias-corrected estimate of each parameter
    estimate: Vec<f64>,
    /// The jackknife bias of each parameter
    bias: Vec<f64>,
    /// The jackknife standard error of each parameter
    standard_error: Vec<f64>,
    /// One row of pseudo-values per block that fitted
    pseudo_values: Vec<Vec<f64>>,
    /// One row of influences per block that fitted
    influence: Vec<Vec<f64>>,
    /// The span and reason of each block whose fit failed
    failed: Vec<((f64, f64), String)>,
    /// Fits kept although their optimizer did not converge
    not_converged: usize,
    /// The block length used
    block_length: usize,
    /// The estimate of each series, with the automatic length
    block_length_estimates: Option<Vec<(String, f64)>>,
}

fn value_error(message: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message)
}

fn predictive_error(error: PredictiveError) -> PyErr {
    match error {
        PredictiveError::Mismatch(message) => value_error(message),
        PredictiveError::Draw { draw, error } => prediction_error(draw, error),
    }
}

/// Weighted draws of the parameters of a calibration problem
///
/// Args:
///     names: One name per column
///     bounds: `(min, max)` of each parameter
///     values: One row per draw
///     log_weights: One log weight per draw, of any scale
///     groups: The group of each draw, or None
///
/// Raises:
///     ValueError: If a value is not finite or outside its bounds, a log
///         weight is NaN or +inf, or the shapes differ
#[pyclass(name = "ParameterSample", frozen)]
pub struct PyParameterSample {
    pub(crate) inner: ParameterSample,
}

#[pymethods]
impl PyParameterSample {
    #[new]
    #[pyo3(signature = (names, bounds, values, log_weights, groups=None))]
    fn new(
        names: Vec<String>,
        bounds: Vec<(f64, f64)>,
        values: Vec<Vec<f64>>,
        log_weights: Vec<f64>,
        groups: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        ParameterSample::new(names, bounds, values, &log_weights, groups)
            .map(|inner| Self { inner })
            .map_err(value_error)
    }

    /// The weights, which sum to 1
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.weights().to_vec()
    }

    /// The natural logarithms of the weights
    #[getter]
    fn log_weights(&self) -> Vec<f64> {
        self.inner.log_weights()
    }

    /// The weighted mean of each parameter
    fn mean(&self) -> Vec<f64> {
        self.inner.mean()
    }

    /// The weighted covariance matrix
    fn covariance(&self) -> Vec<Vec<f64>> {
        self.inner.covariance()
    }

    /// The weighted correlation matrix
    fn correlation(&self) -> Vec<Vec<f64>> {
        self.inner.correlation()
    }

    /// The weighted quantiles: one row per level, one column per parameter
    fn quantiles(&self, levels: Vec<f64>) -> Vec<Vec<f64>> {
        self.inner.quantiles(&levels)
    }

    /// The central interval of `confidence_level`, as `(lower, upper)` rows
    fn interval(&self, confidence_level: f64) -> PyResult<(Vec<f64>, Vec<f64>)> {
        self.inner
            .interval(confidence_level)
            .map(|[lower, upper]| (lower, upper))
            .map_err(value_error)
    }

    /// Kish's effective sample size
    fn effective_size(&self) -> f64 {
        self.inner.effective_size()
    }

    /// The indices of `count` draws taken with replacement by weight
    fn resample(&self, count: usize, seed: u64) -> Vec<usize> {
        self.inner.resample(count, seed)
    }
}

/// The run of every draw of a sample at a set of positions
#[pyclass(name = "Predictive", frozen)]
pub struct PyPredictive {
    inner: Predictive,
}

#[pymethods]
impl PyPredictive {
    /// The positions of the trajectories
    #[getter]
    fn positions(&self) -> Vec<f64> {
        self.inner.positions().to_vec()
    }

    /// The trajectory of every draw, `[draw][position][output]`
    #[getter]
    fn trajectories(&self) -> Vec<Vec<Vec<f64>>> {
        self.inner.trajectories().to_vec()
    }

    /// The weight of every draw
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.weights().to_vec()
    }

    /// The band of `confidence_level`: `(mean, median, lower, upper)`, each
    /// `[position][output]`
    #[allow(clippy::type_complexity)]
    fn band(
        &self,
        confidence_level: f64,
    ) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let band = self.inner.band(confidence_level).map_err(value_error)?;
        Ok((band.mean, band.median, band.lower, band.upper))
    }

    /// The band of the sum of `outputs` (indices), or of its running total
    /// when `cumulative`: `(mean, median, lower, upper)`, one value per
    /// position
    #[allow(clippy::type_complexity)]
    fn series_band(
        &self,
        outputs: Vec<usize>,
        cumulative: bool,
        confidence_level: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let band = self
            .inner
            .series_band(&outputs, cumulative, confidence_level)
            .map_err(value_error)?;
        Ok((band.mean, band.median, band.lower, band.upper))
    }
}

/// Every draw's value of a set of observations
#[pyclass(name = "ObservationDraws", frozen)]
pub struct PyObservationDraws {
    inner: ObservationDraws,
}

#[pymethods]
impl PyObservationDraws {
    /// `[draw][observation]`
    #[getter]
    fn values(&self) -> Vec<Vec<f64>> {
        self.inner.values.clone()
    }

    /// The weight of every draw
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.weights.clone()
    }

    /// The weighted quantiles of each observation: `[level][observation]`
    fn quantiles(&self, levels: Vec<f64>) -> Vec<Vec<f64>> {
        let n = self.inner.values.first().map_or(0, Vec::len);
        let per_observation: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let column: Vec<f64> = self.inner.values.iter().map(|row| row[i]).collect();
                statistics::quantiles(&column, &self.inner.weights, &levels)
            })
            .collect();
        (0..levels.len())
            .map(|l| per_observation.iter().map(|q| q[l]).collect())
            .collect()
    }

    /// The band of each observation: `(mean, median, lower, upper)`, one
    /// value per observation
    #[allow(clippy::type_complexity)]
    fn band(&self, confidence_level: f64) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let band = self.inner.band(confidence_level).map_err(value_error)?;
        Ok((band.mean, band.median, band.lower, band.upper))
    }

    /// 1 for each observation inside its central interval, else 0
    fn coverage(&self, observed: Vec<f64>, confidence_level: f64) -> PyResult<Vec<f64>> {
        scores::coverage(&self.inner, &observed, confidence_level).map_err(value_error)
    }

    /// The interval score of each observation
    fn interval_score(&self, observed: Vec<f64>, confidence_level: f64) -> PyResult<Vec<f64>> {
        scores::interval_score(&self.inner, &observed, confidence_level).map_err(value_error)
    }

    /// The weighted interval score of each observation
    fn weighted_interval_score(
        &self,
        observed: Vec<f64>,
        confidence_levels: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        scores::weighted_interval_score(&self.inner, &observed, &confidence_levels)
            .map_err(value_error)
    }

    /// The CRPS of each observation
    fn crps(&self, observed: Vec<f64>) -> PyResult<Vec<f64>> {
        scores::crps(&self.inner, &observed).map_err(value_error)
    }

    /// The randomized PIT value of each observation
    fn pit(&self, observed: Vec<f64>, seed: u64) -> PyResult<Vec<f64>> {
        scores::pit(&self.inner, &observed, seed).map_err(value_error)
    }
}

/// The weighted quantiles of `values` at `levels`
///
/// The inverse of the weighted empirical distribution, interpolated between
/// the midpoints of the cumulative weights; with equal weights it is NumPy's
/// `method="hazen"`.
///
/// Raises:
///     ValueError: If the lengths differ or no weight is positive
#[pyfunction]
fn weighted_quantiles(values: Vec<f64>, weights: Vec<f64>, levels: Vec<f64>) -> PyResult<Vec<f64>> {
    if values.len() != weights.len() {
        return Err(value_error(format!(
            "One weight per value: {} values, {} weights",
            values.len(),
            weights.len()
        )));
    }
    if !weights.iter().any(|w| *w > 0.0) {
        return Err(value_error(
            "A quantile needs a positive weight".to_string(),
        ));
    }
    Ok(statistics::quantiles(&values, &weights, &levels))
}

/// The weighted mean of `values`
///
/// Raises:
///     ValueError: If the lengths differ or no weight is positive
#[pyfunction]
fn weighted_mean(values: Vec<f64>, weights: Vec<f64>) -> PyResult<f64> {
    if values.len() != weights.len() || !weights.iter().any(|w| *w > 0.0) {
        return Err(value_error(
            "A weighted mean needs one weight per value and a positive weight".to_string(),
        ));
    }
    Ok(statistics::mean(&values, &weights))
}

/// The Monte Carlo error of the quantile at `level` of equally weighted
/// `values`: an order-statistic interval of about 95 %
#[pyfunction]
fn quantile_error(values: Vec<f64>, level: f64) -> PyResult<(f64, f64)> {
    if values.is_empty() || !(0.0..=1.0).contains(&level) {
        return Err(value_error(
            "A quantile error needs values and a level in [0, 1]".to_string(),
        ));
    }
    Ok(statistics::quantile_error(&values, level))
}

/// The mean of `values` per series named by `labels`, in order of first
/// appearance
#[pyfunction]
fn by_series(labels: Vec<String>, values: Vec<f64>) -> PyResult<Vec<(String, f64)>> {
    if labels.len() != values.len() {
        return Err(value_error("One label per value".to_string()));
    }
    Ok(scores::by_series(&labels, &values))
}

/// Register the inference module with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCalibrationProblem>()?;
    m.add_class::<PyParameterSample>()?;
    m.add_class::<PyPredictive>()?;
    m.add_class::<PyObservationDraws>()?;
    m.add_class::<PyBootstrapRun>()?;
    m.add_class::<PyJackknifeRun>()?;
    m.add_function(wrap_pyfunction!(quantile_error, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(by_series, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_mean, m)?)?;
    crate::identifiability::register(m)?;
    Ok(())
}
