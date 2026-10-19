//! Python bindings of a built calibration problem and of the inference
//! methods that read it.
//!
//! [`PyCalibrationProblem`] is built once from a `Calibrator` and passed to
//! every call, so Python never converts the problem twice and never
//! transforms a prediction: the observation operator, the replicates and the
//! starts all run here.

use commol_calibration::{
    CalibrationProblem, DiscrepancyConfig, EvaluationError, StartOutcome, Violation, Workspace,
    optimize_starts, optimize_with_history,
};
use commol_inference::resampling::{
    BayesianBootstrap, BlockDrawing, BlockScheme, Multipliers, ParametricResampler,
    ReplicateRunner, Resampler, ResidualBlocks, ResidualBootstrap, ResidualNoise, ResidualScale,
    Residuals, WildBootstrap, automatic_block_length, block_jackknife, bootstrap,
};
use commol_inference::{
    DistributionNoise, NoNoise, ObservationDraws, ObservationNoise, ParameterSample, Predictive,
    PredictiveError, observation_draws, scores, statistics,
};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::calibration::{
    PyCalibrationConstraint, PyCalibrationParameter, PyCalibrationResultWithHistory, PyLossConfig,
    PyObservationModel, PyObservedDataPoint, PyOptimizationConfig, PyStartOutcome,
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

impl PyCalibrationProblem {
    /// Run `replicates` of the resampler that `resampler` makes from the fit,
    /// and the block length it chose, if it draws blocks.
    #[allow(clippy::too_many_arguments)]
    fn run_bootstrap(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        fit: &[f64],
        replicates: usize,
        starts: usize,
        seed: u64,
        resampler: impl FnOnce(&[f64]) -> PyResult<(Box<dyn Resampler>, Option<BlockLength>)>,
    ) -> PyResult<PyBootstrapRun> {
        let runner = ReplicateRunner::new(&self.inner, &config.inner, fit, starts, seed)
            .map_err(value_error)?;
        let (resampler, chosen) = resampler(runner.fit())?;
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
            block_length: chosen.as_ref().map(|chosen| chosen.length),
            block_length_estimates: chosen.and_then(|chosen| chosen.estimates),
        })
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
    ///     loss_config: Loss, normalization and replicates; None with
    ///         observation models
    ///     initial_condition_remainder: Compartment or bin that takes what
    ///         the other compartments leave of the amounts the model states,
    ///         if any
    ///     observation_models: `(series label, model)` of each series; None
    ///         with a loss
    ///
    /// Raises:
    ///     ValueError: If the problem is not valid for the engine, it has
    ///         both or neither of a loss and observation models, or it has
    ///         observation models and a stochastic engine
    #[new]
    #[pyo3(signature = (engine, observed_data, parameters, constraints, loss_config=None, initial_condition_remainder=None, observation_models=None))]
    fn new(
        engine: PyRef<'_, PyEngine>,
        observed_data: Vec<PyObservedDataPoint>,
        parameters: Vec<PyCalibrationParameter>,
        constraints: Vec<PyCalibrationConstraint>,
        loss_config: Option<PyRef<'_, PyLossConfig>>,
        initial_condition_remainder: Option<String>,
        observation_models: Option<Vec<(String, PyObservationModel)>>,
    ) -> PyResult<Self> {
        let engine = engine.engine();
        let discrepancy = match (&loss_config, observation_models) {
            (Some(loss), None) => DiscrepancyConfig::Loss(loss.inner),
            (None, Some(_)) if engine.engine_type().is_stochastic() => {
                return Err(value_error(
                    "A stochastic engine needs a loss: the likelihood of a stochastic model \
                     integrates its hidden path, which observation models of one run do not \
                     do; set loss_function instead of observation_models"
                        .to_string(),
                ));
            }
            (None, Some(models)) => DiscrepancyConfig::Likelihood(
                models
                    .into_iter()
                    .map(|(label, model)| (label, model.inner))
                    .collect(),
            ),
            _ => {
                return Err(value_error(
                    "A calibration problem has a loss or observation models: exactly one of them"
                        .to_string(),
                ));
            }
        };
        let mut problem = CalibrationProblem::new(
            engine,
            observed_data.into_iter().map(|d| d.inner).collect(),
            parameters.into_iter().map(|p| p.inner).collect(),
            constraints.into_iter().map(|c| c.inner).collect(),
            discrepancy,
        )
        .map_err(value_error)?;
        if let Some(loss) = loss_config {
            problem = problem
                .with_observation_normalization(loss.normalize_observations)
                .and_then(|problem| problem.with_replicates(loss.replicates.clone()))
                .map_err(value_error)?;
        }
        if let Some(remainder) = initial_condition_remainder {
            problem = problem
                .with_initial_condition_remainder(&remainder)
                .map_err(value_error)?;
        }
        Ok(Self { inner: problem })
    }

    /// Optimize from `starts` start points in parallel
    ///
    /// Start 0 is the initial guess; the others are rows of a Latin
    /// hypercube over the bounds drawn from `seed`.
    ///
    /// Args:
    ///     config: The optimizer
    ///     starts: Number of starts, at least 1 (the caller checks it)
    ///     seed: Seed of the starts
    ///
    /// Returns:
    ///     One StartOutcome per start, in order
    ///
    /// Raises:
    ///     ValueError: If a verbose optimizer has more than one start
    fn optimize(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        starts: usize,
        seed: u64,
    ) -> PyResult<Vec<PyStartOutcome>> {
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
        let n = self.inner.observation().count();
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

    /// The run of every draw of `sample` at `positions`, in parallel, each
    /// replicate of a stochastic engine seeded from `seed`
    ///
    /// Raises:
    ///     ValueError: If the sample's parameters are not the problem's
    ///     InfeasibleStateError: If a draw leaves the values its model allows
    fn predictive(
        &self,
        py: Python<'_>,
        sample: &PyParameterSample,
        positions: Vec<f64>,
        seed: u64,
    ) -> PyResult<PyPredictive> {
        py.detach(|| Predictive::new(self.inner.forward(), &sample.inner, &positions, seed))
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
    ///         blocks of residuals of the fit at `fit`, or None
    ///     distribution_noise: Whether to draw from the observation models
    ///         of the problem; with neither noise, no noise
    ///
    /// Raises:
    ///     ValueError: If the sample's parameters are not the problem's, an
    ///         observation names an unknown output or factor, a series the
    ///         fit has no residuals of or a series with no observation
    ///         model, both noises are asked for, or the noise is not valid
    ///     InfeasibleStateError: If a draw leaves the values its model allows
    #[pyo3(signature = (sample, observations, seed, residual_noise=None, distribution_noise=false))]
    fn observation_draws(
        &self,
        py: Python<'_>,
        sample: &PyParameterSample,
        observations: Vec<PyObservedDataPoint>,
        seed: u64,
        residual_noise: Option<(Vec<f64>, usize, String)>,
        distribution_noise: bool,
    ) -> PyResult<PyObservationDraws> {
        let observations: Vec<_> = observations.into_iter().map(|o| o.inner).collect();
        let noise: Box<dyn ObservationNoise> = match residual_noise {
            None if distribution_noise => {
                Box::new(DistributionNoise::new(&self.inner).map_err(value_error)?)
            }
            None => Box::new(NoNoise),
            Some(_) if distribution_noise => {
                return Err(value_error(
                    "Observation draws take one noise: residual or distribution".to_string(),
                ));
            }
            Some((fit, length, scale)) => {
                let scale = ResidualScale::from_name(&scale).map_err(value_error)?;
                let residuals = Residuals::new(&self.inner, &fit, scale).map_err(value_error)?;
                Box::new(
                    ResidualNoise::new(self.inner.observed_data(), &residuals, length)
                        .map_err(value_error)?,
                )
            }
        };
        py.detach(|| {
            observation_draws(
                self.inner.forward(),
                &sample.inner,
                &observations,
                noise.as_ref(),
                seed,
            )
        })
        .map(|inner| PyObservationDraws { inner })
        .map_err(predictive_error)
    }

    /// The log score of each of `observations` under the predictive of
    /// `sample` and the observation models of the problem, each replicate of
    /// a stochastic engine seeded from `seed`
    ///
    /// Raises:
    ///     ValueError: If the sample's parameters are not the problem's, the
    ///         problem has a loss, or an observation names an unknown output
    ///         or factor or a series with no observation model
    ///     InfeasibleStateError: If a draw leaves the values its model allows
    fn log_score(
        &self,
        py: Python<'_>,
        sample: &PyParameterSample,
        observations: Vec<PyObservedDataPoint>,
        seed: u64,
    ) -> PyResult<Vec<f64>> {
        let observations: Vec<_> = observations.into_iter().map(|o| o.inner).collect();
        let noise = DistributionNoise::new(&self.inner).map_err(value_error)?;
        py.detach(|| scores::log_score(&self.inner, &sample.inner, &observations, &noise, seed))
            .map_err(predictive_error)
    }

    /// The weighted log density of each observation at `values`
    ///
    /// Raises:
    ///     ValueError: If the problem has a loss, or the values give no
    ///         prediction
    fn pointwise_log_likelihood(&self, values: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner
            .pointwise_log_likelihood(&values, &mut Workspace::default())
            .map_err(|error| value_error(error.to_string()))
    }

    /// Fit replicates of the data drawn from the observation models at the
    /// fit (parametric bootstrap), from the fit
    ///
    /// Args:
    ///     config: The optimizer
    ///     fit: The fitted values, in problem order
    ///     replicates: Number of replicates
    ///     starts: Starts per replicate, the fit first
    ///     seed: Seed of the replicates
    ///
    /// Raises:
    ///     ValueError: If the problem has a loss or the fit is not feasible
    ///     RuntimeError: If every replicate fails
    fn parametric_bootstrap(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        fit: Vec<f64>,
        replicates: usize,
        starts: usize,
        seed: u64,
    ) -> PyResult<PyBootstrapRun> {
        self.run_bootstrap(py, config, &fit, replicates, starts, seed, |fit| {
            let resampler = ParametricResampler::new(&self.inner, fit).map_err(value_error)?;
            Ok((Box::new(resampler), None))
        })
    }

    /// Fit replicates of the data reweighted by blocks (Bayesian
    /// bootstrap), from the fit
    ///
    /// Args:
    ///     config: The optimizer
    ///     fit: The fitted values, in problem order
    ///     replicates: Number of replicates
    ///     block_length: Positions per block, or None for the automatic
    ///         length
    ///     starts: Starts per replicate, the fit first
    ///     seed: Seed of the replicates
    ///
    /// Raises:
    ///     ValueError: If the block length does not fit the positions
    ///     RuntimeError: If every replicate fails
    #[allow(clippy::too_many_arguments)]
    fn bayesian_bootstrap(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        fit: Vec<f64>,
        replicates: usize,
        block_length: Option<usize>,
        starts: usize,
        seed: u64,
    ) -> PyResult<PyBootstrapRun> {
        let observed = self.inner.observed_data();
        self.run_bootstrap(py, config, &fit, replicates, starts, seed, |fit| {
            let residuals =
                Residuals::new(&self.inner, fit, ResidualScale::Additive).map_err(value_error)?;
            let chosen = BlockLength::resolve(
                observed,
                &residuals,
                block_length,
                BlockDrawing::NonOverlapping,
            );
            let scheme = BlockScheme::new(observed, chosen.length).map_err(value_error)?;
            Ok((
                Box::new(BayesianBootstrap::new(observed, &scheme)),
                Some(chosen),
            ))
        })
    }

    /// Fit replicates of the fit plus blocks of its residuals, from the fit
    ///
    /// Args:
    ///     config: The optimizer
    ///     fit: The fitted values, in problem order
    ///     replicates: Number of replicates
    ///     block_length: Positions per block, or None for the automatic
    ///         length
    ///     starts: Starts per replicate, the fit first
    ///     seed: Seed of the replicates
    ///     residual_scale: "additive" or "multiplicative"
    ///     scheme: "moving" or "stationary"
    ///
    /// Raises:
    ///     ValueError: If a setting is not valid, the block length does not
    ///         fit the positions, or the residuals cannot be taken on their
    ///         scale
    ///     RuntimeError: If every replicate fails
    #[allow(clippy::too_many_arguments)]
    fn residual_bootstrap(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        fit: Vec<f64>,
        replicates: usize,
        block_length: Option<usize>,
        starts: usize,
        seed: u64,
        residual_scale: &str,
        scheme: &str,
    ) -> PyResult<PyBootstrapRun> {
        let observed = self.inner.observed_data();
        let scale = ResidualScale::from_name(residual_scale).map_err(value_error)?;
        let blocks = ResidualBlocks::from_name(scheme).map_err(value_error)?;
        let drawing = match blocks {
            ResidualBlocks::Moving => BlockDrawing::Moving,
            ResidualBlocks::Stationary => BlockDrawing::Stationary,
        };
        self.run_bootstrap(py, config, &fit, replicates, starts, seed, |fit| {
            let residuals = Residuals::new(&self.inner, fit, scale).map_err(value_error)?;
            let chosen = BlockLength::resolve(observed, &residuals, block_length, drawing);
            let block_scheme = BlockScheme::new(observed, chosen.length).map_err(value_error)?;
            Ok((
                Box::new(ResidualBootstrap::new(
                    observed,
                    residuals,
                    &block_scheme,
                    blocks,
                )),
                Some(chosen),
            ))
        })
    }

    /// Fit replicates of the fit plus its residuals times random
    /// multipliers shared by a block (wild bootstrap), from the fit
    ///
    /// Args:
    ///     config: The optimizer
    ///     fit: The fitted values, in problem order
    ///     replicates: Number of replicates
    ///     block_length: Positions per block, or None for the automatic
    ///         length
    ///     starts: Starts per replicate, the fit first
    ///     seed: Seed of the replicates
    ///     multipliers: "rademacher" or "mammen"
    ///
    /// Raises:
    ///     ValueError: If a setting is not valid or the block length does
    ///         not fit the positions
    ///     RuntimeError: If every replicate fails
    #[allow(clippy::too_many_arguments)]
    fn wild_bootstrap(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        fit: Vec<f64>,
        replicates: usize,
        block_length: Option<usize>,
        starts: usize,
        seed: u64,
        multipliers: &str,
    ) -> PyResult<PyBootstrapRun> {
        let observed = self.inner.observed_data();
        let multipliers = Multipliers::from_name(multipliers).map_err(value_error)?;
        self.run_bootstrap(py, config, &fit, replicates, starts, seed, |fit| {
            let residuals =
                Residuals::new(&self.inner, fit, ResidualScale::Additive).map_err(value_error)?;
            let chosen = BlockLength::resolve(
                observed,
                &residuals,
                block_length,
                BlockDrawing::NonOverlapping,
            );
            let scheme = BlockScheme::new(observed, chosen.length).map_err(value_error)?;
            Ok((
                Box::new(WildBootstrap::new(residuals, &scheme, multipliers)),
                Some(chosen),
            ))
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
        );
        let scheme = BlockScheme::new(observed, chosen.length).map_err(value_error)?;
        if scheme.count() < 2 {
            return Err(value_error(format!(
                "block_length {} makes {} block; a jackknife needs at least two",
                scheme.length(),
                scheme.count()
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
pub(crate) struct BlockLength {
    pub(crate) length: usize,
    pub(crate) estimates: Option<Vec<(String, f64)>>,
}

impl BlockLength {
    pub(crate) fn resolve(
        observed: &[commol_calibration::ObservedDataPoint],
        residuals: &Residuals,
        requested: Option<usize>,
        drawing: BlockDrawing,
    ) -> Self {
        match requested {
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
        }
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
    /// The block length used, or None for a bootstrap with no blocks
    block_length: Option<usize>,
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

pub(crate) fn predictive_error(error: PredictiveError) -> PyErr {
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
    fn quantiles(&self, levels: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        self.inner.quantiles(&levels).map_err(value_error)
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
    pub(crate) inner: Predictive,
}

#[pymethods]
impl PyPredictive {
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
    pub(crate) inner: ObservationDraws,
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
///     ValueError: If the lengths differ, no weight is positive or a level is
///         not in [0, 1]
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
    statistics::check_levels(&levels).map_err(value_error)?;
    Ok(statistics::quantiles(&values, &weights, &levels))
}

/// The weighted mean of `values`, one weight per value and some positive
#[pyfunction]
fn weighted_mean(values: Vec<f64>, weights: Vec<f64>) -> f64 {
    debug_assert!(
        values.len() == weights.len() && weights.iter().any(|w| *w > 0.0),
        "member_statistics checks the weights with weighted_quantiles first"
    );
    statistics::mean(&values, &weights)
}

/// The Monte Carlo error of the quantile at `level` of equally weighted
/// `values`: an order-statistic interval of about 95 %
#[pyfunction]
fn quantile_error(values: Vec<f64>, level: f64) -> (f64, f64) {
    debug_assert!(
        !values.is_empty() && (0.0..=1.0).contains(&level),
        "the package passes values and a level in [0, 1]"
    );
    statistics::quantile_error(&values, level)
}

/// The mean of `values` per series named by `labels`, in order of first
/// appearance
#[pyfunction]
fn by_series(labels: Vec<String>, values: Vec<f64>) -> Vec<(String, f64)> {
    debug_assert_eq!(
        labels.len(),
        values.len(),
        "the package passes a label per value"
    );
    scores::by_series(&labels, &values)
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
    crate::smc::register(m)?;
    crate::conformal::register(m)?;
    crate::sensitivity::register(m)?;
    Ok(())
}
