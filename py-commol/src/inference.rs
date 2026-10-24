//! Python bindings of a built calibration problem and of the inference
//! methods that read it.
//!
//! [`PyCalibrationProblem`] is built once from a `Calibrator` and passed to
//! every call, so Python never converts the problem twice and never
//! transforms a prediction: the observation operator, the replicates and the
//! starts all run here.

use crate::parallelism::parallel;
use commol_calibration::{
    CalibrationProblem, DiscrepancyConfig, EvaluationError, Violation, Workspace, optimize_starts,
    optimize_with_history, start_points,
};
use commol_inference::resampling::{
    BayesianBootstrap, BlockDrawing, BlockScheme, FitUnit, Multipliers, ParametricResampler,
    ReplicateRunner, Resampler, ResidualBlocks, ResidualBootstrap, ResidualNoise, ResidualScale,
    Residuals, WildBootstrap, block_jackknife, bootstrap,
};
use commol_inference::{
    DistributionNoise, ErrorKind, MethodError, NoNoise, ObservationDraws, ObservationNoise,
    ParameterSample, Predictive, PredictiveError, observation_draws, scores,
};
use commol_statistics::statistics;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::calibration::{
    PyCalibrationConstraint, PyCalibrationParameter, PyCalibrationResultWithHistory, PyLossConfig,
    PyObservationModel, PyObservedDataPoint, PyOptimizationConfig, PyStarts,
};
use crate::engine::{Engine, InfeasibleStateError, PyEngine};

/// A calibration problem, built once
#[pyclass(name = "CalibrationProblem", frozen)]
pub struct PyCalibrationProblem {
    pub(crate) inner: CalibrationProblem<Engine>,
    seed: u64,
}

/// The Python error of a parameter set that has no prediction:
/// `InfeasibleStateError` when its starting state or its run leaves the
/// values its model allows, with the message of the plan of values or of
/// the run; `ValueError` for a value that is not finite; `RuntimeError`
/// otherwise.
fn prediction_error(index: usize, error: EvaluationError) -> PyErr {
    match error {
        EvaluationError::Violation(Violation::NonFinite) => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Parameter set {index} gives a value that is not finite"
            ))
        }
        EvaluationError::Violation(
            Violation::InfeasibleStart { message, .. } | Violation::InfeasibleRun { message },
        ) => InfeasibleStateError::new_err(format!(
            "Parameter set {index} is not feasible: {message}"
        )),
        EvaluationError::Violation(
            Violation::OutOfBounds { .. }
            | Violation::Constraint { .. }
            | Violation::MeanOutsideModel { .. },
        ) => {
            // coverage: invariant, a run of the forward model returns only the violations above
            unreachable!(
                "a run of the forward model checks no bounds, no constraint and no observation \
                 model"
            )
        }
        EvaluationError::Failed(message) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Parameter set {index}: {message}"),
        ),
    }
}

/// How a bootstrap draws the data of its replicates.
pub(crate) enum ResamplerSpec {
    /// Weights by blocks (Bayesian bootstrap); `None` for the automatic
    /// length.
    Bayesian { block_length: Option<usize> },
    /// Values drawn from the observation models at the fit.
    Parametric,
    /// The fit plus blocks of its residuals.
    Residual {
        block_length: Option<usize>,
        scale: ResidualScale,
        blocks: ResidualBlocks,
    },
    /// The fit plus its residuals times multipliers shared by a block.
    Wild {
        block_length: Option<usize>,
        multipliers: Multipliers,
    },
}

impl ResamplerSpec {
    /// The residual bootstrap of a scale and a scheme by name.
    pub(crate) fn residual(
        block_length: Option<usize>,
        residual_scale: &str,
        scheme: &str,
    ) -> PyResult<Self> {
        Ok(Self::Residual {
            block_length,
            scale: ResidualScale::from_name(residual_scale).map_err(method_error)?,
            blocks: ResidualBlocks::from_name(scheme).map_err(method_error)?,
        })
    }

    /// The wild bootstrap of multipliers by name.
    pub(crate) fn wild(block_length: Option<usize>, multipliers: &str) -> PyResult<Self> {
        Ok(Self::Wild {
            block_length,
            multipliers: Multipliers::from_name(multipliers).map_err(method_error)?,
        })
    }
}

impl PyCalibrationProblem {
    /// The resampler of `spec` around the fit `fit`, and its blocks, if it
    /// draws blocks.
    pub(crate) fn resampler(
        &self,
        spec: &ResamplerSpec,
        fit: &[f64],
    ) -> PyResult<(Box<dyn Resampler>, Option<BlockScheme>)> {
        let observed = self.inner.observed_data();
        let blocks = |scale: ResidualScale, requested: Option<usize>, drawing: BlockDrawing| {
            let residuals = Residuals::new(&self.inner, fit, scale).map_err(method_error)?;
            let scheme = BlockScheme::resolve(observed, residuals.residuals(), requested, drawing)
                .map_err(method_error)?;
            Ok::<_, PyErr>((residuals, scheme))
        };
        match spec {
            ResamplerSpec::Parametric => {
                let resampler = ParametricResampler::new(&self.inner, fit).map_err(method_error)?;
                Ok((Box::new(resampler), None))
            }
            ResamplerSpec::Bayesian { block_length } => {
                let scheme = BlockScheme::at_fit(
                    &self.inner,
                    fit,
                    *block_length,
                    BlockDrawing::NonOverlapping,
                )
                .map_err(method_error)?;
                Ok((
                    Box::new(BayesianBootstrap::new(observed, &scheme)),
                    Some(scheme),
                ))
            }
            ResamplerSpec::Residual {
                block_length,
                scale,
                blocks: residual_blocks,
            } => {
                let drawing = match residual_blocks {
                    ResidualBlocks::Moving => BlockDrawing::Moving,
                    ResidualBlocks::Stationary => BlockDrawing::Stationary,
                };
                let (residuals, scheme) = blocks(*scale, *block_length, drawing)?;
                Ok((
                    Box::new(ResidualBootstrap::new(
                        observed,
                        residuals,
                        &scheme,
                        *residual_blocks,
                    )),
                    Some(scheme),
                ))
            }
            ResamplerSpec::Wild {
                block_length,
                multipliers,
            } => {
                let (residuals, scheme) = blocks(
                    ResidualScale::Additive,
                    *block_length,
                    BlockDrawing::NonOverlapping,
                )?;
                Ok((
                    Box::new(WildBootstrap::new(residuals, &scheme, *multipliers)),
                    Some(scheme),
                ))
            }
        }
    }

    /// Run `replicates` of the resampler that `resampler` makes from the fit,
    /// and the block length of its blocks, if it draws blocks.
    #[allow(clippy::too_many_arguments)]
    fn run_bootstrap(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        fit: &[f64],
        replicates: usize,
        starts: usize,
        seed: u64,
        resampler: impl FnOnce(&[f64]) -> PyResult<(Box<dyn Resampler>, Option<BlockScheme>)>,
    ) -> PyResult<PyBootstrapRun> {
        let runner = ReplicateRunner::new(&self.inner, &config.inner, fit, starts, seed)
            .map_err(method_error)?;
        let (resampler, scheme) = resampler(runner.fit())?;
        let run = parallel(py, || bootstrap(&runner, resampler.as_ref(), replicates))?
            .map_err(method_error)?;
        Ok(PyBootstrapRun {
            values: run.sample.values().to_vec(),
            groups: run
                .sample
                .groups()
                .map(<[usize]>::to_vec)
                .unwrap_or_default(),
            failed: FitUnit::Replicate.entries(&run.failed),
            not_converged: run.not_converged,
            negative_values: run.negative_values,
            block_length: scheme.as_ref().map(BlockScheme::length),
            block_length_estimates: scheme
                .as_ref()
                .and_then(BlockScheme::estimates)
                .map(<[_]>::to_vec),
        })
    }
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
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
    ///     seed: The seed of every random number of the calibration; None
    ///         draws one, which `seed` reports, and a stochastic engine needs
    ///         it set
    ///
    /// Raises:
    ///     ValueError: If the problem is not valid for the engine, it has
    ///         both or neither of a loss and observation models, it has
    ///         observation models and a stochastic engine, or a stochastic
    ///         engine lacks a setting of its replicates
    #[new]
    #[pyo3(signature = (engine, observed_data, parameters, constraints, loss_config=None, initial_condition_remainder=None, observation_models=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        engine: PyRef<'_, PyEngine>,
        observed_data: Vec<PyObservedDataPoint>,
        parameters: Vec<PyCalibrationParameter>,
        constraints: Vec<PyCalibrationConstraint>,
        loss_config: Option<PyRef<'_, PyLossConfig>>,
        initial_condition_remainder: Option<String>,
        observation_models: Option<Vec<(String, PyObservationModel)>>,
        seed: Option<i128>,
    ) -> PyResult<Self> {
        let seed: Option<u64> = crate::counts::optional_count(&commol_core::SEED, seed)?;
        let engine = engine.engine();
        let discrepancy = match (&loss_config, observation_models) {
            (Some(loss), None) => DiscrepancyConfig::Loss(loss.inner),
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
                .and_then(|problem| {
                    problem.with_replicate_settings(loss.replicates, seed, loss.summary)
                })
                .map_err(value_error)?;
        }
        if let Some(remainder) = initial_condition_remainder {
            problem = problem
                .with_initial_condition_remainder(&remainder)
                .map_err(value_error)?;
        }
        Ok(Self {
            inner: problem,
            seed: commol_random::seed_or_drawn(seed),
        })
    }

    /// The seed of every random number of the calibration: the seed given,
    /// or the one drawn when it was built
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Optimize from `starts` start points in parallel, and choose the best
    ///
    /// Start 0 is the initial guess; the others are rows of a Latin
    /// hypercube over the bounds. `seed` gives the hypercube and the seed of
    /// each start of a particle swarm.
    ///
    /// Args:
    ///     config: The optimizer
    ///     starts: Number of starts, at least 1 (the caller checks it)
    ///     seed: Seed of the starts
    ///
    /// Returns:
    ///     The outcome of each start, in order, and the index of the best
    ///
    /// Raises:
    ///     ValueError: If a verbose optimizer has more than one start
    ///     RuntimeError: If every start fails; the message gives the reason
    ///         of each
    fn optimize(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        starts: i64,
        seed: i128,
    ) -> PyResult<PyStarts> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
        let starts: usize = crate::counts::count(&commol_calibration::ranges::STARTS, starts)?;
        if config.verbose && starts > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "A verbose optimizer shows one start; set starts=1 or verbose=False",
            ));
        }
        let points = start_points(self.inner.parameters(), starts, seed);
        parallel(py, || {
            if config.verbose {
                crate::python_observer::optimize_with_python_observer(
                    &self.inner,
                    &config.inner,
                    points,
                    seed,
                    config.header_interval,
                )
            } else {
                optimize_starts(&self.inner, &config.inner, points, seed)
            }
        })?
        .map(|inner| PyStarts { inner })
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Optimize from the initial guess, as start 0 of `optimize` with
    /// `seed`, and keep every evaluation of a feasible point
    ///
    /// Raises:
    ///     RuntimeError: If the optimization fails
    fn optimize_with_history(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        seed: i128,
    ) -> PyResult<PyCalibrationResultWithHistory> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
        parallel(py, || {
            optimize_with_history(&self.inner, &config.inner, seed)
        })?
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
        parallel(py, || {
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
        })?
        .map_err(|(index, error)| prediction_error(index, error))
    }

    /// The run of every draw of `sample` at `positions`, in parallel, each
    /// replicate of a stochastic engine seeded from `seed`
    ///
    /// Raises:
    ///     ValueError: Before any draw runs, if the sample's parameters are
    ///         not the problem's, or the positions are not sorted and unique
    ///         or one is a position the engine rejects
    ///     InfeasibleStateError: If a draw leaves the values its model allows
    fn predictive(
        &self,
        py: Python<'_>,
        sample: &PyParameterSample,
        positions: Vec<f64>,
        seed: i128,
    ) -> PyResult<PyPredictive> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
        parallel(py, || {
            Predictive::new(self.inner.forward(), &sample.inner, &positions, seed)
        })?
        .map(|inner| PyPredictive { inner })
        .map_err(method_error)
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
        seed: i128,
        residual_noise: Option<(Vec<f64>, i64, String)>,
        distribution_noise: bool,
    ) -> PyResult<PyObservationDraws> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
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
                let length = crate::counts::count(&commol_inference::ranges::BLOCK_LENGTH, length)?;
                let scale = ResidualScale::from_name(&scale).map_err(method_error)?;
                let residuals = Residuals::new(&self.inner, &fit, scale).map_err(method_error)?;
                Box::new(
                    ResidualNoise::new(self.inner.observed_data(), &residuals, length)
                        .map_err(method_error)?,
                )
            }
        };
        parallel(py, || {
            observation_draws(
                self.inner.forward(),
                &sample.inner,
                &observations,
                noise.as_ref(),
                seed,
            )
        })?
        .map(|inner| PyObservationDraws { inner })
        .map_err(method_error)
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
        seed: i128,
    ) -> PyResult<Vec<f64>> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
        let observations: Vec<_> = observations.into_iter().map(|o| o.inner).collect();
        let noise = DistributionNoise::new(&self.inner).map_err(value_error)?;
        parallel(py, || {
            scores::log_score(&self.inner, &sample.inner, &observations, &noise, seed)
        })?
        .map_err(method_error)
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
    ///     RuntimeError: If the run at the fit fails or every replicate fails
    fn parametric_bootstrap(
        &self,
        py: Python<'_>,
        config: &PyOptimizationConfig,
        fit: Vec<f64>,
        replicates: i64,
        starts: i64,
        seed: i128,
    ) -> PyResult<PyBootstrapRun> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
        let replicates: usize =
            crate::counts::count(&commol_inference::ranges::REPLICATES, replicates)?;
        let starts: usize = crate::counts::count(&commol_calibration::ranges::STARTS, starts)?;
        self.run_bootstrap(py, config, &fit, replicates, starts, seed, |fit| {
            self.resampler(&ResamplerSpec::Parametric, fit)
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
        replicates: i64,
        block_length: Option<i64>,
        starts: i64,
        seed: i128,
    ) -> PyResult<PyBootstrapRun> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
        let replicates: usize =
            crate::counts::count(&commol_inference::ranges::REPLICATES, replicates)?;
        let block_length: Option<usize> =
            crate::counts::optional_count(&commol_inference::ranges::BLOCK_LENGTH, block_length)?;
        let starts: usize = crate::counts::count(&commol_calibration::ranges::STARTS, starts)?;
        self.run_bootstrap(py, config, &fit, replicates, starts, seed, |fit| {
            self.resampler(&ResamplerSpec::Bayesian { block_length }, fit)
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
        replicates: i64,
        block_length: Option<i64>,
        starts: i64,
        seed: i128,
        residual_scale: &str,
        scheme: &str,
    ) -> PyResult<PyBootstrapRun> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
        let replicates: usize =
            crate::counts::count(&commol_inference::ranges::REPLICATES, replicates)?;
        let block_length: Option<usize> =
            crate::counts::optional_count(&commol_inference::ranges::BLOCK_LENGTH, block_length)?;
        let starts: usize = crate::counts::count(&commol_calibration::ranges::STARTS, starts)?;
        let spec = ResamplerSpec::residual(block_length, residual_scale, scheme)?;
        self.run_bootstrap(py, config, &fit, replicates, starts, seed, |fit| {
            self.resampler(&spec, fit)
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
        replicates: i64,
        block_length: Option<i64>,
        starts: i64,
        seed: i128,
        multipliers: &str,
    ) -> PyResult<PyBootstrapRun> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
        let replicates: usize =
            crate::counts::count(&commol_inference::ranges::REPLICATES, replicates)?;
        let block_length: Option<usize> =
            crate::counts::optional_count(&commol_inference::ranges::BLOCK_LENGTH, block_length)?;
        let starts: usize = crate::counts::count(&commol_calibration::ranges::STARTS, starts)?;
        let spec = ResamplerSpec::wild(block_length, multipliers)?;
        self.run_bootstrap(py, config, &fit, replicates, starts, seed, |fit| {
            self.resampler(&spec, fit)
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
        block_length: Option<i64>,
        starts: i64,
        seed: i128,
    ) -> PyResult<PyJackknifeRun> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
        let block_length: Option<usize> =
            crate::counts::optional_count(&commol_inference::ranges::BLOCK_LENGTH, block_length)?;
        let starts: usize = crate::counts::count(&commol_calibration::ranges::STARTS, starts)?;
        let runner = ReplicateRunner::new(&self.inner, &config.inner, &fit, starts, seed)
            .map_err(method_error)?;
        let scheme = BlockScheme::at_fit(
            &self.inner,
            runner.fit(),
            block_length,
            BlockDrawing::NonOverlapping,
        )
        .map_err(method_error)?;
        let jackknife =
            parallel(py, || block_jackknife(&runner, &scheme))?.map_err(method_error)?;
        let spans = scheme.spans();
        Ok(PyJackknifeRun {
            spans: jackknife.blocks.iter().map(|&b| spans[b]).collect(),
            estimate: jackknife.statistics.estimate,
            bias: jackknife.statistics.bias,
            standard_error: jackknife.statistics.standard_error,
            pseudo_values: jackknife.statistics.pseudo_values,
            influence: jackknife.statistics.influence,
            failed: FitUnit::Block.entries(&jackknife.failed),
            not_converged: jackknife.not_converged,
            block_length: scheme.length(),
            block_length_estimates: scheme.estimates().map(<[_]>::to_vec),
        })
    }
}

/// The replicates of a bootstrap, as the runner returns them
#[pyclass(name = "BootstrapRun", frozen, get_all)]
pub struct PyBootstrapRun {
    /// One row per replicate that did not fail, in problem order
    values: Vec<Vec<f64>>,
    /// The replicate of each row
    groups: Vec<usize>,
    /// Each replicate that failed: "Replicate <index> failed: <reason>"
    failed: Vec<String>,
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
    /// Each block whose fit failed: "Block <index> failed: <reason>"
    failed: Vec<String>,
    /// Fits kept although their optimizer did not converge
    not_converged: usize,
    /// The block length used
    block_length: usize,
    /// The estimate of each series, with the automatic length
    block_length_estimates: Option<Vec<(String, f64)>>,
}

pub(crate) fn value_error(message: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message)
}

/// The Python error of the error of a method, with the message of Rust: a
/// draw that has no run as [`prediction_error`] says, `ValueError` for an
/// error of the kind [`ErrorKind::Invalid`], `RuntimeError` for one of the
/// kind [`ErrorKind::Failed`]. Every binding of a method maps its errors
/// here.
pub(crate) fn method_error(error: impl Into<MethodError>) -> PyErr {
    match error.into() {
        MethodError::Prediction(PredictiveError::Draw { draw, error }) => {
            prediction_error(draw, error)
        }
        error => match error.kind() {
            ErrorKind::Invalid => value_error(error.to_string()),
            ErrorKind::Failed => {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
            }
        },
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

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
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
    fn resample(&self, count: i64, seed: i128) -> PyResult<Vec<usize>> {
        let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
        let count: usize = crate::counts::count(&commol_inference::ranges::COUNT, count)?;
        Ok(self.inner.resample(count, seed))
    }
}

/// The run of every draw of a sample at a set of positions
#[pyclass(name = "Predictive", frozen)]
pub struct PyPredictive {
    pub(crate) inner: Predictive,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
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

    /// The seed of each draw of the sample: replicate `i` of draw `j` of a
    /// stochastic engine runs with `replicate_seed(draw_seeds[j], i)`
    #[getter]
    fn draw_seeds(&self) -> Vec<u64> {
        self.inner.draw_seeds().to_vec()
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

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
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

    /// The randomized PIT value of each observation, its ties randomized
    /// from the seed of the draws
    fn pit(&self, observed: Vec<f64>) -> PyResult<Vec<f64>> {
        scores::pit(&self.inner, &observed).map_err(value_error)
    }
}

/// One weight per value, and some weight positive.
fn check_weights(values: &[f64], weights: &[f64]) -> PyResult<()> {
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
    Ok(())
}

/// The weighted quantiles of `values` at `levels`
///
/// The inverse of the weighted empirical distribution, interpolated between
/// the midpoints of the cumulative weights; with equal weights it is NumPy's
/// `method="hazen"`.
///
/// Raises:
///     ValueError: If the lengths differ, no weight is positive, a level is
///         not in [0, 1] or a value is NaN
#[pyfunction]
fn weighted_quantiles(values: Vec<f64>, weights: Vec<f64>, levels: Vec<f64>) -> PyResult<Vec<f64>> {
    check_weights(&values, &weights)?;
    statistics::check_levels(&levels).map_err(value_error)?;
    statistics::quantiles(&values, &weights, &levels)
        .map_err(|error| value_error(error.to_string()))
}

/// The two quantiles of the central interval of `confidence_level`
///
/// The one check of a confidence level.
///
/// Raises:
///     ValueError: If `confidence_level` is not strictly between 0 and 1
#[pyfunction]
fn interval_levels(confidence_level: f64) -> PyResult<(f64, f64)> {
    commol_core::CONFIDENCE_LEVEL
        .check(confidence_level)
        .map_err(value_error)?;
    let [lower, upper] = statistics::interval_levels(confidence_level);
    Ok((lower, upper))
}

/// Check that every level of `levels` is a quantile level, from 0 to 1
///
/// The one check of a quantile level.
///
/// Raises:
///     ValueError: If a level is outside [0, 1]; the message gives it
#[pyfunction]
fn check_levels(levels: Vec<f64>) -> PyResult<()> {
    statistics::check_levels(&levels).map_err(value_error)
}

/// The weighted summary of one column of runs
#[pyclass(name = "WeightedSummary", frozen, get_all)]
pub struct PyWeightedSummary {
    /// The weighted mean
    mean: f64,
    /// The weighted median
    median: f64,
    /// The quantile `(1 - c)/2`
    lower: f64,
    /// The quantile `(1 + c)/2`
    upper: f64,
    /// The smallest value of positive weight
    minimum: f64,
    /// The largest value of positive weight
    maximum: f64,
}

/// The weighted summary of `values`: the mean, the median, the central
/// interval of `confidence_level`, and the smallest and largest values of
/// positive weight
///
/// The reduction of every band, with the one quantile of commol.
///
/// Raises:
///     ValueError: If the lengths differ, no weight is positive,
///         `confidence_level` is not strictly between 0 and 1 or a value is
///         NaN
#[pyfunction]
fn weighted_summary(
    values: Vec<f64>,
    weights: Vec<f64>,
    confidence_level: f64,
) -> PyResult<PyWeightedSummary> {
    commol_core::CONFIDENCE_LEVEL
        .check(confidence_level)
        .map_err(value_error)?;
    let levels = statistics::interval_levels(confidence_level);
    check_weights(&values, &weights)?;
    let summary = statistics::summary(&values, &weights, levels)
        .map_err(|error| value_error(error.to_string()))?;
    Ok(PyWeightedSummary {
        mean: summary.mean,
        median: summary.median,
        lower: summary.lower,
        upper: summary.upper,
        minimum: summary.minimum,
        maximum: summary.maximum,
    })
}

/// Kish's effective size of `weights`, `(Σw)²/Σw²`
#[pyfunction]
fn effective_size(weights: Vec<f64>) -> f64 {
    statistics::effective_size(&weights)
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
    m.add_function(wrap_pyfunction!(interval_levels, m)?)?;
    m.add_function(wrap_pyfunction!(check_levels, m)?)?;
    m.add_function(wrap_pyfunction!(by_series, m)?)?;
    m.add_class::<PyWeightedSummary>()?;
    m.add_function(wrap_pyfunction!(weighted_summary, m)?)?;
    m.add_function(wrap_pyfunction!(effective_size, m)?)?;
    crate::identifiability::register(m)?;
    crate::smc::register(m)?;
    crate::local::register(m)?;
    crate::conformal::register(m)?;
    crate::quantity::register(m)?;
    crate::sensitivity::register(m)?;
    crate::profiles::register(m)?;
    Ok(())
}
