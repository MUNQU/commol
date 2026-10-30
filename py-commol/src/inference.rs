//! Python bindings of a built calibration problem and of the inference
//! methods that read it.
//!
//! [`PyCalibrationProblem`] is built once from the JSON of a
//! `CalibrationProblem`, with its optimizer and its seed, and passed to
//! every call, so Python never converts the problem twice and never
//! transforms a prediction: the observation operator, the replicates, the
//! starts and the fit of every method all run here. Each method takes the
//! problem, the JSON of its settings ([`commol_inference::settings`]) and
//! the handles of the results it reads.

use crate::parallelism::parallel;
use commol_calibration::{
    CalibrationProblem, EvaluationError, OptimizationConfig, ProblemSettings, Progress, Starts,
    Violation, Workspace, observed_data, optimize_starts, optimize_with_history, start_points,
};
use commol_inference::resampling::{
    BayesianBootstrap, BlockDrawing, BlockScheme, FitUnit, Multipliers, ParametricResampler,
    ReplicateRunner, Resampler, ResidualBlocks, ResidualBootstrap, ResidualNoise, ResidualScale,
    Residuals, WildBootstrap, bootstrap as run_replicates,
};
use commol_inference::settings::resampling::{BlockJackknifeSettings, Bootstrap, NoiseSettings};
use commol_inference::{
    DistributionNoise, ErrorKind, MethodError, NoNoise, ObservationDraws, ObservationNoise,
    ParameterSample, Predictive, PredictiveError, observation_draws, scores,
};
use commol_statistics::statistics;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::calibration::{PyCalibrationResultWithHistory, PyStarts};
use crate::engine::{Engine, InfeasibleStateError, PyEngine};

/// A calibration problem, built once, with its optimizer and its seed
#[pyclass(name = "CalibrationProblem", frozen)]
pub struct PyCalibrationProblem {
    pub(crate) inner: CalibrationProblem<Engine>,
    /// The optimizer of every fit.
    optimizer: OptimizationConfig,
    /// The progress table of each stage of the optimizer.
    progress: Progress,
    /// The seed of every random number of the calibration.
    pub(crate) seed: u64,
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
    /// How the bootstrap of `settings` draws its replicates.
    pub(crate) fn of(settings: &Bootstrap) -> Self {
        match settings {
            Bootstrap::Bayesian(settings) => Self::Bayesian {
                block_length: settings.block_length.0,
            },
            Bootstrap::Parametric(_) => Self::Parametric,
            Bootstrap::Residual(settings) => Self::Residual {
                block_length: settings.block_length.0,
                scale: settings.residual_scale,
                blocks: settings.scheme,
            },
            Bootstrap::Wild(settings) => Self::Wild {
                block_length: settings.block_length.0,
                multipliers: settings.multipliers,
            },
        }
    }
}

/// The values of the best start of `starts`, in problem order.
pub(crate) fn best_values(starts: &Starts) -> Vec<f64> {
    starts.best().best_parameters.clone()
}

/// The end of each start of `starts` that did not fail, in order.
pub(crate) fn start_ends(starts: &Starts) -> Vec<Vec<f64>> {
    starts
        .outcomes()
        .iter()
        .filter_map(|outcome| outcome.result.as_ref().ok())
        .map(|result| result.best_parameters.clone())
        .collect()
}

/// A list of names as Python writes it, `['a', 'b']`, for a message that
/// names them as Python does.
pub(crate) fn python_list(names: &[String]) -> String {
    let quoted: Vec<String> = names.iter().map(|name| format!("'{name}'")).collect();
    format!("[{}]", quoted.join(", "))
}

impl PyCalibrationProblem {
    /// The optimizer of every fit.
    pub(crate) fn optimizer(&self) -> &OptimizationConfig {
        &self.optimizer
    }

    /// The names of the parameters, in problem order.
    pub(crate) fn names(&self) -> Vec<String> {
        self.inner.parameter_names()
    }

    /// Fit the problem from `starts` start points in parallel with its
    /// optimizer and its seed, and keep every start.
    ///
    /// # Errors
    ///
    /// A `ValueError` when a verbose optimizer has more than one start, and
    /// a `RuntimeError` when every start fails.
    pub(crate) fn fit(&self, py: Python<'_>, starts: usize) -> PyResult<Starts> {
        if self.progress.any() && starts > 1 {
            return Err(value_error(
                "A verbose optimizer shows one start; set starts=1 or verbose=False".to_string(),
            ));
        }
        let points = start_points(self.inner.parameters(), starts, self.seed);
        parallel(py, || {
            if self.progress.any() {
                crate::python_observer::optimize_with_python_observer(
                    &self.inner,
                    &self.optimizer,
                    points,
                    self.seed,
                    self.progress,
                )
            } else {
                optimize_starts(&self.inner, &self.optimizer, points, self.seed)
            }
        })?
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// The runner of the refits of a method from `fit`, with `starts`
    /// starts per refit.
    ///
    /// # Errors
    ///
    /// As [`ReplicateRunner::new`].
    pub(crate) fn runner(
        &self,
        fit: &[f64],
        starts: usize,
    ) -> PyResult<ReplicateRunner<'_, Engine>> {
        ReplicateRunner::new(&self.inner, &self.optimizer, fit, starts, self.seed)
            .map_err(method_error)
    }

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

    /// The noise of `settings`, the JSON of an observation noise.
    fn noise(&self, settings: &str) -> PyResult<Box<dyn ObservationNoise>> {
        Ok(match crate::json::read::<NoiseSettings>(settings)? {
            NoiseSettings::None => Box::new(NoNoise),
            NoiseSettings::Distribution => {
                Box::new(DistributionNoise::new(&self.inner).map_err(value_error)?)
            }
            NoiseSettings::Residual(noise) => {
                let fit = noise.fit.values(&self.names()).map_err(value_error)?;
                let residuals = Residuals::new(&self.inner, &fit, noise.residual_scale)
                    .map_err(method_error)?;
                Box::new(
                    ResidualNoise::new(self.inner.observed_data(), &residuals, noise.block_length)
                        .map_err(method_error)?,
                )
            }
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
    ///     problem: The JSON of the `CalibrationProblem`: its observations,
    ///         parameters, constraints, loss or observation models,
    ///         optimizer, replicates and seed; a seed of None draws one,
    ///         which `seed` reports, and a stochastic engine needs it set
    ///
    /// Raises:
    ///     ValueError: If the JSON is not a problem, a setting is outside
    ///         its rule (the message names its path), the problem is not
    ///         valid for the engine, it has both or neither of a loss and
    ///         observation models, it has observation models and a
    ///         stochastic engine, or a stochastic engine lacks a setting of
    ///         its replicates
    #[new]
    fn new(engine: PyRef<'_, PyEngine>, problem: &str) -> PyResult<Self> {
        let settings: ProblemSettings = crate::json::read(problem)?;
        let optimizer = settings.optimization_config.config().map_err(value_error)?;
        let inner = settings.problem(engine.engine()).map_err(value_error)?;
        Ok(Self {
            inner,
            optimizer,
            progress: settings.optimization_config.progress(),
            seed: commol_random::seed_or_drawn(settings.seed),
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
    /// hypercube over the bounds. The seed of the problem gives the
    /// hypercube and the seed of each start of a particle swarm.
    ///
    /// Args:
    ///     starts: Number of starts, at least 1
    ///
    /// Returns:
    ///     The outcome of each start, in order, and the index of the best
    ///
    /// Raises:
    ///     ValueError: If `starts` is below 1, or a verbose optimizer has
    ///         more than one start
    ///     RuntimeError: If every start fails; the message gives the reason
    ///         of each
    fn optimize(&self, py: Python<'_>, starts: i64) -> PyResult<PyStarts> {
        let starts: usize = crate::counts::count(&commol_calibration::ranges::STARTS, starts)?;
        self.fit(py, starts).map(|inner| PyStarts { inner })
    }

    /// Optimize from the initial guess, as start 0 of `optimize`, and keep
    /// every evaluation of a feasible point
    ///
    /// Raises:
    ///     RuntimeError: If the optimization fails
    fn optimize_with_history(&self, py: Python<'_>) -> PyResult<PyCalibrationResultWithHistory> {
        parallel(py, || {
            optimize_with_history(&self.inner, &self.optimizer, self.seed)
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
    /// replicate of a stochastic engine seeded from the seed of the problem
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
    ) -> PyResult<PyPredictive> {
        parallel(py, || {
            Predictive::new(self.inner.forward(), &sample.inner, &positions, self.seed)
        })?
        .map(|inner| PyPredictive { inner })
        .map_err(method_error)
    }

    /// Every draw's value of `observations`, read through the observation
    /// operator, with observation noise drawn from the seed of the problem
    ///
    /// Args:
    ///     sample: The draws
    ///     observations: The JSON of the observations, which need not be the
    ///         problem's
    ///     noise: The JSON of the noise: `NoNoise`, `ResidualNoise` or
    ///         `DistributionNoise`
    ///
    /// Raises:
    ///     ValueError: If the sample's parameters are not the problem's, an
    ///         observation names an unknown output or factor, a series the
    ///         fit has no residuals of or a series with no observation
    ///         model, or the noise is not valid
    ///     InfeasibleStateError: If a draw leaves the values its model allows
    fn observation_draws(
        &self,
        py: Python<'_>,
        sample: &PyParameterSample,
        observations: &str,
        noise: &str,
    ) -> PyResult<PyObservationDraws> {
        let observations = observed_data(&crate::json::read::<Vec<_>>(observations)?);
        let noise = self.noise(noise)?;
        parallel(py, || {
            observation_draws(
                self.inner.forward(),
                &sample.inner,
                &observations,
                noise.as_ref(),
                self.seed,
            )
        })?
        .map(|inner| PyObservationDraws { inner })
        .map_err(method_error)
    }

    /// Check `observations` against the problem, with no run: the positions
    /// the engine accepts and what the observation operator reads
    /// (`ForwardModel::with_observations`)
    ///
    /// Args:
    ///     observations: The JSON of the observations
    ///
    /// Raises:
    ///     ValueError: If the engine rejects a position, or an observation
    ///         names an unknown output or factor, or a parameter that is not
    ///         a factor
    fn check_observations(&self, observations: &str) -> PyResult<()> {
        let observations = observed_data(&crate::json::read::<Vec<_>>(observations)?);
        self.inner
            .forward()
            .with_observations(&observations)
            .map(|_| ())
            .map_err(value_error)
    }

    /// The log score of each of `observations` under the predictive of
    /// `sample` and the observation models of the problem, each replicate of
    /// a stochastic engine seeded from the seed of the problem
    ///
    /// Args:
    ///     sample: The draws
    ///     observations: The JSON of the observations
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
        observations: &str,
    ) -> PyResult<Vec<f64>> {
        let observations = observed_data(&crate::json::read::<Vec<_>>(observations)?);
        let noise = DistributionNoise::new(&self.inner).map_err(value_error)?;
        parallel(py, || {
            scores::log_score(&self.inner, &sample.inner, &observations, &noise, self.seed)
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
}

/// Fit the problem, then fit replicates of the data a bootstrap draws
/// around the fit
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `BayesianBootstrap`, `ParametricBootstrap`,
///         `ResidualBootstrap` or `WildBootstrap`, by its `kind`
///
/// Returns:
///     The starts of the fit, and the replicates
///
/// Raises:
///     ValueError: If a setting is not valid, the problem does not fit the
///         bootstrap (a parametric bootstrap needs observation models), the
///         fit is not feasible, the block length does not fit the positions,
///         or the residuals cannot be taken on their scale
///     RuntimeError: If the fit, the run at the fit or every replicate fails
#[pyfunction]
fn bootstrap(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(PyStarts, PyBootstrapRun)> {
    let settings: Bootstrap = crate::json::read(settings)?;
    let starts = problem.fit(py, settings.starts())?;
    let runner = problem.runner(&best_values(&starts), settings.starts())?;
    let (resampler, scheme) = problem.resampler(&ResamplerSpec::of(&settings), runner.fit())?;
    let run = parallel(py, || {
        run_replicates(&runner, resampler.as_ref(), settings.replicates())
    })?
    .map_err(method_error)?;
    let run = PyBootstrapRun {
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
    };
    Ok((PyStarts { inner: starts }, run))
}

/// Fit the problem, then fit it once per block left out
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `BlockJackknife`
///
/// Returns:
///     The starts of the fit, and the jackknife
///
/// Raises:
///     ValueError: If a setting is not valid, or the block length does not
///         fit the positions or makes fewer than two blocks
///     RuntimeError: If the fit fails or fewer than two refits succeed
#[pyfunction]
fn block_jackknife(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(PyStarts, PyJackknifeRun)> {
    let settings: BlockJackknifeSettings = crate::json::read(settings)?;
    let starts = problem.fit(py, settings.starts)?;
    let runner = problem.runner(&best_values(&starts), settings.starts)?;
    let scheme = BlockScheme::at_fit(
        &problem.inner,
        runner.fit(),
        settings.block_length.0,
        BlockDrawing::NonOverlapping,
    )
    .map_err(method_error)?;
    let jackknife = parallel(py, || {
        commol_inference::resampling::block_jackknife(&runner, &scheme)
    })?
    .map_err(method_error)?;
    let spans = scheme.spans();
    let run = PyJackknifeRun {
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
    };
    Ok((PyStarts { inner: starts }, run))
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
    m.add_function(wrap_pyfunction!(bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(block_jackknife, m)?)?;
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
