//! Python bindings for commol-calibration (parameter optimization).

use pyo3::prelude::*;

use crate::engine::PyEngine;

/// Type of value being calibrated
// coverage: code that a PyO3 attribute generates: the __int__ and __repr__ of the enum, which no
// call uses
#[pyclass(name = "CalibrationParameterType", from_py_object)]
#[derive(Clone, Copy)]
pub enum PyCalibrationParameterType {
    Parameter,
    InitialCondition,
    Factor,
    Noise,
}

impl From<PyCalibrationParameterType> for commol_calibration::CalibrationParameterType {
    fn from(py_type: PyCalibrationParameterType) -> Self {
        match py_type {
            PyCalibrationParameterType::Parameter => {
                commol_calibration::CalibrationParameterType::Parameter
            }
            PyCalibrationParameterType::InitialCondition => {
                commol_calibration::CalibrationParameterType::InitialCondition
            }
            PyCalibrationParameterType::Factor => {
                commol_calibration::CalibrationParameterType::Factor
            }
            PyCalibrationParameterType::Noise => {
                commol_calibration::CalibrationParameterType::Noise
            }
        }
    }
}

/// Observed data point for calibration
#[pyclass(name = "ObservedDataPoint", from_py_object)]
#[derive(Clone)]
pub struct PyObservedDataPoint {
    pub inner: commol_calibration::ObservedDataPoint,
}

#[pymethods]
impl PyObservedDataPoint {
    /// Create a new observed data point
    ///
    /// Args:
    ///     at: Position of the observation on the axis
    ///     label: Name of the observed series
    ///     outputs: Outputs summed before comparison; at least one
    ///     value: Observed value
    ///     weight: Optional weight for this observation (default: 1.0)
    ///     factor_id: Optional factor parameter ID that multiplies the model output
    ///     window: Optional window length for a cumulative output
    ///     trials: Optional trials of a binomial observation model that
    ///         reads them from each observation
    #[new]
    #[pyo3(signature = (at, label, outputs, value, weight=None, factor_id=None, window=None, trials=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        at: f64,
        label: String,
        outputs: Vec<String>,
        value: f64,
        weight: Option<f64>,
        factor_id: Option<String>,
        window: Option<f64>,
        trials: Option<u64>,
    ) -> Self {
        let mut inner = commol_calibration::ObservedDataPoint::of_sum(at, label, outputs, value);
        inner.weight = weight.unwrap_or(1.0);
        inner.factor_id = factor_id;
        inner.window = window;
        inner.trials = trials;
        Self { inner }
    }
}

/// A noise argument of an observation model: a number, or the id of a noise
/// parameter.
#[derive(FromPyObject)]
pub enum PyArgument {
    Value(f64),
    Parameter(String),
}

impl From<PyArgument> for commol_calibration::Argument {
    fn from(argument: PyArgument) -> Self {
        match argument {
            PyArgument::Value(value) => Self::Value(value),
            PyArgument::Parameter(id) => Self::Parameter(id),
        }
    }
}

/// How the observed values of one series scatter around their predicted
/// mean
#[pyclass(name = "ObservationModel", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyObservationModel {
    pub inner: commol_calibration::ObservationModel,
}

#[pymethods]
impl PyObservationModel {
    /// Normal of mean the prediction and standard deviation `sd`
    #[staticmethod]
    fn normal(sd: PyArgument) -> Self {
        Self {
            inner: commol_calibration::ObservationModel::Normal { sd: sd.into() },
        }
    }

    /// Log-normal whose median, with `median`, or mean is the prediction
    #[staticmethod]
    fn log_normal(sigma: PyArgument, median: bool) -> Self {
        Self {
            inner: commol_calibration::ObservationModel::LogNormal {
                sigma: sigma.into(),
                center: if median {
                    commol_calibration::Center::Median
                } else {
                    commol_calibration::Center::Mean
                },
            },
        }
    }

    /// Student-t of center the prediction
    #[staticmethod]
    fn student_t(df: PyArgument, scale: PyArgument) -> Self {
        Self {
            inner: commol_calibration::ObservationModel::StudentT {
                df: df.into(),
                scale: scale.into(),
            },
        }
    }

    /// Poisson of mean the prediction
    #[staticmethod]
    fn poisson() -> Self {
        Self {
            inner: commol_calibration::ObservationModel::Poisson,
        }
    }

    /// Negative binomial of mean the prediction and variance `μ + φμ²`
    #[staticmethod]
    fn negative_binomial(overdispersion: PyArgument) -> Self {
        Self {
            inner: commol_calibration::ObservationModel::NegativeBinomial {
                overdispersion: overdispersion.into(),
            },
        }
    }

    /// Binomial of mean the prediction: `trials` for the series, or None to
    /// read the trials of each observation
    #[staticmethod]
    #[pyo3(signature = (trials=None))]
    fn binomial(trials: Option<u64>) -> Self {
        Self {
            inner: commol_calibration::ObservationModel::Binomial { trials },
        }
    }
}

/// Parameter to calibrate with bounds
#[pyclass(name = "CalibrationParameter", from_py_object)]
#[derive(Clone)]
pub struct PyCalibrationParameter {
    pub inner: commol_calibration::CalibrationParameter,
}

#[pymethods]
impl PyCalibrationParameter {
    /// Create a new calibration parameter
    ///
    /// Args:
    ///     id: Parameter identifier (parameter ID for parameters, bin ID for initial conditions)
    ///     parameter_type: Type of value being calibrated
    ///     min_bound: Minimum allowed value
    ///     max_bound: Maximum allowed value
    ///     initial_guess: Optional starting value for optimization
    #[new]
    #[pyo3(signature = (id, parameter_type, min_bound, max_bound, initial_guess=None))]
    fn new(
        id: String,
        parameter_type: PyCalibrationParameterType,
        min_bound: f64,
        max_bound: f64,
        initial_guess: Option<f64>,
    ) -> Self {
        Self {
            inner: if let Some(guess) = initial_guess {
                commol_calibration::CalibrationParameter::with_type_and_guess(
                    id,
                    parameter_type.into(),
                    min_bound,
                    max_bound,
                    guess,
                )
            } else {
                commol_calibration::CalibrationParameter::with_type(
                    id,
                    parameter_type.into(),
                    min_bound,
                    max_bound,
                )
            },
        }
    }
}

/// Calibration constraint defined as a mathematical expression
#[pyclass(name = "CalibrationConstraint", from_py_object)]
#[derive(Clone)]
pub struct PyCalibrationConstraint {
    pub inner: commol_calibration::CalibrationConstraint,
}

#[pymethods]
impl PyCalibrationConstraint {
    /// Create a new calibration constraint
    ///
    /// Args:
    ///     id: Unique identifier for this constraint
    ///     expression: Mathematical expression that must evaluate >= 0
    ///     weight: Penalty weight multiplier (default: 1.0)
    ///     at: Optional positions on the axis at which to evaluate the
    ///         constraint; it can then read output values
    #[new]
    #[pyo3(signature = (id, expression, weight=1.0, at=None))]
    fn new(id: String, expression: String, weight: f64, at: Option<Vec<f64>>) -> Self {
        let mut constraint =
            commol_calibration::CalibrationConstraint::new(id, expression).with_weight(weight);

        if let Some(positions) = at {
            constraint = constraint.with_positions(positions);
        }

        Self { inner: constraint }
    }
}

/// Loss function configuration
#[pyclass(name = "LossConfig", from_py_object)]
#[derive(Clone)]
pub struct PyLossConfig {
    pub inner: commol_calibration::LossConfig,
    /// Whether to apply per-series observation normalization to the metric.
    pub normalize_observations: bool,
    /// The replicates of each evaluation of a stochastic engine, or `None`.
    pub replicates: Option<commol_calibration::Replicates>,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyLossConfig {
    /// Sum of squared errors
    #[staticmethod]
    fn sse() -> Self {
        Self {
            inner: commol_calibration::LossConfig::SumSquaredError,
            normalize_observations: false,
            replicates: None,
        }
    }

    /// Root mean squared error
    #[staticmethod]
    fn rmse() -> Self {
        Self {
            inner: commol_calibration::LossConfig::RootMeanSquaredError,
            normalize_observations: false,
            replicates: None,
        }
    }

    /// Mean absolute error
    #[staticmethod]
    fn mae() -> Self {
        Self {
            inner: commol_calibration::LossConfig::MeanAbsoluteError,
            normalize_observations: false,
            replicates: None,
        }
    }

    /// Return a copy with per-series observation normalization enabled or disabled.
    ///
    /// Normalization is orthogonal to the metric: each residual is divided by the
    /// RMS of its series' observed values, so series of very different magnitudes
    /// contribute comparably to whichever error measurement is selected.
    #[pyo3(signature = (enabled=true))]
    fn normalized(&self, enabled: bool) -> Self {
        Self {
            normalize_observations: enabled,
            ..self.clone()
        }
    }

    /// Return a copy that runs replicates of a stochastic engine at each
    /// evaluation, with common random numbers
    ///
    /// Each evaluation runs `count` replicates, the same seeds at every
    /// evaluation (derived from `seed`), and compares the observations with
    /// their summary.
    ///
    /// Args:
    ///     count: How many replicates, at least 1
    ///     seed: The seed the replicate seeds derive from
    ///     summary: "mean", "median", "quantile" or "crps"
    ///     quantile: The level of a "quantile" summary, in [0, 1]
    ///
    /// Raises:
    ///     ValueError: If count is 0, the summary is unknown, or the quantile
    ///         level is missing or outside [0, 1]
    #[pyo3(signature = (count, seed, summary, quantile=None))]
    fn with_replicates(
        &self,
        count: usize,
        seed: u64,
        summary: &str,
        quantile: Option<f64>,
    ) -> PyResult<Self> {
        let summary = commol_calibration::ReplicateSummary::from_name(summary, quantile)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let replicates = commol_calibration::Replicates::new(count, seed, summary)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self {
            replicates: Some(replicates),
            ..self.clone()
        })
    }

    /// Whether per-series observation normalization is enabled.
    #[getter]
    fn normalize_observations(&self) -> bool {
        self.normalize_observations
    }
}

/// Nelder-Mead optimization configuration
#[pyclass(name = "NelderMeadConfig", from_py_object)]
#[derive(Clone)]
pub struct PyNelderMeadConfig {
    pub inner: commol_calibration::NelderMeadConfig,
    pub verbose: bool,
    pub header_interval: u64,
}

#[pymethods]
impl PyNelderMeadConfig {
    /// Create Nelder-Mead configuration
    ///
    /// Args:
    ///     max_iterations: Maximum number of iterations (default: 1000)
    ///     sd_tolerance: Convergence tolerance (default: 1e-6)
    ///     initial_simplex_size: Size of the initial simplex, as a fraction of each bound width (default: 0.1)
    ///     alpha: Reflection coefficient (default: None, uses argmin's default)
    ///     gamma: Expansion coefficient (default: None, uses argmin's default)
    ///     rho: Contraction coefficient (default: None, uses argmin's default)
    ///     sigma: Shrink coefficient (default: None, uses argmin's default)
    ///     verbose: Enable verbose output (default: false)
    ///     header_interval: Number of iterations between table header repeats (default: 100)
    #[new]
    #[pyo3(signature = (max_iterations=1000, sd_tolerance=1e-6, initial_simplex_size=0.1, alpha=None, gamma=None, rho=None, sigma=None, verbose=false, header_interval=100))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_iterations: u64,
        sd_tolerance: f64,
        initial_simplex_size: f64,
        alpha: Option<f64>,
        gamma: Option<f64>,
        rho: Option<f64>,
        sigma: Option<f64>,
        verbose: bool,
        header_interval: u64,
    ) -> Self {
        Self {
            inner: commol_calibration::NelderMeadConfig {
                max_iterations,
                sd_tolerance,
                initial_simplex_size,
                alpha,
                gamma,
                rho,
                sigma,
            },
            verbose,
            header_interval,
        }
    }
}

/// PSO Inertia Weight Strategy - Constant
#[pyclass(name = "PSOInertiaConstant", from_py_object)]
#[derive(Clone)]
pub struct PyPSOInertiaConstant {
    #[pyo3(get)]
    pub factor: f64,
}

#[pymethods]
impl PyPSOInertiaConstant {
    #[new]
    fn new(factor: f64) -> Self {
        Self { factor }
    }
}

/// PSO Inertia Weight Strategy - Chaotic
#[pyclass(name = "PSOInertiaChaotic", from_py_object)]
#[derive(Clone)]
pub struct PyPSOInertiaChaotic {
    #[pyo3(get)]
    pub w_min: f64,
    #[pyo3(get)]
    pub w_max: f64,
}

#[pymethods]
impl PyPSOInertiaChaotic {
    #[new]
    fn new(w_min: f64, w_max: f64) -> Self {
        Self { w_min, w_max }
    }
}

/// PSO Acceleration Strategy - Constant
#[pyclass(name = "PSOAccelerationConstant", from_py_object)]
#[derive(Clone)]
pub struct PyPSOAccelerationConstant {
    #[pyo3(get)]
    pub cognitive: f64,
    #[pyo3(get)]
    pub social: f64,
}

#[pymethods]
impl PyPSOAccelerationConstant {
    #[new]
    fn new(cognitive: f64, social: f64) -> Self {
        Self { cognitive, social }
    }
}

/// PSO Acceleration Strategy - Time Varying (TVAC)
#[pyclass(name = "PSOAccelerationTimeVarying", from_py_object)]
#[derive(Clone)]
pub struct PyPSOAccelerationTimeVarying {
    #[pyo3(get)]
    pub c1_initial: f64,
    #[pyo3(get)]
    pub c1_final: f64,
    #[pyo3(get)]
    pub c2_initial: f64,
    #[pyo3(get)]
    pub c2_final: f64,
}

#[pymethods]
impl PyPSOAccelerationTimeVarying {
    #[new]
    fn new(c1_initial: f64, c1_final: f64, c2_initial: f64, c2_final: f64) -> Self {
        Self {
            c1_initial,
            c1_final,
            c2_initial,
            c2_final,
        }
    }
}

/// PSO Mutation Configuration
#[pyclass(name = "PSOMutation", from_py_object)]
#[derive(Clone)]
pub struct PyPSOMutation {
    #[pyo3(get)]
    pub strategy: String,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub probability: f64,
    #[pyo3(get)]
    pub application: String,
}

#[pymethods]
impl PyPSOMutation {
    #[new]
    fn new(strategy: String, scale: f64, probability: f64, application: String) -> Self {
        Self {
            strategy,
            scale,
            probability,
            application,
        }
    }
}

/// PSO Velocity Configuration
#[pyclass(name = "PSOVelocity", from_py_object)]
#[derive(Clone)]
pub struct PyPSOVelocity {
    #[pyo3(get)]
    pub clamp_factor: Option<f64>,
    #[pyo3(get)]
    pub mutation_threshold: Option<f64>,
}

#[pymethods]
impl PyPSOVelocity {
    #[new]
    #[pyo3(signature = (clamp_factor=None, mutation_threshold=None))]
    fn new(clamp_factor: Option<f64>, mutation_threshold: Option<f64>) -> Self {
        Self {
            clamp_factor,
            mutation_threshold,
        }
    }
}

/// An inertia strategy of a swarm.
#[derive(FromPyObject)]
pub enum PyInertia<'py> {
    /// A constant weight.
    Constant(PyRef<'py, PyPSOInertiaConstant>),
    /// A chaotic weight.
    Chaotic(PyRef<'py, PyPSOInertiaChaotic>),
}

/// An acceleration strategy of a swarm.
#[derive(FromPyObject)]
pub enum PyAcceleration<'py> {
    /// Constant coefficients.
    Constant(PyRef<'py, PyPSOAccelerationConstant>),
    /// Time-varying coefficients.
    TimeVarying(PyRef<'py, PyPSOAccelerationTimeVarying>),
}

/// Particle Swarm Optimization configuration
#[pyclass(name = "ParticleSwarmConfig", from_py_object)]
#[derive(Clone)]
pub struct PyParticleSwarmConfig {
    pub inner: commol_calibration::ParticleSwarmConfig,
    pub verbose: bool,
    pub header_interval: u64,
}

#[pymethods]
impl PyParticleSwarmConfig {
    /// Create Particle Swarm Optimization configuration
    ///
    /// Args:
    ///     num_particles: Number of particles in the swarm (default: 20)
    ///     max_iterations: Maximum number of iterations (default: 1000)
    ///     verbose: Enable verbose output (default: false)
    ///     inertia: Optional inertia strategy (PSOInertiaConstant or PSOInertiaChaotic)
    ///     acceleration: Optional acceleration strategy (PSOAccelerationConstant or PSOAccelerationTimeVarying)
    ///     mutation: Optional mutation configuration (PSOMutation)
    ///     velocity: Optional velocity configuration (PSOVelocity)
    ///     initialization: Initialization strategy (default: "uniform")
    ///     seed: Random seed for reproducibility (default: None, uses system entropy)
    #[new]
    #[pyo3(signature = (num_particles=20, max_iterations=1000, verbose=false, inertia=None, acceleration=None, mutation=None, velocity=None, initialization="uniform", seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_particles: usize,
        max_iterations: u64,
        verbose: bool,
        inertia: Option<PyInertia>,
        acceleration: Option<PyAcceleration>,
        mutation: Option<PyRef<PyPSOMutation>>,
        velocity: Option<PyRef<PyPSOVelocity>>,
        initialization: &str,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let value_error = PyErr::new::<pyo3::exceptions::PyValueError, _>;
        let (mutation_strategy, mutation_probability, mutation_application) = match mutation {
            Some(mutation) => (
                commol_calibration::MutationStrategy::from_name(&mutation.strategy, mutation.scale)
                    .map_err(value_error)?,
                mutation.probability,
                commol_calibration::MutationApplication::from_name(&mutation.application)
                    .map_err(value_error)?,
            ),
            None => (
                commol_calibration::MutationStrategy::None,
                0.0,
                commol_calibration::MutationApplication::None,
            ),
        };
        let config = commol_calibration::ParticleSwarmConfig {
            num_particles,
            max_iterations,
            target_cost: None,
            inertia_strategy: inertia.map(|inertia| match inertia {
                PyInertia::Constant(constant) => {
                    commol_calibration::InertiaWeightStrategy::Constant(constant.factor)
                }
                PyInertia::Chaotic(chaotic) => commol_calibration::InertiaWeightStrategy::Chaotic {
                    w_min: chaotic.w_min,
                    w_max: chaotic.w_max,
                },
            }),
            acceleration_strategy: acceleration.map(|acceleration| match acceleration {
                PyAcceleration::Constant(constant) => {
                    commol_calibration::AccelerationStrategy::Constant {
                        cognitive: constant.cognitive,
                        social: constant.social,
                    }
                }
                PyAcceleration::TimeVarying(tvac) => {
                    commol_calibration::AccelerationStrategy::TimeVarying {
                        c1_initial: tvac.c1_initial,
                        c1_final: tvac.c1_final,
                        c2_initial: tvac.c2_initial,
                        c2_final: tvac.c2_final,
                    }
                }
            }),
            initialization_strategy: commol_calibration::InitializationStrategy::from_name(
                initialization,
            )
            .map_err(value_error)?,
            velocity_clamp_factor: velocity.as_ref().and_then(|velocity| velocity.clamp_factor),
            velocity_mutation_threshold: velocity
                .as_ref()
                .and_then(|velocity| velocity.mutation_threshold),
            mutation_strategy,
            mutation_probability,
            mutation_application,
            seed,
        };
        Ok(Self {
            inner: config,
            verbose,
            header_interval: 100,
        })
    }
}

/// Optimization algorithm configuration
#[pyclass(name = "OptimizationConfig", from_py_object)]
#[derive(Clone)]
pub struct PyOptimizationConfig {
    pub inner: commol_calibration::OptimizationConfig,
    /// Whether the optimization writes its progress to Python's stdout.
    pub verbose: bool,
    pub header_interval: u64,
}

#[pymethods]
impl PyOptimizationConfig {
    /// Create optimization config with Nelder-Mead algorithm
    #[staticmethod]
    fn nelder_mead(config: Option<PyNelderMeadConfig>) -> Self {
        let header_interval = config.as_ref().map(|c| c.header_interval).unwrap_or(100);
        let verbose = config.as_ref().is_some_and(|c| c.verbose);
        Self {
            inner: commol_calibration::OptimizationConfig::NelderMead(
                config.map(|c| c.inner).unwrap_or_default(),
            ),
            verbose,
            header_interval,
        }
    }

    /// Create optimization config with Particle Swarm algorithm
    #[staticmethod]
    fn particle_swarm(config: Option<PyParticleSwarmConfig>) -> Self {
        let header_interval = config.as_ref().map(|c| c.header_interval).unwrap_or(100);
        let verbose = config.as_ref().is_some_and(|c| c.verbose);
        Self {
            inner: commol_calibration::OptimizationConfig::ParticleSwarm(
                config.map(|c| c.inner).unwrap_or_default(),
            ),
            verbose,
            header_interval,
        }
    }
}

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

/// Initial state a compartmental engine starts from once calibration
/// values are applied
///
/// The values are applied exactly as the calibration objective scores them:
/// clamped to their bounds, with the remainder initial condition
/// auto-calculated, and written in parameter order.
///
/// Args:
///     engine: Engine built from a compartmental model, before calibration
///     parameters: Calibration parameters, in problem order
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
    parameters: Vec<PyCalibrationParameter>,
    values: Vec<f64>,
    initial_condition_remainder: Option<String>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let parameters: Vec<_> = parameters.into_iter().map(|p| p.inner).collect();
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

/// Check that the engine resolves every value of its model kind, such as an
/// initial condition, and the remainder
///
/// Args:
///     engine: The simulation engine
///     parameters: Calibration parameters, in problem order
///     initial_condition_remainder: The id that takes what the values leave
///         of the sum the model kind keeps (a compartment or bin of a
///         compartmental model), if any
///
/// Raises:
///     ValueError: If the kind does not accept a type of value, or a value
///         or the remainder cannot be resolved; for example a category of a
///         stratification that is not binary
#[pyfunction]
#[pyo3(signature = (engine, parameters, initial_condition_remainder=None))]
fn check_model_values(
    engine: PyRef<'_, PyEngine>,
    parameters: Vec<PyCalibrationParameter>,
    initial_condition_remainder: Option<String>,
) -> PyResult<()> {
    let parameters: Vec<_> = parameters.into_iter().map(|p| p.inner).collect();
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
///     parameters: Initial-condition calibration parameters, in problem order
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
    parameters: Vec<PyCalibrationParameter>,
    initial_condition_remainder: Option<String>,
) -> PyResult<(Vec<Vec<usize>>, Vec<usize>)> {
    let parameters: Vec<_> = parameters.into_iter().map(|p| p.inner).collect();
    commol_calibration::check_model_values(
        &engine.engine(),
        &parameters,
        initial_condition_remainder.as_deref(),
    )
    .and_then(|plan| crate::engine::compartmental_coverage(&plan))
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

/// Check every position of a calibration problem with the engine
///
/// Args:
///     engine: Engine built from the model definition
///     observed_data: Observations, at their positions and windows
///     constraints: Constraints, at their positions
///
/// Raises:
///     ValueError: If the engine does not report at a position; the message
///         names the observation or constraint that holds it
#[pyfunction]
fn check_positions(
    engine: PyRef<'_, PyEngine>,
    observed_data: Vec<PyObservedDataPoint>,
    constraints: Vec<PyCalibrationConstraint>,
) -> PyResult<()> {
    let observed_data: Vec<_> = observed_data.into_iter().map(|d| d.inner).collect();
    let constraints: Vec<_> = constraints.into_iter().map(|c| c.inner).collect();
    commol_calibration::check_positions(&engine.engine(), &observed_data, &constraints)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

/// Register calibration module with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCalibrationParameterType>()?;
    m.add_class::<PyObservedDataPoint>()?;
    m.add_class::<PyCalibrationParameter>()?;
    m.add_class::<PyCalibrationConstraint>()?;
    m.add_class::<PyLossConfig>()?;
    m.add_class::<PyObservationModel>()?;
    m.add_class::<PyNelderMeadConfig>()?;
    m.add_class::<PyPSOInertiaConstant>()?;
    m.add_class::<PyPSOInertiaChaotic>()?;
    m.add_class::<PyPSOAccelerationConstant>()?;
    m.add_class::<PyPSOAccelerationTimeVarying>()?;
    m.add_class::<PyPSOMutation>()?;
    m.add_class::<PyPSOVelocity>()?;
    m.add_class::<PyParticleSwarmConfig>()?;
    m.add_class::<PyOptimizationConfig>()?;
    m.add_class::<PyCalibrationEvaluation>()?;
    m.add_class::<PyCalibrationResultWithHistory>()?;
    m.add_class::<PyCalibrationResult>()?;
    m.add_class::<PyStartOutcome>()?;
    m.add_function(wrap_pyfunction!(calibrated_initial_state, m)?)?;
    m.add_function(wrap_pyfunction!(check_model_values, m)?)?;
    m.add_function(wrap_pyfunction!(initial_condition_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(check_positions, m)?)?;
    Ok(())
}
