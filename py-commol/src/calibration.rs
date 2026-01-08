//! Python bindings for epimodel-calibration (parameter optimization).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::difference::PyDifferenceEquations;

/// Type of value being calibrated
#[pyclass(name = "CalibrationParameterType")]
#[derive(Clone, Copy)]
pub enum PyCalibrationParameterType {
    Parameter,
    InitialCondition,
    Scale,
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
            PyCalibrationParameterType::Scale => {
                commol_calibration::CalibrationParameterType::Scale
            }
        }
    }
}

/// Observed data point for calibration
#[pyclass(name = "ObservedDataPoint")]
#[derive(Clone)]
pub struct PyObservedDataPoint {
    pub inner: commol_calibration::ObservedDataPoint,
}

#[pymethods]
impl PyObservedDataPoint {
    /// Create a new observed data point
    ///
    /// Args:
    ///     step: Step of observation
    ///     compartment: Name of the compartment being observed
    ///     value: Observed value
    ///     weight: Optional weight for this observation (default: 1.0)
    ///     scale_id: Optional scale parameter ID to apply to model output
    #[new]
    #[pyo3(signature = (step, compartment, value, weight=None, scale_id=None))]
    fn new(
        step: u32,
        compartment: String,
        value: f64,
        weight: Option<f64>,
        scale_id: Option<String>,
    ) -> Self {
        let inner = match (weight, scale_id) {
            (Some(w), Some(s)) => commol_calibration::ObservedDataPoint::with_weight_and_scale(
                step,
                compartment,
                value,
                w,
                s,
            ),
            (Some(w), None) => {
                commol_calibration::ObservedDataPoint::with_weight(step, compartment, value, w)
            }
            (None, Some(s)) => {
                commol_calibration::ObservedDataPoint::with_scale(step, compartment, value, s)
            }
            (None, None) => commol_calibration::ObservedDataPoint::new(step, compartment, value),
        };
        Self { inner }
    }

    fn __repr__(&self) -> String {
        format!(
            "ObservedDataPoint(time_step={}, compartment='{}', value={})",
            self.inner.time_step, self.inner.compartment, self.inner.value
        )
    }
}

/// Parameter to calibrate with bounds
#[pyclass(name = "CalibrationParameter")]
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

    fn __repr__(&self) -> String {
        format!(
            "CalibrationParameter(id='{}', type={:?}, bounds=[{}, {}])",
            self.inner.id, self.inner.parameter_type, self.inner.min_bound, self.inner.max_bound
        )
    }
}

/// Calibration constraint defined as a mathematical expression
#[pyclass(name = "CalibrationConstraint")]
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
    ///     description: Optional human-readable description
    ///     weight: Penalty weight multiplier (default: 1.0)
    ///     time_steps: Optional time steps at which to evaluate (for time-dependent constraints)
    #[new]
    #[pyo3(signature = (id, expression, description=None, weight=1.0, time_steps=None))]
    fn new(
        id: String,
        expression: String,
        description: Option<String>,
        weight: f64,
        time_steps: Option<Vec<u32>>,
    ) -> Self {
        let mut constraint = commol_calibration::CalibrationConstraint::new(id, expression);

        if let Some(desc) = description {
            constraint = constraint.with_description(desc);
        }

        constraint = constraint.with_weight(weight);

        if let Some(steps) = time_steps {
            constraint = constraint.with_time_steps(steps);
        }

        Self { inner: constraint }
    }

    fn __repr__(&self) -> String {
        if let Some(ref desc) = self.inner.description {
            format!(
                "CalibrationConstraint(id='{}', expression='{}', description='{}')",
                self.inner.id, self.inner.expression, desc
            )
        } else {
            format!(
                "CalibrationConstraint(id='{}', expression='{}')",
                self.inner.id, self.inner.expression
            )
        }
    }
}

/// Loss function configuration
#[pyclass(name = "LossConfig")]
#[derive(Clone)]
pub struct PyLossConfig {
    pub inner: commol_calibration::LossConfig,
}

#[pymethods]
impl PyLossConfig {
    /// Sum of squared errors
    #[staticmethod]
    fn sse() -> Self {
        Self {
            inner: commol_calibration::LossConfig::SumSquaredError,
        }
    }

    /// Root mean squared error
    #[staticmethod]
    fn rmse() -> Self {
        Self {
            inner: commol_calibration::LossConfig::RootMeanSquaredError,
        }
    }

    /// Mean absolute error
    #[staticmethod]
    fn mae() -> Self {
        Self {
            inner: commol_calibration::LossConfig::MeanAbsoluteError,
        }
    }

    /// Weighted sum of squared errors
    #[staticmethod]
    fn weighted_sse() -> Self {
        Self {
            inner: commol_calibration::LossConfig::WeightedSSE,
        }
    }
}

/// Nelder-Mead optimization configuration
#[pyclass(name = "NelderMeadConfig")]
#[derive(Clone)]
pub struct PyNelderMeadConfig {
    pub inner: commol_calibration::NelderMeadConfig,
    pub header_interval: u64,
}

#[pymethods]
impl PyNelderMeadConfig {
    /// Create Nelder-Mead configuration
    ///
    /// Args:
    ///     max_iterations: Maximum number of iterations (default: 1000)
    ///     sd_tolerance: Convergence tolerance (default: 1e-6)
    ///     simplex_perturbation: Multiplier for initial simplex vertices (default: 1.1 = 10% perturbation)
    ///     alpha: Reflection coefficient (default: None, uses argmin's default)
    ///     gamma: Expansion coefficient (default: None, uses argmin's default)
    ///     rho: Contraction coefficient (default: None, uses argmin's default)
    ///     sigma: Shrink coefficient (default: None, uses argmin's default)
    ///     verbose: Enable verbose output (default: false)
    ///     header_interval: Number of iterations between table header repeats (default: 100)
    #[new]
    #[pyo3(signature = (max_iterations=1000, sd_tolerance=1e-6, simplex_perturbation=1.1, alpha=None, gamma=None, rho=None, sigma=None, verbose=false, header_interval=100))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_iterations: u64,
        sd_tolerance: f64,
        simplex_perturbation: f64,
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
                simplex_perturbation,
                alpha,
                gamma,
                rho,
                sigma,
                verbose,
            },
            header_interval,
        }
    }

    /// Get the header interval
    #[getter]
    fn header_interval(&self) -> u64 {
        self.header_interval
    }
}

/// PSO Inertia Weight Strategy - Constant
#[pyclass(name = "PSOInertiaConstant")]
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
#[pyclass(name = "PSOInertiaChaotic")]
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
#[pyclass(name = "PSOAccelerationConstant")]
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
#[pyclass(name = "PSOAccelerationTimeVarying")]
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
#[pyclass(name = "PSOMutation")]
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
#[pyclass(name = "PSOVelocity")]
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

/// Particle Swarm Optimization configuration
#[pyclass(name = "ParticleSwarmConfig")]
#[derive(Clone)]
pub struct PyParticleSwarmConfig {
    pub inner: commol_calibration::ParticleSwarmConfig,
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
        inertia: Option<PyObject>,
        acceleration: Option<PyObject>,
        mutation: Option<Py<PyPSOMutation>>,
        velocity: Option<Py<PyPSOVelocity>>,
        initialization: &str,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            // Parse initialization strategy
            let init_strategy = match initialization {
                "uniform" | "uniform_random" => {
                    commol_calibration::InitializationStrategy::UniformRandom
                }
                "latin_hypercube" | "lhs" => {
                    commol_calibration::InitializationStrategy::LatinHypercube
                }
                "opposition_based" | "obl" => {
                    commol_calibration::InitializationStrategy::OppositionBased
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown initialization strategy: {}. Valid options: 'uniform', 'latin_hypercube', 'opposition_based'",
                        initialization
                    )));
                }
            };

            let mut config = commol_calibration::ParticleSwarmConfig {
                num_particles,
                max_iterations,
                target_cost: None,
                inertia_strategy: None,
                acceleration_strategy: None,
                initialization_strategy: init_strategy,
                velocity_clamp_factor: None,
                velocity_mutation_threshold: None,
                mutation_strategy: commol_calibration::MutationStrategy::None,
                mutation_probability: 0.0,
                mutation_application: commol_calibration::MutationApplication::None,
                seed,
                verbose,
            };

            // Handle inertia strategy
            if let Some(inertia_obj) = inertia {
                if let Ok(constant) = inertia_obj.extract::<PyRef<PyPSOInertiaConstant>>(py) {
                    config.inertia_strategy = Some(
                        commol_calibration::InertiaWeightStrategy::Constant(constant.factor),
                    );
                } else if let Ok(chaotic) = inertia_obj.extract::<PyRef<PyPSOInertiaChaotic>>(py) {
                    config.inertia_strategy =
                        Some(commol_calibration::InertiaWeightStrategy::Chaotic {
                            w_min: chaotic.w_min,
                            w_max: chaotic.w_max,
                        });
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "inertia must be PSOInertiaConstant or PSOInertiaChaotic",
                    ));
                }
            }

            // Handle acceleration strategy
            if let Some(accel_obj) = acceleration {
                if let Ok(constant) = accel_obj.extract::<PyRef<PyPSOAccelerationConstant>>(py) {
                    config.acceleration_strategy =
                        Some(commol_calibration::AccelerationStrategy::Constant {
                            cognitive: constant.cognitive,
                            social: constant.social,
                        });
                } else if let Ok(tvac) =
                    accel_obj.extract::<PyRef<PyPSOAccelerationTimeVarying>>(py)
                {
                    config.acceleration_strategy =
                        Some(commol_calibration::AccelerationStrategy::TimeVarying {
                            c1_initial: tvac.c1_initial,
                            c1_final: tvac.c1_final,
                            c2_initial: tvac.c2_initial,
                            c2_final: tvac.c2_final,
                        });
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "acceleration must be PSOAccelerationConstant or PSOAccelerationTimeVarying",
                    ));
                }
            }

            // Handle mutation
            if let Some(mut_ref) = mutation {
                let mut_obj = mut_ref.borrow(py);
                config.mutation_strategy = match mut_obj.strategy.as_str() {
                    "gaussian" => commol_calibration::MutationStrategy::Gaussian(mut_obj.scale),
                    "cauchy" => commol_calibration::MutationStrategy::Cauchy(mut_obj.scale),
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Invalid mutation strategy: {}",
                            mut_obj.strategy
                        )));
                    }
                };
                config.mutation_probability = mut_obj.probability;
                config.mutation_application = match mut_obj.application.as_str() {
                    "global_best" => commol_calibration::MutationApplication::GlobalBestOnly,
                    "all_particles" => commol_calibration::MutationApplication::AllParticles,
                    "below_average" => commol_calibration::MutationApplication::BelowAverage,
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Invalid mutation application: {}",
                            mut_obj.application
                        )));
                    }
                };
            }

            // Handle velocity
            if let Some(vel_ref) = velocity {
                let vel_obj = vel_ref.borrow(py);
                config.velocity_clamp_factor = vel_obj.clamp_factor;
                config.velocity_mutation_threshold = vel_obj.mutation_threshold;
            }

            Ok(Self {
                inner: config,
                header_interval: 100,
            })
        })
    }

    /// Get the header interval
    #[getter]
    fn header_interval(&self) -> u64 {
        self.header_interval
    }

    /// Set chaotic inertia weight using logistic map
    ///
    /// Args:
    ///     w_min: Minimum inertia weight
    ///     w_max: Maximum inertia weight
    fn set_chaotic_inertia(&mut self, w_min: f64, w_max: f64) {
        self.inner.inertia_strategy =
            Some(commol_calibration::InertiaWeightStrategy::Chaotic { w_min, w_max });
    }

    /// Enable Time-Varying Acceleration Coefficients (TVAC)
    ///
    /// Args:
    ///     c1_initial: Initial cognitive factor
    ///     c1_final: Final cognitive factor
    ///     c2_initial: Initial social factor
    ///     c2_final: Final social factor
    fn set_tvac(&mut self, c1_initial: f64, c1_final: f64, c2_initial: f64, c2_final: f64) {
        self.inner.acceleration_strategy =
            Some(commol_calibration::AccelerationStrategy::TimeVarying {
                c1_initial,
                c1_final,
                c2_initial,
                c2_final,
            });
    }

    /// Set initialization strategy
    ///
    /// Args:
    ///     strategy: One of "uniform" (default), "latin_hypercube", or "opposition_based"
    fn set_initialization_strategy(&mut self, strategy: &str) -> PyResult<()> {
        self.inner.initialization_strategy = match strategy {
            "uniform" | "uniform_random" => {
                commol_calibration::InitializationStrategy::UniformRandom
            }
            "latin_hypercube" | "lhs" => commol_calibration::InitializationStrategy::LatinHypercube,
            "opposition_based" | "obl" => {
                commol_calibration::InitializationStrategy::OppositionBased
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown initialization strategy: {}. Valid options: 'uniform', 'latin_hypercube', 'opposition_based'",
                    strategy
                )));
            }
        };
        Ok(())
    }

    /// Enable velocity clamping
    ///
    /// Args:
    ///     clamp_factor: Fraction of search space range (typically 0.1 to 0.2)
    fn set_velocity_clamping(&mut self, clamp_factor: f64) {
        self.inner.velocity_clamp_factor = Some(clamp_factor);
    }

    /// Enable velocity mutation when velocity approaches zero
    ///
    /// Args:
    ///     threshold: Velocity threshold below which reinitialization occurs (typically 0.001 to 0.01)
    fn set_velocity_mutation(&mut self, threshold: f64) {
        self.inner.velocity_mutation_threshold = Some(threshold);
    }

    /// Enable mutation to help escape local optima
    ///
    /// Args:
    ///     strategy: Either "gaussian" or "cauchy"
    ///     scale: Standard deviation (for gaussian) or scale parameter (for cauchy)
    ///     probability: Mutation probability per iteration (0.0 to 1.0)
    ///     application: One of "global_best" (default), "all_particles", or "below_average"
    #[pyo3(signature = (strategy, scale, probability, application="global_best"))]
    fn set_mutation(
        &mut self,
        strategy: &str,
        scale: f64,
        probability: f64,
        application: &str,
    ) -> PyResult<()> {
        self.inner.mutation_strategy = match strategy {
            "gaussian" => commol_calibration::MutationStrategy::Gaussian(scale),
            "cauchy" => commol_calibration::MutationStrategy::Cauchy(scale),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown mutation strategy: {}. Valid options: 'gaussian', 'cauchy'",
                    strategy
                )));
            }
        };

        self.inner.mutation_probability = probability;

        self.inner.mutation_application = match application {
            "global_best" | "global_best_only" => {
                commol_calibration::MutationApplication::GlobalBestOnly
            }
            "all" | "all_particles" => commol_calibration::MutationApplication::AllParticles,
            "below_average" => commol_calibration::MutationApplication::BelowAverage,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown mutation application: {}. Valid options: 'global_best', 'all_particles', 'below_average'",
                    application
                )));
            }
        };

        Ok(())
    }
}

/// Optimization algorithm configuration
#[pyclass(name = "OptimizationConfig")]
#[derive(Clone)]
pub struct PyOptimizationConfig {
    pub inner: commol_calibration::OptimizationConfig,
    pub header_interval: u64,
}

#[pymethods]
impl PyOptimizationConfig {
    /// Create optimization config with Nelder-Mead algorithm
    #[staticmethod]
    fn nelder_mead(config: Option<PyNelderMeadConfig>) -> Self {
        let header_interval = config.as_ref().map(|c| c.header_interval).unwrap_or(100);
        Self {
            inner: commol_calibration::OptimizationConfig::NelderMead(
                config.map(|c| c.inner).unwrap_or_default(),
            ),
            header_interval,
        }
    }

    /// Create optimization config with Particle Swarm algorithm
    #[staticmethod]
    fn particle_swarm(config: Option<PyParticleSwarmConfig>) -> Self {
        let header_interval = config.as_ref().map(|c| c.header_interval).unwrap_or(100);
        Self {
            inner: commol_calibration::OptimizationConfig::ParticleSwarm(
                config.map(|c| c.inner).unwrap_or_default(),
            ),
            header_interval,
        }
    }
}

/// Information about a Pareto front solution
#[pyclass(name = "ParetoSolution")]
#[derive(Clone)]
pub struct PyParetoSolution {
    pub inner: commol_calibration::ParetoSolution,
}

#[pymethods]
impl PyParetoSolution {
    #[getter]
    fn ensemble_size(&self) -> usize {
        self.inner.ensemble_size
    }

    #[getter]
    fn ci_width(&self) -> f64 {
        self.inner.ci_width
    }

    #[getter]
    fn coverage(&self) -> f64 {
        self.inner.coverage
    }

    #[getter]
    fn size_penalty(&self) -> f64 {
        self.inner.size_penalty
    }

    #[getter]
    fn selected_indices(&self) -> Vec<usize> {
        self.inner.selected_indices.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ParetoSolution(size={}, ci_width={:.6}, coverage={:.4}, penalty={:.2})",
            self.inner.ensemble_size,
            self.inner.ci_width,
            self.inner.coverage,
            self.inner.size_penalty
        )
    }
}

/// Result from ensemble selection including Pareto front
#[pyclass(name = "EnsembleSelectionResult")]
#[derive(Clone)]
pub struct PyEnsembleSelectionResult {
    pub inner: commol_calibration::EnsembleSelectionResult,
}

#[pymethods]
impl PyEnsembleSelectionResult {
    #[getter]
    fn selected_ensemble(&self) -> Vec<usize> {
        self.inner.selected_ensemble.clone()
    }

    #[getter]
    fn pareto_front(&self) -> Vec<PyParetoSolution> {
        self.inner
            .pareto_front
            .iter()
            .map(|sol| PyParetoSolution { inner: sol.clone() })
            .collect()
    }

    #[getter]
    fn selected_pareto_index(&self) -> usize {
        self.inner.selected_pareto_index
    }

    fn __repr__(&self) -> String {
        format!(
            "EnsembleSelectionResult(selected_size={}, pareto_front_size={}, selected_index={})",
            self.inner.selected_ensemble.len(),
            self.inner.pareto_front.len(),
            self.inner.selected_pareto_index
        )
    }
}

/// Calibration evaluation (single optimization run)
#[pyclass(name = "CalibrationEvaluation")]
#[derive(Clone)]
pub struct PyCalibrationEvaluation {
    pub inner: commol_calibration::CalibrationEvaluation,
}

#[pymethods]
impl PyCalibrationEvaluation {
    #[new]
    fn new(parameters: Vec<f64>, loss: f64, predictions: Vec<Vec<f64>>) -> Self {
        Self {
            inner: commol_calibration::CalibrationEvaluation {
                parameters,
                loss,
                predictions,
            },
        }
    }

    #[getter]
    fn parameters(&self) -> Vec<f64> {
        self.inner.parameters.clone()
    }

    #[getter]
    fn loss(&self) -> f64 {
        self.inner.loss
    }

    #[getter]
    fn predictions(&self) -> Vec<Vec<f64>> {
        self.inner.predictions.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CalibrationEvaluation(loss={:.6}, num_params={})",
            self.inner.loss,
            self.inner.parameters.len()
        )
    }
}

/// Calibration result with evaluation history
#[pyclass(name = "CalibrationResultWithHistory")]
pub struct PyCalibrationResultWithHistory {
    inner: commol_calibration::CalibrationResultWithHistory,
}

#[pymethods]
impl PyCalibrationResultWithHistory {
    #[getter]
    fn best_parameters(&self) -> HashMap<String, f64> {
        self.inner
            .parameter_names
            .iter()
            .zip(self.inner.best_parameters.iter())
            .map(|(name, &value)| (name.clone(), value))
            .collect()
    }

    #[getter]
    fn best_parameters_list(&self) -> Vec<f64> {
        self.inner.best_parameters.clone()
    }

    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names.clone()
    }

    #[getter]
    fn final_loss(&self) -> f64 {
        self.inner.final_loss
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn termination_reason(&self) -> String {
        self.inner.termination_reason.clone()
    }

    #[getter]
    fn evaluations(&self) -> Vec<PyCalibrationEvaluation> {
        self.inner
            .evaluations
            .iter()
            .map(|e| PyCalibrationEvaluation { inner: e.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "CalibrationResultWithHistory(loss={:.6}, iterations={}, converged={}, num_evaluations={})",
            self.inner.final_loss,
            self.inner.iterations,
            self.inner.converged,
            self.inner.evaluations.len()
        )
    }
}

/// Calibration result
#[pyclass(name = "CalibrationResult")]
pub struct PyCalibrationResult {
    inner: commol_calibration::CalibrationResult,
}

#[pymethods]
impl PyCalibrationResult {
    /// Get the best parameter values as a dictionary
    #[getter]
    fn best_parameters(&self) -> HashMap<String, f64> {
        self.inner.parameters_map()
    }

    /// Get the best parameter values as a list
    #[getter]
    fn best_parameters_list(&self) -> Vec<f64> {
        self.inner.best_parameters.clone()
    }

    /// Get the parameter names
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names.clone()
    }

    /// Get the final loss value
    #[getter]
    fn final_loss(&self) -> f64 {
        self.inner.final_loss
    }

    /// Get the number of iterations performed
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    /// Check if the optimization converged
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Get the termination reason
    #[getter]
    fn termination_reason(&self) -> String {
        self.inner.termination_reason.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CalibrationResult(loss={:.6}, iterations={}, converged={})",
            self.inner.final_loss, self.inner.iterations, self.inner.converged
        )
    }

    /// Convert to Python dictionary
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("best_parameters", self.best_parameters())?;
        dict.set_item("parameter_names", self.parameter_names())?;
        dict.set_item("final_loss", self.final_loss())?;
        dict.set_item("iterations", self.iterations())?;
        dict.set_item("converged", self.converged())?;
        dict.set_item("termination_reason", self.termination_reason())?;
        Ok(dict.into())
    }
}

/// Calibrate a model against observed data
///
/// Args:
///     engine: The simulation engine (e.g., DifferenceEquations)
///     observed_data: List of observed data points
///     parameters: List of parameters to calibrate
///     constraints: List of calibration constraints
///     loss_config: Loss function configuration
///     optimization_config: Optimization algorithm configuration
///     initial_population_size: Initial population size from the model
///
/// Returns:
///     CalibrationResult with best parameters and optimization statistics
#[pyfunction]
fn calibrate(
    engine: &PyDifferenceEquations,
    observed_data: Vec<PyObservedDataPoint>,
    parameters: Vec<PyCalibrationParameter>,
    constraints: Vec<PyCalibrationConstraint>,
    loss_config: &PyLossConfig,
    optimization_config: &PyOptimizationConfig,
    initial_population_size: u64,
) -> PyResult<PyCalibrationResult> {
    // Extract inner Rust types
    let observed_data: Vec<_> = observed_data.into_iter().map(|d| d.inner).collect();
    let parameters: Vec<_> = parameters.into_iter().map(|p| p.inner).collect();
    let constraints: Vec<_> = constraints.into_iter().map(|c| c.inner).collect();

    // Clone the engine since CalibrationProblem takes ownership
    let engine_clone = engine.inner().clone();

    // Create calibration problem
    let problem = commol_calibration::CalibrationProblem::new(
        engine_clone,
        observed_data,
        parameters,
        constraints,
        loss_config.inner,
        initial_population_size,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    // Check if verbose mode is enabled
    let verbose = match &optimization_config.inner {
        commol_calibration::OptimizationConfig::NelderMead(config) => config.verbose,
        commol_calibration::OptimizationConfig::ParticleSwarm(config) => config.verbose,
    };

    // Run optimization with Python observer if verbose, otherwise use standard optimize
    let result = if verbose {
        crate::python_observer::optimize_with_python_observer(
            problem,
            optimization_config.inner.clone(),
            optimization_config.header_interval,
        )
    } else {
        commol_calibration::optimize(problem, optimization_config.inner.clone())
    }
    .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    Ok(PyCalibrationResult { inner: result })
}

/// Calibrate model parameters with evaluation history tracking
///
/// This function performs the same optimization as `calibrate`, but returns
/// all objective function evaluations that occurred during optimization,
/// not just the final best result.
///
/// Args:
///     engine: The simulation engine (DifferenceEquations)
///     observed_data: List of observed data points to fit
///     parameters: List of parameters to calibrate with bounds
///     constraints: List of constraints on parameters/compartments
///     loss_config: Loss function configuration
///     optimization_config: Optimization algorithm configuration
///     initial_population_size: Initial population for fraction calculations
///
/// Returns:
///     CalibrationResultWithHistory containing best parameters and all evaluations
#[pyfunction]
fn calibrate_with_history(
    engine: &PyDifferenceEquations,
    observed_data: Vec<PyObservedDataPoint>,
    parameters: Vec<PyCalibrationParameter>,
    constraints: Vec<PyCalibrationConstraint>,
    loss_config: &PyLossConfig,
    optimization_config: &PyOptimizationConfig,
    initial_population_size: u64,
) -> PyResult<PyCalibrationResultWithHistory> {
    // Extract inner Rust types
    let observed_data: Vec<_> = observed_data.into_iter().map(|d| d.inner).collect();
    let parameters: Vec<_> = parameters.into_iter().map(|p| p.inner).collect();
    let constraints: Vec<_> = constraints.into_iter().map(|c| c.inner).collect();

    // Clone the engine since CalibrationProblem takes ownership
    let engine_clone = engine.inner().clone();

    // Create calibration problem
    let problem = commol_calibration::CalibrationProblem::new(
        engine_clone,
        observed_data,
        parameters,
        constraints,
        loss_config.inner,
        initial_population_size,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    // Run optimization with history tracking
    let result =
        commol_calibration::optimize_with_history(problem, optimization_config.inner.clone())
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

    Ok(PyCalibrationResultWithHistory { inner: result })
}

/// Run multiple calibration attempts in parallel (for probabilistic calibration)
///
/// This is the performance-critical bottleneck of probabilistic calibration.
/// Runs N calibration runs in parallel using Rayon, each with a different seed.
///
/// Args:
///     engine: The simulation engine (DifferenceEquations)
///     observed_data: List of observed data points
///     parameters: List of parameters to calibrate
///     constraints: List of constraints on parameters/compartments
///     loss_config: Loss function configuration
///     optimization_config: Optimization algorithm configuration
///     initial_population_size: Initial population for fraction calculations
///     n_runs: Number of calibration runs to perform
///     seed: Base random seed (each run gets seed + run_index)
///
/// Returns:
///     List of CalibrationResultWithHistory for each successful run
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_multiple_calibrations(
    engine: &PyDifferenceEquations,
    observed_data: Vec<PyObservedDataPoint>,
    parameters: Vec<PyCalibrationParameter>,
    constraints: Vec<PyCalibrationConstraint>,
    loss_config: &PyLossConfig,
    optimization_config: &PyOptimizationConfig,
    initial_population_size: u64,
    n_runs: usize,
    seed: u64,
) -> PyResult<Vec<PyCalibrationResultWithHistory>> {
    // Convert to Rust types
    let rust_observed_data: Vec<_> = observed_data.into_iter().map(|d| d.inner).collect();
    let rust_parameters: Vec<_> = parameters.into_iter().map(|p| p.inner).collect();
    let rust_constraints: Vec<_> = constraints.into_iter().map(|c| c.inner).collect();

    // Create the base calibration problem
    let engine_clone = engine.inner().clone();
    let base_problem = commol_calibration::CalibrationProblem::new(
        engine_clone,
        rust_observed_data,
        rust_parameters,
        rust_constraints,
        loss_config.inner,
        initial_population_size,
    )
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

    // Run multiple calibrations in parallel
    let results = commol_calibration::run_multiple_calibrations(
        &base_problem,
        &optimization_config.inner,
        n_runs,
        seed,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Convert to Python types
    Ok(results
        .into_iter()
        .map(|r| PyCalibrationResultWithHistory { inner: r })
        .collect())
}

/// Select optimal ensemble using NSGA-II (for probabilistic calibration)
///
/// Uses NSGA-II multi-objective optimization to find a Pareto-optimal ensemble
/// of parameter sets that balances narrow confidence intervals with good coverage.
///
/// Args:
///     candidates: List of CalibrationEvaluation objects with predictions
///     observed_data_tuples: List of (time_step, compartment_idx, value) tuples
///     population_size: NSGA-II population size
///     generations: Number of NSGA-II generations
///     confidence_level: Confidence level for CI calculation (e.g., 0.95)
///     seed: Random seed for reproducibility
///     pareto_preference: Preference for Pareto front selection (0.0-1.0)
///     ensemble_size_mode: Mode for determining ensemble size ("fixed", "bounded", or "automatic")
///     ensemble_size: Fixed ensemble size (required if mode='fixed', otherwise None)
///     ensemble_size_min: Minimum ensemble size (required if mode='bounded', otherwise None)
///     ensemble_size_max: Maximum ensemble size (required if mode='bounded', otherwise None)
///     ci_margin_factor: Safety margin factor for CI width bounds (default: 0.1)
///     ci_sample_sizes: Sample sizes for CI bounds estimation (default: [10, 20, 50, 100])
///     nsga_crossover_probability: NSGA-II crossover probability (default: 0.9)
///
/// Returns:
///     EnsembleSelectionResult containing selected ensemble and Pareto front
#[pyfunction]
#[pyo3(signature = (candidates, observed_data_tuples, population_size, generations, confidence_level, seed, pareto_preference, ensemble_size_mode, ensemble_size=None, ensemble_size_min=None, ensemble_size_max=None, ci_margin_factor=0.1, ci_sample_sizes=None, nsga_crossover_probability=0.9))]
#[allow(clippy::too_many_arguments)]
fn select_optimal_ensemble(
    candidates: Vec<PyCalibrationEvaluation>,
    observed_data_tuples: Vec<(usize, usize, f64)>,
    population_size: usize,
    generations: usize,
    confidence_level: f64,
    seed: u64,
    pareto_preference: f64,
    ensemble_size_mode: &str,
    ensemble_size: Option<usize>,
    ensemble_size_min: Option<usize>,
    ensemble_size_max: Option<usize>,
    ci_margin_factor: f64,
    ci_sample_sizes: Option<Vec<usize>>,
    nsga_crossover_probability: f64,
) -> PyResult<PyEnsembleSelectionResult> {
    // Convert Python CalibrationEvaluation to Rust
    let rust_candidates: Vec<_> = candidates.into_iter().map(|e| e.inner).collect();

    // Parse ensemble size mode
    let size_mode = match ensemble_size_mode {
        "fixed" => {
            let size = ensemble_size.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "ensemble_size must be specified when ensemble_size_mode='fixed'",
                )
            })?;
            if size < 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "ensemble_size must be >= 2, got {}",
                    size
                )));
            }
            commol_calibration::EnsembleSizeMode::Fixed { size }
        }
        "bounded" => {
            let min = ensemble_size_min.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "ensemble_size_min must be specified when ensemble_size_mode='bounded'",
                )
            })?;
            let max = ensemble_size_max.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "ensemble_size_max must be specified when ensemble_size_mode='bounded'",
                )
            })?;
            if min < 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "ensemble_size_min must be >= 2, got {}",
                    min
                )));
            }
            if max < min {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "ensemble_size_max ({}) must be >= ensemble_size_min ({})",
                    max, min
                )));
            }
            commol_calibration::EnsembleSizeMode::Bounded { min, max }
        }
        "automatic" => commol_calibration::EnsembleSizeMode::Automatic,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid ensemble_size_mode: '{}'. Must be 'fixed', 'bounded', or 'automatic'",
                ensemble_size_mode
            )));
        }
    };

    // Build ensemble selection config from parameters
    let ensemble_config = commol_calibration::EnsembleSelectionConfig {
        ci_margin_factor,
        ci_sample_sizes: ci_sample_sizes.unwrap_or_else(|| vec![10, 20, 50, 100]),
        nsga_crossover_probability,
        // Use defaults for cluster representative parameters (not used in select_optimal_ensemble)
        k_neighbors_min: 5,
        k_neighbors_max: 10,
        sparsity_weight: 2.0,
        stratum_fit_weight: 10.0,
    };

    let optimal_config = commol_calibration::OptimalEnsembleConfig {
        population_size,
        generations,
        confidence_level,
        seed,
        pareto_preference,
        size_mode,
        ensemble_config: &ensemble_config,
    };

    // Run NSGA-II ensemble selection
    let result = commol_calibration::select_optimal_ensemble(
        rust_candidates,
        observed_data_tuples,
        &optimal_config,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Return full result with Pareto front
    Ok(PyEnsembleSelectionResult { inner: result })
}

/// Deduplicate calibration evaluations using grid-based spatial hashing
///
/// This is a performance-critical operation that uses O(n) average-case
/// grid-based spatial hashing to remove duplicate parameter sets.
///
/// Args:
///     evaluations: List of CalibrationEvaluation objects to deduplicate
///     tolerance: Relative tolerance for considering parameters as duplicates
///
/// Returns:
///     List of unique CalibrationEvaluation objects
#[pyfunction]
fn deduplicate_evaluations(
    evaluations: Vec<PyCalibrationEvaluation>,
    tolerance: f64,
) -> PyResult<Vec<PyCalibrationEvaluation>> {
    // Convert Python evaluations to Rust
    let rust_evaluations: Vec<_> = evaluations.into_iter().map(|e| e.inner).collect();

    // Run deduplication in Rust (fast!)
    let unique_evaluations =
        commol_calibration::deduplicate_evaluations(rust_evaluations, tolerance);

    // Convert back to Python
    Ok(unique_evaluations
        .into_iter()
        .map(|e| PyCalibrationEvaluation { inner: e })
        .collect())
}

/// Generate predictions for multiple parameter sets in parallel
///
/// This function runs simulations for each parameter set in parallel using Rust/Rayon,
/// which is much faster than sequential execution in Python.
///
/// Args:
///     engine: DifferenceEquations simulation engine
///     parameter_sets: List of parameter value lists (one per parameter set)
///     parameter_names: List of parameter names (must match model parameter IDs)
///     time_steps: Number of time steps to simulate
///
/// Returns:
///     List of predictions, where each prediction is a 2D list [time_step][compartment_idx]
#[pyfunction]
fn generate_predictions_parallel(
    engine: &PyDifferenceEquations,
    parameter_sets: Vec<Vec<f64>>,
    parameter_names: Vec<String>,
    time_steps: u32,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let base_engine = engine.inner();

    let predictions = commol_calibration::generate_predictions_parallel(
        base_engine,
        parameter_sets,
        parameter_names,
        time_steps,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(predictions)
}

/// Select cluster representatives using Rust for performance
///
/// Args:
///     evaluations: List of CalibrationEvaluation objects
///     cluster_labels: Cluster assignment for each evaluation (from KMeans)
///     max_representatives: Maximum total representatives to select
///     elite_fraction: Fraction of best solutions to always include (0.0-1.0)
///     strategy: Distribution strategy ("proportional" or "equal")
///     selection_method: Diversity method ("crowding_distance" or "maximin_distance")
///     quality_temperature: Temperature for quality weighting in maximin (default: 1.0)
///     seed: Random seed for reproducibility (default: 42)
///     k_neighbors_min: Minimum k for k-NN density estimation (default: 5)
///     k_neighbors_max: Maximum k for k-NN density estimation (default: 10)
///     sparsity_weight: Exponential weight for sparsity bonus (default: 2.0)
///     stratum_fit_weight: Weight for stratum fit in latin_hypercube (default: 10.0)
///
/// Returns:
///     List of indices of selected representatives
#[pyfunction]
#[pyo3(signature = (evaluations, cluster_labels, max_representatives, elite_fraction, strategy, selection_method="crowding_distance", quality_temperature=1.0, seed=42, k_neighbors_min=5, k_neighbors_max=10, sparsity_weight=2.0, stratum_fit_weight=10.0))]
#[allow(clippy::too_many_arguments)]
fn select_cluster_representatives(
    evaluations: Vec<PyCalibrationEvaluation>,
    cluster_labels: Vec<usize>,
    max_representatives: usize,
    elite_fraction: f64,
    strategy: &str,
    selection_method: &str,
    quality_temperature: f64,
    seed: u64,
    k_neighbors_min: usize,
    k_neighbors_max: usize,
    sparsity_weight: f64,
    stratum_fit_weight: f64,
) -> PyResult<Vec<usize>> {
    // Convert to Rust types
    let rust_evaluations: Vec<_> = evaluations.into_iter().map(|e| e.inner).collect();

    // Build config from parameters
    let config = commol_calibration::EnsembleSelectionConfig {
        // CI parameters not used in cluster representatives
        ci_margin_factor: 0.1,
        ci_sample_sizes: vec![10, 20, 50, 100],
        nsga_crossover_probability: 0.9,
        // Cluster representative parameters
        k_neighbors_min,
        k_neighbors_max,
        sparsity_weight,
        stratum_fit_weight,
    };

    let cluster_config = commol_calibration::ClusterRepresentativeConfig {
        max_representatives,
        elite_fraction,
        strategy,
        selection_method,
        quality_temperature,
        seed,
        ensemble_config: &config,
    };

    let indices = commol_calibration::select_cluster_representatives(
        rust_evaluations,
        cluster_labels,
        &cluster_config,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(indices)
}

/// Register calibration module with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCalibrationParameterType>()?;
    m.add_class::<PyObservedDataPoint>()?;
    m.add_class::<PyCalibrationParameter>()?;
    m.add_class::<PyCalibrationConstraint>()?;
    m.add_class::<PyLossConfig>()?;
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
    m.add_class::<PyParetoSolution>()?;
    m.add_class::<PyEnsembleSelectionResult>()?;
    m.add_function(wrap_pyfunction!(calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_with_history, m)?)?;
    m.add_function(wrap_pyfunction!(run_multiple_calibrations, m)?)?;
    m.add_function(wrap_pyfunction!(select_optimal_ensemble, m)?)?;
    m.add_function(wrap_pyfunction!(deduplicate_evaluations, m)?)?;
    m.add_function(wrap_pyfunction!(generate_predictions_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(select_cluster_representatives, m)?)?;
    Ok(())
}
