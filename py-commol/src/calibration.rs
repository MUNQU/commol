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
    #[new]
    #[pyo3(signature = (step, compartment, value, weight=None))]
    fn new(step: u32, compartment: String, value: f64, weight: Option<f64>) -> Self {
        Self {
            inner: if let Some(w) = weight {
                commol_calibration::ObservedDataPoint::with_weight(step, compartment, value, w)
            } else {
                commol_calibration::ObservedDataPoint::new(step, compartment, value)
            },
        }
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
    fn sum_squared_error() -> Self {
        Self {
            inner: commol_calibration::LossConfig::SumSquaredError,
        }
    }

    /// Root mean squared error
    #[staticmethod]
    fn root_mean_squared_error() -> Self {
        Self {
            inner: commol_calibration::LossConfig::RootMeanSquaredError,
        }
    }

    /// Mean absolute error
    #[staticmethod]
    fn mean_absolute_error() -> Self {
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
    ///     alpha: Reflection coefficient (default: None, uses argmin's default)
    ///     gamma: Expansion coefficient (default: None, uses argmin's default)
    ///     rho: Contraction coefficient (default: None, uses argmin's default)
    ///     sigma: Shrink coefficient (default: None, uses argmin's default)
    ///     verbose: Enable verbose output (default: false)
    ///     header_interval: Number of iterations between table header repeats (default: 100)
    #[new]
    #[pyo3(signature = (max_iterations=1000, sd_tolerance=1e-6, alpha=None, gamma=None, rho=None, sigma=None, verbose=false, header_interval=100))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_iterations: u64,
        sd_tolerance: f64,
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
    ///     target_cost: Target cost for early stopping (default: None)
    ///     inertia_factor: Inertia weight applied to velocity (default: None, uses argmin's default)
    ///     cognitive_factor: Attraction to personal best (default: None, uses argmin's default)
    ///     social_factor: Attraction to swarm best (default: None, uses argmin's default)
    ///     verbose: Enable verbose output (default: false)
    ///     header_interval: Number of iterations between table header repeats (default: 100)
    #[new]
    #[pyo3(signature = (num_particles=20, max_iterations=1000, target_cost=None, inertia_factor=None, cognitive_factor=None, social_factor=None, verbose=false, header_interval=100))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_particles: usize,
        max_iterations: u64,
        target_cost: Option<f64>,
        inertia_factor: Option<f64>,
        cognitive_factor: Option<f64>,
        social_factor: Option<f64>,
        verbose: bool,
        header_interval: u64,
    ) -> Self {
        Self {
            inner: commol_calibration::ParticleSwarmConfig {
                num_particles,
                max_iterations,
                target_cost,
                inertia_factor,
                cognitive_factor,
                social_factor,
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
///     loss_config: Loss function configuration
///     optimization_config: Optimization algorithm configuration
///
/// Returns:
///     CalibrationResult with best parameters and optimization statistics
#[pyfunction]
fn calibrate(
    engine: &PyDifferenceEquations,
    observed_data: Vec<PyObservedDataPoint>,
    parameters: Vec<PyCalibrationParameter>,
    loss_config: &PyLossConfig,
    optimization_config: &PyOptimizationConfig,
    initial_population_size: u64,
) -> PyResult<PyCalibrationResult> {
    // Extract inner Rust types
    let observed_data: Vec<_> = observed_data.into_iter().map(|d| d.inner).collect();
    let parameters: Vec<_> = parameters.into_iter().map(|p| p.inner).collect();

    // Clone the engine since CalibrationProblem takes ownership
    let engine_clone = engine.inner().clone();

    // Create calibration problem
    let problem = commol_calibration::CalibrationProblem::new(
        engine_clone,
        observed_data,
        parameters,
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

/// Register calibration module with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCalibrationParameterType>()?;
    m.add_class::<PyObservedDataPoint>()?;
    m.add_class::<PyCalibrationParameter>()?;
    m.add_class::<PyLossConfig>()?;
    m.add_class::<PyNelderMeadConfig>()?;
    m.add_class::<PyParticleSwarmConfig>()?;
    m.add_class::<PyOptimizationConfig>()?;
    m.add_class::<PyCalibrationResult>()?;
    m.add_function(wrap_pyfunction!(calibrate, m)?)?;
    Ok(())
}
