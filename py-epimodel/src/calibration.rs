//! Python bindings for epimodel-calibration (parameter optimization).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::difference::PyDifferenceEquations;

/// Observed data point for calibration
#[pyclass(name = "ObservedDataPoint")]
#[derive(Clone)]
pub struct PyObservedDataPoint {
    pub inner: epimodel_calibration::ObservedDataPoint,
}

#[pymethods]
impl PyObservedDataPoint {
    /// Create a new observed data point
    ///
    /// Args:
    ///     step: Step of observation
    ///     compartment_index: Index of the compartment being observed
    ///     value: Observed value
    ///     weight: Optional weight for this observation (default: 1.0)
    #[new]
    #[pyo3(signature = (step, compartment_index, value, weight=None))]
    fn new(step: u32, compartment_index: usize, value: f64, weight: Option<f64>) -> Self {
        Self {
            inner: if let Some(w) = weight {
                epimodel_calibration::ObservedDataPoint::with_weight(
                    step,
                    compartment_index,
                    value,
                    w,
                )
            } else {
                epimodel_calibration::ObservedDataPoint::new(step, compartment_index, value)
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ObservedDataPoint(time_step={}, compartment={}, value={})",
            self.inner.time_step, self.inner.compartment_index, self.inner.value
        )
    }
}

/// Parameter to calibrate with bounds
#[pyclass(name = "CalibrationParameter")]
#[derive(Clone)]
pub struct PyCalibrationParameter {
    pub inner: epimodel_calibration::CalibrationParameter,
}

#[pymethods]
impl PyCalibrationParameter {
    /// Create a new calibration parameter
    ///
    /// Args:
    ///     id: Parameter identifier (must match model parameter ID)
    ///     min_bound: Minimum allowed value
    ///     max_bound: Maximum allowed value
    ///     initial_guess: Optional starting value for optimization
    #[new]
    #[pyo3(signature = (id, min_bound, max_bound, initial_guess=None))]
    fn new(id: String, min_bound: f64, max_bound: f64, initial_guess: Option<f64>) -> Self {
        Self {
            inner: if let Some(guess) = initial_guess {
                epimodel_calibration::CalibrationParameter::with_initial_guess(
                    id, min_bound, max_bound, guess,
                )
            } else {
                epimodel_calibration::CalibrationParameter::new(id, min_bound, max_bound)
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CalibrationParameter(id='{}', bounds=[{}, {}])",
            self.inner.id, self.inner.min_bound, self.inner.max_bound
        )
    }
}

/// Loss function configuration
#[pyclass(name = "LossConfig")]
#[derive(Clone)]
pub struct PyLossConfig {
    pub inner: epimodel_calibration::LossConfig,
}

#[pymethods]
impl PyLossConfig {
    /// Sum of squared errors
    #[staticmethod]
    fn sum_squared_error() -> Self {
        Self {
            inner: epimodel_calibration::LossConfig::SumSquaredError,
        }
    }

    /// Root mean squared error
    #[staticmethod]
    fn root_mean_squared_error() -> Self {
        Self {
            inner: epimodel_calibration::LossConfig::RootMeanSquaredError,
        }
    }

    /// Mean absolute error
    #[staticmethod]
    fn mean_absolute_error() -> Self {
        Self {
            inner: epimodel_calibration::LossConfig::MeanAbsoluteError,
        }
    }

    /// Weighted sum of squared errors
    #[staticmethod]
    fn weighted_sse() -> Self {
        Self {
            inner: epimodel_calibration::LossConfig::WeightedSSE,
        }
    }
}

/// Nelder-Mead optimization configuration
#[pyclass(name = "NelderMeadConfig")]
#[derive(Clone)]
pub struct PyNelderMeadConfig {
    pub inner: epimodel_calibration::NelderMeadConfig,
}

#[pymethods]
impl PyNelderMeadConfig {
    /// Create Nelder-Mead configuration
    ///
    /// Args:
    ///     max_iterations: Maximum number of iterations (default: 1000)
    ///     sd_tolerance: Convergence tolerance (default: 1e-6)
    ///     alpha: Reflection coefficient (default: 1.0)
    ///     gamma: Expansion coefficient (default: 2.0)
    ///     rho: Contraction coefficient (default: 0.5)
    ///     sigma: Shrink coefficient (default: 0.5)
    ///     verbose: Enable verbose output (default: false)
    #[new]
    #[pyo3(signature = (max_iterations=1000, sd_tolerance=1e-6, alpha=None, gamma=None, rho=None, sigma=None, verbose=false))]
    fn new(
        max_iterations: u64,
        sd_tolerance: f64,
        alpha: Option<f64>,
        gamma: Option<f64>,
        rho: Option<f64>,
        sigma: Option<f64>,
        verbose: bool,
    ) -> Self {
        Self {
            inner: epimodel_calibration::NelderMeadConfig {
                max_iterations,
                sd_tolerance,
                alpha,
                gamma,
                rho,
                sigma,
                verbose,
            },
        }
    }
}

/// Particle Swarm Optimization configuration
#[pyclass(name = "ParticleSwarmConfig")]
#[derive(Clone)]
pub struct PyParticleSwarmConfig {
    pub inner: epimodel_calibration::ParticleSwarmConfig,
}

#[pymethods]
impl PyParticleSwarmConfig {
    /// Create Particle Swarm Optimization configuration
    ///
    /// Args:
    ///     num_particles: Number of particles in the swarm (default: 40)
    ///     max_iterations: Maximum number of iterations (default: 1000)
    ///     target_cost: Target cost for early stopping (optional)
    ///     inertia_factor: Inertia weight applied to velocity (default: ~0.721)
    ///     cognitive_factor: Attraction to personal best (default: ~1.193)
    ///     social_factor: Attraction to swarm best (default: ~1.193)
    ///     verbose: Enable verbose output (default: false)
    #[new]
    #[pyo3(signature = (num_particles=40, max_iterations=1000, target_cost=None, inertia_factor=None, cognitive_factor=None, social_factor=None, verbose=false))]
    fn new(
        num_particles: usize,
        max_iterations: u64,
        target_cost: Option<f64>,
        inertia_factor: Option<f64>,
        cognitive_factor: Option<f64>,
        social_factor: Option<f64>,
        verbose: bool,
    ) -> Self {
        Self {
            inner: epimodel_calibration::ParticleSwarmConfig {
                num_particles,
                max_iterations,
                target_cost,
                inertia_factor,
                cognitive_factor,
                social_factor,
                verbose,
            },
        }
    }
}

/// Optimization algorithm configuration
#[pyclass(name = "OptimizationConfig")]
#[derive(Clone)]
pub struct PyOptimizationConfig {
    pub inner: epimodel_calibration::OptimizationConfig,
}

#[pymethods]
impl PyOptimizationConfig {
    /// Create optimization config with Nelder-Mead algorithm
    #[staticmethod]
    fn nelder_mead(config: Option<PyNelderMeadConfig>) -> Self {
        Self {
            inner: epimodel_calibration::OptimizationConfig::NelderMead(
                config
                    .map(|c| c.inner)
                    .unwrap_or_else(epimodel_calibration::NelderMeadConfig::default),
            ),
        }
    }

    /// Create optimization config with Particle Swarm algorithm
    #[staticmethod]
    fn particle_swarm(config: Option<PyParticleSwarmConfig>) -> Self {
        Self {
            inner: epimodel_calibration::OptimizationConfig::ParticleSwarm(
                config
                    .map(|c| c.inner)
                    .unwrap_or_else(epimodel_calibration::ParticleSwarmConfig::default),
            ),
        }
    }
}

/// Calibration result
#[pyclass(name = "CalibrationResult")]
pub struct PyCalibrationResult {
    inner: epimodel_calibration::CalibrationResult,
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
) -> PyResult<PyCalibrationResult> {
    // Extract inner Rust types
    let observed_data: Vec<_> = observed_data.into_iter().map(|d| d.inner).collect();
    let parameters: Vec<_> = parameters.into_iter().map(|p| p.inner).collect();

    // Clone the engine since CalibrationProblem takes ownership
    let engine_clone = engine.inner().clone();

    // Create calibration problem
    let problem = epimodel_calibration::CalibrationProblem::new(
        engine_clone,
        observed_data,
        parameters,
        loss_config.inner,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    // Run optimization
    let result = epimodel_calibration::optimize(problem, optimization_config.inner.clone())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(PyCalibrationResult { inner: result })
}

/// Register calibration module with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
