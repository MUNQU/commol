//! Python bindings for epimodel-difference (discrete-time solver).

use crate::core::PyModel;
use pyo3::prelude::*;

/// Wrapper for commol_difference::DifferenceEquations
///
/// Discrete-time compartment model solver using difference equations.
#[pyclass(name = "DifferenceEquations")]
pub struct PyDifferenceEquations {
    pub(crate) inner: commol_difference::DifferenceEquations,
}

#[pymethods]
impl PyDifferenceEquations {
    /// Create a new DifferenceEquations solver from a Model
    ///
    /// Args:
    ///     model: The compartment model to simulate
    ///
    /// Returns:
    ///     A new DifferenceEquations instance
    #[new]
    fn new(model: &PyModel) -> Self {
        Self {
            inner: commol_difference::DifferenceEquations::from_model(&model.inner),
        }
    }

    /// Run the simulation for a given number of steps
    ///
    /// Args:
    ///     num_steps: Number of time steps to simulate
    ///
    /// Returns:
    ///     List of population vectors, one for each time step (including initial state)
    fn run(&mut self, num_steps: u32) -> PyResult<Vec<Vec<f64>>> {
        self.inner
            .run(num_steps)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Execute a single simulation step
    fn step(&mut self) -> PyResult<()> {
        self.inner
            .step()
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Get the current population distribution
    ///
    /// Returns:
    ///     Vector of population values for each compartment
    #[getter]
    fn population(&self) -> Vec<f64> {
        self.inner.population()
    }

    /// Get the list of compartment names
    ///
    /// Returns:
    ///     List of compartment names (e.g., ["S", "I_young", "R_old"])
    #[getter]
    fn compartments(&self) -> Vec<String> {
        self.inner.compartments()
    }

    /// Reset the simulation to initial conditions
    fn reset(&mut self) {
        use commol_core::SimulationEngine;
        self.inner.reset();
    }

    /// Set a parameter value
    ///
    /// Args:
    ///     parameter_id: The parameter identifier
    ///     value: The new parameter value
    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> PyResult<()> {
        use commol_core::SimulationEngine;
        self.inner
            .set_parameter(parameter_id, value)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Get all current parameter values
    ///
    /// Returns:
    ///     Dictionary mapping parameter names to values
    fn get_parameters(&self) -> std::collections::HashMap<String, f64> {
        use commol_core::SimulationEngine;
        self.inner.get_parameters().clone()
    }

    /// Get the current simulation step
    ///
    /// Returns:
    ///     Current step number
    fn current_step(&self) -> f64 {
        use commol_core::SimulationEngine;
        self.inner.current_step()
    }

    /// Set the initial condition for a specific compartment
    ///
    /// Args:
    ///     compartment_index: The index of the compartment (0-based)
    ///     value: The new initial population value
    ///
    /// Raises:
    ///     ValueError: If compartment_index is invalid or value is negative
    fn set_initial_condition(&mut self, compartment_index: usize, value: f64) -> PyResult<()> {
        use commol_core::SimulationEngine;
        self.inner
            .set_initial_condition(compartment_index, value)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn __repr__(&self) -> String {
        format!(
            "DifferenceEquations(compartments={}, step={})",
            self.inner.compartments().len(),
            {
                use commol_core::SimulationEngine;
                self.inner.current_step()
            }
        )
    }
}

impl PyDifferenceEquations {
    /// Get a reference to the inner Rust engine (for internal use)
    pub(crate) fn inner(&self) -> &commol_difference::DifferenceEquations {
        &self.inner
    }
}

/// Register difference equation solver with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDifferenceEquations>()?;
    Ok(())
}
