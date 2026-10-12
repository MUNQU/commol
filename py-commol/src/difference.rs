//! Python bindings for commol-difference (discrete-time solver).

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
    ///
    /// Raises:
    ///     RuntimeError: If an initial-condition override names a compartment
    ///         that is not part of the model expansion
    #[new]
    fn new(model: &PyModel) -> PyResult<Self> {
        Ok(Self {
            inner: commol_difference::DifferenceEquations::try_from_model(&model.inner)
                .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?,
        })
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

    /// Get the list of simulation output names, including accumulators.
    #[getter]
    fn output_names(&self) -> Vec<String> {
        self.inner.output_names()
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

/// Expand a model into its compartments and their initial populations
///
/// This is the authority for the initial state of a model, including
/// per-compartment overrides.
///
/// Args:
///     model: The compartment model
///
/// Returns:
///     One `(name, bin, categories, population)` tuple per expanded compartment,
///     in engine order. `categories` holds the `(stratification, category)`
///     pairs applied to the compartment, in declaration order.
///
/// Raises:
///     ValueError: If an override names an unknown or repeated compartment,
///         holds a fraction outside [0, 1], or makes the initial population
///         exceed population_size
#[pyfunction]
#[allow(clippy::type_complexity)]
fn initial_state(model: &PyModel) -> PyResult<Vec<(String, String, Vec<(String, String)>, f64)>> {
    let state = commol_difference::initial_state(&model.inner)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(state
        .into_iter()
        .map(|compartment| {
            (
                compartment.name,
                compartment.bin,
                compartment.categories,
                compartment.population,
            )
        })
        .collect())
}

/// Expand the bins and accumulators of a model over its stratifications
///
/// This is the authority for the names a model expands into.
///
/// Args:
///     model: The compartment model
///
/// Returns:
///     A `(compartments, accumulators)` tuple. Each holds one
///     `(name, source, categories)` tuple per expanded output, in engine order,
///     where `source` is the bin or accumulator id and `categories` the
///     `(stratification, category)` pairs applied to it.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn expanded_outputs(
    model: &PyModel,
) -> (
    Vec<(String, String, Vec<(String, String)>)>,
    Vec<(String, String, Vec<(String, String)>)>,
) {
    let as_tuples = |outputs: Vec<commol_difference::ExpandedOutput>| {
        outputs
            .into_iter()
            .map(|output| (output.name, output.source, output.categories))
            .collect()
    };
    (
        as_tuples(commol_difference::expand_compartments(&model.inner)),
        as_tuples(commol_difference::expand_accumulators(&model.inner)),
    )
}

/// Every transition flow of a model, exactly as the engine simulates it
///
/// Args:
///     model: The compartment model, of any typology
///
/// Returns:
///     One `(transition, source, target, rate, per_capita_of)` tuple per flow
///     piece, in transition order. `source` or `target` is None for a
///     source-less or target-less transition. `per_capita_of` is the
///     compartment a per-capita rate is multiplied by, or None for an absolute
///     flow; the pieces of a transition with several sources share it.
///
/// Raises:
///     ValueError: If a transition mixes bins and categories, or a category
///         transition has no source or no target
#[pyfunction]
#[allow(clippy::type_complexity)]
fn transition_flows(
    model: &PyModel,
) -> PyResult<
    Vec<(
        String,
        Option<String>,
        Option<String>,
        String,
        Option<String>,
    )>,
> {
    let flows = commol_difference::describe_flows(&model.inner)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(flows
        .into_iter()
        .map(|flow| {
            (
                flow.transition,
                flow.source,
                flow.target,
                flow.rate,
                flow.per_capita_of,
            )
        })
        .collect())
}

/// Register difference equation solver with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDifferenceEquations>()?;
    m.add_function(wrap_pyfunction!(initial_state, m)?)?;
    m.add_function(wrap_pyfunction!(expanded_outputs, m)?)?;
    m.add_function(wrap_pyfunction!(transition_flows, m)?)?;
    Ok(())
}
