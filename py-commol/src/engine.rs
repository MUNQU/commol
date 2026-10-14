//! The engines of the list in [`commol_core::EngineType`], for Python.
//!
//! [`Engine`] holds the engine a model names, so each binding exists once for
//! every engine. [`PyEngine`] builds it from a model, and [`PyEngineType`]
//! lets Python ask the list instead of branching on engine names.

use std::collections::HashMap;

use commol_core::{EngineType, OutputLabels, SimulationEngine};
use commol_difference::DifferenceEquations;
use commol_differential::DifferentialEquations;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// The engine of one entry of [`EngineType`].
#[derive(Clone)]
pub enum Engine {
    /// Discrete time.
    Difference(DifferenceEquations),
    /// Continuous time.
    Differential(DifferentialEquations),
}

impl Engine {
    /// Build the engine the model names.
    ///
    /// # Errors
    ///
    /// Returns a message when the engine cannot compile the model.
    pub fn build(model: &commol_core::Model) -> Result<Self, String> {
        match model.engine.engine_type() {
            EngineType::DifferenceEquations => {
                DifferenceEquations::try_from_model(model).map(Self::Difference)
            }
            EngineType::DifferentialEquations => {
                DifferentialEquations::try_from_model(model).map(Self::Differential)
            }
        }
    }

    /// The entry of the list this engine is.
    pub fn engine_type(&self) -> EngineType {
        match self {
            Self::Difference(_) => EngineType::DifferenceEquations,
            Self::Differential(_) => EngineType::DifferentialEquations,
        }
    }
}

/// Call `$body` with `$engine` bound to the engine inside an [`Engine`].
macro_rules! dispatch {
    ($self:expr, $engine:ident => $body:expr) => {
        match $self {
            Engine::Difference($engine) => $body,
            Engine::Differential($engine) => $body,
        }
    };
}

impl SimulationEngine for Engine {
    fn check_position(&self, position: f64) -> Result<(), String> {
        dispatch!(self, engine => engine.check_position(position))
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        dispatch!(self, engine => engine.run_at_positions_into_buffer(positions, buffer))
    }

    fn step(&mut self) -> Result<(), String> {
        dispatch!(self, engine => SimulationEngine::step(engine))
    }

    fn compartments(&self) -> Vec<String> {
        dispatch!(self, engine => SimulationEngine::compartments(engine))
    }

    fn output_labels(&self) -> Vec<OutputLabels> {
        dispatch!(self, engine => engine.output_labels())
    }

    fn output_names(&self) -> Vec<String> {
        dispatch!(self, engine => SimulationEngine::output_names(engine))
    }

    fn population(&self) -> Vec<f64> {
        dispatch!(self, engine => SimulationEngine::population(engine))
    }

    fn reset(&mut self) {
        dispatch!(self, engine => engine.reset())
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        dispatch!(self, engine => engine.set_parameter(parameter_id, value))
    }

    fn get_parameters(&self) -> &HashMap<String, f64> {
        dispatch!(self, engine => engine.get_parameters())
    }

    fn current_position(&self) -> f64 {
        dispatch!(self, engine => engine.current_position())
    }

    fn set_initial_condition(
        &mut self,
        compartment_index: usize,
        value: f64,
    ) -> Result<(), String> {
        dispatch!(self, engine => engine.set_initial_condition(compartment_index, value))
    }
}

/// The engine a model names, ready to simulate it.
#[pyclass(name = "Engine")]
pub struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    /// Build the engine the model names, and compile the model into it
    ///
    /// Args:
    ///     model: The model to simulate
    ///
    /// Raises:
    ///     RuntimeError: If the model cannot be compiled (for example an
    ///         initial-condition override names a compartment that is not
    ///         part of the model expansion)
    #[new]
    fn new(model: &crate::core::PyModel) -> PyResult<Self> {
        Ok(Self {
            inner: Engine::build(&model.inner).map_err(PyRuntimeError::new_err)?,
        })
    }

    /// Name of the engine, an entry of `EngineType.names()`
    #[getter]
    fn engine_type(&self) -> &'static str {
        self.inner.engine_type().name()
    }

    /// Run on the default grid of a number of steps
    ///
    /// Args:
    ///     num_steps: Number of steps to simulate
    ///
    /// Returns:
    ///     The positions of the grid, from the current one (included)
    ///     one unit apart, and one output row per position
    fn run(&mut self, num_steps: u32) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let trajectory =
            SimulationEngine::run(&mut self.inner, num_steps).map_err(PyRuntimeError::new_err)?;
        Ok((trajectory.positions, trajectory.rows))
    }

    /// Run through the last of the given positions on the axis
    ///
    /// Args:
    ///     positions: Sorted, unique positions to report at, not before
    ///         the current one
    ///
    /// Returns:
    ///     One output row per position
    ///
    /// Raises:
    ///     ValueError: If a position is not valid for the engine (a
    ///         difference engine reports only at integer steps), or the
    ///         positions are not sorted and unique
    ///     RuntimeError: If the simulation fails
    fn run_at(&mut self, positions: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        let engine = &mut self.inner;
        commol_core::check_positions(&positions, engine.current_position(), |position| {
            engine.check_position(position)
        })
        .map_err(PyValueError::new_err)?;
        engine.run_at(&positions).map_err(PyRuntimeError::new_err)
    }

    /// Check that the engine can report at a position on the axis
    ///
    /// Args:
    ///     position: A position on the axis
    ///
    /// Raises:
    ///     ValueError: If the engine does not report there
    fn check_position(&self, position: f64) -> PyResult<()> {
        self.inner
            .check_position(position)
            .map_err(PyValueError::new_err)
    }

    /// Advance one step
    fn step(&mut self) -> PyResult<()> {
        SimulationEngine::step(&mut self.inner).map_err(PyRuntimeError::new_err)
    }

    /// Current population of each compartment
    #[getter]
    fn population(&self) -> Vec<f64> {
        SimulationEngine::population(&self.inner)
    }

    /// Names of the compartments (e.g., ["S", "I_young", "R_old"])
    #[getter]
    fn compartments(&self) -> Vec<String> {
        SimulationEngine::compartments(&self.inner)
    }

    /// Names of the simulation outputs, including accumulators
    #[getter]
    fn output_names(&self) -> Vec<String> {
        SimulationEngine::output_names(&self.inner)
    }

    /// Reset the simulation to initial conditions
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Set a parameter value
    ///
    /// Args:
    ///     parameter_id: The parameter identifier
    ///     value: The new parameter value
    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> PyResult<()> {
        self.inner
            .set_parameter(parameter_id, value)
            .map_err(PyValueError::new_err)
    }

    /// All current parameter values, by name
    fn get_parameters(&self) -> HashMap<String, f64> {
        self.inner.get_parameters().clone()
    }

    /// Current position on the axis
    fn current_position(&self) -> f64 {
        self.inner.current_position()
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
        self.inner
            .set_initial_condition(compartment_index, value)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "Engine(type={}, compartments={}, position={})",
            self.inner.engine_type().name(),
            SimulationEngine::compartments(&self.inner).len(),
            self.inner.current_position()
        )
    }
}

impl PyEngine {
    /// A copy of the Rust engine.
    pub fn engine(&self) -> Engine {
        self.inner.clone()
    }
}

/// One entry of the list of engines a model can name
#[pyclass(name = "EngineType", frozen)]
pub struct PyEngineType {
    inner: EngineType,
}

#[pymethods]
impl PyEngineType {
    /// The entry with a name
    ///
    /// Args:
    ///     name: Name of the engine, for example "DifferentialEquations"
    ///
    /// Raises:
    ///     ValueError: If no entry has the name; the message lists them all
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: EngineType::from_name(name).map_err(PyValueError::new_err)?,
        })
    }

    /// Names of every entry of the list, in order
    #[staticmethod]
    fn names() -> Vec<&'static str> {
        EngineType::ALL.iter().map(|engine| engine.name()).collect()
    }

    /// Name of the engine
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// The kind of model the engine runs, for example "compartmental"
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind().name()
    }

    /// What a rate means to the engine: "per_step" (the amount or fraction
    /// a flow moves in one step) or "continuous" (an instantaneous rate per
    /// unit of the axis)
    #[getter]
    fn rate_meaning(&self) -> &'static str {
        self.inner.rate_meaning().name()
    }

    /// Left-hand side of the equation of a variable
    ///
    /// Args:
    ///     variable: Name of the variable, already formatted
    ///     latex: Whether to write it in LaTeX
    fn equation_lhs(&self, variable: &str, latex: bool) -> String {
        self.inner.equation_lhs(variable, latex)
    }

    /// Check settings against the engine
    ///
    /// Args:
    ///     settings_json: The settings as a JSON object, or None for the
    ///         defaults
    ///
    /// Raises:
    ///     ValueError: If the settings are not valid for the engine; when
    ///         they are settings of another engine, the message names both
    #[pyo3(signature = (settings_json=None))]
    fn check_settings(&self, settings_json: Option<&str>) -> PyResult<()> {
        let settings = settings_json
            .map(serde_json::from_str::<serde_json::Value>)
            .transpose()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .settings(settings.as_ref())
            .map(|_| ())
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!("EngineType('{}')", self.inner.name())
    }
}

/// Register the engine classes with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyEngineType>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use commol_core::{EngineSettings, Model, ModelEngine, OdeSolver};

    /// A → B at the per-capita rate 0.1, run by `engine`.
    fn decay_model(engine: EngineType) -> Model {
        let mut model: Model = serde_json::from_str(
            r#"{"name":"m","population":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
            "accumulators":[],"stratifications":[],"transitions":[],
            "initial_conditions":{"population_size":1000,"bin_fractions":[
            {"bin":"A","fraction":1.0},{"bin":"B","fraction":0.0}],
            "stratification_fractions":[]}},"parameters":[],
            "dynamics":{"transitions":[{"id":"ab","source":["A"],"target":["B"],
            "rate":"0.1","stratified_rates":null,"per_compartment":null}]},
            "engine":{"type":"DifferenceEquations"}}"#,
        )
        .expect("valid model JSON");
        model.engine = ModelEngine::new(engine);
        model
    }

    #[test]
    fn every_entry_builds_its_engine_and_runs() {
        for engine_type in EngineType::ALL {
            let mut engine = Engine::build(&decay_model(engine_type)).unwrap();
            assert_eq!(engine.engine_type(), engine_type);
            let rows = SimulationEngine::run(&mut engine, 1).unwrap().rows;
            assert!(rows[1][0] < rows[0][0], "{engine_type:?} moved nothing");
        }
    }

    #[test]
    fn the_engine_takes_the_settings_of_the_model() {
        let mut model = decay_model(EngineType::DifferentialEquations);
        let settings = OdeSolver {
            rtol: 1e-9,
            ..OdeSolver::default()
        };
        model.engine = ModelEngine::with_settings(EngineSettings::DifferentialEquations(settings));
        let Engine::Differential(engine) = Engine::build(&model).unwrap() else {
            panic!("expected the differential engine");
        };
        assert_eq!(*engine.settings(), settings);
    }
}
