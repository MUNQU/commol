//! The engines of the list in [`commol_core::EngineType`], for Python.
//!
//! [`Engine`] holds the engine a model names, so each binding exists once for
//! every engine. [`PyEngine`] builds it from a model, and [`PyEngineType`]
//! lets Python ask the list instead of branching on engine names.

use std::collections::HashMap;

use commol_core::{
    EngineType, ModelKind, ModelValue, SimulationEngine, SimulationError, ValuePlan,
};
use commol_difference::DifferenceEquations;
use commol_differential::DifferentialEquations;
use commol_model::{CompartmentalEngine, InitialConditionPlan};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[cfg(feature = "test-kind")]
use crate::test_kind::{TestEngine, TestValuePlan};

pyo3::create_exception!(
    commol,
    InfeasibleStateError,
    PyValueError,
    "A run left the values its model allows, such as an amount below zero."
);

/// The Python error for a run that stopped: [`InfeasibleStateError`] for an
/// infeasible state, `RuntimeError` for any other error.
pub(crate) fn simulation_error(error: SimulationError) -> PyErr {
    match error {
        SimulationError::Infeasible { message, .. } => InfeasibleStateError::new_err(message),
        SimulationError::Failed(message) => PyRuntimeError::new_err(message),
    }
}

/// The engine of one entry of [`EngineType`].
#[derive(Clone)]
pub enum Engine {
    /// Discrete time.
    Difference(DifferenceEquations),
    /// Continuous time.
    Differential(DifferentialEquations),
    /// The engine of the test model kind.
    #[cfg(feature = "test-kind")]
    Test(TestEngine),
}

/// The values of the model kind of an [`Engine`], resolved by it.
#[derive(Clone, Debug)]
pub enum EngineValuePlan {
    /// Initial conditions of an aggregate model.
    Aggregate(InitialConditionPlan),
    /// The test model kind has none.
    #[cfg(feature = "test-kind")]
    Test(TestValuePlan),
}

impl ValuePlan for EngineValuePlan {
    fn infeasibility(&self, excess: f64) -> String {
        match self {
            Self::Aggregate(plan) => plan.infeasibility(excess),
            #[cfg(feature = "test-kind")]
            Self::Test(plan) => plan.infeasibility(excess),
        }
    }
}

impl Engine {
    /// Build the engine the model names, for the kind of its structure.
    ///
    /// An engine names a method; the implementation is picked by the pair
    /// of the engine and the kind of the structure.
    ///
    /// # Errors
    ///
    /// Returns a message when the engine does not run the kind of the model
    /// (it names the engine, both kinds and the engines of the kind of the
    /// model), or cannot compile the model.
    pub fn build(model: &commol_core::Model) -> Result<Self, String> {
        model.check_kind()?;
        match (model.engine.engine_type(), model.structure.kind()) {
            (EngineType::DifferenceEquations, ModelKind::Aggregate) => {
                DifferenceEquations::try_from_model(model).map(Self::Difference)
            }
            (EngineType::DifferentialEquations, ModelKind::Aggregate) => {
                DifferentialEquations::try_from_model(model).map(Self::Differential)
            }
            #[cfg(feature = "test-kind")]
            (EngineType::TestEngine, ModelKind::Test) => {
                TestEngine::try_from_model(model).map(Self::Test)
            }
            #[allow(unreachable_patterns)]
            (engine, kind) => Err(format!(
                "The {} engine has no implementation for {} models; the engines of {} \
                 models are {:?}",
                engine.name(),
                kind.name(),
                kind.name(),
                EngineType::engines_of(kind)
                    .into_iter()
                    .map(EngineType::name)
                    .collect::<Vec<_>>()
            )),
        }
    }

    /// The entry of the list this engine is.
    pub fn engine_type(&self) -> EngineType {
        match self {
            Self::Difference(_) => EngineType::DifferenceEquations,
            Self::Differential(_) => EngineType::DifferentialEquations,
            #[cfg(feature = "test-kind")]
            Self::Test(_) => EngineType::TestEngine,
        }
    }
}

/// Call `$body` with `$engine` bound to the engine inside an [`Engine`].
macro_rules! dispatch {
    ($self:expr, $engine:ident => $body:expr) => {
        match $self {
            Engine::Difference($engine) => $body,
            Engine::Differential($engine) => $body,
            #[cfg(feature = "test-kind")]
            Engine::Test($engine) => $body,
        }
    };
}

/// Call `$body` with `$engine` bound to the compartmental engine inside an
/// [`Engine`], or return an error that names the engine.
macro_rules! compartmental {
    ($self:expr, $engine:ident => $body:expr) => {
        match $self {
            Engine::Difference($engine) => Ok::<_, String>($body),
            Engine::Differential($engine) => Ok::<_, String>($body),
            #[cfg(feature = "test-kind")]
            other => Err(format!(
                "The {} engine of this model has no compartments; only an aggregate model \
                 has them",
                other.engine_type().name(),
            )),
        }
    };
}

/// The initial state of a compartmental engine, and the compartments a
/// plan of initial conditions covers.
///
/// # Errors
///
/// Returns a message when the engine or the plan is not of an aggregate
/// model.
pub fn compartmental_state(
    engine: &Engine,
    plan: &EngineValuePlan,
) -> Result<(Vec<f64>, Vec<usize>), String> {
    let state = compartmental!(engine, engine => engine.state())?;
    match plan {
        EngineValuePlan::Aggregate(plan) => Ok((state, plan.covered_compartments())),
        #[cfg(feature = "test-kind")]
        EngineValuePlan::Test(_) => Err("The plan is not of an aggregate model".to_string()),
    }
}

impl SimulationEngine for Engine {
    type ValuePlan = EngineValuePlan;

    fn check_position(&self, position: f64) -> Result<(), String> {
        dispatch!(self, engine => engine.check_position(position))
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        dispatch!(self, engine => engine.run_at_positions_into_buffer(positions, buffer))
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        dispatch!(self, engine => SimulationEngine::step(engine))
    }

    fn output_names(&self) -> Vec<String> {
        dispatch!(self, engine => SimulationEngine::output_names(engine))
    }

    fn state_outputs(&self) -> usize {
        dispatch!(self, engine => engine.state_outputs())
    }

    fn output_sums(&self) -> Vec<(String, Vec<usize>)> {
        dispatch!(self, engine => engine.output_sums())
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

    fn plan_values(
        &self,
        values: &[ModelValue],
        remainder: Option<&str>,
    ) -> Result<EngineValuePlan, String> {
        match self {
            Self::Difference(engine) => engine
                .plan_values(values, remainder)
                .map(EngineValuePlan::Aggregate),
            Self::Differential(engine) => engine
                .plan_values(values, remainder)
                .map(EngineValuePlan::Aggregate),
            #[cfg(feature = "test-kind")]
            Self::Test(engine) => engine
                .plan_values(values, remainder)
                .map(EngineValuePlan::Test),
        }
    }

    fn apply_values(&mut self, plan: &EngineValuePlan, values: &[f64]) -> Result<f64, String> {
        match (self, plan) {
            (Self::Difference(engine), EngineValuePlan::Aggregate(plan)) => {
                engine.apply_values(plan, values)
            }
            (Self::Differential(engine), EngineValuePlan::Aggregate(plan)) => {
                engine.apply_values(plan, values)
            }
            #[cfg(feature = "test-kind")]
            (Self::Test(engine), EngineValuePlan::Test(plan)) => engine.apply_values(plan, values),
            #[cfg(feature = "test-kind")]
            (engine, _) => Err(format!(
                "The plan of values is not a plan of the {} engine",
                engine.engine_type().name()
            )),
        }
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
    ///
    /// Raises:
    ///     InfeasibleStateError: If the state leaves the values the model
    ///         allows, such as an amount below zero
    ///     RuntimeError: If the simulation fails for another reason
    fn run(&mut self, num_steps: u32) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let trajectory =
            SimulationEngine::run(&mut self.inner, num_steps).map_err(simulation_error)?;
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
    ///     InfeasibleStateError: If the state leaves the values the model
    ///         allows, such as an amount below zero
    ///     RuntimeError: If the simulation fails for another reason
    fn run_at(&mut self, positions: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        let engine = &mut self.inner;
        commol_core::check_positions(&positions, engine.current_position(), |position| {
            engine.check_position(position)
        })
        .map_err(PyValueError::new_err)?;
        engine.run_at(&positions).map_err(simulation_error)
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
    ///
    /// Raises:
    ///     InfeasibleStateError: If the state leaves the values the model
    ///         allows, such as an amount below zero
    ///     RuntimeError: If the step fails for another reason
    fn step(&mut self) -> PyResult<()> {
        SimulationEngine::step(&mut self.inner).map_err(simulation_error)
    }

    /// Current amount of each compartment of an aggregate model
    ///
    /// Raises:
    ///     ValueError: If the model is not aggregate
    #[getter]
    fn state(&self) -> PyResult<Vec<f64>> {
        compartmental!(&self.inner, engine => engine.state()).map_err(PyValueError::new_err)
    }

    /// Names of the compartments of an aggregate model (e.g., ["A",
    /// "B_g1", "C_g2"])
    ///
    /// Raises:
    ///     ValueError: If the model is not aggregate
    #[getter]
    fn compartments(&self) -> PyResult<Vec<String>> {
        compartmental!(&self.inner, engine => engine.compartments()).map_err(PyValueError::new_err)
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
    ///     value: The new initial amount
    ///
    /// Raises:
    ///     ValueError: If the model is not aggregate, compartment_index is
    ///         invalid or value is negative
    fn set_initial_condition(&mut self, compartment_index: usize, value: f64) -> PyResult<()> {
        compartmental!(&mut self.inner, engine => engine.set_initial_condition(compartment_index, value))
            .and_then(|result| result)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "Engine(type={}, outputs={}, position={})",
            self.inner.engine_type().name(),
            self.inner.output_names().len(),
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

    /// The kinds of model the engine runs, for example ["aggregate"]
    #[getter]
    fn kinds(&self) -> Vec<&'static str> {
        self.inner.kinds().iter().map(|kind| kind.name()).collect()
    }

    /// What a rate means to the engine: "per_step" (the amount or fraction
    /// a flow moves in one step) or "continuous" (an instantaneous rate per
    /// unit of the axis)
    #[getter]
    fn rate_meaning(&self) -> &'static str {
        self.inner.rate_meaning().name()
    }

    /// The rate of one outflow whose residence has a mean span, as the
    /// engine reads a rate
    ///
    /// Args:
    ///     span: Mean span of the residence, in steps
    ///
    /// Raises:
    ///     ValueError: If the span is not positive and finite
    fn rate_from_mean_span(&self, span: f64) -> PyResult<f64> {
        self.inner
            .rate_meaning()
            .rate_from_mean_span(span)
            .map_err(PyValueError::new_err)
    }

    /// The rate of one outflow that moves a probability of the bin over a
    /// period, as the engine reads a rate
    ///
    /// Args:
    ///     probability: Fraction of the bin that leaves over the period
    ///     period: Length of the period, in steps
    ///
    /// Raises:
    ///     ValueError: If the period is not positive and finite, the
    ///         probability is outside [0, 1], or the rate is continuous and
    ///         the probability is 1
    fn rate_from_probability(&self, probability: f64, period: f64) -> PyResult<f64> {
        self.inner
            .rate_meaning()
            .rate_from_probability(probability, period)
            .map_err(PyValueError::new_err)
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

    /// Check that the engine runs a model kind
    ///
    /// Args:
    ///     kind: Name of the kind of a model structure, for example
    ///         "aggregate"
    ///
    /// Raises:
    ///     ValueError: If the kind is unknown, or the engine does not run
    ///         it; the message names the engine, the kinds and the engines
    ///         of the kind
    fn check_kind(&self, kind: &str) -> PyResult<()> {
        let kind = ModelKind::from_name(kind).map_err(PyValueError::new_err)?;
        self.inner.check_kind(kind).map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!("EngineType('{}')", self.inner.name())
    }
}

/// One kind of model the engines of the list run
#[pyclass(name = "ModelKind", frozen)]
pub struct PyModelKind {
    inner: ModelKind,
}

#[pymethods]
impl PyModelKind {
    /// The kind with a name
    ///
    /// Args:
    ///     name: Name of the kind, for example "aggregate"
    ///
    /// Raises:
    ///     ValueError: If no kind has the name; the message lists them all
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ModelKind::from_name(name).map_err(PyValueError::new_err)?,
        })
    }

    /// Names of every kind, in order
    #[staticmethod]
    fn names() -> Vec<&'static str> {
        ModelKind::ALL.iter().map(|kind| kind.name()).collect()
    }

    /// Names of the engines that run the kind, in the order of the list
    fn engines(&self) -> Vec<&'static str> {
        EngineType::engines_of(self.inner)
            .into_iter()
            .map(EngineType::name)
            .collect()
    }

    /// Name of the kind
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// Types of value of the kind that calibration may set besides
    /// parameters and factors, for example ["initial_condition"]
    fn value_types(&self) -> Vec<&'static str> {
        self.inner
            .value_types()
            .iter()
            .map(|value_type| value_type.name())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("ModelKind('{}')", self.inner.name())
    }
}

/// Register the engine classes with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyEngineType>()?;
    m.add_class::<PyModelKind>()?;
    m.add(
        "InfeasibleStateError",
        m.py().get_type::<InfeasibleStateError>(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use commol_core::{EngineSettings, Model, ModelEngine, OdeSolver};

    /// A → B at the proportional rate 0.1, run by `engine`.
    fn decay_model(engine: EngineType) -> Model {
        let mut model: Model = serde_json::from_str(
            r#"{"name":"m","parameters":[],"engine":{"type":"DifferenceEquations"},
            "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
            "accumulators":[],"stratifications":[],
            "transitions":[{"id":"ab","source":["A"],"target":["B"],
            "rate":"0.1","stratified_rates":null,"per_compartment":null}]},
            "structure":{"kind":"aggregate","total":1000,"bin_fractions":{"A":1.0,"B":0.0}}}"#,
        )
        .expect("valid model JSON");
        model.engine = ModelEngine::new(engine);
        model
    }

    #[test]
    fn every_entry_builds_its_engine_and_runs() {
        for &engine_type in EngineType::ALL {
            if !engine_type.runs(ModelKind::Aggregate) {
                continue;
            }
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

    /// A test model: one output that starts at 10 and decays at rate `k`.
    #[cfg(feature = "test-kind")]
    fn test_model() -> Model {
        serde_json::from_str(
            r#"{"name":"t","parameters":[{"id":"k","value":0.5}],
            "engine":{"type":"TestEngine"},
            "structure":{"kind":"test","outputs":[{"id":"x","initial":10.0,"rate":"k"}]}}"#,
        )
        .expect("valid model JSON")
    }

    #[cfg(feature = "test-kind")]
    #[test]
    fn an_engine_rejects_a_model_of_another_kind() {
        let mut test_model = test_model();
        test_model.engine = ModelEngine::new(EngineType::DifferenceEquations);
        let error = Engine::build(&test_model).err().expect("a kind mismatch");
        for name in ["DifferenceEquations", "aggregate", "test", "TestEngine"] {
            assert!(error.contains(name), "{error}");
        }

        let mut aggregate = decay_model(EngineType::DifferenceEquations);
        aggregate.engine = ModelEngine::new(EngineType::TestEngine);
        let error = Engine::build(&aggregate).err().expect("a kind mismatch");
        for name in [
            "TestEngine",
            "aggregate",
            "test",
            "DifferenceEquations",
            "DifferentialEquations",
        ] {
            assert!(error.contains(name), "{error}");
        }
    }

    #[cfg(feature = "test-kind")]
    #[test]
    fn the_test_engine_decays_each_output() {
        let mut engine = Engine::build(&test_model()).unwrap();
        let rows = engine.run_at(&[0.0, 2.0]).unwrap();
        assert_eq!(rows[0], vec![10.0]);
        assert!((rows[1][0] - 10.0 * (-1.0f64).exp()).abs() < 1e-12);
        assert!(engine.plan_values(&[], Some("x")).is_err());
    }
}
