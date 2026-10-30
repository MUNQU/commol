//! The engines of the list in [`commol_core::EngineType`], for Python.
//!
//! [`Engine`] holds the engine a model names, so each binding exists once for
//! every engine. [`PyEngine`] builds it from a model, and [`PyEngineType`]
//! lets Python ask the list instead of branching on engine names.

use crate::parallelism::parallel;
use commol_core::{
    Accuracy, AccuracySettings, EngineType, ModelKind, ModelValue, RateKeeps, SimulationEngine,
    SimulationError, ValuePlan,
};
use commol_difference::DifferenceEquations;
use commol_differential::DifferentialEquations;
use commol_model::{CompartmentalEngine, InitialConditionPlan};
use commol_stochastic::{StochasticEvents, StochasticSteps};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[cfg(feature = "test-kind")]
use crate::test_kind::{TestEngine, TestStochasticEngine, TestValuePlan};

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
    /// Continuous time, at random.
    StochasticEvents(StochasticEvents),
    /// Discrete time, at random.
    StochasticSteps(StochasticSteps),
    /// The engine of the test model kind.
    #[cfg(feature = "test-kind")]
    Test(TestEngine),
    /// The stochastic engine of the test model kind.
    #[cfg(feature = "test-kind")]
    TestStochastic(TestStochasticEngine),
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
            // coverage: an arm for the test model kind: its engines apply values with no excess
            // (test_kind.rs, apply_values), so no infeasibility is asked
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
            (EngineType::StochasticEvents, ModelKind::Aggregate) => {
                StochasticEvents::try_from_model(model).map(Self::StochasticEvents)
            }
            (EngineType::StochasticSteps, ModelKind::Aggregate) => {
                StochasticSteps::try_from_model(model).map(Self::StochasticSteps)
            }
            #[cfg(feature = "test-kind")]
            (EngineType::TestEngine, ModelKind::Test) => {
                TestEngine::try_from_model(model).map(Self::Test)
            }
            #[cfg(feature = "test-kind")]
            (EngineType::TestStochasticEngine, ModelKind::Test) => {
                TestStochasticEngine::try_from_model(model).map(Self::TestStochastic)
            }
            // coverage: an invariant, model.check_kind() accepts only the pairs above
            #[allow(unreachable_patterns)]
            _ => unreachable!("check_kind accepts only the pairs above"),
        }
    }

    /// The entry of the list this engine is.
    pub fn engine_type(&self) -> EngineType {
        match self {
            Self::Difference(_) => EngineType::DifferenceEquations,
            Self::Differential(_) => EngineType::DifferentialEquations,
            Self::StochasticEvents(_) => EngineType::StochasticEvents,
            Self::StochasticSteps(_) => EngineType::StochasticSteps,
            #[cfg(feature = "test-kind")]
            Self::Test(_) => EngineType::TestEngine,
            #[cfg(feature = "test-kind")]
            Self::TestStochastic(_) => EngineType::TestStochasticEngine,
        }
    }
}

/// Call `$body` with `$engine` bound to the engine inside an [`Engine`].
macro_rules! dispatch {
    ($self:expr, $engine:ident => $body:expr) => {
        match $self {
            Engine::Difference($engine) => $body,
            Engine::Differential($engine) => $body,
            Engine::StochasticEvents($engine) => $body,
            Engine::StochasticSteps($engine) => $body,
            #[cfg(feature = "test-kind")]
            Engine::Test($engine) => $body,
            #[cfg(feature = "test-kind")]
            Engine::TestStochastic($engine) => $body,
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
            Engine::StochasticEvents($engine) => Ok::<_, String>($body),
            Engine::StochasticSteps($engine) => Ok::<_, String>($body),
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
    let state = compartmental!(engine, engine => engine.initial_state())?;
    match plan {
        EngineValuePlan::Aggregate(plan) => Ok((state, plan.covered_compartments())),
        // coverage: an arm for the test model kind: compartmental! rejects the engine of a test
        // plan first
        #[cfg(feature = "test-kind")]
        EngineValuePlan::Test(_) => Err("The plan is not of an aggregate model".to_string()),
    }
}

/// The compartments each initial-condition value of a plan sets, in the
/// order of the values, and those of the remainder.
///
/// # Errors
///
/// Returns a message when the plan is not of an aggregate model.
pub fn compartmental_coverage(
    plan: &EngineValuePlan,
) -> Result<(Vec<Vec<usize>>, Vec<usize>), String> {
    match plan {
        EngineValuePlan::Aggregate(plan) => Ok(plan.coverage()),
        // coverage: an arm for the test model kind: only Aggregate asks the coverage (aggregate.py,
        // initial_condition_coverage)
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

    fn matrices(&self) -> Vec<commol_core::ScaledMatrix> {
        dispatch!(self, engine => engine.matrices())
    }

    fn accuracy(&self) -> Accuracy {
        dispatch!(self, engine => engine.accuracy())
    }

    fn accuracy_settings(&self) -> Option<AccuracySettings> {
        dispatch!(self, engine => engine.accuracy_settings())
    }

    fn at_accuracy(&self, accuracy: Accuracy) -> Option<Self> {
        match self {
            Self::Difference(engine) => engine.at_accuracy(accuracy).map(Self::Difference),
            Self::Differential(engine) => engine.at_accuracy(accuracy).map(Self::Differential),
            Self::StochasticEvents(engine) => {
                engine.at_accuracy(accuracy).map(Self::StochasticEvents)
            }
            Self::StochasticSteps(engine) => {
                engine.at_accuracy(accuracy).map(Self::StochasticSteps)
            }
            #[cfg(feature = "test-kind")]
            Self::Test(engine) => engine.at_accuracy(accuracy).map(Self::Test),
            #[cfg(feature = "test-kind")]
            Self::TestStochastic(engine) => engine.at_accuracy(accuracy).map(Self::TestStochastic),
        }
    }

    fn reset(&mut self) {
        dispatch!(self, engine => engine.reset())
    }

    fn set_seed(&mut self, seed: u64) {
        dispatch!(self, engine => engine.set_seed(seed))
    }

    fn engine_type(&self) -> Option<EngineType> {
        Some(Engine::engine_type(self))
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        dispatch!(self, engine => engine.set_parameter(parameter_id, value))
    }

    fn check_values(&self) -> Result<(), String> {
        dispatch!(self, engine => engine.check_values())
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
            Self::StochasticEvents(engine) => engine
                .plan_values(values, remainder)
                .map(EngineValuePlan::Aggregate),
            Self::StochasticSteps(engine) => engine
                .plan_values(values, remainder)
                .map(EngineValuePlan::Aggregate),
            #[cfg(feature = "test-kind")]
            Self::Test(engine) => engine
                .plan_values(values, remainder)
                .map(EngineValuePlan::Test),
            #[cfg(feature = "test-kind")]
            Self::TestStochastic(engine) => engine
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
            (Self::StochasticEvents(engine), EngineValuePlan::Aggregate(plan)) => {
                engine.apply_values(plan, values)
            }
            (Self::StochasticSteps(engine), EngineValuePlan::Aggregate(plan)) => {
                engine.apply_values(plan, values)
            }
            #[cfg(feature = "test-kind")]
            (Self::Test(engine), EngineValuePlan::Test(plan)) => engine.apply_values(plan, values),
            #[cfg(feature = "test-kind")]
            (Self::TestStochastic(engine), EngineValuePlan::Test(plan)) => {
                engine.apply_values(plan, values)
            }
            // coverage: an arm for the test model kind: an engine applies only the plans of its own
            // plan_values
            #[cfg(feature = "test-kind")]
            (engine, _) => Err(format!(
                "The plan of values is not a plan of the {} engine",
                engine.engine_type().name()
            )),
        }
    }
}

/// The engine a model names, ready to simulate it.
/// The positions of the replicates: a number of steps on the default grid of
/// the engine, or the positions themselves.
#[derive(FromPyObject)]
pub enum ReplicateGrid {
    /// The default grid of this many steps.
    Steps(i64),
    /// Sorted, unique positions.
    Positions(Vec<f64>),
}

#[pyclass(name = "Engine")]
pub struct PyEngine {
    inner: Engine,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyEngine {
    /// Build the engine the model names, and compile the model into it
    ///
    /// Args:
    ///     model: The JSON of the model to simulate
    ///
    /// Raises:
    ///     ValueError: If the JSON is not a model
    ///     RuntimeError: If the model cannot be compiled (for example an
    ///         initial-condition override names a compartment that is not
    ///         part of the model expansion)
    #[new]
    fn new(model: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Engine::build(&crate::core::model(model)?).map_err(PyRuntimeError::new_err)?,
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
    ///     ValueError: If a parameter or an initial amount has no value
    ///     InfeasibleStateError: If the state leaves the values the model
    ///         allows, such as an amount below zero
    ///     RuntimeError: If the simulation fails for another reason
    fn run(&mut self, num_steps: i64) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let num_steps: u32 = crate::counts::count(&commol_core::NUM_STEPS, num_steps)?;
        self.inner.check_values().map_err(PyValueError::new_err)?;
        let trajectory =
            SimulationEngine::run(&mut self.inner, num_steps).map_err(simulation_error)?;
        Ok((trajectory.positions, trajectory.rows))
    }

    /// Run replicates of a stochastic engine in parallel, each from the
    /// initial state with a seed of its own
    ///
    /// Replicate `i` takes the seed `replicate_seed(seed, i)`, the SplitMix64
    /// of the SplitMix64 of `seed` plus `i`, so two seeds share no replicate,
    /// and a replicate gives the same run however many replicates run.
    ///
    /// Args:
    ///     seed: The seed of the replicates, from 0 to 2**64 - 1; None draws
    ///         one
    ///     count: How many replicates to run
    ///     grid: A number of steps, to run on the default grid of that many
    ///         steps, or the sorted, unique positions to report at
    ///
    /// Returns:
    ///     The seed of the replicates, the positions, the seed of each
    ///     replicate, and one list of output rows per replicate
    ///
    /// Raises:
    ///     ValueError: If a parameter or an initial amount has no value,
    ///         or a position is not valid for the engine
    ///     InfeasibleStateError: If a replicate leaves the values the model
    ///         allows
    ///     RuntimeError: If a replicate fails for another reason
    #[allow(clippy::type_complexity)]
    fn run_replicates(
        &self,
        py: Python<'_>,
        seed: Option<i128>,
        count: i64,
        grid: ReplicateGrid,
    ) -> PyResult<(u64, Vec<f64>, Vec<u64>, Vec<Vec<Vec<f64>>>)> {
        let seed: Option<u64> = crate::counts::optional_count(&commol_core::SEED, seed)?;
        let count: u64 = crate::counts::count(&commol_core::REPLICATE_COUNT, count)?;
        self.inner.check_values().map_err(PyValueError::new_err)?;
        let seed = commol_random::seed_or_drawn(seed);
        let mut start = self.inner.clone();
        start.reset();
        let positions = match grid {
            ReplicateGrid::Steps(num_steps) => {
                start.grid(crate::counts::count(&commol_core::NUM_STEPS, num_steps)?)
            }
            ReplicateGrid::Positions(positions) => positions,
        };
        commol_core::check_positions(&positions, 0.0, |position| start.check_position(position))
            .map_err(PyValueError::new_err)?;
        let seeds: Vec<u64> = (0..count)
            .map(|index| commol_random::replicate_seed(seed, index))
            .collect();
        let runs = parallel(py, || {
            commol_core::run_replicates(&start, &positions, &seeds)
        })?
        .map_err(simulation_error)?;
        Ok((seed, positions, seeds, runs))
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
    ///     ValueError: If a parameter or an initial amount has no value, a
    ///         position is not valid for the engine (a difference engine
    ///         reports only at integer steps), or the positions are not
    ///         sorted and unique
    ///     InfeasibleStateError: If the state leaves the values the model
    ///         allows, such as an amount below zero
    ///     RuntimeError: If the simulation fails for another reason
    fn run_at(&mut self, positions: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        let engine = &mut self.inner;
        engine.check_values().map_err(PyValueError::new_err)?;
        commol_core::check_positions(&positions, engine.current_position(), |position| {
            engine.check_position(position)
        })
        .map_err(PyValueError::new_err)?;
        engine.run_at(&positions).map_err(simulation_error)
    }

    /// Advance one step
    ///
    /// Raises:
    ///     ValueError: If a parameter or an initial amount has no value
    ///     InfeasibleStateError: If the state leaves the values the model
    ///         allows, such as an amount below zero
    ///     RuntimeError: If the step fails for another reason
    fn step(&mut self) -> PyResult<()> {
        self.inner.check_values().map_err(PyValueError::new_err)?;
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
    ///
    /// A stochastic engine also restores the generator of its seed, so a run
    /// after a reset repeats the run before it.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Seed the random numbers of the engine, and restore its generator
    ///
    /// The seed acts from the current position: the state there stays, and
    /// every draw after it comes from the new seed. A deterministic engine
    /// ignores the seed.
    ///
    /// Args:
    ///     seed: The seed, from 0 to 2**64 - 1; None draws one
    ///
    /// Returns:
    ///     The seed the engine runs with
    ///
    /// Raises:
    ///     ValueError: With the message of the rule of a seed, when the seed
    ///         is outside its range
    #[pyo3(signature = (seed=None))]
    fn set_seed(&mut self, seed: Option<i128>) -> PyResult<u64> {
        let seed =
            commol_random::seed_or_drawn(crate::counts::optional_count(&commol_core::SEED, seed)?);
        self.inner.set_seed(seed);
        Ok(seed)
    }

    /// Current position on the axis
    fn current_position(&self) -> f64 {
        self.inner.current_position()
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

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyEngineType {
    /// The entry with a name
    ///
    /// Args:
    ///     name: The JSON of the name of the engine, for example
    ///         `"DifferentialEquations"` in quotes
    ///
    /// Raises:
    ///     ValueError: If no entry has the name; the message lists them all
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: crate::json::read(name)?,
        })
    }

    /// Names of every entry of the list, in order
    #[staticmethod]
    fn names() -> Vec<&'static str> {
        EngineType::ALL.iter().map(|engine| engine.name()).collect()
    }

    /// The kinds of model the engine runs, for example ["aggregate"]
    #[getter]
    fn kinds(&self) -> Vec<&'static str> {
        self.inner.kinds().iter().map(|kind| kind.name()).collect()
    }

    /// Whether the engine draws random numbers, so a run depends on its seed
    #[getter]
    fn is_stochastic(&self) -> bool {
        self.inner.is_stochastic()
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
    ///     keep: The JSON of what the rate keeps of the residence:
    ///         "survival" or "mean"
    ///
    /// Raises:
    ///     ValueError: If `keep` is another value, the span is not positive
    ///         and finite, or a per-step rate keeps the mean of a span shorter
    ///         than one step
    fn rate_from_mean_span(&self, span: f64, keep: &str) -> PyResult<f64> {
        let keep: RateKeeps = crate::json::read(keep)?;
        self.inner
            .rate_meaning()
            .rate_from_mean_span(span, keep)
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

    /// How the engine advances the state: "difference_equation",
    /// "differential_equation", "stochastic_events" or "rules"
    #[getter]
    fn dynamics_form(&self) -> &'static str {
        self.inner.dynamics_form().name()
    }

    /// How the engine advances, in words, for a description of a model
    #[getter]
    fn schedule(&self) -> &'static str {
        self.inner.schedule()
    }

    /// Left-hand side of the equation of a variable, or None when the form
    /// of the dynamics of the engine has no equations
    ///
    /// Args:
    ///     variable: Name of the variable, already formatted
    ///     latex: Whether to write it in LaTeX
    fn equation_lhs(&self, variable: &str, latex: bool) -> Option<String> {
        self.inner.dynamics_form().equation_lhs(variable, latex)
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
    ///     kind: The JSON of the name of the kind of a model structure, for
    ///         example `"aggregate"` in quotes
    ///
    /// Raises:
    ///     ValueError: If the kind is unknown, or the engine does not run
    ///         it; the message names the engine, the kinds and the engines
    ///         of the kind
    fn check_kind(&self, kind: &str) -> PyResult<()> {
        let kind: ModelKind = crate::json::read(kind)?;
        self.inner.check_kind(kind).map_err(PyValueError::new_err)
    }
}

/// One kind of model the engines of the list run
#[pyclass(name = "ModelKind", frozen)]
pub struct PyModelKind {
    inner: ModelKind,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyModelKind {
    /// The kind with a name
    ///
    /// Args:
    ///     name: The JSON of the name of the kind, for example `"aggregate"`
    ///         in quotes
    ///
    /// Raises:
    ///     ValueError: If no kind has the name; the message lists them all
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: crate::json::read(name)?,
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

    /// Types of value of the kind that calibration may set besides
    /// parameters and factors, for example ["initial_condition"]
    fn value_types(&self) -> Vec<&'static str> {
        self.inner
            .value_types()
            .iter()
            .map(|value_type| value_type.name())
            .collect()
    }
}

/// Register the engine classes with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ODE_DEFAULT_RTOL", commol_core::OdeSolver::DEFAULT_RTOL)?;
    m.add("ODE_DEFAULT_ATOL", commol_core::OdeSolver::DEFAULT_ATOL)?;
    use commol_core::StochasticEventsSettings as Events;
    m.add("STOCHASTIC_EVENTS_DEFAULT_EPSILON", Events::DEFAULT_EPSILON)?;
    m.add(
        "STOCHASTIC_EVENTS_DEFAULT_TOLERANCE",
        Events::DEFAULT_TOLERANCE,
    )?; // coverage: an error that only PyO3 returns, when the module cannot take it
    m.add_class::<PyEngine>()?;
    m.add_class::<PyEngineType>()?;
    m.add_class::<PyModelKind>()?;
    m.add(
        "InfeasibleStateError",
        m.py().get_type::<InfeasibleStateError>(),
    )?; // coverage: an error that only PyO3 returns, when the module cannot take it
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use commol_core::{EngineSettings, Model, ModelEngine, OdeSolver};

    /// An accuracy tighter than the default tolerances.
    const TIGHTER: Accuracy = Accuracy {
        relative: 1e-9,
        absolute: 1e-9,
    };

    /// A → B at the proportional rate 0.1, run by `engine`.
    fn decay_model(engine: EngineType) -> Model {
        let mut model: Model = serde_json::from_str(
            r#"{"name":"m","parameters":[],"engine":{"type":"DifferenceEquations"},
            "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
            "accumulators":[],"stratifications":[],
            "transitions":[{"id":"ab","source":["A"],"target":["B"],
            "rate":"0.1","stratified_rates":null,"per_compartment":null}]},
            "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#,
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
            // Only the engine that integrates with tolerances runs at another accuracy.
            let ode = engine_type == EngineType::DifferentialEquations;
            let tighter = engine.at_accuracy(TIGHTER);
            assert_eq!(
                tighter.map(|engine| engine.accuracy()),
                ode.then_some(TIGHTER),
                "{engine_type:?}"
            );
            assert_eq!(engine.accuracy_settings().is_some(), ode, "{engine_type:?}");
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
            panic!("expected the differential engine"); // coverage: the failure of a test
        };
        assert_eq!(
            SimulationEngine::accuracy(&engine),
            Accuracy {
                relative: settings.rtol,
                absolute: settings.atol
            }
        );
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
        assert!(engine.at_accuracy(TIGHTER).is_none());
    }

    /// A stochastic test model: 100 units of `x` leave at rate `k`.
    #[cfg(feature = "test-kind")]
    fn stochastic_model() -> Model {
        let mut model = test_model();
        model.engine = ModelEngine::new(EngineType::TestStochasticEngine);
        if let commol_core::ModelStructure::Test(structure) = &mut model.structure {
            structure.outputs[0].initial = 100.0;
        }
        model
    }

    #[cfg(feature = "test-kind")]
    fn stochastic_run(engine: &mut Engine, seed: u64) -> Vec<Vec<f64>> {
        engine.set_seed(seed);
        engine.reset();
        engine.run_at(&[0.5, 1.0, 2.0, 4.0]).unwrap()
    }

    #[cfg(feature = "test-kind")]
    #[test]
    fn a_stochastic_engine_repeats_a_run_from_its_seed() {
        let mut engine = Engine::build(&stochastic_model()).unwrap();
        assert!(engine.engine_type().is_stochastic());
        assert!(engine.at_accuracy(TIGHTER).is_none());

        let first = stochastic_run(&mut engine, 7);
        let again = stochastic_run(&mut engine, 7);
        let other = stochastic_run(&mut engine, 8);

        assert_eq!(first, again);
        assert_ne!(first, other);
        engine.reset();
        assert_eq!(engine.run_at(&[0.5, 1.0, 2.0, 4.0]).unwrap(), other);
    }

    #[cfg(feature = "test-kind")]
    #[test]
    fn a_stochastic_run_does_not_depend_on_where_it_reports() {
        let mut engine = Engine::build(&stochastic_model()).unwrap();
        engine.set_seed(3);
        engine.reset();
        let coarse = engine.run_at(&[4.0]).unwrap();
        engine.reset();
        let fine = engine.run_at(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        assert_eq!(coarse[0], fine[3]);
    }

    #[test]
    fn an_aggregate_stochastic_engine_repeats_a_run_from_its_seed() {
        for &engine_type in EngineType::ALL {
            if !engine_type.runs(ModelKind::Aggregate) || !engine_type.is_stochastic() {
                continue;
            }
            let mut engine = Engine::build(&decay_model(engine_type)).unwrap();
            assert_eq!(engine.engine_type(), engine_type);
            let mut run = |seed| {
                engine.set_seed(seed);
                engine.reset();
                SimulationEngine::run(&mut engine, 20).unwrap().rows
            };
            let first = run(1);
            assert_eq!(run(1), first);
            assert_ne!(run(2), first);
        }
    }

    #[test]
    fn a_deterministic_engine_ignores_the_seed() {
        for &engine_type in EngineType::ALL {
            if !engine_type.runs(ModelKind::Aggregate) || engine_type.is_stochastic() {
                continue;
            }
            let mut engine = Engine::build(&decay_model(engine_type)).unwrap();
            engine.set_seed(1);
            let first = SimulationEngine::run(&mut engine, 3).unwrap().rows;
            engine.set_seed(2);
            engine.reset();
            let second = SimulationEngine::run(&mut engine, 3).unwrap().rows;
            assert_eq!(first, second);
        }
    }
}
