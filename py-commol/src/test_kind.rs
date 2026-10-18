//! The engines of the test model kind, compiled only with the `test-kind`
//! feature.
//!
//! Each output starts at its initial value and decays exponentially at the
//! rate of a parameter: `x(t) = initial * exp(-rate * t)` ([`TestEngine`]).
//! In [`TestStochasticEngine`] each unit of an output leaves after an
//! exponential time of that rate, so the mean of its runs is the same curve.
//! The engines exist to test that the generic layers run a kind that is not
//! aggregate, and a stochastic engine.

use std::collections::HashMap;

use commol_core::{
    DecayOutput, Model, ModelStructure, ModelValue, ParameterValue, SimulationEngine,
    SimulationError, TestStructure, ValuePlan,
};
use commol_random::{Generator, SumTree, sample};

/// The engine of the test model kind.
#[derive(Clone)]
pub struct TestEngine {
    outputs: Vec<DecayOutput>,
    parameters: HashMap<String, f64>,
    position: f64,
}

/// The test kind takes no values besides parameters, so its plan is empty.
#[derive(Clone, Debug)]
pub struct TestValuePlan;

impl ValuePlan for TestValuePlan {
    fn infeasibility(&self, excess: f64) -> String {
        format!("A test model has no infeasible state, got {excess}")
    }
}

impl TestEngine {
    /// Build the engine of a test model.
    ///
    /// # Errors
    ///
    /// Returns a message when the model is not a test model, or its process
    /// is not empty: a test model runs no process.
    pub fn try_from_model(model: &Model) -> Result<Self, String> {
        let (structure, parameters) = test_model(model)?;
        Ok(Self {
            outputs: structure.outputs.clone(),
            parameters,
            position: 0.0,
        })
    }

    /// A test model takes no values besides parameters, so its plan is empty.
    fn plan_test_values(
        values: &[ModelValue],
        remainder: Option<&str>,
    ) -> Result<TestValuePlan, String> {
        if let Some(value) = values.first() {
            return Err(format!(
                "A test model takes no values of type {}; '{}' is one",
                value.value_type.name(),
                value.id
            ));
        }
        if let Some(remainder) = remainder {
            return Err(format!("A test model has no remainder; got '{remainder}'"));
        }
        Ok(TestValuePlan)
    }

    fn row(&self, position: f64) -> Result<Vec<f64>, String> {
        self.outputs
            .iter()
            .map(|output| {
                let rate = self
                    .parameters
                    .get(&output.rate)
                    .ok_or_else(|| format!("Parameter '{}' has no value", output.rate))?;
                Ok(output.initial * (-rate * position).exp())
            })
            .collect()
    }
}

/// The structure and the constant parameters of a test model.
fn test_model(model: &Model) -> Result<(&TestStructure, HashMap<String, f64>), String> {
    let ModelStructure::Test(structure) = &model.structure else {
        return Err(format!(
            "Model '{}' is a {} model, not a test model",
            model.name,
            model.structure.kind().name()
        ));
    };
    if !model.process.is_empty() {
        return Err(format!(
            "Model '{}' is a test model, which runs no process; its process must be empty",
            model.name
        ));
    }
    let parameters = model
        .parameters
        .iter()
        .filter_map(|parameter| match parameter.value {
            Some(ParameterValue::Constant(value)) => Some((parameter.id.clone(), value)),
            _ => None,
        })
        .collect();
    Ok((structure, parameters))
}

fn check_position(position: f64) -> Result<(), String> {
    if position.is_finite() && position >= 0.0 {
        Ok(())
    } else {
        Err(format!(
            "a test model reports at finite, non-negative positions; {position} is not one"
        ))
    }
}

impl SimulationEngine for TestEngine {
    type ValuePlan = TestValuePlan;

    fn check_position(&self, position: f64) -> Result<(), String> {
        check_position(position)
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        commol_core::check_positions(positions, self.position, check_position)?;
        buffer.clear();
        for &position in positions {
            buffer.push(self.row(position)?);
        }
        if let Some(&last) = positions.last() {
            self.position = last;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        self.position += 1.0;
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        self.outputs
            .iter()
            .map(|output| output.id.clone())
            .collect()
    }

    fn state_outputs(&self) -> usize {
        self.outputs.len()
    }

    fn reset(&mut self) {
        self.position = 0.0;
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.parameters.insert(parameter_id.to_string(), value);
        Ok(())
    }

    fn current_position(&self) -> f64 {
        self.position
    }

    fn plan_values(
        &self,
        values: &[ModelValue],
        remainder: Option<&str>,
    ) -> Result<TestValuePlan, String> {
        Self::plan_test_values(values, remainder)
    }

    fn apply_values(&mut self, _plan: &TestValuePlan, _values: &[f64]) -> Result<f64, String> {
        Ok(0.0)
    }
}

/// The stochastic engine of the test model kind.
///
/// Each output holds a whole number of units, its initial value rounded, and
/// each unit leaves after an exponential time at the rate of the output's
/// parameter. The engine draws one event at a time (the next time from the
/// sum of the rates, the output from a [`SumTree`]), so a run is exact and
/// reproducible from its seed, whatever positions it reports at.
#[derive(Clone)]
pub struct TestStochasticEngine {
    outputs: Vec<DecayOutput>,
    initial: Vec<u64>,
    counts: Vec<u64>,
    parameters: HashMap<String, f64>,
    position: f64,
    seed: u64,
    generator: Generator,
    rates: SumTree,
    /// The position of the next event, drawn from the rates of the state;
    /// `None` until it is drawn.
    next_event: Option<f64>,
}

impl TestStochasticEngine {
    /// Build the stochastic engine of a test model, seeded with 0.
    ///
    /// # Errors
    ///
    /// Returns a message when the model is not a test model, its process is
    /// not empty, or an initial value is negative or not finite.
    pub fn try_from_model(model: &Model) -> Result<Self, String> {
        let (structure, parameters) = test_model(model)?;
        let initial = structure
            .outputs
            .iter()
            .map(|output| {
                if output.initial.is_finite() && output.initial >= 0.0 {
                    Ok(output.initial.round() as u64)
                } else {
                    Err(format!(
                        "Output '{}' starts at {}; a stochastic test model counts units",
                        output.id, output.initial
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut engine = Self {
            outputs: structure.outputs.clone(),
            counts: initial.clone(),
            initial,
            parameters,
            position: 0.0,
            seed: 0,
            generator: Generator::from_seed(0),
            rates: SumTree::new(structure.outputs.len()),
            next_event: None,
        };
        engine.reset();
        Ok(engine)
    }

    /// The rate at which one unit of output `index` leaves.
    fn rate(&self, index: usize) -> Result<f64, String> {
        let id = &self.outputs[index].rate;
        self.parameters
            .get(id)
            .copied()
            .ok_or_else(|| format!("Parameter '{id}' has no value"))
    }

    /// Set the rate of every output in the tree from the state.
    fn refresh_rates(&mut self) -> Result<(), String> {
        for index in 0..self.outputs.len() {
            let rate = self.rate(index)? * self.counts[index] as f64;
            self.rates.set(index, rate);
        }
        Ok(())
    }

    /// Run the events up to `target`, and stop there.
    fn advance_to(&mut self, target: f64) -> Result<(), SimulationError> {
        loop {
            let next = match self.next_event {
                Some(next) => next,
                None => {
                    self.refresh_rates()?;
                    let next = self.position
                        + sample::exponential(&mut self.generator, self.rates.total());
                    self.next_event = Some(next);
                    next
                }
            };
            if next > target {
                self.position = target;
                return Ok(());
            }
            self.position = next;
            let point = self.generator.uniform() * self.rates.total();
            let index = self
                .rates
                .find(point)
                .ok_or_else(|| SimulationError::Failed("no event to pick".to_string()))?;
            self.counts[index] -= 1;
            self.next_event = None;
        }
    }

    fn row(&self) -> Vec<f64> {
        self.counts.iter().map(|count| *count as f64).collect()
    }
}

impl SimulationEngine for TestStochasticEngine {
    type ValuePlan = TestValuePlan;

    fn check_position(&self, position: f64) -> Result<(), String> {
        check_position(position)
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        commol_core::check_positions(positions, self.position, check_position)?;
        buffer.clear();
        for &position in positions {
            self.advance_to(position)?;
            buffer.push(self.row());
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        self.advance_to(self.position + 1.0)
    }

    fn output_names(&self) -> Vec<String> {
        self.outputs
            .iter()
            .map(|output| output.id.clone())
            .collect()
    }

    fn state_outputs(&self) -> usize {
        self.outputs.len()
    }

    fn reset(&mut self) {
        self.position = 0.0;
        self.counts.clone_from(&self.initial);
        self.generator = Generator::from_seed(self.seed);
        self.next_event = None;
    }

    fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.generator = Generator::from_seed(seed);
        self.next_event = None;
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.parameters.insert(parameter_id.to_string(), value);
        self.next_event = None;
        Ok(())
    }

    fn current_position(&self) -> f64 {
        self.position
    }

    fn plan_values(
        &self,
        values: &[ModelValue],
        remainder: Option<&str>,
    ) -> Result<TestValuePlan, String> {
        TestEngine::plan_test_values(values, remainder)
    }

    fn apply_values(&mut self, _plan: &TestValuePlan, _values: &[f64]) -> Result<f64, String> {
        Ok(0.0)
    }
}
