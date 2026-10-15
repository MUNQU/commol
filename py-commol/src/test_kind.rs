//! The engine of the test model kind, compiled only with the `test-kind`
//! feature.
//!
//! Each output starts at its initial value and decays exponentially at the
//! rate of a parameter: `x(t) = initial * exp(-rate * t)`. The engine exists
//! to test that the generic layers run a kind that is not compartmental.

use std::collections::HashMap;

use commol_core::{
    DecayOutput, Model, ModelStructure, ModelValue, ParameterValue, SimulationEngine,
    SimulationError, ValuePlan,
};

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
    /// Returns a message when the model is not a test model.
    pub fn try_from_model(model: &Model) -> Result<Self, String> {
        let ModelStructure::Test(structure) = &model.structure else {
            return Err(format!(
                "Model '{}' is a {} model, not a test model",
                model.name,
                model.structure.kind().name()
            ));
        };
        let parameters = model
            .parameters
            .iter()
            .filter_map(|parameter| match parameter.value {
                Some(ParameterValue::Constant(value)) => Some((parameter.id.clone(), value)),
                _ => None,
            })
            .collect();
        Ok(Self {
            outputs: structure.outputs.clone(),
            parameters,
            position: 0.0,
        })
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

    fn get_parameters(&self) -> &HashMap<String, f64> {
        &self.parameters
    }

    fn current_position(&self) -> f64 {
        self.position
    }

    fn plan_values(
        &self,
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

    fn apply_values(&mut self, _plan: &TestValuePlan, _values: &[f64]) -> Result<f64, String> {
        Ok(0.0)
    }
}
