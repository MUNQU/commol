//! What the trait of an engine gives an engine that takes only the defaults.

use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};

/// An engine of one output, 1, that sets no value of a model kind.
#[derive(Clone)]
struct One;

#[derive(Clone, Debug)]
struct NoValues;

impl ValuePlan for NoValues {
    fn infeasibility(&self, excess: f64) -> String {
        format!("The state is {excess} from feasible")
    }
}

impl SimulationEngine for One {
    type ValuePlan = NoValues;

    fn check_position(&self, _position: f64) -> Result<(), String> {
        Ok(())
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        buffer.clear();
        buffer.extend(positions.iter().map(|_| vec![1.0]));
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        vec!["A".to_string()]
    }

    fn state_outputs(&self) -> usize {
        1
    }

    fn reset(&mut self) {}

    fn set_parameter(&mut self, parameter_id: &str, _value: f64) -> Result<(), String> {
        Err(format!("The model has no parameter '{parameter_id}'"))
    }

    fn current_position(&self) -> f64 {
        0.0
    }

    fn plan_values(
        &self,
        _values: &[ModelValue],
        _remainder: Option<&str>,
    ) -> Result<NoValues, String> {
        Ok(NoValues)
    }

    fn apply_values(&mut self, _plan: &NoValues, _values: &[f64]) -> Result<f64, String> {
        Ok(0.0)
    }
}

#[test]
fn an_engine_that_leaves_nothing_to_calibrate_has_every_value() {
    assert_eq!(One.check_values(), Ok(()));
}
