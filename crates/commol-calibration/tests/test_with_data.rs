//! A copy of a problem with other observed values and weights keeps every
//! other layer, and checks the data it gets.

use std::collections::HashMap;

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, LossConfig, ObservedDataPoint, Workspace,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};

/// One output `x = a + b·t` at every position `t`.
#[derive(Clone, Debug)]
struct LineEngine {
    parameters: HashMap<String, f64>,
}

#[derive(Clone, Debug)]
struct NoPlan;

impl ValuePlan for NoPlan {
    fn infeasibility(&self, excess: f64) -> String {
        format!("{excess}")
    }
}

impl SimulationEngine for LineEngine {
    fn check_values(&self) -> Result<(), String> {
        Ok(())
    }

    type ValuePlan = NoPlan;

    fn check_position(&self, _position: f64) -> Result<(), String> {
        Ok(())
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        let (a, b) = (self.parameters["a"], self.parameters["b"]);
        buffer.clear();
        buffer.extend(positions.iter().map(|t| vec![a + b * t]));
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        vec!["x".to_string()]
    }

    fn state_outputs(&self) -> usize {
        1
    }

    fn reset(&mut self) {}

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.parameters.insert(parameter_id.to_string(), value);
        Ok(())
    }

    fn current_position(&self) -> f64 {
        0.0
    }

    fn plan_values(&self, _: &[ModelValue], _: Option<&str>) -> Result<NoPlan, String> {
        Ok(NoPlan)
    }

    fn apply_values(&mut self, _: &NoPlan, _: &[f64]) -> Result<f64, String> {
        Ok(0.0)
    }
}

fn problem() -> CalibrationProblem<LineEngine> {
    let mut observed: Vec<ObservedDataPoint> = [1.0, 3.0, 2.0, 5.0]
        .iter()
        .enumerate()
        .map(|(t, &value)| ObservedDataPoint::new(t as f64, "x".to_string(), value))
        .collect();
    observed[2].weight = 2.0;
    CalibrationProblem::new(
        LineEngine {
            parameters: HashMap::from([("a".to_string(), 0.0), ("b".to_string(), 0.0)]),
        },
        observed,
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -10.0, 10.0),
        ],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
    .with_observation_normalization(true)
    .unwrap()
}

#[test]
fn with_data_of_the_same_data_is_the_same_problem() {
    let problem = problem();
    let values: Vec<f64> = problem.observed_data().iter().map(|o| o.value).collect();
    let weights: Vec<f64> = problem.observed_data().iter().map(|o| o.weight).collect();
    let copy = problem.with_data(&values, &weights).unwrap();
    let mut workspace = Workspace::default();
    for point in [[0.0, 0.0], [1.0, 1.0], [-2.0, 0.5]] {
        assert_eq!(
            problem.evaluate(&point, &mut workspace).unwrap(),
            copy.evaluate(&point, &mut workspace).unwrap()
        );
    }
}

#[test]
fn with_data_compares_with_the_new_values_and_weights() {
    let problem = problem().with_observation_normalization(false).unwrap();
    let copy = problem
        .with_data(&[0.0, 1.0, 2.0, 3.0], &[1.0, 0.0, 1.0, 1.0])
        .unwrap();
    let mut workspace = Workspace::default();
    // The line 0 + 1·t fits every value; the weight 0 hides any value.
    assert_eq!(copy.evaluate(&[0.0, 1.0], &mut workspace).unwrap(), 0.0);
    let hidden = problem
        .with_data(&[0.0, 100.0, 2.0, 3.0], &[1.0, 0.0, 1.0, 1.0])
        .unwrap();
    assert_eq!(hidden.evaluate(&[0.0, 1.0], &mut workspace).unwrap(), 0.0);
    assert_eq!(copy.observed_data()[1].weight, 0.0);
    assert_eq!(problem.observed_data()[1].value, 3.0);
}

#[test]
fn with_data_checks_its_values_and_weights() {
    let problem = problem();
    let error = problem.with_data(&[1.0, 2.0], &[1.0, 1.0]).unwrap_err();
    assert!(error.contains("4 observations, 2 values"), "{error}");
    let error = problem.with_data(&[1.0; 4], &[1.0, 1.0]).unwrap_err();
    assert!(error.contains("4 values, 2 weights"), "{error}");

    let error = problem
        .with_data(&[1.0, f64::NAN, 2.0, 3.0], &[1.0; 4])
        .unwrap_err();
    assert!(error.contains("Observation 'x' at 1"), "{error}");
    assert!(error.contains("finite"), "{error}");

    let error = problem
        .with_data(&[1.0; 4], &[1.0, 1.0, -0.5, 1.0])
        .unwrap_err();
    assert!(error.contains("Observation 'x' at 2"), "{error}");
    assert!(error.contains("not negative"), "{error}");

    let error = problem.with_data(&[1.0; 4], &[0.0; 4]).unwrap_err();
    assert!(error.contains("Every weight is zero"), "{error}");
}
