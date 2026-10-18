//! The calibration writes values only through `SimulationEngine`: parameters
//! by id, and the values of the model kind through `plan_values` and
//! `apply_values`. The engine here is of a kind that is not compartmental.

use std::collections::HashMap;

use argmin::core::CostFunction;
use commol_calibration::{
    check_model_values, configured_engine, CalibrationConstraint, CalibrationParameter,
    CalibrationParameterType, CalibrationProblem, LossConfig, ObservedDataPoint,
    OptimizationObjective,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};

/// An engine with two outputs: `x = start * k`, its state, and `y = 2`,
/// which is not. Its kind has one value, `start`, which is feasible up to 1.
/// It keeps the remainder of the last plan it applied.
#[derive(Clone)]
struct StartEngine {
    start: f64,
    parameters: HashMap<String, f64>,
    remainder: Option<String>,
}

impl StartEngine {
    fn new() -> Self {
        Self {
            start: 0.5,
            parameters: HashMap::from([("k".to_string(), 1.0)]),
            remainder: None,
        }
    }
}

/// The ids and the remainder the engine was asked to resolve.
#[derive(Clone, Debug, PartialEq)]
struct StartPlan {
    ids: Vec<String>,
    remainder: Option<String>,
}

impl ValuePlan for StartPlan {
    fn infeasibility(&self, excess: f64) -> String {
        format!("start is {excess} above its limit")
    }
}

impl SimulationEngine for StartEngine {
    type ValuePlan = StartPlan;

    fn check_position(&self, _position: f64) -> Result<(), String> {
        Ok(())
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        let row = vec![self.start * self.parameters["k"], 2.0];
        buffer.clear();
        buffer.extend(positions.iter().map(|_| row.clone()));
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        vec!["x".to_string(), "y".to_string()]
    }

    fn state_outputs(&self) -> usize {
        1
    }

    fn output_sums(&self) -> Vec<(String, Vec<usize>)> {
        vec![("xy".to_string(), vec![0, 1])]
    }

    fn reset(&mut self) {}

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.parameters.insert(parameter_id.to_string(), value);
        Ok(())
    }

    fn get_parameters(&self) -> &HashMap<String, f64> {
        &self.parameters
    }

    fn current_position(&self) -> f64 {
        0.0
    }

    fn plan_values(
        &self,
        values: &[ModelValue],
        remainder: Option<&str>,
    ) -> Result<StartPlan, String> {
        if let Some(value) = values.iter().find(|value| value.id != "start") {
            return Err(format!("this kind has no value '{}'", value.id));
        }
        Ok(StartPlan {
            ids: values.iter().map(|value| value.id.clone()).collect(),
            remainder: remainder.map(str::to_string),
        })
    }

    fn apply_values(&mut self, plan: &StartPlan, values: &[f64]) -> Result<f64, String> {
        assert_eq!(values.len(), plan.ids.len());
        self.remainder.clone_from(&plan.remainder);
        if let Some(&start) = values.first() {
            self.start = start;
        }
        Ok((self.start - 1.0).max(0.0))
    }
}

fn parameter(id: &str, parameter_type: CalibrationParameterType) -> CalibrationParameter {
    CalibrationParameter::with_type_and_guess(id.to_string(), parameter_type, 0.0, 2.0, 0.5)
}

fn start_and_k() -> Vec<CalibrationParameter> {
    vec![
        parameter("k", CalibrationParameterType::Parameter),
        parameter("start", CalibrationParameterType::InitialCondition),
        parameter("f", CalibrationParameterType::Factor),
    ]
}

fn problem(
    constraints: Vec<CalibrationConstraint>,
) -> Result<CalibrationProblem<StartEngine>, String> {
    CalibrationProblem::new(
        StartEngine::new(),
        vec![ObservedDataPoint::new(1.0, "x".to_string(), 0.5)],
        start_and_k(),
        constraints,
        LossConfig::SumSquaredError,
    )
}

#[test]
fn only_values_of_the_kind_reach_the_engine_and_parameters_are_set_by_id() {
    let (engine, applied, plan) =
        configured_engine(&StartEngine::new(), &start_and_k(), None, &[0.3, 0.8, 1.5]).unwrap();

    assert_eq!(plan.ids, vec!["start".to_string()]);
    assert_eq!(engine.start, 0.8);
    assert_eq!(engine.parameters["k"], 0.3);
    assert!(!engine.parameters.contains_key("f"));
    assert_eq!(applied, vec![0.3, 0.8, 1.5]);
}

#[test]
fn values_are_clamped_to_their_bounds_before_they_reach_the_engine() {
    let (engine, applied, _) =
        configured_engine(&StartEngine::new(), &start_and_k(), None, &[3.0, -1.0, 0.5]).unwrap();

    assert_eq!(applied, vec![2.0, 0.0, 0.5]);
    assert_eq!(engine.start, 0.0);
}

#[test]
fn the_remainder_reaches_the_engine() {
    let (_, _, plan) = configured_engine(
        &StartEngine::new(),
        &start_and_k(),
        Some("rest"),
        &[1.0, 0.5, 1.0],
    )
    .unwrap();

    assert_eq!(plan.remainder.as_deref(), Some("rest"));
    let problem = problem(vec![])
        .unwrap()
        .with_initial_condition_remainder("rest")
        .unwrap();
    let engine = problem.forward().configure(&[1.0, 0.5, 1.0]).unwrap();
    assert_eq!(engine.remainder.as_deref(), Some("rest"));
}

#[test]
fn a_value_the_engine_does_not_resolve_stops_the_problem() {
    let parameters = vec![parameter(
        "other",
        CalibrationParameterType::InitialCondition,
    )];

    let error = check_model_values(&StartEngine::new(), &parameters, None).unwrap_err();
    assert!(error.contains("no value 'other'"), "{error}");

    let error = CalibrationProblem::new(
        StartEngine::new(),
        vec![ObservedDataPoint::new(1.0, "x".to_string(), 0.5)],
        parameters,
        vec![],
        LossConfig::SumSquaredError,
    )
    .err()
    .expect("the engine rejects the value");
    assert!(error.contains("no value 'other'"), "{error}");
}

#[test]
fn an_infeasible_state_is_penalized_and_its_reason_comes_from_the_engine() {
    let problem = problem(vec![]).unwrap();

    let objective = OptimizationObjective::new(&problem);
    let feasible = objective.cost(&vec![1.0, 0.5, 1.0]).unwrap();
    let infeasible = objective.cost(&vec![1.0, 1.5, 1.0]).unwrap();
    assert_eq!(feasible, 0.0);
    assert!(infeasible > 1e9, "{infeasible}");

    let error = problem.configured_engine(&[1.0, 1.5, 1.0]).err().unwrap();
    assert!(error.contains("start is 0.5 above its limit"), "{error}");
}

#[test]
fn a_constraint_reads_the_sums_and_the_state_total_the_engine_names() {
    // At k = 1 and start = 0.5: x = 0.5, y = 2, xy = 2.5 and total = x = 0.5.
    let penalty = |expression: &str| {
        let problem = problem(vec![CalibrationConstraint::new(
            "c".to_string(),
            expression.to_string(),
        )
        .with_positions(vec![1.0])])
        .unwrap();
        OptimizationObjective::new(&problem)
            .cost(&vec![1.0, 0.5, 1.0])
            .unwrap()
    };

    assert_eq!(penalty("2.5 - xy"), 0.0);
    assert_eq!(penalty("0.5 - total"), 0.0);
    assert!(penalty("xy - 3") > 0.0);
    assert!(penalty("total - 1") > 0.0);

    let error = problem(vec![CalibrationConstraint::new(
        "c".to_string(),
        "zz - 1".to_string(),
    )
    .with_positions(vec![1.0])])
    .err()
    .unwrap();
    assert!(error.contains("unknown variable 'zz'"), "{error}");
}
