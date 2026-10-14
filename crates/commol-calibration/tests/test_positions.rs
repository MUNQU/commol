//! Each engine checks the positions of a calibration problem, and its error
//! names the observation or constraint that holds a rejected one.

use commol_calibration::{
    CalibrationConstraint, CalibrationParameter, CalibrationProblem, LossConfig, ObservedDataPoint,
};
use commol_core::{Model, SimulationEngine};
use commol_difference::DifferenceEquations;
use commol_differential::DifferentialEquations;

/// A → B at the per-capita rate `k`, run by the given engine.
fn decay_model(engine: &str) -> Model {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":0.1}}],
        "engine":{{"type":"{engine}"}},
        "structure":{{"kind":"compartmental",
        "bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"ab","source":["A"],
        "target":["B"],"rate":"k","stratified_rates":null,"per_compartment":null}}],
        "initial_conditions":{{"total":1000,"bin_fractions":[
        {{"bin":"A","fraction":1.0}},{{"bin":"B","fraction":0.0}}],
        "stratification_fractions":[]}}}}}}"#
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

fn problem<E: SimulationEngine>(
    engine: E,
    observed: Vec<ObservedDataPoint>,
    constraints: Vec<CalibrationConstraint>,
) -> Result<CalibrationProblem<E>, String> {
    CalibrationProblem::new(
        engine,
        observed,
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        constraints,
        LossConfig::SumSquaredError,
    )
}

fn difference() -> DifferenceEquations {
    DifferenceEquations::try_from_model(&decay_model("DifferenceEquations")).unwrap()
}

fn differential() -> DifferentialEquations {
    DifferentialEquations::try_from_model(&decay_model("DifferentialEquations")).unwrap()
}

fn at(position: f64) -> ObservedDataPoint {
    ObservedDataPoint::new(position, "A".to_string(), 500.0)
}

fn bounded(positions: Vec<f64>) -> CalibrationConstraint {
    CalibrationConstraint::new("floor".to_string(), "A - 1".to_string()).with_positions(positions)
}

#[test]
fn a_difference_engine_names_the_observation_at_a_non_integer_position() {
    let error = problem(difference(), vec![at(4.0), at(2.5)], vec![])
        .err()
        .expect("2.5 is rejected");
    assert!(error.contains("Observation 'A'"), "{error}");
    assert!(error.contains("2.5 is not one"), "{error}");
}

#[test]
fn a_difference_engine_names_the_observation_whose_window_starts_between_steps() {
    let observation = ObservedDataPoint::with_window(4.0, "A".to_string(), 500.0, 1.5);
    let error = problem(difference(), vec![observation], vec![])
        .err()
        .expect("a window start at 2.5 is rejected");
    assert!(
        error.contains("Observation 'A' at 4 with the window 1.5"),
        "{error}"
    );
    assert!(error.contains("2.5 is not one"), "{error}");
}

#[test]
fn a_difference_engine_names_the_constraint_at_a_non_integer_position() {
    let error = problem(difference(), vec![at(4.0)], vec![bounded(vec![1.0, 3.5])])
        .err()
        .expect("3.5 is rejected");
    assert!(error.contains("Constraint 'floor'"), "{error}");
    assert!(error.contains("3.5 is not one"), "{error}");
}

#[test]
fn a_differential_engine_accepts_non_integer_positions() {
    let observation = ObservedDataPoint::with_window(4.0, "A".to_string(), 500.0, 1.5);
    problem(
        differential(),
        vec![at(2.5), observation],
        vec![bounded(vec![0.25, 3.5])],
    )
    .expect("every position is accepted");
}

#[test]
fn both_engines_name_the_observation_at_a_negative_position() {
    let error = problem(difference(), vec![at(-1.0)], vec![]).err().unwrap();
    assert!(error.contains("Observation 'A'"), "{error}");
    let error = problem(differential(), vec![at(-1.0)], vec![])
        .err()
        .unwrap();
    assert!(error.contains("Observation 'A'"), "{error}");
    assert!(error.contains("-1 is not one"), "{error}");
}

#[test]
fn a_window_longer_than_its_position_is_rejected() {
    let observation = ObservedDataPoint::with_window(1.0, "A".to_string(), 500.0, 2.0);
    let error = problem(differential(), vec![observation], vec![])
        .err()
        .expect("a window start at -1 is rejected");
    assert!(
        error.contains("Observation 'A' at 1 with the window 2"),
        "{error}"
    );
}
