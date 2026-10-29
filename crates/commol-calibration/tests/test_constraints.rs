//! A constraint reads the entries and the spectral radius of a matrix
//! between categories at each point: the matrix scaled by the values of the
//! point for the scales it calibrates, and by the values of the model for
//! the others.

use commol_calibration::{
    CalibrationConstraint, CalibrationParameter, CalibrationProblem, EvaluationError, Feasibility,
    LossConfig, ObservedDataPoint, Violation, Workspace, configured_engine,
};
use commol_core::{Model, SimulationEngine};
use commol_difference::DifferenceEquations;
use commol_model::CompartmentalEngine;

/// The rate of the engines of the tests that read a matrix sum.
const SUM: &str = "k * matrix_sum(M, B / total)";

/// A → B in `g1` and `g2`, at `rate`, with the matrix `M` = [[2, 1], [1, 3]]
/// whose row of `g1` the parameter `a1` scales and whose column of `g2` the
/// parameter `b2` scales. `k` is to calibrate; `a1` and `b2` have the values
/// given, or none.
fn engine_of(rate: &str, a1: Option<f64>, b2: Option<f64>) -> DifferenceEquations {
    let json_value = |value: Option<f64>| value.map_or("null".to_string(), |v| v.to_string());
    let (a1, b2) = (json_value(a1), json_value(b2));
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":null}},{{"id":"a1","value":{a1}}},
        {{"id":"b2","value":{b2}}}],
        "engine":{{"type":"DifferenceEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "stratifications":[{{"id":"group","categories":["g1","g2"]}}],
        "matrices":[{{"id":"M","stratification":"group",
        "values":{{"g1":{{"g1":2.0,"g2":1.0}},"g2":{{"g1":1.0,"g2":3.0}}}},
        "row_scales":{{"g1":"a1"}},"column_scales":{{"g2":"b2"}}}}],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],
        "rate":"{rate}"}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":900.0,"B":100.0}}}}}}"#
    );
    let model: Model = serde_json::from_str(&json).expect("valid model JSON");
    DifferenceEquations::try_from_model(&model).unwrap()
}

/// `engine_of` at the rate `SUM`, with `a1` to calibrate.
fn engine(b2: Option<f64>) -> DifferenceEquations {
    engine_of(SUM, None, b2)
}

fn parameters() -> Vec<CalibrationParameter> {
    vec![
        CalibrationParameter::new("k".to_string(), 0.0, 1.0),
        CalibrationParameter::new("a1".to_string(), 0.25, 8.0),
    ]
}

/// The feasibility of `constraint` on the engine of `b2`, calibrating `k`
/// and `a1`.
fn feasibility(constraint: &str, b2: Option<f64>) -> Result<Feasibility, String> {
    feasibility_of(&engine(b2), &parameters(), constraint)
}

fn feasibility_of(
    engine: &DifferenceEquations,
    parameters: &[CalibrationParameter],
    constraint: &str,
) -> Result<Feasibility, String> {
    Feasibility::new(
        parameters,
        &[CalibrationConstraint::new(
            "c".to_string(),
            constraint.to_string(),
        )],
        &engine.output_names(),
        engine.state_outputs(),
        engine.output_sums(),
        engine.matrices(),
        &[],
    )
}

/// How much the point `(k, a1)` violates `constraint`; 0 when it holds.
fn violation(feasibility: &Feasibility, k: f64, a1: f64) -> f64 {
    match feasibility.check_parameters(&mut feasibility.context(&[k, a1])) {
        Ok(()) => 0.0,
        Err(EvaluationError::Violation(Violation::Constraint { amount })) => amount,
        Err(other) => panic!("{other}"),
    }
}

fn close(value: f64, expected: f64) {
    assert!((value - expected).abs() < 1e-12, "{value} != {expected}");
}

#[test]
fn the_radius_follows_a_calibrated_scale() {
    // `-M_radius` is violated by the radius itself.
    let radius = feasibility("-M_radius", Some(1.0)).unwrap();

    close(violation(&radius, 0.5, 1.0), (5.0 + 5.0_f64.sqrt()) / 2.0);
    // [[4, 2], [1, 3]]: the eigenvalues 5 and 2.
    close(violation(&radius, 0.5, 2.0), 5.0);
    // The column of `g2` takes the value of `b2` in the model: [[4, 4], [1, 6]],
    // of eigenvalues 5 ± √5.
    let wider = feasibility("-M_radius", Some(2.0)).unwrap();
    close(violation(&wider, 0.5, 2.0), 5.0 + 5.0_f64.sqrt());
    // An entry is the scaled entry.
    let entry = feasibility("-M_g1_g2", Some(2.0)).unwrap();
    close(violation(&entry, 0.5, 2.0), 4.0);
}

#[test]
fn a_bound_on_the_growth_factor_is_one_constraint() {
    let bound = feasibility("2 - k * M_radius", Some(1.0)).unwrap();

    close(violation(&bound, 0.5, 2.0), 0.5);
    assert_eq!(violation(&bound, 0.25, 2.0), 0.0);
}

#[test]
fn the_constraint_and_the_rate_read_one_radius() {
    // A rate of `k * M_radius` reads the slot of the radius in the engine.
    let rate = "k * M_radius";
    let observed = vec![ObservedDataPoint::new(1.0, "B_g1".to_string(), 150.0)];
    let constraint = CalibrationConstraint::new("c".to_string(), "-M_radius".to_string());
    let problem = CalibrationProblem::new(
        engine_of(rate, None, Some(1.0)),
        observed,
        parameters(),
        vec![constraint],
        LossConfig::SumSquaredError,
    )
    .unwrap();
    let mut workspace = Workspace::default();
    let Err(EvaluationError::Violation(Violation::Constraint { amount })) =
        problem.evaluate(&[0.5, 2.0], &mut workspace)
    else {
        panic!("the radius violates -M_radius");
    };

    let (configured, _, _) = configured_engine(
        &engine_of(rate, None, Some(1.0)),
        &parameters(),
        None,
        &[0.5, 2.0],
    )
    .unwrap();
    let state = configured.initial_state();
    let mut rates = vec![0.0; 2];
    configured
        .model()
        .clone()
        .transition_rates(0.0, 0.0, &state, &mut rates)
        .unwrap();
    // [[4, 2], [1, 3]]: the eigenvalues 5 and 2.
    close(amount, 5.0);
    for rate in rates {
        close(rate / 0.5, amount);
    }
}

#[test]
fn a_matrix_that_no_calibrated_parameter_scales_keeps_its_radius() {
    // `a1` is 2 in the model; the calibration sets only `k`.
    let engine = engine_of(SUM, Some(2.0), Some(1.0));
    let only_k = [CalibrationParameter::new("k".to_string(), 0.0, 1.0)];
    let radius = feasibility_of(&engine, &only_k, "-M_radius").unwrap();

    for k in [0.1, 0.9] {
        let amount = match radius.check_parameters(&mut radius.context(&[k])) {
            Err(EvaluationError::Violation(Violation::Constraint { amount })) => amount,
            other => panic!("{other:?}"),
        };
        close(amount, 5.0);
    }
    // Two variables of one matrix: the entry 4 of [[4, 2], [1, 3]] and its
    // radius 5.
    let two = feasibility_of(&engine, &only_k, "M_g1_g1 - M_radius").unwrap();
    match two.check_parameters(&mut two.context(&[0.5])) {
        Err(EvaluationError::Violation(Violation::Constraint { amount })) => close(amount, 1.0),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_constraint_does_not_read_a_matrix_whose_scale_has_no_value() {
    assert_eq!(
        feasibility("2 - M_radius", None).unwrap_err(),
        "Constraint 'c' reads 'M_radius', but the scale 'b2' of matrix 'M' has no value; \
         calibrate it or give it a value"
    );
    assert_eq!(
        feasibility("2 - N_radius", Some(1.0)).unwrap_err(),
        "Constraint 'c' references unknown variable 'N_radius' (not a parameter, a variable \
         of a matrix or a simulation output)"
    );
}
