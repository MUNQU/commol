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

/// A → B in `g1` and `g2`, at the rate `k * matrix_sum(M, B / total)`, with
/// the matrix `M` = [[2, 1], [1, 3]] whose row of `g1` the parameter `a1`
/// scales and whose column of `g2` the parameter `b2` scales. `a1` and `k`
/// are to calibrate; `b2` has the value `b2`, or none.
fn engine(b2: Option<f64>) -> DifferenceEquations {
    let b2 = b2.map_or("null".to_string(), |value| value.to_string());
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":null}},{{"id":"a1","value":null}},
        {{"id":"b2","value":{b2}}}],
        "engine":{{"type":"DifferenceEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "stratifications":[{{"id":"group","categories":["g1","g2"]}}],
        "matrices":[{{"id":"M","stratification":"group",
        "values":{{"g1":{{"g1":2.0,"g2":1.0}},"g2":{{"g1":1.0,"g2":3.0}}}},
        "row_scales":{{"g1":"a1"}},"column_scales":{{"g2":"b2"}}}}],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],
        "rate":"k * matrix_sum(M, B / total)"}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":900.0,"B":100.0}}}}}}"#
    );
    let model: Model = serde_json::from_str(&json).expect("valid model JSON");
    DifferenceEquations::try_from_model(&model).unwrap()
}

fn parameters() -> Vec<CalibrationParameter> {
    vec![
        CalibrationParameter::new("k".to_string(), 0.0, 1.0),
        CalibrationParameter::new("a1".to_string(), 0.25, 8.0),
    ]
}

/// The feasibility of `constraint` on the engine of `b2`.
fn feasibility(constraint: &str, b2: Option<f64>) -> Result<Feasibility, String> {
    let engine = engine(b2);
    Feasibility::new(
        &parameters(),
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
    match feasibility.check_parameters(&[k, a1]) {
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
    let observed = vec![ObservedDataPoint::new(1.0, "B_g1".to_string(), 150.0)];
    let constraint = CalibrationConstraint::new("c".to_string(), "-M_radius".to_string());
    let problem = CalibrationProblem::new(
        engine(Some(1.0)),
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

    let (configured, _, _) =
        configured_engine(&engine(Some(1.0)), &parameters(), None, &[0.5, 2.0]).unwrap();
    let entries = configured.matrices()[0].entries(|_| None).unwrap();
    assert_eq!(entries, [4.0, 2.0, 1.0, 3.0]);
    close(amount, 5.0);
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
