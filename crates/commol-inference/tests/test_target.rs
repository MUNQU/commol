//! A target reads the discrepancy of a problem only inside its support.

use commol_calibration::{
    CalibrationConstraint, CalibrationParameter, CalibrationProblem, LossConfig, ObservedDataPoint,
    Workspace,
};
use commol_core::Model;
use commol_difference::DifferenceEquations;
use commol_inference::discrepancy_in_support;

/// A → B at the proportional rate `k` in `[0, 1]`, from A = 1000, with
/// `A(2) = 600` observed and the constraint `k ≤ 0.5`.
fn problem() -> CalibrationProblem<DifferenceEquations> {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"DifferenceEquations"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],
        "target":["B"],"rate":"k","stratified_rates":null,"per_compartment":null}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#;
    let model: Model = serde_json::from_str(json).expect("valid model JSON");
    CalibrationProblem::new(
        DifferenceEquations::try_from_model(&model).unwrap(),
        vec![ObservedDataPoint::new(2.0, "A".to_string(), 600.0)],
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        vec![CalibrationConstraint::new(
            "cap".to_string(),
            "0.5 - k".to_string(),
        )],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

#[test]
fn the_discrepancy_is_read_inside_the_support_only() {
    let problem = problem();
    let mut workspace = Workspace::default();
    let mut at = |k: f64| discrepancy_in_support(&problem, &[k], &mut workspace).unwrap();

    // A(2) = 1000 · 0.8² = 640, so the squared error is 40².
    let inside = at(0.2).expect("k = 0.2 is in the support");
    assert!((inside - 1600.0).abs() < 1e-6, "{inside}");
    assert_eq!(at(0.7), None, "k = 0.7 breaks the constraint");
    assert_eq!(at(1.5), None, "k = 1.5 is outside the bounds");
}

#[test]
fn a_run_that_fails_is_an_error_not_a_point_outside_the_support() {
    let mut model: Model = serde_json::from_str(
        r#"{"name":"m","parameters":[{"id":"k","value":0.1},{"id":"z","value":null}],
        "engine":{"type":"DifferenceEquations"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k * z"}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#,
    )
    .expect("valid model JSON");
    model.process.transitions[0].stratified_rates = None;
    let problem = CalibrationProblem::new(
        DifferenceEquations::try_from_model(&model).unwrap(),
        vec![ObservedDataPoint::new(2.0, "A".to_string(), 600.0)],
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap();

    let error = discrepancy_in_support(&problem, &[0.2], &mut Workspace::default()).unwrap_err();

    assert!(error.contains("not found"), "{error}");
}
