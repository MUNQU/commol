//! A method that varies a value asks the engine whether it takes every
//! number of it (`SimulationEngine::check_varied_values`); a write of known
//! values does not, and the engine checks each number it writes.

use commol_calibration::{
    CalibrationParameter, CalibrationParameterType, CalibrationProblem, ForwardModel, LossConfig,
    ObservedDataPoint, ValueSource, check_model_values, configured_engine, plan_model_values,
};
use commol_core::Model;
use commol_stochastic::StochasticSteps;

/// A → B at the rate `k`, in whole units from A = 1000, with no rounding.
fn whole_only() -> StochasticSteps {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"StochasticSteps"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],
        "target":["B"],"rate":"k","stratified_rates":null,"per_compartment":null}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#;
    let model: Model = serde_json::from_str(json).expect("valid model JSON");
    StochasticSteps::try_from_model(&model).unwrap()
}

/// The initial amount of B in `[0, 50]`: whole bounds, which the point
/// nearest 0 of [`check_model_values`] does not catch.
fn amount_of_b() -> Vec<CalibrationParameter> {
    vec![CalibrationParameter::with_type(
        "B".to_string(),
        CalibrationParameterType::InitialCondition,
        0.0,
        50.0,
    )]
}

fn assert_needs_a_rounding(error: &str) {
    for part in ["StochasticSteps", "'B'", "largest_remainder", "random"] {
        assert!(error.contains(part), "{part} in {error}");
    }
}

#[test]
fn every_method_that_varies_an_amount_needs_a_rounding() {
    let problem = CalibrationProblem::new(
        whole_only(),
        vec![ObservedDataPoint::new(1.0, "A".to_string(), 900.0)],
        amount_of_b(),
        vec![],
        LossConfig::SumSquaredError,
    );
    let unobserved = ForwardModel::unobserved(
        whole_only(),
        amount_of_b(),
        None,
        ValueSource::Varied,
        Some(2),
        Some(7),
    );

    assert_needs_a_rounding(&check_model_values(&whole_only(), &amount_of_b(), None).unwrap_err());
    assert_needs_a_rounding(&problem.err().unwrap());
    assert_needs_a_rounding(&unobserved.err().unwrap());
}

#[test]
fn known_values_are_checked_when_they_are_written() {
    let (engine, applied, _) = configured_engine(&whole_only(), &amount_of_b(), None, &[12.0])
        .expect("a whole amount needs no rounding");
    let error = configured_engine(&whole_only(), &amount_of_b(), None, &[12.3])
        .err()
        .unwrap();

    assert_eq!(applied, [12.0]);
    assert_eq!(
        commol_model::CompartmentalEngine::initial_state(&engine),
        [1000.0, 12.0]
    );
    assert!(error.contains("12.3 is not whole"), "{error}");
    assert!(plan_model_values(&whole_only(), &amount_of_b(), None).is_ok());
}

#[test]
fn a_forward_model_of_known_values_writes_its_first_numbers() {
    let known = |numbers: &[f64]| {
        ForwardModel::unobserved(
            whole_only(),
            amount_of_b(),
            None,
            ValueSource::Known(numbers),
            Some(2),
            Some(7),
        )
    };

    let not_whole = known(&[12.3]).err().unwrap();
    let too_few = known(&[]).err().unwrap();

    assert!(known(&[12.0]).is_ok());
    assert!(not_whole.contains("12.3 is not whole"), "{not_whole}");
    assert_eq!(too_few, "Expected 1 known values, one per parameter, got 0");
}
