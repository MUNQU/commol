//! A method that varies a value asks the engine whether it takes every
//! number of it (`SimulationEngine::check_varied_values`); a write of known
//! values does not, and the engine checks each number it writes. Each
//! builder that varies values writes them at the point of their bounds
//! nearest 0, with the remainder, to a copy of the engine, which then has
//! every value a run needs (`SimulationEngine::check_values`). Every
//! builder and query of the values refuses a remainder that keeps no amount
//! the model states.

use commol_calibration::{
    CalibrationParameter, CalibrationParameterType, CalibrationProblem, ForwardModel, LossConfig,
    ObservedDataPoint, ValueSource, configured_engine, plan_model_values,
};
use commol_core::Model;
use commol_difference::DifferenceEquations;
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

/// The initial amount of B in `[0, 50]`: whole bounds, so the point of the
/// bounds nearest 0, which a problem writes to check that no value is
/// missing, is whole and does not catch it.
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
        None,
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
fn a_forward_model_of_known_values_checks_every_set_before_any_run() {
    let known = |sets: &[Vec<f64>]| {
        ForwardModel::unobserved(
            whole_only(),
            amount_of_b(),
            None,
            ValueSource::Known(sets),
            Some(2),
            Some(7),
        )
    };

    let second_not_whole = known(&[vec![10.0], vec![12.3]]).err().unwrap();
    let too_few = known(&[vec![10.0], vec![]]).err().unwrap();
    let none = known(&[]).err().unwrap();

    assert!(known(&[vec![10.0], vec![12.0]]).is_ok());
    assert!(
        second_not_whole.starts_with("Parameter set 1: ")
            && second_not_whole.contains("12.3 is not whole"),
        "{second_not_whole}"
    );
    assert_eq!(
        too_few,
        "Parameter set 1 needs one number per parameter (1), got 0"
    );
    assert_eq!(
        none,
        "There is no parameter set to write: give at least one"
    );
}

/// A → B at the rate `k * m` in discrete steps, with `k = 0.1` and `m`,
/// which has no value, from `amounts`, the JSON of the amount of each bin.
fn decay(amounts: &str, m: &str) -> DifferenceEquations {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":0.1}},{{"id":"m","value":{m}}}],
        "engine":{{"type":"DifferenceEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"ab","source":["A"],
        "target":["B"],"rate":"k * m","stratified_rates":null,"per_compartment":null}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{amounts}}}}}"#
    );
    let model: Model = serde_json::from_str(&json).expect("valid model JSON");
    DifferenceEquations::try_from_model(&model).unwrap()
}

/// The message of a builder that varies `parameters` on `engine`, or `None`
/// when it builds.
type Build = fn(DifferenceEquations, Vec<CalibrationParameter>, Option<&str>) -> Option<String>;

fn problem_of(
    engine: DifferenceEquations,
    parameters: Vec<CalibrationParameter>,
    remainder: Option<&str>,
) -> Option<String> {
    CalibrationProblem::new(
        engine,
        vec![ObservedDataPoint::new(1.0, "B".to_string(), 100.0)],
        parameters,
        remainder,
        vec![],
        LossConfig::SumSquaredError,
    )
    .err()
}

fn analysis_of(
    engine: DifferenceEquations,
    parameters: Vec<CalibrationParameter>,
    remainder: Option<&str>,
) -> Option<String> {
    ForwardModel::unobserved(
        engine,
        parameters,
        remainder,
        ValueSource::Varied,
        None,
        None,
    )
    .err()
}

#[test]
fn a_problem_that_leaves_a_value_missing_is_refused_when_it_is_built() {
    let k = || vec![CalibrationParameter::new("k".to_string(), 0.0, 3.0)];
    let k_and_b = || {
        vec![
            CalibrationParameter::new("k".to_string(), 0.0, 3.0),
            CalibrationParameter::with_type(
                "B".to_string(),
                CalibrationParameterType::InitialCondition,
                0.0,
                1000.0,
            ),
        ]
    };
    let whole = r#"{"A":1000.0,"B":0.0}"#;
    // A is left None: only the remainder gives it a value, what B leaves of
    // the 1000 the model states.
    let a_left_none = r#"{"A":null,"B":1000.0}"#;
    let missing = |what: &str| {
        Some(format!(
            "The model cannot run: {what} have no value. Give them a value, or calibrate \
             them and write the calibrated values back"
        ))
    };

    for (builder, build) in [
        ("CalibrationProblem::new", problem_of as Build),
        ("ForwardModel::unobserved", analysis_of),
    ] {
        assert_eq!(
            build(decay(whole, "null"), k(), None),
            missing("the parameters [\"m\"]"),
            "{builder}"
        );
        assert_eq!(
            build(decay(a_left_none, "1.0"), k_and_b(), None),
            missing("the amounts of the bins [\"A\"]"),
            "{builder}"
        );
        assert_eq!(
            build(decay(a_left_none, "1.0"), k_and_b(), Some("A")),
            None,
            "{builder}"
        );
        assert_eq!(build(decay(whole, "1.0"), k(), None), None, "{builder}");
    }
}

/// The message of the plan of a query of the values (`plan_model_values`),
/// or `None` when it resolves them.
fn plan_of(
    engine: DifferenceEquations,
    parameters: Vec<CalibrationParameter>,
    remainder: Option<&str>,
) -> Option<String> {
    plan_model_values(&engine, &parameters, remainder).err()
}

/// The message of the write-back (`configured_engine`) of the point of the
/// bounds nearest 0, or `None` when it writes it.
fn written_of(
    engine: DifferenceEquations,
    parameters: Vec<CalibrationParameter>,
    remainder: Option<&str>,
) -> Option<String> {
    let nearest_zero: Vec<f64> = parameters
        .iter()
        .map(|parameter| 0.0_f64.max(parameter.min_bound).min(parameter.max_bound))
        .collect();
    configured_engine(&engine, &parameters, remainder, &nearest_zero).err()
}

#[test]
fn a_remainder_that_would_give_a_value_no_one_gave_is_refused() {
    let k = || vec![CalibrationParameter::new("k".to_string(), 0.0, 3.0)];
    let k_and_b = || {
        vec![
            CalibrationParameter::new("k".to_string(), 0.0, 3.0),
            CalibrationParameter::with_type(
                "B".to_string(),
                CalibrationParameterType::InitialCondition,
                0.0,
                1000.0,
            ),
        ]
    };
    let refused = |bins: &str| {
        Some(format!(
            "Initial-condition remainder 'A' would give a value no one gave: it takes what \
             the initial-condition values leave of the amounts the model states for them and \
             for it, and the amounts of the bins {bins} have no value. Give one of them an \
             amount, or add an initial-condition value of a bin that has one"
        ))
    };
    let a_left_none = || decay(r#"{"A":null,"B":1000.0}"#, "1.0");
    let both_left_none = || decay(r#"{"A":null,"B":null}"#, "1.0");
    let a_stated = || decay(r#"{"A":1000.0,"B":0.0}"#, "1.0");

    for (builder, build) in [
        ("CalibrationProblem::new", problem_of as Build),
        ("ForwardModel::unobserved", analysis_of),
        ("plan_model_values", plan_of),
        ("configured_engine", written_of),
    ] {
        // Only k varies, so A would take 1000 less the 1000 of B: 0.
        assert_eq!(
            build(a_left_none(), k(), Some("A")),
            refused("[\"A\"]"),
            "{builder}"
        );
        // B varies, but the model states no amount of A or B: the sum is 0.
        assert_eq!(
            build(both_left_none(), k_and_b(), Some("A")),
            refused("[\"A\", \"B\"]"),
            "{builder}"
        );
        // The model states the amount of A, which A takes back.
        assert_eq!(build(a_stated(), k(), Some("A")), None, "{builder}");
        // A takes what B leaves of the 1000 the model states for B.
        assert_eq!(
            build(a_left_none(), k_and_b(), Some("A")),
            None,
            "{builder}"
        );
    }
}
