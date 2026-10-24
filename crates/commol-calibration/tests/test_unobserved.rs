//! A forward model with no observations writes values as a calibration does
//! and runs at the positions it is given.

use commol_calibration::{
    CalibrationParameter, CalibrationParameterType, CalibrationProblem, ForwardModel, LossConfig,
    ObservedDataPoint,
};
use commol_core::Model;
use commol_difference::DifferenceEquations;
use commol_stochastic::StochasticSteps;

/// A → B at the proportional rate `k`, in discrete steps from A = 1000.
fn engine() -> DifferenceEquations {
    DifferenceEquations::try_from_model(&model("DifferenceEquations")).unwrap()
}

/// The model of [`engine`], in whole units on the stochastic engine of steps.
fn stochastic() -> StochasticSteps {
    StochasticSteps::try_from_model(&model("StochasticSteps")).unwrap()
}

fn model(engine: &str) -> Model {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"ENGINE"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],
        "target":["B"],"rate":"k","stratified_rates":null,"per_compartment":null}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#;
    serde_json::from_str(&json.replace("ENGINE", engine)).expect("valid model JSON")
}

fn parameters() -> Vec<CalibrationParameter> {
    vec![
        CalibrationParameter::new("k".to_string(), 0.0, 1.0),
        CalibrationParameter::with_type(
            "A".to_string(),
            CalibrationParameterType::InitialCondition,
            0.0,
            2000.0,
        ),
    ]
}

#[test]
fn an_unobserved_model_runs_as_the_forward_model_of_a_problem() {
    let unobserved = ForwardModel::unobserved(engine(), parameters(), None, None, None).unwrap();
    let problem = CalibrationProblem::new(
        engine(),
        vec![ObservedDataPoint::new(1.0, "A".to_string(), 0.0)],
        parameters(),
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap();

    let values = [0.2, 500.0];
    let runs = unobserved.run_at(&values, &[0.0, 2.0]).unwrap();

    assert_eq!(
        runs,
        problem.forward().run_at(&values, &[0.0, 2.0]).unwrap()
    );
    // A = 500 · 0.8², B = 500 − A.
    assert!((runs[0][1][0] - 320.0).abs() < 1e-9);
    assert_eq!(unobserved.observation().count(), 0);
}

#[test]
fn an_unobserved_model_has_no_factors_or_noise_parameters() {
    let factor = CalibrationParameter::with_type(
        "f".to_string(),
        CalibrationParameterType::Factor,
        0.5,
        2.0,
    );

    let error = ForwardModel::unobserved(engine(), vec![factor], None, None, None)
        .err()
        .unwrap();

    assert!(error.contains("'f' is a factor"), "{error}");
    let noise = CalibrationParameter::with_type(
        "sigma".to_string(),
        CalibrationParameterType::Noise,
        0.1,
        2.0,
    );
    let error = ForwardModel::unobserved(engine(), vec![noise], None, None, None)
        .err()
        .unwrap();
    assert!(error.contains("'sigma' is a noise parameter"), "{error}");
}

#[test]
fn an_unobserved_model_needs_parameters_and_replicates() {
    let no_parameters = ForwardModel::unobserved(engine(), vec![], None, None, None)
        .err()
        .unwrap();
    let no_replicates = ForwardModel::unobserved(engine(), parameters(), None, Some(0), None)
        .err()
        .unwrap();

    assert!(no_parameters.contains("No parameters"), "{no_parameters}");
    assert_eq!(no_replicates, "replicates is at least 1, got 0");
}

#[test]
fn a_stochastic_engine_needs_replicates_and_a_seed() {
    let no_replicates = ForwardModel::unobserved(stochastic(), parameters(), None, None, Some(7))
        .err()
        .unwrap();
    let no_seed = ForwardModel::unobserved(stochastic(), parameters(), None, Some(3), None)
        .err()
        .unwrap();

    assert!(no_replicates.contains("set replicates"), "{no_replicates}");
    assert!(no_seed.contains("set seed"), "{no_seed}");
}

#[test]
fn a_deterministic_engine_runs_once_whatever_the_replicates() {
    let model = ForwardModel::unobserved(engine(), parameters(), None, Some(3), Some(7)).unwrap();

    assert!(model.replicates().is_none());
}

#[test]
fn an_unknown_initial_condition_gives_the_engine_message() {
    let unknown = CalibrationParameter::with_type(
        "C".to_string(),
        CalibrationParameterType::InitialCondition,
        0.0,
        1.0,
    );

    let error = ForwardModel::unobserved(engine(), vec![unknown], None, None, None)
        .err()
        .unwrap();

    assert!(error.contains('C'), "{error}");
}

#[test]
fn the_replicates_of_an_unobserved_model_are_seeded_from_its_seed() {
    let model =
        ForwardModel::unobserved(stochastic(), parameters(), None, Some(3), Some(7)).unwrap();

    let seeds = &model.replicates().unwrap().seeds;

    assert_eq!(seeds.len(), 3);
    assert_eq!(seeds[1], commol_random::replicate_seed(7, 1));
}

#[test]
fn observations_of_an_unknown_output_name_it() {
    let model = ForwardModel::unobserved(engine(), parameters(), None, None, None).unwrap();

    let error = model
        .with_observations(&[ObservedDataPoint::new(1.0, "C".to_string(), 0.0)])
        .err()
        .unwrap();

    assert!(error.contains("output name 'C'"), "{error}");
}
