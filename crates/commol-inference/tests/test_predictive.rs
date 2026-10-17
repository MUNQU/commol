//! The predictive of a sample reads runs through the forward model and
//! observations through the observation operator of the problem.

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, LossConfig, ObservedDataPoint, Workspace,
};
use commol_core::Model;
use commol_difference::DifferenceEquations;
use commol_inference::{NoNoise, ParameterSample, Predictive, observation_draws};

/// A → B at the proportional rate `k`, in discrete steps from A = 1000.
fn problem() -> CalibrationProblem<DifferenceEquations> {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"DifferenceEquations"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],
        "target":["B"],"rate":"k","stratified_rates":null,"per_compartment":null}]},
        "structure":{"kind":"aggregate","total":1000,"bin_fractions":{"A":1.0,"B":0.0}}}"#;
    let model: Model = serde_json::from_str(json).expect("valid model JSON");
    let mut windowed = ObservedDataPoint::new(3.0, "B".to_string(), 0.0);
    windowed.window = Some(2.0);
    CalibrationProblem::new(
        DifferenceEquations::try_from_model(&model).unwrap(),
        vec![
            ObservedDataPoint::new(2.0, "A".to_string(), 640.0),
            windowed,
        ],
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

fn sample(values: Vec<Vec<f64>>) -> ParameterSample {
    let log_weights = vec![0.0; values.len()];
    ParameterSample::new(
        vec!["k".to_string()],
        vec![(0.0, 1.0)],
        values,
        &log_weights,
        None,
    )
    .unwrap()
}

#[test]
fn the_band_of_one_draw_is_that_draw() {
    let problem = problem();
    let predictive = Predictive::new(&problem, &sample(vec![vec![0.2]]), &[1.0, 2.0]).unwrap();
    let band = predictive.band(0.9).unwrap();

    // A = 1000 · 0.8ᵗ.
    assert!((band.median[1][0] - 640.0).abs() < 1e-9);
    assert_eq!(band.lower, band.upper);
    assert_eq!(band.lower, predictive.trajectories()[0]);
}

#[test]
fn observation_draws_are_the_predictions_the_loss_compares() {
    let problem = problem();
    let sample = sample(vec![vec![0.2], vec![0.5]]);
    let draws = observation_draws(&problem, &sample, problem.observed_data(), &NoNoise, 1).unwrap();

    for (draw, values) in draws.values.iter().zip(sample.values()) {
        let mut workspace = Workspace::default();
        let predicted = problem
            .forward()
            .run_observed(values, &mut workspace)
            .unwrap();
        assert_eq!(draw.as_slice(), predicted);
    }
    // B over the window from 1 to 3 at k = 0.2: 1000 (0.8 − 0.512).
    assert!((draws.values[0][1] - 288.0).abs() < 1e-9);
}

#[test]
fn a_sample_of_other_parameters_is_named_in_the_error() {
    let other = ParameterSample::new(
        vec!["r".to_string()],
        vec![(0.0, 1.0)],
        vec![vec![0.2]],
        &[0.0],
        None,
    )
    .unwrap();

    let error = Predictive::new(&problem(), &other, &[1.0])
        .expect_err("the sample does not match the problem")
        .to_string();

    assert!(error.contains(r#"["r"]; the problem has ["k"]"#), "{error}");
}
