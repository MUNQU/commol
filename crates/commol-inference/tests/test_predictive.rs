//! The predictive of a sample reads runs through the forward model and
//! observations through the observation operator of the problem.

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, EvaluationError, LossConfig, ObservedDataPoint,
    Violation, Workspace,
};
use commol_core::Model;
use commol_difference::DifferenceEquations;
use commol_inference::{
    NoNoise, ObservationDraws, ParameterSample, Predictive, PredictiveError, draw_seeds,
    observation_draws, scores,
};
use commol_random::replicate_seed;

/// A → B at the proportional rate `k`, in discrete steps from A = 1000.
fn problem() -> CalibrationProblem<DifferenceEquations> {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"DifferenceEquations"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],
        "target":["B"],"rate":"k","stratified_rates":null,"per_compartment":null}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#;
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
    let predictive =
        Predictive::new(problem.forward(), &sample(vec![vec![0.2]]), &[1.0, 2.0], 0).unwrap();
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
    let draws = observation_draws(
        problem.forward(),
        &sample,
        problem.observed_data(),
        &NoNoise,
        1,
    )
    .unwrap();

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

    let error = Predictive::new(problem().forward(), &other, &[1.0], 0)
        .expect_err("the sample does not match the problem")
        .to_string();

    assert!(
        error.contains(r#"["r"]; the forward model has ["k"]"#),
        "{error}"
    );
}

#[test]
fn positions_the_engine_cannot_run_are_rejected_before_any_draw() {
    let problem = problem();
    let draws = sample(vec![vec![0.1], vec![0.2]]);
    let cases: [(&[f64], &str); 3] = [
        (&[0.0, 1.0, 1.0], "sorted and unique, got 1 before 1"),
        (&[2.0, 1.0], "sorted and unique, got 2 before 1"),
        (&[0.5], "integer steps; 0.5 is not one"),
    ];

    for (positions, message) in cases {
        let error = Predictive::new(problem.forward(), &draws, positions, 0)
            .expect_err("the engine cannot run these positions");

        assert!(
            matches!(&error, PredictiveError::Mismatch(text) if text.contains(message)),
            "{positions:?}: {error:?}"
        );
    }
}

#[test]
fn a_draw_without_a_run_is_named_with_its_reason() {
    let failed = PredictiveError::Draw {
        draw: 1,
        error: EvaluationError::Failed("the engine stopped".to_string()),
    };
    assert_eq!(failed.to_string(), "Draw 1: the engine stopped");
    let infeasible = PredictiveError::Draw {
        draw: 2,
        error: EvaluationError::Violation(Violation::NonFinite),
    };
    assert_eq!(
        infeasible.to_string(),
        "Draw 2: Infeasible point: NonFinite"
    );
}

#[test]
fn a_forward_model_observed_again_at_its_observations_predicts_the_same() {
    // The second observation counts B over the window (1, 3].
    let problem = problem();
    let again = problem
        .forward()
        .with_observations(problem.observed_data())
        .unwrap();
    let mut workspace = Workspace::default();

    let first = problem
        .forward()
        .run_observed(&[0.1], &mut workspace)
        .unwrap()
        .to_vec();
    let second = again.run_observed(&[0.1], &mut workspace).unwrap();
    assert_eq!(first, second);

    let windowed = problem
        .forward()
        .with_observations(&problem.observed_data()[1..])
        .unwrap();
    let alone = windowed.run_observed(&[0.1], &mut workspace).unwrap();
    assert_eq!(alone, &first[1..]);
}

#[test]
fn a_band_of_an_output_the_rows_lack_is_an_error() {
    let problem = problem();
    let predictive =
        Predictive::new(problem.forward(), &sample(vec![vec![0.2]]), &[1.0, 2.0], 0).unwrap();

    let error = predictive.series_band(&[9], false, 0.9).err();

    assert_eq!(
        error.as_deref(),
        Some("Output 9 is out of range; a row has 2 outputs")
    );
}

/// A → B at `k`, from A = 1000, with the chain binomial, `replicates`
/// replicates per evaluation.
fn stochastic_problem(replicates: usize) -> CalibrationProblem<commol_stochastic::StochasticSteps> {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"StochasticSteps","settings":null},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k"}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#;
    let model: Model = serde_json::from_str(json).expect("valid model JSON");
    CalibrationProblem::new(
        commol_stochastic::StochasticSteps::try_from_model(&model).unwrap(),
        vec![ObservedDataPoint::new(5.0, "A".to_string(), 0.0)],
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
    .with_replicates(Some(
        commol_calibration::Replicates::new(
            replicates,
            3,
            commol_calibration::ReplicateSummary::Mean,
        )
        .unwrap(),
    ))
    .unwrap()
}

#[test]
fn every_draw_of_a_stochastic_predictive_draws_its_own_noise() {
    let problem = stochastic_problem(2);
    // Three draws of one parameter value: with the replicates of the
    // problem they would run the same two paths.
    let draws = sample(vec![vec![0.2]; 3]);

    let predictive = Predictive::new(problem.forward(), &draws, &[5.0], 11).unwrap();
    let again = Predictive::new(problem.forward(), &draws, &[5.0], 11).unwrap();
    let other = Predictive::new(problem.forward(), &draws, &[5.0], 12).unwrap();

    let a_at_5: Vec<f64> = predictive
        .trajectories()
        .iter()
        .map(|rows| rows[0][0])
        .collect();
    assert_eq!(a_at_5.len(), 6);
    let distinct: std::collections::BTreeSet<u64> = a_at_5.iter().map(|v| v.to_bits()).collect();
    assert!(distinct.len() > 2, "{a_at_5:?}");
    assert_eq!(predictive, again);
    assert_ne!(predictive, other);
    // The draws of observations run the same paths as the predictive.
    let observed = observation_draws(
        problem.forward(),
        &draws,
        problem.observed_data(),
        &NoNoise,
        11,
    )
    .unwrap();
    let observed_a: Vec<f64> = observed.values.iter().map(|row| row[0]).collect();
    assert_eq!(observed_a, a_at_5);
}

#[test]
fn a_draw_keeps_its_runs_when_the_replicates_change() {
    let draws = sample(vec![vec![0.2], vec![0.3], vec![0.4]]);
    let two = Predictive::new(stochastic_problem(2).forward(), &draws, &[5.0], 11).unwrap();
    let four = Predictive::new(stochastic_problem(4).forward(), &draws, &[5.0], 11).unwrap();
    let fewer_draws = sample(vec![vec![0.2], vec![0.3]]);
    let of_two_draws =
        Predictive::new(stochastic_problem(2).forward(), &fewer_draws, &[5.0], 11).unwrap();

    assert_eq!(two.draw_seeds(), draw_seeds(11, 3).as_slice());
    assert_eq!(four.draw_seeds(), two.draw_seeds());
    assert_eq!(of_two_draws.draw_seeds(), &two.draw_seeds()[..2]);
    for draw in 0..3 {
        assert_eq!(
            &two.trajectories()[2 * draw..2 * draw + 2],
            &four.trajectories()[4 * draw..4 * draw + 2],
        );
    }
    assert_eq!(of_two_draws.trajectories(), &two.trajectories()[..4]);
    // Replicate i of draw j runs with the seed replicate i of the seed of
    // the draw.
    let problem = stochastic_problem(4);
    let forward = problem.forward();
    for (draw, values) in draws.values().iter().enumerate() {
        for replicate in 0..4 {
            let seed = replicate_seed(four.draw_seeds()[draw], replicate as u64);
            let run = forward.run_once_at(values, &[5.0], Some(seed)).unwrap();
            assert_eq!(run, four.trajectories()[4 * draw + replicate]);
        }
    }
}

#[test]
fn the_pit_of_ties_follows_the_seed_of_the_draws() {
    // Whole units: the draws of A at 5 tie with each other and with the
    // observed value, the first draw's.
    let problem = stochastic_problem(1);
    let sample = sample(vec![vec![0.2]; 40]);
    let draws_of = |seed| {
        observation_draws(
            problem.forward(),
            &sample,
            problem.observed_data(),
            &NoNoise,
            seed,
        )
        .unwrap()
    };
    let draws = draws_of(11);
    let observed = [draws.values[0][0]];
    let below: f64 = draws
        .values
        .iter()
        .zip(&draws.weights)
        .filter(|(row, _)| row[0] < observed[0])
        .map(|(_, w)| w)
        .sum();
    let at: f64 = draws
        .values
        .iter()
        .zip(&draws.weights)
        .filter(|(row, _)| row[0] == observed[0])
        .map(|(_, w)| w)
        .sum();

    let pit = scores::pit(&draws, &observed).unwrap();

    assert_eq!(draws.seed, 11);
    assert_eq!(scores::pit(&draws_of(11), &observed).unwrap(), pit);
    assert!(pit[0] >= below && pit[0] <= below + at, "{pit:?}");
    // The same values drawn with another seed randomize the tie otherwise.
    let reseeded = ObservationDraws {
        seed: 12,
        ..draws.clone()
    };
    assert_ne!(scores::pit(&reseeded, &observed).unwrap(), pit);
}
