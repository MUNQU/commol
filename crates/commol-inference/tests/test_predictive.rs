//! The predictive of a sample reads runs through the forward model and
//! observations through the observation operator of the problem.

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, EvaluationError, LossConfig, ObservedDataPoint,
    Violation, Workspace,
};
use commol_core::Model;
use commol_core::NoSteps;
use commol_core::{STOPPED, Steps};
use commol_difference::DifferenceEquations;
use commol_inference::runs::RunEvent;
use commol_inference::{
    Band, NoNoise, ObservationDraws, ParameterSample, Predictive, PredictiveError, draw_seeds,
    observation_draws, scores, share_among_replicates,
};
use commol_random::replicate_seed;
use std::sync::Mutex;

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
        None,
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
    let predictive = Predictive::new(
        problem.forward(),
        &sample(vec![vec![0.2]]),
        &[1.0, 2.0],
        0,
        &NoSteps,
    )
    .unwrap();
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

    let error = Predictive::new(problem().forward(), &other, &[1.0], 0, &NoSteps)
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
        let error = Predictive::new(problem.forward(), &draws, positions, 0, &NoSteps)
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
    let predictive = Predictive::new(
        problem.forward(),
        &sample(vec![vec![0.2]]),
        &[1.0, 2.0],
        0,
        &NoSteps,
    )
    .unwrap();

    let error = predictive.series_band(&[9], false, 0.9).err();

    assert_eq!(
        error.as_deref(),
        Some("Output 9 is out of range; a row has 2 outputs")
    );
}

#[test]
fn a_band_of_a_series_checks_its_confidence_level() {
    let problem = problem();
    let predictive = Predictive::new(
        problem.forward(),
        &sample(vec![vec![0.2]]),
        &[1.0, 2.0],
        0,
        &NoSteps,
    )
    .unwrap();

    let error = predictive.series_band(&[0], false, 1.5).unwrap_err();

    assert!(error.contains("1.5"), "{error}");
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
        None,
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

    let predictive = Predictive::new(problem.forward(), &draws, &[5.0], 11, &NoSteps).unwrap();
    let again = Predictive::new(problem.forward(), &draws, &[5.0], 11, &NoSteps).unwrap();
    let other = Predictive::new(problem.forward(), &draws, &[5.0], 12, &NoSteps).unwrap();

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
    let two = Predictive::new(
        stochastic_problem(2).forward(),
        &draws,
        &[5.0],
        11,
        &NoSteps,
    )
    .unwrap();
    let four = Predictive::new(
        stochastic_problem(4).forward(),
        &draws,
        &[5.0],
        11,
        &NoSteps,
    )
    .unwrap();
    let fewer_draws = sample(vec![vec![0.2], vec![0.3]]);
    let of_two_draws = Predictive::new(
        stochastic_problem(2).forward(),
        &fewer_draws,
        &[5.0],
        11,
        &NoSteps,
    )
    .unwrap();

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

    let pit_of = |draws: &ObservationDraws| {
        scores::ObservationBounds::of_draws(draws, &observed)
            .unwrap()
            .pit()
    };
    let pit = pit_of(&draws);

    assert_eq!(draws.seed, 11);
    assert_eq!(pit_of(&draws_of(11)), pit);
    assert!(pit[0] >= below && pit[0] <= below + at, "{pit:?}");
    // The same values drawn with another seed randomize the tie otherwise.
    let reseeded = ObservationDraws {
        seed: 12,
        ..draws.clone()
    };
    assert_ne!(pit_of(&reseeded), pit);
}

#[test]
fn observation_draws_reject_a_position_the_engine_rejects() {
    let problem = problem();
    let between_steps = [ObservedDataPoint::new(2.5, "A".to_string(), 0.0)];

    let error = observation_draws(
        problem.forward(),
        &sample(vec![vec![0.2]]),
        &between_steps,
        &NoNoise,
        0,
    )
    .unwrap_err();

    let PredictiveError::Mismatch(message) = error else {
        panic!("expected a mismatch, got {error:?}");
    };
    assert!(message.contains("2.5"), "{message}");
}

#[test]
fn each_replicate_has_its_share_of_the_weight_of_its_draw() {
    // Draw 0 has 4 replicates at 1; draw 1 has one replicate at 0.
    let runs = vec![vec![vec![vec![1.0]]; 4], vec![vec![vec![0.0]]]];

    let (trajectories, weights) = share_among_replicates(runs, &[0.2, 0.8]);
    let band = Band::of(&trajectories, &weights, 0.5).unwrap();

    assert_eq!(weights, [0.05, 0.05, 0.05, 0.05, 0.8]);
    assert!((band.mean[0][0] - 0.2).abs() < 1e-12);
}

/// A receiver whose loops have stopped before their first step.
struct Stopped;

impl<E> Steps<E> for Stopped {
    fn stopped(&self) -> bool {
        true
    }
}

/// The events of a loop of model runs, as text: the beginning first, then
/// the points that ran, sorted.
#[derive(Default)]
struct KeptRuns(Mutex<Vec<String>>);

impl KeptRuns {
    fn events(self) -> Vec<String> {
        let mut events = self.0.into_inner().unwrap();
        events[1..].sort();
        events
    }
}

impl Steps<RunEvent> for KeptRuns {
    fn send(&self, event: RunEvent) {
        self.0.lock().unwrap().push(match event {
            RunEvent::Began { total } => format!("began {total}"),
            RunEvent::Ran { index } => format!("ran {index}"),
        });
    }
}

#[test]
fn a_predictive_sends_each_draw_and_runs_none_once_the_steps_stop() {
    let problem = problem();
    let draws = sample(vec![vec![0.2], vec![0.3], vec![0.4]]);
    let kept = KeptRuns::default();
    Predictive::new(problem.forward(), &draws, &[1.0], 0, &kept).unwrap();
    assert_eq!(kept.events(), ["began 3", "ran 0", "ran 1", "ran 2"]);

    let error = Predictive::new(problem.forward(), &draws, &[1.0], 0, &Stopped).unwrap_err();
    assert!(error.to_string().contains(STOPPED), "{error}");
}

/// The problem of the increment of B over 20 steps in four parts of five,
/// read by `model`, at `k = 0.1`: the means of the parts fall from 410 to
/// 50.
fn sum_problem(
    model: commol_calibration::ObservationModel,
) -> CalibrationProblem<DifferenceEquations> {
    let base = problem();
    let observation = ObservedDataPoint {
        parts: 4,
        ..ObservedDataPoint::with_window(20.0, "B".to_string(), 800.0, 20.0)
    };
    CalibrationProblem::new(
        base.forward().base_engine().clone(),
        vec![observation],
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        None,
        vec![],
        commol_calibration::DiscrepancyConfig::Likelihood(vec![("B".to_string(), model)]),
    )
    .unwrap()
}

#[test]
fn draws_of_a_sum_have_the_variance_of_their_parts() {
    use commol_calibration::{Argument, ObservationModel, Sum};
    use commol_inference::{DistributionNoise, ObservationNoise};
    use commol_random::Generator;

    let phi = 0.3;
    let negative_binomial = |sum| ObservationModel::NegativeBinomial {
        overdispersion: Argument::Value(phi),
        sum: Some(sum),
    };
    let cases = [
        (negative_binomial(Sum::Exact), Some(phi)),
        (negative_binomial(Sum::Moments), Some(phi)),
        (ObservationModel::Poisson, Some(0.0)),
        (
            ObservationModel::Normal {
                sd: Argument::Value(3.0),
            },
            None,
        ),
    ];
    for (model, overdispersion) in cases {
        let problem = sum_problem(model);
        let mut workspace = Workspace::default();
        problem.predict_once(&[0.1], None, &mut workspace).unwrap();
        let (predicted, parts) = (workspace.predictions, workspace.parts);
        assert_eq!(parts.len(), 4);
        let mean: f64 = parts.iter().sum();
        let expected_variance = match overdispersion {
            Some(phi) => mean + phi * parts.iter().map(|mu| mu * mu).sum::<f64>(),
            None => 4.0 * 9.0,
        };
        let noise = DistributionNoise::new(&problem).unwrap();
        let observations = problem.observed_data();
        let mut generator = Generator::from_seed(5);
        let count = 40_000;
        let mut out = [0.0];
        let draws: Vec<f64> = (0..count)
            .map(|_| {
                noise.observe(
                    observations,
                    &[0.1],
                    &predicted,
                    &parts,
                    &mut generator,
                    &mut out,
                );
                out[0]
            })
            .collect();
        let draw_mean = draws.iter().sum::<f64>() / count as f64;
        let variance =
            draws.iter().map(|d| (d - draw_mean).powi(2)).sum::<f64>() / (count - 1) as f64;
        // Four standard errors of the mean, and of the variance (about
        // √(2/n) of it, wider for the heavier tails of a negative binomial).
        let mean_error = 4.0 * (expected_variance / count as f64).sqrt();
        assert!((draw_mean - mean).abs() < mean_error, "{draw_mean} {mean}");
        let variance_error = 4.0 * 3.0 * (2.0 / count as f64).sqrt() * expected_variance;
        assert!(
            (variance - expected_variance).abs() < variance_error,
            "{variance} {expected_variance}"
        );
    }
}

#[test]
fn draws_of_a_sum_have_the_distribution_of_their_reading() {
    use commol_calibration::{Argument, Noise, ObservationModel, Sum};
    use commol_inference::{DistributionNoise, ObservationNoise};
    use commol_random::Generator;

    // Unequal parts and a large φ, where the moments of the sum are not its
    // distribution: each reading draws the count 0 as often as its density
    // says, and the two densities there are far apart.
    let (phi, parts) = (2.0, [0.2, 0.5, 1.0, 30.0]);
    let count = 40_000;
    let zero = |sum| {
        Noise::NegativeBinomial {
            overdispersion: phi,
            sum: Some(sum),
        }
        .sum_log_density(0.0, &parts)
        .unwrap()
        .exp()
    };
    let error = |p: f64| 4.0 * (p * (1.0 - p) / count as f64).sqrt();
    let (exact, moments) = (zero(Sum::Exact), zero(Sum::Moments));
    assert!((exact - moments).abs() > 2.0 * (error(exact) + error(moments)));
    for (sum, expected) in [(Sum::Exact, exact), (Sum::Moments, moments)] {
        let problem = sum_problem(ObservationModel::NegativeBinomial {
            overdispersion: Argument::Value(phi),
            sum: Some(sum),
        });
        let noise = DistributionNoise::new(&problem).unwrap();
        let predicted = [parts.iter().sum::<f64>()];
        let mut generator = Generator::from_seed(9);
        let mut out = [0.0];
        let zeros = (0..count)
            .filter(|_| {
                noise.observe(
                    problem.observed_data(),
                    &[0.1],
                    &predicted,
                    &parts,
                    &mut generator,
                    &mut out,
                );
                out[0] == 0.0
            })
            .count();
        let share = zeros as f64 / count as f64;
        assert!(
            (share - expected).abs() < error(expected),
            "{sum:?} {share} {expected}"
        );
    }
}

/// A of the decay of `problem` at step 1, 900 at `k = 0.1`, times the
/// factor `c` and a known scale per observation: `n` observations at
/// step 1, whose means rise from 18·c to 990·c, read by `model`.
fn factor_problem(
    model: commol_calibration::ObservationModel,
    n: usize,
) -> CalibrationProblem<DifferenceEquations> {
    factor_problem_of(model, n, ObservedDataPoint::new(1.0, "A".to_string(), 1.0))
}

/// The problem of `factor_problem` with `n` copies of `observation`, each
/// with its scale.
fn factor_problem_of(
    model: commol_calibration::ObservationModel,
    n: usize,
    observation: ObservedDataPoint,
) -> CalibrationProblem<DifferenceEquations> {
    use commol_calibration::{CalibrationParameterType, DiscrepancyConfig};
    let label = observation.label.clone();
    let observations = (0..n)
        .map(|i| ObservedDataPoint {
            factor_id: Some("c".to_string()),
            scale: 0.02 + 1.08 * i as f64 / n as f64,
            ..observation.clone()
        })
        .collect();
    CalibrationProblem::new(
        problem().forward().base_engine().clone(),
        observations,
        vec![CalibrationParameter::with_type(
            "c".to_string(),
            CalibrationParameterType::Factor,
            0.001,
            10.0,
        )],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![(label, model)]),
    )
    .unwrap()
}

fn factor_sample(values: Vec<f64>, log_weights: &[f64]) -> ParameterSample {
    ParameterSample::new(
        vec!["c".to_string()],
        vec![(0.001, 10.0)],
        values.into_iter().map(|c| vec![c]).collect(),
        log_weights,
        None,
    )
    .unwrap()
}

/// The predicted mean of each observation of `problem` at the factor `c`,
/// and the means of their parts.
fn means_at(problem: &CalibrationProblem<DifferenceEquations>, c: f64) -> (Vec<f64>, Vec<f64>) {
    let mut workspace = Workspace::default();
    problem.predict_once(&[c], None, &mut workspace).unwrap();
    (workspace.predictions, workspace.parts)
}

#[test]
fn the_bounds_of_a_sample_mix_its_draws_by_weight() {
    use commol_calibration::{Noise, ObservationModel};
    use commol_inference::DistributionNoise;

    let problem = factor_problem(ObservationModel::Poisson, 3);
    let noise = DistributionNoise::new(&problem).unwrap();
    let observations: Vec<ObservedDataPoint> = problem
        .observed_data()
        .iter()
        .zip([10.0, 40.0, 400.0])
        .map(|(obs, value)| ObservedDataPoint {
            value,
            ..obs.clone()
        })
        .collect();
    let (first, second) = (0.2, 0.9);
    let sample = factor_sample(vec![first, second], &[0.3_f64.ln(), 0.7_f64.ln()]);

    let bounds = scores::observation_bounds(&problem, &sample, &observations, &noise, 3).unwrap();

    let (low, _) = means_at(&problem, first);
    let (high, _) = means_at(&problem, second);
    let mixture = bounds.mixture();
    let places = bounds.places();
    let pit = bounds.pit();
    for (i, obs) in observations.iter().enumerate() {
        let (a, b) = (
            Noise::Poisson.tails(obs.value, low[i]),
            Noise::Poisson.tails(obs.value, high[i]),
        );
        assert_eq!(bounds.tails[0][i], a);
        assert_eq!(bounds.tails[1][i], b);
        let close = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() <= 1e-15 * expected.abs().max(1e-300),
                "{i}: {actual} against {expected}"
            );
        };
        close(mixture[i].below, 0.3 * a.below + 0.7 * b.below);
        close(mixture[i].at, 0.3 * a.at + 0.7 * b.at);
        close(mixture[i].above, 0.3 * a.above + 0.7 * b.above);
        close(pit[i], mixture[i].below + places[i] * mixture[i].at);
    }
    assert_eq!(bounds.seed, 3);
}

#[test]
fn copies_of_a_point_have_its_bounds() {
    use commol_calibration::{Argument, ObservationModel};
    use commol_inference::DistributionNoise;

    let problem = factor_problem(
        ObservationModel::NegativeBinomial {
            overdispersion: Argument::Value(0.4),
            sum: None,
        },
        5,
    );
    let noise = DistributionNoise::new(&problem).unwrap();
    let observations = problem.observed_data();
    let bounds_of = |copies: usize| {
        scores::observation_bounds(
            &problem,
            &factor_sample(vec![0.05; copies], &vec![0.0; copies]),
            observations,
            &noise,
            8,
        )
        .unwrap()
    };
    let (point, copies) = (bounds_of(1), bounds_of(20));

    for (one, many) in point.mixture().iter().zip(copies.mixture()) {
        for (a, b) in [
            (one.below, many.below),
            (one.at, many.at),
            (one.above, many.above),
        ] {
            assert!((a - b).abs() <= 1e-14 * a.abs(), "{a} {b}");
        }
    }
    for (a, b) in point.pit().iter().zip(copies.pit()) {
        assert!((a - b).abs() <= 1e-14, "{a} {b}");
    }
}

/// The Kolmogorov–Smirnov distance of `values` from the standard normal.
fn normal_distance(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len() as f64;
    sorted
        .iter()
        .enumerate()
        .map(|(i, &z)| {
            let f = commol_statistics::special::normal_cdf(z);
            (f - i as f64 / n).max((i + 1) as f64 / n - f)
        })
        .fold(0.0, f64::max)
}

#[test]
fn quantile_residuals_of_unseen_data_are_standard_normal() {
    use commol_calibration::{Argument, Center, ObservationModel, Sum};
    use commol_inference::{DistributionNoise, ObservationNoise};
    use commol_random::Generator;
    use commol_random::sample::{gamma, normal, poisson};

    let n = 400;
    // The distance of 1 % of the Kolmogorov–Smirnov test of n values.
    let critical = 1.628 / (n as f64).sqrt();
    let value = |x: f64| Argument::Value(x);
    let negative_binomial = |phi: f64, sum| ObservationModel::NegativeBinomial {
        overdispersion: value(phi),
        sum,
    };
    // At a point: data drawn from each model at the factor 0.3 (means from
    // 5 to 300) and read there; the observations of several parts are the
    // windows of B over steps 1 to 20 in four parts.
    let models = [
        ObservationModel::Normal { sd: value(4.0) },
        ObservationModel::LogNormal {
            sigma: value(0.4),
            center: Center::Mean,
        },
        ObservationModel::LogNormal {
            sigma: value(0.4),
            center: Center::Median,
        },
        ObservationModel::StudentT {
            df: value(3.0),
            scale: value(2.0),
        },
        ObservationModel::Poisson,
        negative_binomial(0.5, None),
        negative_binomial(5e-9, None),
        ObservationModel::Binomial { trials: Some(1000) },
    ];
    let windowed = |model: ObservationModel| {
        let window = ObservedDataPoint {
            parts: 4,
            ..ObservedDataPoint::with_window(20.0, "B".to_string(), 1.0, 20.0)
        };
        factor_problem_of(model, n, window)
    };
    let mut problems: Vec<(String, CalibrationProblem<DifferenceEquations>)> = models
        .into_iter()
        .map(|model| (format!("{model:?}"), factor_problem(model, n)))
        .collect();
    for model in [
        negative_binomial(0.5, Some(Sum::Exact)),
        negative_binomial(0.5, Some(Sum::Moments)),
        ObservationModel::Poisson,
        ObservationModel::Normal { sd: value(4.0) },
    ] {
        problems.push((format!("{model:?} of four parts"), windowed(model)));
    }
    let point = factor_sample(vec![0.3], &[0.0]);
    for (index, (name, problem)) in problems.iter().enumerate() {
        let noise = DistributionNoise::new(problem).unwrap();
        let (predicted, parts) = means_at(problem, 0.3);
        let mut drawn = vec![0.0; n];
        let mut generator = Generator::from_seed(300 + index as u64);
        noise.observe(
            problem.observed_data(),
            &[0.3],
            &predicted,
            &parts,
            &mut generator,
            &mut drawn,
        );
        let observations: Vec<ObservedDataPoint> = problem
            .observed_data()
            .iter()
            .zip(&drawn)
            .map(|(obs, &value)| ObservedDataPoint {
                value,
                ..obs.clone()
            })
            .collect();
        let residuals =
            scores::observation_bounds(problem, &point, &observations, &noise, index as u64)
                .unwrap()
                .quantile_residuals();
        let distance = normal_distance(&residuals);
        assert!(distance < critical, "{name}: {distance} ≥ {critical}");
    }

    // Held out from a conjugate posterior and read with its draws: the
    // factor of a normal model of a known sd, under a flat prior, and of a
    // Poisson model, under the gamma prior of shape 1 and rate 0.
    let draws = 2000;
    let truth = 0.3;
    let mut generator = Generator::from_seed(7);
    for (name, model) in [
        ("normal", ObservationModel::Normal { sd: value(20.0) }),
        ("Poisson", ObservationModel::Poisson),
    ] {
        let problem = factor_problem(model, n);
        let noise = DistributionNoise::new(&problem).unwrap();
        let (unit, _) = means_at(&problem, 1.0);
        // The training data: one draw of each observation at the truth.
        let trained: Vec<f64> = unit
            .iter()
            .map(|&m| match name {
                "normal" => truth * m + 20.0 * normal(&mut generator),
                _ => poisson(&mut generator, truth * m) as f64,
            })
            .collect();
        let posterior = |generator: &mut Generator| -> f64 {
            match name {
                // N(Σ y m / Σ m², σ²/Σ m²).
                "normal" => {
                    let squares: f64 = unit.iter().map(|m| m * m).sum();
                    let centre: f64 =
                        trained.iter().zip(&unit).map(|(y, m)| y * m).sum::<f64>() / squares;
                    centre + 20.0 / squares.sqrt() * normal(generator)
                }
                // Gamma(1 + Σ y, Σ m).
                _ => gamma(
                    generator,
                    1.0 + trained.iter().sum::<f64>(),
                    unit.iter().sum::<f64>(),
                ),
            }
        };
        let values: Vec<f64> = (0..draws).map(|_| posterior(&mut generator)).collect();
        let sample = factor_sample(values, &vec![0.0; draws]);
        // Each held-out value: a fresh draw of the posterior, then of the
        // model there.
        let observations: Vec<ObservedDataPoint> = problem
            .observed_data()
            .iter()
            .zip(&unit)
            .map(|(obs, &m)| {
                let c = posterior(&mut generator);
                let value = match name {
                    "normal" => c * m + 20.0 * normal(&mut generator),
                    _ => poisson(&mut generator, c * m) as f64,
                };
                ObservedDataPoint {
                    value,
                    ..obs.clone()
                }
            })
            .collect();
        let residuals = scores::observation_bounds(&problem, &sample, &observations, &noise, 1)
            .unwrap()
            .quantile_residuals();
        let distance = normal_distance(&residuals);
        assert!(distance < critical, "{name}: {distance} ≥ {critical}");
    }
}
