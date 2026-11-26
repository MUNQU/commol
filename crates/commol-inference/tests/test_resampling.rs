//! Blocks, resamplers, the replicate runner, the jackknife and residual
//! noise, on engines whose output is a closed form of the parameters.

use std::collections::HashMap;

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, LossConfig, NelderMeadConfig, ObservedDataPoint,
    OptimizationConfig, Workspace, optimize_starts, start_points,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};
use commol_inference::conformal::{
    ConformalSettings, NonconformityScore, Pooling, enbpi, jackknife_plus, rolling_conformal,
};
use commol_inference::local::{
    CurvatureKind, curvature_at, sandwich, sandwich_approximation, scaled_block_scores,
};
use commol_inference::profiles::{ProfiledQuantity, bootstrap_thresholds};
use commol_inference::resampling::{
    BayesianBootstrap, BlockScheme, Multipliers, Perturbation, ReplicateRunner, Resampler,
    ResidualBlocks, ResidualBootstrap, ResidualNoise, ResidualScale, Residuals, WildBootstrap,
    block_jackknife, bootstrap,
};
use commol_inference::sensitivity::Matrix;
use commol_inference::smc::information_matching_eta;
use commol_inference::{NoNoise, ObservationNoise, ParameterSample, observation_draws};
use commol_random::Generator;
use commol_random::Stream;
use commol_random::sample::normal;

type Formula = fn(&HashMap<String, f64>, f64) -> f64;

/// One output `x = formula(parameters, t)` at every position `t`; a run
/// whose output is not a number fails.
#[derive(Clone, Debug)]
struct ClosedForm {
    formula: Formula,
    parameters: HashMap<String, f64>,
}

#[derive(Clone, Debug)]
struct NoPlan;

impl ValuePlan for NoPlan {
    fn infeasibility(&self, excess: f64) -> String {
        format!("{excess}")
    }
}

impl SimulationEngine for ClosedForm {
    fn check_varied_values(&self, _plan: &Self::ValuePlan) -> Result<(), String> {
        Ok(())
    }

    fn check_values(&self) -> Result<(), String> {
        Ok(())
    }

    type ValuePlan = NoPlan;

    fn check_position(&self, _position: f64) -> Result<(), String> {
        Ok(())
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        buffer.clear();
        for &t in positions {
            let value = (self.formula)(&self.parameters, t);
            if value.is_nan() {
                return Err(SimulationError::Failed(
                    "the run is not a number".to_string(),
                ));
            }
            buffer.push(vec![value]);
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        vec!["x".to_string()]
    }

    fn state_outputs(&self) -> usize {
        1
    }

    fn reset(&mut self) {}

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.parameters.insert(parameter_id.to_string(), value);
        Ok(())
    }

    fn current_position(&self) -> f64 {
        0.0
    }

    fn plan_values(&self, _: &[ModelValue], _: Option<&str>) -> Result<NoPlan, String> {
        Ok(NoPlan)
    }

    fn apply_values(&mut self, _: &NoPlan, _: &[f64]) -> Result<f64, String> {
        Ok(0.0)
    }
}

fn line(p: &HashMap<String, f64>, t: f64) -> f64 {
    p["a"] + p["b"] * t
}

fn sine(p: &HashMap<String, f64>, t: f64) -> f64 {
    (p["k"] * t).sin()
}

/// The problem of fitting `formula` to `values` at positions 0, 1, 2, ….
fn problem(
    formula: Formula,
    parameters: &[(&str, f64, f64)],
    values: &[f64],
) -> CalibrationProblem<ClosedForm> {
    CalibrationProblem::new(
        ClosedForm {
            formula,
            parameters: parameters
                .iter()
                .map(|(id, min, _)| (id.to_string(), *min))
                .collect(),
        },
        values
            .iter()
            .enumerate()
            .map(|(t, &value)| ObservedDataPoint::new(t as f64, "x".to_string(), value))
            .collect(),
        parameters
            .iter()
            .map(|(id, min, max)| CalibrationParameter::new(id.to_string(), *min, *max))
            .collect(),
        None,
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

fn line_problem(values: &[f64]) -> CalibrationProblem<ClosedForm> {
    problem(line, &[("a", -50.0, 50.0), ("b", -10.0, 10.0)], values)
}

/// The problem of fitting the line to `observations`.
fn line_problem_of(observations: Vec<ObservedDataPoint>) -> CalibrationProblem<ClosedForm> {
    CalibrationProblem::new(
        ClosedForm {
            formula: line,
            parameters: HashMap::from([("a".to_string(), 0.0), ("b".to_string(), 0.0)]),
        },
        observations,
        vec![
            CalibrationParameter::new("a".to_string(), -50.0, 50.0),
            CalibrationParameter::new("b".to_string(), -10.0, 10.0),
        ],
        None,
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

fn config() -> OptimizationConfig {
    OptimizationConfig::NelderMead(NelderMeadConfig {
        sd_tolerance: 1e-12,
        max_iterations: 2000,
        ..NelderMeadConfig::default()
    })
}

/// An AR(1) series with coefficient `phi` and unit innovations.
fn ar1(phi: f64, count: usize, seed: u64) -> Vec<f64> {
    let mut previous = 0.0;
    let mut generator = Generator::from_seed(seed);
    (0..count)
        .map(|_| normal(&mut generator))
        .map(|e| {
            previous = phi * previous + e;
            previous
        })
        .collect()
}

#[test]
fn blocks_cover_every_observation_once_across_series() {
    let observations: Vec<ObservedDataPoint> = ["x", "y"]
        .iter()
        .flat_map(|label| {
            (0..10).map(move |t| ObservedDataPoint::new(t as f64, label.to_string(), 0.0))
        })
        .collect();
    let scheme = BlockScheme::new(&observations, 3).unwrap();

    assert_eq!(scheme.count(), 4);
    let mut seen: Vec<usize> = scheme.blocks().iter().flatten().copied().collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..20).collect::<Vec<_>>());
    for block in scheme.blocks() {
        let positions: Vec<f64> = block.iter().map(|&i| observations[i].at).collect();
        let labels: Vec<&str> = block
            .iter()
            .map(|&i| observations[i].label.as_str())
            .collect();
        assert!(labels.contains(&"x") && labels.contains(&"y"));
        let (min, max) = positions
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        assert!(max - min <= 2.0, "{positions:?}");
    }
}

#[test]
fn a_given_block_length_gives_the_blocks_of_its_windows() {
    // `x` at 0 to 4 and 8 to 12, `y` at 3, 9, 12 and 14: 11 distinct
    // positions, with a gap that no window of the axis would skip.
    let positions = [0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut observations: Vec<ObservedDataPoint> = positions
        .iter()
        .zip(ar1(0.5, 10, 4))
        .map(|(&t, e)| ObservedDataPoint::new(t, "x".to_string(), 2.0 + 0.5 * t + e))
        .collect();
    observations.extend([3.0, 9.0, 12.0, 14.0].map(|t| {
        ObservedDataPoint::of_sum(t, "y".to_string(), vec!["x".to_string()], 2.5 + 0.5 * t)
    }));
    let problem = line_problem_of(observations);
    let fit = [2.0, 0.5];
    let curvature = curvature_at(&problem, &fit, CurvatureKind::GaussNewton).unwrap();
    let gradients = curvature.scaled_gradients();

    // The `η` of each length, and its blocks as windows of the distinct
    // positions, which hold the observations of `y` as 10, 11, 12 and 13.
    let windows: [(usize, f64, Vec<Vec<usize>>); 2] = [
        (
            3,
            0.11636936919447627,
            vec![
                vec![0, 1, 2],
                vec![3, 4, 5, 10],
                vec![6, 7, 8, 11],
                vec![9, 12, 13],
            ],
        ),
        (
            4,
            0.07649049084066743,
            vec![
                vec![0, 1, 2, 3, 10],
                vec![4, 5, 6, 7, 11],
                vec![8, 9, 12, 13],
            ],
        ),
    ];
    for (length, eta, blocks) in windows {
        let scheme = BlockScheme::new(problem.observed_data(), length).unwrap();
        assert_eq!(scheme.blocks(), blocks);

        let run =
            sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, length).unwrap();
        let expected = sandwich(&curvature, &scheme).unwrap();
        assert_eq!((run.block_length, run.blocks), (length, blocks.len()));
        assert_eq!(run.gaussian.covariance(), expected.covariance());

        let found = information_matching_eta(&problem, &fit, length).unwrap();
        assert_eq!(found.block_length, length);
        assert!((found.eta / eta - 1.0).abs() < 1e-9, "{}", found.eta);
        // The scores the two readers sum, to the bit.
        let mut scores = Matrix::zeros(blocks.len(), 2);
        for (b, block) in blocks.iter().enumerate() {
            for &i in block {
                for j in 0..2 {
                    scores[(b, j)] += gradients[(i, j)];
                }
            }
        }
        assert_eq!(scaled_block_scores(&curvature, &scheme), scores);
    }
}

#[test]
fn a_block_of_scores_of_zero_counts_in_the_sandwich() {
    // The line runs through the observations of the first two of five
    // blocks: their scores are 0 and their curvature is not, so they read the
    // degrees of freedom they read with residuals.
    let run = |first: [f64; 4]| {
        let values: Vec<f64> = first
            .iter()
            .chain(&[0.3, -0.2, 0.5, 0.1, -0.4, 0.2])
            .enumerate()
            .map(|(t, e)| 2.0 + 0.5 * t as f64 + e)
            .collect();
        let problem = line_problem(&values);
        sandwich_approximation(&problem, &[2.0, 0.5], CurvatureKind::GaussNewton, 2).unwrap()
    };

    let [zeros, residuals] = [[0.0; 4], [0.2, -0.1, 0.3, -0.3]].map(run);

    assert_eq!(zeros.blocks, 5);
    assert_eq!(
        zeros.gaussian.degrees_of_freedom(),
        residuals.gaussian.degrees_of_freedom()
    );
}

#[test]
fn bayesian_weights_have_mean_one_and_are_shared_by_a_block() {
    let problem = line_problem(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let scheme = BlockScheme::new(problem.observed_data(), 2).unwrap();
    let resampler = BayesianBootstrap::new(&problem, &[1.0, 1.0], &scheme).unwrap();
    let Perturbation::Weights(weights) = resampler.replicate(&mut Generator::from_seed(3)) else {
        panic!("a Bayesian bootstrap draws weights");
    };
    assert!((weights.iter().sum::<f64>() / 6.0 - 1.0).abs() < 1e-12);
    for pair in weights.chunks(2) {
        assert_eq!(pair[0], pair[1]);
    }
    assert_ne!(weights[0], weights[2]);
}

#[test]
fn a_bootstrap_has_one_draw_per_replicate_grouped_by_replicate() {
    let values: Vec<f64> = (0..12)
        .map(|t| 1.0 + 0.5 * t as f64 + if t % 2 == 0 { 0.3 } else { -0.3 })
        .collect();
    let problem = line_problem(&values);
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[1.0, 0.5], 1, 9).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 3).unwrap();

    let run = bootstrap(
        &runner,
        &BayesianBootstrap::new(&problem, &[1.0, 1.0], &scheme).unwrap(),
        6,
    )
    .unwrap();

    assert_eq!(run.sample.len(), 6);
    assert_eq!(run.sample.groups(), Some(&[0, 1, 2, 3, 4, 5][..]));
    assert!(run.failed.is_empty());
    assert_eq!(run.negative_values, 0);
    let spread = run.sample.covariance();
    assert!(spread[0][0] > 0.0 && spread[1][1] > 0.0);
}

#[test]
fn a_negative_synthetic_value_is_kept_and_counted() {
    // Values near zero with residuals of ±2: an additive replicate goes
    // below zero.
    let values: Vec<f64> = (0..10)
        .map(|t| 0.5 + if t % 2 == 0 { 2.0 } else { -2.0 })
        .collect();
    let problem = line_problem(&values);
    let config = config();
    let fit = [0.5, 0.0];
    let runner = ReplicateRunner::new(&problem, &config, &fit, 1, 4).unwrap();
    let residuals = Residuals::new(&problem, &fit, ResidualScale::Additive).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 1).unwrap();
    let resampler = ResidualBootstrap::new(
        problem.observed_data(),
        residuals,
        &scheme,
        ResidualBlocks::Moving,
    )
    .unwrap();

    let run = bootstrap(&runner, &resampler, 8).unwrap();

    assert!(run.negative_values > 0);
    assert_eq!(run.sample.len(), 8);
}

#[test]
fn residual_blocks_are_runs_of_consecutive_residuals() {
    let values: Vec<f64> = (0..12).map(|t| 3.0 + (t * t) as f64 * 0.1).collect();
    let problem = line_problem(&values);
    let fit = [3.0, 0.0];
    let residuals = Residuals::new(&problem, &fit, ResidualScale::Additive).unwrap();
    let original = residuals.residuals().to_vec();
    let scheme = BlockScheme::new(problem.observed_data(), 4).unwrap();

    for blocks in [ResidualBlocks::Moving, ResidualBlocks::Stationary] {
        let resampler =
            ResidualBootstrap::new(problem.observed_data(), residuals.clone(), &scheme, blocks)
                .unwrap();
        let Perturbation::Values(synthetic) = resampler.replicate(&mut Generator::from_seed(1))
        else {
            panic!("a residual bootstrap draws values");
        };
        // Each synthetic residual is one of the fit's, and the source of
        // each moving block is a run of 4 consecutive positions.
        let source: Vec<usize> = synthetic
            .iter()
            .map(|value| {
                original
                    .iter()
                    .position(|r| (3.0 + r - value).abs() < 1e-12)
                    .expect("a residual of the fit")
            })
            .collect();
        if blocks == ResidualBlocks::Moving {
            for run in source.chunks(4) {
                for pair in run.windows(2) {
                    assert_eq!(pair[1], pair[0] + 1, "{source:?}");
                }
            }
        }
    }
}

#[test]
fn a_residual_is_added_or_multiplies_by_its_exponential() {
    assert_eq!(ResidualScale::Additive.apply(2.0, 3.0), 5.0);
    let value = ResidualScale::Multiplicative.apply(2.0, 3f64.ln());
    assert!((value - 6.0).abs() < 1e-12, "{value}");
}

#[test]
fn the_multiplicative_scale_needs_positive_values() {
    let problem = line_problem(&[1.0, 0.0, 2.0]);
    let error = Residuals::new(&problem, &[1.0, 0.0], ResidualScale::Multiplicative)
        .unwrap_err()
        .to_string();
    assert!(error.contains("Observation 'x' at 1"), "{error}");
    assert!(error.contains("residual_scale=\"additive\""), "{error}");
}

#[test]
fn wild_multipliers_keep_each_residual_at_its_position() {
    let values = [1.0, 4.0, 2.0, 7.0, 3.0, 5.0];
    let problem = line_problem(&values);
    let fit = [0.0, 0.0];
    let residuals = Residuals::new(&problem, &fit, ResidualScale::Additive).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 2).unwrap();
    let resampler = WildBootstrap::new(residuals, &scheme, Multipliers::Rademacher);

    for seed in 0..5 {
        let Perturbation::Values(synthetic) = resampler.replicate(&mut Generator::from_seed(seed))
        else {
            panic!("a wild bootstrap draws values");
        };
        for (block, pair) in synthetic.chunks(2).enumerate() {
            let signs: Vec<f64> = pair
                .iter()
                .zip(&values[2 * block..])
                .map(|(s, v)| s / v)
                .collect();
            assert!(signs[0].abs() == 1.0 && signs[0] == signs[1], "{signs:?}");
        }
    }
}

#[test]
fn mammen_multipliers_multiply_the_residuals() {
    let values = [1.0, 4.0, 2.0, 7.0];
    let problem = line_problem(&values);
    let residuals = Residuals::new(&problem, &[0.0, 0.0], ResidualScale::Additive).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 1).unwrap();
    let resampler = WildBootstrap::new(residuals, &scheme, Multipliers::Mammen);
    let root5 = 5f64.sqrt();
    let (low, high) = (-(root5 - 1.0) / 2.0, (root5 + 1.0) / 2.0);

    for seed in 0..5 {
        let Perturbation::Values(synthetic) = resampler.replicate(&mut Generator::from_seed(seed))
        else {
            panic!("a wild bootstrap draws values");
        };
        for (s, v) in synthetic.iter().zip(values) {
            let factor = s / v;
            assert!(
                (factor - low).abs() < 1e-12 || (factor - high).abs() < 1e-12,
                "{factor}"
            );
        }
    }
}

#[test]
fn rademacher_multipliers_are_minus_one_or_one_as_often() {
    let mut generator = Generator::from_seed(12);
    let draws: Vec<f64> = (0..10_000)
        .map(|_| Multipliers::Rademacher.draw(&mut generator))
        .collect();
    assert!(draws.iter().all(|d| d.abs() == 1.0));
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    assert!(mean.abs() < 0.05, "{mean}");
}

#[test]
fn mammen_multipliers_have_their_moments() {
    let mut generator = Generator::from_seed(11);
    let draws: Vec<f64> = (0..200_000)
        .map(|_| Multipliers::Mammen.draw(&mut generator))
        .collect();
    let moment = |k: i32| draws.iter().map(|v| v.powi(k)).sum::<f64>() / draws.len() as f64;
    assert!(moment(1).abs() < 0.01, "{}", moment(1));
    assert!((moment(2) - 1.0).abs() < 0.01, "{}", moment(2));
    assert!((moment(3) - 1.0).abs() < 0.02, "{}", moment(3));
}

/// A resampler whose odd replicates get a value that is not finite.
struct HalfBroken {
    values: Vec<f64>,
    always: bool,
}

impl Resampler for HalfBroken {
    fn replicate(&self, generator: &mut Generator) -> Perturbation {
        let mut values = self.values.clone();
        if self.always || generator.uniform() < 0.5 {
            values[0] = f64::NAN;
        }
        Perturbation::Values(values)
    }
}

/// A resampler that draws the same values every time.
struct Same(Vec<f64>);

impl Resampler for Same {
    fn replicate(&self, _: &mut Generator) -> Perturbation {
        Perturbation::Values(self.0.clone())
    }
}

#[test]
fn a_bootstrap_counts_values_below_zero_and_fits_that_did_not_converge() {
    let problem = line_problem(&[1.0, 2.0, 3.0, 4.0]);
    let same = Same(vec![-1.0, -2.0, 0.0, 3.0]);
    let run = |config: &OptimizationConfig| {
        let runner = ReplicateRunner::new(&problem, config, &[1.0, 1.0], 1, 3).unwrap();
        bootstrap(&runner, &same, 4).unwrap()
    };
    let one_iteration = OptimizationConfig::NelderMead(NelderMeadConfig {
        max_iterations: 1,
        ..NelderMeadConfig::default()
    });

    let converged = run(&config());
    assert_eq!(converged.negative_values, 8);
    assert_eq!(converged.not_converged, 0);
    assert_eq!(run(&one_iteration).not_converged, 4);
}

#[test]
fn a_failed_replicate_is_reported_not_sampled() {
    let values = [1.0, 2.0, 3.0, 4.0];
    let problem = line_problem(&values);
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[1.0, 1.0], 1, 2).unwrap();

    let run = bootstrap(
        &runner,
        &HalfBroken {
            values: values.to_vec(),
            always: false,
        },
        12,
    )
    .unwrap();
    assert!(!run.failed.is_empty());
    assert_eq!(run.sample.len() + run.failed.len(), 12);
    let groups = run.sample.groups().unwrap();
    for (index, reason) in &run.failed {
        assert!(!groups.contains(index));
        assert!(reason.contains("finite"), "{reason}");
    }

    let error = bootstrap(
        &runner,
        &HalfBroken {
            values: values.to_vec(),
            always: true,
        },
        3,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.starts_with(
            "Every replicate failed: 0 of 3 replicates fitted, and the method needs 1; the \
             first reasons: replicate 0: "
        ),
        "{error}"
    );
}

#[test]
fn a_replicate_keeps_its_best_start() {
    let values: Vec<f64> = (0..12).map(|t| (2.0 * t as f64).sin()).collect();
    let problem = problem(sine, &[("k", 0.0, 3.0)], &values);
    let config = config();
    let same = Perturbation::Weights(vec![1.0; values.len()]);
    // A fit in a poor basin: one start stays there, eight find a better one.
    let one = ReplicateRunner::new(&problem, &config, &[0.4], 1, 5)
        .unwrap()
        .fit_one(0, &same)
        .unwrap();
    let eight = ReplicateRunner::new(&problem, &config, &[0.4], 8, 5)
        .unwrap()
        .fit_one(0, &same)
        .unwrap();
    assert!(eight.final_loss < one.final_loss - 1.0, "{one:?} {eight:?}");
    assert!((eight.best_parameters[0] - 2.0).abs() < 1e-4, "{eight:?}");
}

#[test]
fn a_replicate_fit_is_the_calibration_of_its_data() {
    let values: Vec<f64> = (0..12).map(|t| (2.0 * t as f64).sin()).collect();
    let problem = problem(sine, &[("k", 0.0, 3.0)], &values);
    let config = config();
    let weights = Perturbation::Weights((0..values.len()).map(|i| 0.5 + (i % 3) as f64).collect());
    let runner = ReplicateRunner::new(&problem, &config, &[0.4], 4, 5).unwrap();
    let replicate = runner.fit_one(2, &weights).unwrap();

    // The calibration of the data of replicate 2, from the fit and the
    // hypercube, with the seed of the fit of replicate 2.
    let seed = Stream::ReplicateFits.replicate(5, 2);
    let data = runner.replicate_problem(&weights).unwrap();
    let mut points = start_points(data.parameters(), 4, seed);
    points[0] = runner.fit().to_vec();
    let calibration = optimize_starts(&data, &config, points, seed).unwrap();

    assert_eq!(
        format!("{replicate:?}"),
        format!("{:?}", calibration.best())
    );
}

#[test]
fn one_seed_gives_one_bootstrap() {
    let values = [1.0, 2.5, 2.8, 4.1, 5.2, 5.9];
    let problem = line_problem(&values);
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[1.0, 1.0], 2, 21).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 2).unwrap();
    let resampler = BayesianBootstrap::new(&problem, &[1.0, 1.0], &scheme).unwrap();
    let first = bootstrap(&runner, &resampler, 5).unwrap();
    let again = bootstrap(&runner, &resampler, 5).unwrap();
    assert_eq!(first.sample, again.sample);
}

#[test]
fn the_jackknife_of_a_constant_fit_is_the_classical_standard_error() {
    let values = [2.0, 4.0, 9.0, 1.0, 5.0, 3.0];
    let problem = problem(
        |p: &HashMap<String, f64>, _| p["m"],
        &[("m", -20.0, 20.0)],
        &values,
    );
    let config = config();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let runner = ReplicateRunner::new(&problem, &config, &[mean], 1, 0).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 1).unwrap();

    let jackknife = block_jackknife(&runner, &scheme).unwrap();

    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let classical = (variance / n).sqrt();
    let se = jackknife.statistics.standard_error[0];
    assert!(
        (se - classical).abs() / classical < 1e-6,
        "{se} {classical}"
    );
    assert_eq!(jackknife.blocks, [0, 1, 2, 3, 4, 5]);
    // Leaving out 9, the largest value, moves the mean the most.
    let influence: Vec<f64> = jackknife
        .statistics
        .influence
        .iter()
        .map(|r| r[0])
        .collect();
    let largest = influence
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .unwrap()
        .0;
    assert_eq!(largest, 2);
}

#[test]
fn a_jackknife_needs_two_blocks() {
    let problem = line_problem(&[1.0, 2.0, 3.0]);
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[1.0, 1.0], 1, 0).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 3).unwrap();
    let error = block_jackknife(&runner, &scheme).unwrap_err().to_string();
    assert!(error.contains("block_length 3 makes 1 block"), "{error}");
}

#[test]
fn two_blocks_are_enough_and_fits_that_did_not_converge_are_counted() {
    let problem = line_problem(&[1.0, 2.0, 3.0, 4.0]);
    let scheme = BlockScheme::new(problem.observed_data(), 2).unwrap();
    let count = |config: &OptimizationConfig| {
        let runner = ReplicateRunner::new(&problem, config, &[1.0, 1.0], 1, 0).unwrap();
        block_jackknife(&runner, &scheme).unwrap().not_converged
    };
    let one_iteration = OptimizationConfig::NelderMead(NelderMeadConfig {
        max_iterations: 1,
        ..NelderMeadConfig::default()
    });

    assert_eq!(count(&config()), 0);
    assert_eq!(count(&one_iteration), 2);
}

fn draws_of(problem: &CalibrationProblem<ClosedForm>, values: Vec<f64>) -> ParameterSample {
    ParameterSample::new(
        problem.parameter_names(),
        problem.parameter_bounds(),
        vec![values],
        &[0.0],
        None,
    )
    .unwrap()
}

#[test]
fn residual_noise_adds_residuals_of_the_fit_to_held_out_positions() {
    let values = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    let problem = line_problem(&values);
    let fit = [1.0, 1.0];
    let residuals = Residuals::new(&problem, &fit, ResidualScale::Additive).unwrap();
    let noise = ResidualNoise::new(problem.observed_data(), &residuals, 2).unwrap();
    let held_out: Vec<ObservedDataPoint> = (6..10)
        .map(|t| ObservedDataPoint::new(t as f64, "x".to_string(), 0.0))
        .collect();
    let sample = draws_of(&problem, fit.to_vec());

    let plain = observation_draws(problem.forward(), &sample, &held_out, &NoNoise, 3).unwrap();
    let noisy = observation_draws(problem.forward(), &sample, &held_out, &noise, 3).unwrap();

    for (shown, predicted) in noisy.values[0].iter().zip(&plain.values[0]) {
        let added = shown - predicted;
        assert!(
            residuals
                .residuals()
                .iter()
                .any(|r| (r - added).abs() < 1e-12),
            "{added}"
        );
    }
}

#[test]
fn residual_noise_rejects_a_series_with_no_residuals() {
    let problem = line_problem(&[1.0, 2.0, 3.0]);
    let residuals = Residuals::new(&problem, &[1.0, 1.0], ResidualScale::Additive).unwrap();
    let noise = ResidualNoise::new(problem.observed_data(), &residuals, 1).unwrap();
    let other = [ObservedDataPoint::new(1.0, "y".to_string(), 0.0)];
    let error = noise.check(&other).unwrap_err();
    assert!(error.contains("series 'y'"), "{error}");
}

/// The problem of fitting a line to the series `x` at positions 0 to 5,
/// with a second observation of `x` at 3.
fn shared_position_problem() -> CalibrationProblem<ClosedForm> {
    let mut observations: Vec<ObservedDataPoint> = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]
        .iter()
        .enumerate()
        .map(|(t, &value)| ObservedDataPoint::new(t as f64, "x".to_string(), value))
        .collect();
    observations.push(ObservedDataPoint::new(3.0, "x".to_string(), 9.0));
    line_problem_of(observations)
}

#[test]
fn readers_of_a_series_by_position_refuse_two_observations_at_one_position() {
    let problem = shared_position_problem();
    let observed = problem.observed_data();
    let fit = [1.0, 1.0];
    let residuals = Residuals::new(&problem, &fit, ResidualScale::Additive).unwrap();
    let named = |error: String| {
        assert!(
            error.contains("the series 'x' has two observations at 3"),
            "{error}"
        );
    };

    named(
        ResidualNoise::new(observed, &residuals, 1)
            .unwrap_err()
            .to_string(),
    );
    let scheme = BlockScheme::new(observed, 1).unwrap();
    named(
        ResidualBootstrap::new(observed, residuals.clone(), &scheme, ResidualBlocks::Moving)
            .unwrap_err()
            .to_string(),
    );
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &fit, 1, 4).unwrap();
    let settings =
        ConformalSettings::new(NonconformityScore::Absolute, Pooling::Series, 0.5).unwrap();
    named(
        rolling_conformal(&runner, &[2.0], &[6.0], settings)
            .unwrap_err()
            .to_string(),
    );

    // A block of a given length holds both observations, and the wild
    // bootstrap multiplies each by the sign of its block: each keeps its own
    // residual.
    let given = BlockScheme::new(observed, 2).unwrap();
    assert_eq!(given.blocks()[1], [2, 3, 6]);
    let wild = WildBootstrap::new(residuals.clone(), &given, Multipliers::Rademacher);
    let Perturbation::Values(values) = wild.replicate(&mut Generator::from_seed(2)) else {
        panic!("a wild bootstrap draws values");
    };
    let fitted = residuals.fitted();
    let own = |i: usize| (values[i] - fitted[i]) / residuals.residuals()[i];
    assert_eq!(own(3).abs(), 1.0);
    assert_eq!(own(6), own(3));
}

#[test]
fn residual_noise_refuses_two_held_out_observations_of_a_series_at_one_position() {
    let problem = line_problem(&[1.0, 2.0, 3.0]);
    let residuals = Residuals::new(&problem, &[1.0, 1.0], ResidualScale::Additive).unwrap();
    let noise = ResidualNoise::new(problem.observed_data(), &residuals, 1).unwrap();
    let held_out = [
        ObservedDataPoint::new(4.0, "x".to_string(), 0.0),
        ObservedDataPoint::new(5.0, "x".to_string(), 0.0),
        ObservedDataPoint::new(4.0, "x".to_string(), 0.0),
    ];

    let error = noise.check(&held_out).unwrap_err();

    assert!(
        error.contains("the series 'x' has two observations at 4"),
        "{error}"
    );
    noise.check(&held_out[..2]).unwrap();
}

#[test]
fn residuals_of_a_stochastic_free_fit_are_observed_minus_fitted() {
    let problem = line_problem(&[1.0, 3.0, 4.0]);
    let residuals = Residuals::new(&problem, &[1.0, 1.0], ResidualScale::Additive).unwrap();
    assert_eq!(residuals.fitted(), [1.0, 2.0, 3.0]);
    assert_eq!(residuals.residuals(), [0.0, 1.0, 1.0]);
    let mut workspace = Workspace::default();
    let copy = problem
        .with_data(&residuals.values(residuals.residuals()), &[1.0; 3])
        .unwrap();
    assert_eq!(
        copy.evaluate(&[1.0, 1.0], &mut workspace).unwrap(),
        problem.evaluate(&[1.0, 1.0], &mut workspace).unwrap()
    );
}

#[test]
fn a_runner_takes_a_start_and_one_fit_value_per_parameter() {
    let problem = line_problem(&[1.0, 2.0, 3.0]);
    let config = config();

    let starts = ReplicateRunner::new(&problem, &config, &[1.0, 1.0], 0, 0)
        .err()
        .map(|error| error.to_string());
    let fit = ReplicateRunner::new(&problem, &config, &[1.0], 1, 0)
        .err()
        .map(|error| error.to_string());

    assert_eq!(
        starts.as_deref(),
        Some("starts is a whole number of at least 1, got 0")
    );
    assert_eq!(
        fit.as_deref(),
        Some("The fit has 1 values for 2 parameters")
    );
}

#[test]
fn a_jackknife_whose_fits_all_fail_says_every_block_failed() {
    let problem = line_problem(&[1.0, 2.0, 3.0, 4.0]);
    let scheme = BlockScheme::new(problem.observed_data(), 2).unwrap();
    let broken = OptimizationConfig::NelderMead(NelderMeadConfig {
        initial_simplex_size: 0.0,
        ..NelderMeadConfig::default()
    });
    let runner = ReplicateRunner::new(&problem, &broken, &[1.0, 1.0], 2, 0).unwrap();

    let error = block_jackknife(&runner, &scheme).unwrap_err().to_string();

    assert!(
        error.starts_with(
            "Every block failed: 0 of 2 blocks fitted, and the method needs 2; the first \
             reasons: block 0: "
        ),
        "{error}"
    );
}

/// `c` up to 22; above it, a run that is not a number.
fn capped(p: &HashMap<String, f64>, _: f64) -> f64 {
    if p["c"] > 22.0 { f64::NAN } else { p["c"] }
}

#[test]
fn a_jackknife_with_one_fitted_block_says_how_many_fitted() {
    // The fit is the mean, 20. Leaving out 100 moves it down to 0; leaving
    // out a 0 moves it up to 25, past 22, where the run fails.
    let problem = problem(capped, &[("c", -50.0, 50.0)], &[0.0, 0.0, 0.0, 0.0, 100.0]);
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[20.0], 1, 0).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 1).unwrap();

    let error = block_jackknife(&runner, &scheme).unwrap_err().to_string();

    assert!(
        error.starts_with(
            "Only 1 of 5 blocks fitted, and the method needs 2; the first reasons: block 0: "
        ),
        "{error}"
    );
}

#[test]
fn every_replicate_method_counts_the_same_fits() {
    let values = [1.0, 2.2, 2.9, 4.1, 5.0, 5.8, 7.2, 8.0];
    let problem = line_problem(&values);
    let scheme = BlockScheme::new(problem.observed_data(), 2).unwrap();
    let resampler = BayesianBootstrap::new(&problem, &[1.0, 1.0], &scheme).unwrap();
    let settings =
        ConformalSettings::new(NonconformityScore::Absolute, Pooling::Series, 0.5).unwrap();
    let held = [ProfiledQuantity::Parameter(0)];
    // The number of fits each method kept, and how many of them did not
    // converge.
    let counts = |config: &OptimizationConfig| -> Vec<(usize, usize)> {
        let runner = ReplicateRunner::new(&problem, config, &[1.0, 1.0], 1, 4).unwrap();
        let sample = bootstrap(&runner, &resampler, 3).unwrap();
        let jackknife = block_jackknife(&runner, &scheme).unwrap();
        let plus = jackknife_plus(&runner, &scheme, &[8.0], settings).unwrap();
        let rolling = rolling_conformal(&runner, &[3.0, 5.0], &[8.0], settings).unwrap();
        let ensemble = enbpi(&runner, &scheme, 12, &[8.0], settings).unwrap();
        let thresholds = bootstrap_thresholds(&runner, &resampler, 3, &held, &[1.0], 0.5).unwrap();
        vec![
            (sample.sample.len(), sample.not_converged),
            (jackknife.blocks.len(), jackknife.not_converged),
            (plus.blocks, plus.run.not_converged),
            (2 - rolling.failed.len(), rolling.not_converged),
            (12 - ensemble.failed.len(), ensemble.not_converged),
            (thresholds.replicates.len(), thresholds.not_converged),
        ]
    };
    let one_iteration = OptimizationConfig::NelderMead(NelderMeadConfig {
        max_iterations: 1,
        ..NelderMeadConfig::default()
    });

    for (fits, not_converged) in counts(&config()) {
        assert!(fits > 0);
        assert_eq!(not_converged, 0);
    }
    for (fits, not_converged) in counts(&one_iteration) {
        assert!(fits > 0);
        assert_eq!(not_converged, fits);
    }
}
