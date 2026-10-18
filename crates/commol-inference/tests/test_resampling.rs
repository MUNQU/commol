//! Blocks, resamplers, the replicate runner, the jackknife and residual
//! noise, on engines whose output is a closed form of the parameters.

use std::collections::HashMap;

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, LossConfig, NelderMeadConfig, ObservedDataPoint,
    OptimizationConfig, Workspace,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};
use commol_inference::resampling::{
    BayesianBootstrap, BlockDrawing, BlockScheme, Multipliers, Perturbation, ReplicateRunner,
    Resampler, ResidualBlocks, ResidualBootstrap, ResidualNoise, ResidualScale, Residuals,
    WildBootstrap, automatic_block_length, automatic_length, block_jackknife, bootstrap,
};
use commol_inference::{NoNoise, ObservationNoise, ParameterSample, observation_draws};
use commol_random::Generator;

type Formula = fn(&HashMap<String, f64>, f64) -> f64;

/// One output `x = formula(parameters, t)` at every position `t`.
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
        buffer.extend(
            positions
                .iter()
                .map(|&t| vec![(self.formula)(&self.parameters, t)]),
        );
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
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

fn line_problem(values: &[f64]) -> CalibrationProblem<ClosedForm> {
    problem(line, &[("a", -50.0, 50.0), ("b", -10.0, 10.0)], values)
}

fn config() -> OptimizationConfig {
    OptimizationConfig::NelderMead(NelderMeadConfig {
        sd_tolerance: 1e-12,
        max_iterations: 2000,
        ..NelderMeadConfig::default()
    })
}

/// `count` standard normal draws (Box–Muller).
fn normals(generator: &mut Generator, count: usize) -> Vec<f64> {
    (0..count)
        .map(|_| {
            let (u, v) = (generator.uniform_positive(), generator.uniform());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        })
        .collect()
}

/// An AR(1) series with coefficient `phi` and unit innovations.
fn ar1(phi: f64, count: usize, seed: u64) -> Vec<f64> {
    let mut previous = 0.0;
    normals(&mut Generator::from_seed(seed), count)
        .into_iter()
        .map(|e| {
            previous = phi * previous + e;
            previous
        })
        .collect()
}

fn residual_observations(series: &[f64]) -> Vec<ObservedDataPoint> {
    series
        .iter()
        .enumerate()
        .map(|(t, _)| ObservedDataPoint::new(t as f64, "x".to_string(), 0.0))
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

    assert_eq!(scheme.len(), 4);
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
fn automatic_length_matches_the_reference_implementation() {
    // `arch` 8.0.0, `optimal_block_length` of this series: stationary
    // 2.8995633204762306, circular 3.3191714301343724. Its bandwidth rule
    // starts one lag earlier than Politis and White (2004); on this series
    // both rules choose the same bandwidth.
    let series = ar1(0.0, 400, 2);
    let stationary = automatic_length(&series, BlockDrawing::Stationary);
    let moving = automatic_length(&series, BlockDrawing::Moving);
    assert!(
        (stationary - 2.899_563_320_476_230_6).abs() < 1e-9,
        "{stationary}"
    );
    assert!((moving - 3.319_171_430_134_372_4).abs() < 1e-9, "{moving}");
}

#[test]
fn automatic_length_of_a_strong_autocorrelation_matches_the_reference_implementation() {
    // `arch` 8.0.0, `optimal_block_length` of this series: stationary
    // 19.37076455883211, circular 22.173990079642422.
    let series = ar1(0.95, 300, 11);
    let stationary = automatic_length(&series, BlockDrawing::Stationary);
    let moving = automatic_length(&series, BlockDrawing::Moving);
    let non_overlapping = automatic_length(&series, BlockDrawing::NonOverlapping);
    assert!(
        (stationary - 19.370_764_558_832_11).abs() < 1e-9,
        "{stationary}"
    );
    assert!((moving - 22.173_990_079_642_422).abs() < 1e-9, "{moving}");
    assert_eq!(non_overlapping, stationary);
}

#[test]
fn automatic_length_of_a_short_series_matches_the_reference_implementation() {
    // `arch` 8.0.0, `optimal_block_length` of 20 values: stationary
    // 1.6740553467716044, circular 1.9163149982720122.
    let series = ar1(0.5, 20, 0);
    let stationary = automatic_length(&series, BlockDrawing::Stationary);
    let moving = automatic_length(&series, BlockDrawing::Moving);
    assert!(
        (stationary - 1.674_055_346_771_604_4).abs() < 1e-9,
        "{stationary}"
    );
    assert!((moving - 1.916_314_998_272_012_2).abs() < 1e-9, "{moving}");
    // Here the estimate passes a third of the size; `arch` rounds that cap up,
    // to 7.
    let capped = automatic_length(&ar1(0.3, 20, 1), BlockDrawing::Stationary);
    assert!((capped - 20.0 / 3.0).abs() < 1e-12, "{capped}");
}

#[test]
fn automatic_length_is_at_most_three_times_the_root_of_the_size() {
    // `arch` rounds the cap up, to 43.
    let series = ar1(0.8, 200, 5);
    let length = automatic_length(&series, BlockDrawing::Stationary);
    assert!((length - 3.0 * 200f64.sqrt()).abs() < 1e-12, "{length}");
    // Below 81 values a third of the size is the smaller cap; four values
    // are enough for an estimate.
    assert_eq!(automatic_length(&ar1(0.5, 6, 5), BlockDrawing::Moving), 2.0);
    assert_eq!(
        automatic_length(&ar1(0.5, 4, 3), BlockDrawing::Moving),
        4.0 / 3.0
    );
}

#[test]
fn the_median_length_of_white_noise_is_one() {
    let mut lengths: Vec<usize> = (0..51)
        .map(|seed| {
            let series = ar1(0.0, 400, seed);
            automatic_block_length(
                &residual_observations(&series),
                &series,
                BlockDrawing::Moving,
            )
            .length
        })
        .collect();
    lengths.sort_unstable();
    assert_eq!(lengths[25], 1, "{lengths:?}");
}

#[test]
fn automatic_length_grows_with_autocorrelation() {
    let weak = ar1(0.3, 400, 7);
    let strong = ar1(0.8, 400, 7);
    let mut observations = residual_observations(&weak);
    observations.extend(
        strong
            .iter()
            .enumerate()
            .map(|(t, _)| ObservedDataPoint::new(t as f64, "y".to_string(), 0.0)),
    );
    let residuals: Vec<f64> = weak.iter().chain(&strong).copied().collect();

    let chosen = automatic_block_length(&observations, &residuals, BlockDrawing::Stationary);

    assert_eq!(chosen.estimates.len(), 2);
    let (x, y) = (&chosen.estimates[0], &chosen.estimates[1]);
    assert_eq!((x.0.as_str(), y.0.as_str()), ("x", "y"));
    assert!(y.1 > 2.0 * x.1, "{chosen:?}");
    assert_eq!(chosen.length, y.1.round() as usize);
}

#[test]
fn bayesian_weights_have_mean_one_and_are_shared_by_a_block() {
    let problem = line_problem(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let scheme = BlockScheme::new(problem.observed_data(), 2).unwrap();
    let resampler = BayesianBootstrap::new(problem.observed_data(), &scheme);
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
        &BayesianBootstrap::new(problem.observed_data(), &scheme),
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
    );

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
            ResidualBootstrap::new(problem.observed_data(), residuals.clone(), &scheme, blocks);
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
    let error = Residuals::new(&problem, &[1.0, 0.0], ResidualScale::Multiplicative).unwrap_err();
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
    .unwrap_err();
    assert!(error.contains("Every replicate failed (3)"), "{error}");
    assert!(error.contains("replicate 0"), "{error}");
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
fn one_seed_gives_one_bootstrap() {
    let values = [1.0, 2.5, 2.8, 4.1, 5.2, 5.9];
    let problem = line_problem(&values);
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[1.0, 1.0], 2, 21).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 2).unwrap();
    let resampler = BayesianBootstrap::new(problem.observed_data(), &scheme);
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
    let error = block_jackknife(&runner, &scheme).unwrap_err();
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

    let plain = observation_draws(&problem, &sample, &held_out, &NoNoise, 3).unwrap();
    let noisy = observation_draws(&problem, &sample, &held_out, &noise, 3).unwrap();

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
