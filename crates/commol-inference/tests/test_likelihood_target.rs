//! The posterior of observation models on the SMC core, their noise, the
//! parametric bootstrap and the log score, on an engine whose output is a
//! line.

use std::collections::HashMap;

use commol_calibration::{
    Argument, CalibrationParameter, CalibrationParameterType, CalibrationProblem, Center,
    DiscrepancyConfig, LossConfig, ObservationModel, ObservedDataPoint, Workspace,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};
use commol_inference::resampling::{ParametricResampler, Perturbation, Resampler};
use commol_inference::scores::log_score;
use commol_inference::smc::bayesian_posterior;
use commol_inference::smc::{LikelihoodTarget, RandomWalk, SmcSampler, TemperedTarget};
use commol_inference::{DistributionNoise, Family, ObservationNoise, ParameterSample, Priors};
use commol_random::Generator;

/// One output `x = a + b·t` at every position `t`.
#[derive(Clone, Debug)]
struct Line {
    parameters: HashMap<String, f64>,
}

#[derive(Clone, Debug)]
struct NoPlan;

impl ValuePlan for NoPlan {
    fn infeasibility(&self, excess: f64) -> String {
        format!("{excess}")
    }
}

impl SimulationEngine for Line {
    type ValuePlan = NoPlan;

    fn check_position(&self, _position: f64) -> Result<(), String> {
        Ok(())
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        let a = self.parameters["a"];
        let b = self.parameters.get("b").copied().unwrap_or(0.0);
        buffer.clear();
        buffer.extend(positions.iter().map(|t| vec![a + b * t]));
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

fn observed(values: &[f64]) -> Vec<ObservedDataPoint> {
    values
        .iter()
        .enumerate()
        .map(|(t, &value)| ObservedDataPoint::new(t as f64, "x".to_string(), value))
        .collect()
}

/// A problem whose prediction is the constant `a`.
fn constant(
    values: &[f64],
    model: ObservationModel,
    noise: &[(&str, f64, f64)],
) -> CalibrationProblem<Line> {
    let mut parameters = vec![CalibrationParameter::new("a".to_string(), -20.0, 20.0)];
    parameters.extend(noise.iter().map(|&(id, min, max)| {
        CalibrationParameter::with_type(id.to_string(), CalibrationParameterType::Noise, min, max)
    }));
    CalibrationProblem::new(
        Line {
            parameters: HashMap::from([("a".to_string(), 0.0)]),
        },
        observed(values),
        parameters,
        vec![],
        DiscrepancyConfig::Likelihood(vec![("x".to_string(), model)]),
    )
    .unwrap()
}

fn loss_problem() -> CalibrationProblem<Line> {
    CalibrationProblem::new(
        Line {
            parameters: HashMap::from([("a".to_string(), 0.0)]),
        },
        observed(&[1.0]),
        vec![CalibrationParameter::new("a".to_string(), -1.0, 1.0)],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

fn sd(value: f64) -> ObservationModel {
    ObservationModel::Normal {
        sd: Argument::Value(value),
    }
}

#[test]
fn the_target_keeps_the_log_likelihood() {
    let problem = constant(&[1.0, 2.0], sd(1.0), &[]);
    let priors = Priors::new(
        &problem.parameter_names(),
        &problem.parameter_bounds(),
        &[("a".to_string(), Family::Uniform)],
    )
    .unwrap();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();
    let mut workspace = Workspace::default();
    let kept = target
        .evaluate(&[1.5], &mut Generator::from_seed(1), &mut workspace)
        .unwrap()
        .unwrap();

    // Two residuals of 0.5 with sd 1.
    let expected = -(2.0f64 * std::f64::consts::PI).ln() - 0.25;
    assert!((kept - expected).abs() < 1e-12, "{kept}");
    assert_eq!(target.log_kernel(kept, 0.5), 0.5 * kept);
    assert_eq!(
        target.evaluate(&[30.0], &mut Generator::from_seed(1), &mut workspace),
        Ok(None)
    );
}

#[test]
fn methods_of_observation_models_refuse_a_loss() {
    let problem = loss_problem();
    let priors = Priors::new(
        &problem.parameter_names(),
        &problem.parameter_bounds(),
        &[("a".to_string(), Family::Uniform)],
    )
    .unwrap();
    let error = LikelihoodTarget::new(&problem, &priors).err().unwrap();
    assert!(error.contains("needs observation models"), "{error}");
    let error = DistributionNoise::new(&problem).unwrap_err();
    assert!(error.contains("needs observation models"), "{error}");
    let error = ParametricResampler::new(&problem, &[0.0]).unwrap_err();
    assert!(error.contains("needs observation models"), "{error}");
}

#[test]
fn the_posterior_of_a_normal_mean_and_its_evidence_have_their_closed_forms() {
    // y ~ N(a, 1) with the prior a ~ N(0, 2²): a conjugate pair.
    let values = [1.3, 0.4, 2.1, 1.7, 0.9];
    let problem = constant(&values, sd(1.0), &[]);
    let priors = Priors::new(
        &problem.parameter_names(),
        &problem.parameter_bounds(),
        &[("a".to_string(), Family::Normal { mean: 0.0, sd: 2.0 })],
    )
    .unwrap();
    let mut sampler = SmcSampler::new(2000, RandomWalk::new(), 0.5, 0.5, 0.01).unwrap();

    let run = bayesian_posterior(&problem, &priors, &mut sampler, 7).unwrap();

    let n = values.len() as f64;
    let precision = n + 1.0 / 4.0;
    let mean = values.iter().sum::<f64>() / precision;
    let weights = run.weights();
    let sampled: f64 = run.values.iter().zip(&weights).map(|(v, w)| v[0] * w).sum();
    assert!((sampled - mean).abs() < 0.05, "{sampled} against {mean}");
    // The evidence: y ~ N(0, I + 4·11ᵀ).
    let sum: f64 = values.iter().sum();
    let squares: f64 = values.iter().map(|y| y * y).sum();
    let quadratic = squares - 4.0 * sum * sum / (1.0 + 4.0 * n);
    let log_evidence =
        -0.5 * n * (2.0 * std::f64::consts::PI).ln() - 0.5 * (1.0 + 4.0 * n).ln() - 0.5 * quadratic;
    assert!(
        (run.log_evidence - log_evidence).abs() < 0.05,
        "{} against {log_evidence}",
        run.log_evidence
    );
}

/// The mean and variance of `count` draws of `problem`'s noise of one
/// observation whose predicted mean is `mean`, at the parameter `values`.
fn moments(problem: &CalibrationProblem<Line>, values: &[f64], mean: f64) -> (f64, f64) {
    let noise = DistributionNoise::new(problem).unwrap();
    let observations = &problem.observed_data()[..1];
    noise.check(observations).unwrap();
    let mut generator = Generator::from_seed(11);
    let count = 40_000;
    let mut out = [0.0];
    let draws: Vec<f64> = (0..count)
        .map(|_| {
            noise.observe(observations, values, &[mean], &mut generator, &mut out);
            out[0]
        })
        .collect();
    let average = draws.iter().sum::<f64>() / count as f64;
    let variance = draws.iter().map(|x| (x - average).powi(2)).sum::<f64>() / count as f64;
    (average, variance)
}

#[test]
fn distribution_noise_draws_have_the_moments_of_each_model() {
    let k = || Argument::Parameter("k".to_string());
    let cases: Vec<(ObservationModel, f64, f64, f64, f64)> = vec![
        // (model, the noise parameter k, mean, expected mean, expected variance)
        (ObservationModel::Normal { sd: k() }, 1.5, 4.0, 4.0, 2.25),
        (
            ObservationModel::LogNormal {
                sigma: k(),
                center: Center::Mean,
            },
            0.3,
            5.0,
            5.0,
            (0.09f64.exp() - 1.0) * 25.0,
        ),
        (
            ObservationModel::StudentT {
                df: Argument::Value(6.0),
                scale: k(),
            },
            2.0,
            -1.0,
            -1.0,
            4.0 * 6.0 / 4.0,
        ),
        (
            ObservationModel::NegativeBinomial {
                overdispersion: k(),
            },
            0.2,
            8.0,
            8.0,
            8.0 + 0.2 * 64.0,
        ),
    ];
    for (model, k, mean, expected_mean, expected_variance) in cases {
        let problem = constant(&[mean.abs()], model.clone(), &[("k", 0.01, 10.0)]);
        let (average, variance) = moments(&problem, &[0.0, k], mean);
        let tolerance = 5.0 * (expected_variance / 40_000.0).sqrt();
        assert!(
            (average - expected_mean).abs() < tolerance,
            "{model:?}: mean {average} against {expected_mean}"
        );
        assert!(
            (variance / expected_variance - 1.0).abs() < 0.05,
            "{model:?}: variance {variance} against {expected_variance}"
        );
    }
    let counts: Vec<(ObservationModel, f64, f64)> = vec![
        (ObservationModel::Poisson, 3.5, 3.5),
        (
            ObservationModel::NegativeBinomial {
                overdispersion: Argument::Value(0.0),
            },
            3.5,
            3.5,
        ),
        (
            ObservationModel::Binomial { trials: Some(20) },
            6.0,
            6.0 * 0.7,
        ),
    ];
    for (model, mean, expected_variance) in counts {
        let problem = constant(&[1.0], model.clone(), &[]);
        let (average, variance) = moments(&problem, &[0.0], mean);
        assert!((average - mean).abs() < 0.05, "{model:?}: mean {average}");
        assert!(
            (variance / expected_variance - 1.0).abs() < 0.05,
            "{model:?}: variance {variance}"
        );
    }
    // The median of a log-normal of center "median" is its prediction.
    let problem = constant(
        &[1.0],
        ObservationModel::LogNormal {
            sigma: Argument::Value(0.8),
            center: Center::Median,
        },
        &[],
    );
    let (average, _) = moments(&problem, &[0.0], 2.0);
    let expected = 2.0 * (0.32f64).exp();
    assert!((average / expected - 1.0).abs() < 0.03, "{average}");
}

#[test]
fn a_binomial_of_no_trials_draws_zero() {
    let mut problem_data = observed(&[0.0]);
    problem_data[0].trials = Some(0);
    let problem = CalibrationProblem::new(
        Line {
            parameters: HashMap::from([("a".to_string(), 0.0)]),
        },
        problem_data,
        vec![CalibrationParameter::new("a".to_string(), -1.0, 1.0)],
        vec![],
        DiscrepancyConfig::Likelihood(vec![(
            "x".to_string(),
            ObservationModel::Binomial { trials: None },
        )]),
    )
    .unwrap();
    assert_eq!(moments(&problem, &[0.0], 0.0), (0.0, 0.0));
}

#[test]
fn a_mean_outside_a_model_draws_at_the_nearest_allowed_one() {
    let problem = constant(&[1.0], ObservationModel::Poisson, &[]);
    let (average, variance) = moments(&problem, &[0.0], -3.0);
    assert_eq!((average, variance), (0.0, 0.0));
}

#[test]
fn distribution_noise_needs_a_model_for_each_series() {
    let problem = constant(&[1.0], ObservationModel::Poisson, &[]);
    let noise = DistributionNoise::new(&problem).unwrap();
    let other = [ObservedDataPoint::of_sum(
        1.0,
        "y".to_string(),
        vec!["x".to_string()],
        1.0,
    )];
    let error = noise.check(&other).unwrap_err();
    assert!(
        error.contains("the series 'y' has no observation model"),
        "{error}"
    );
}

#[test]
fn parametric_replicates_draw_the_data_at_the_fit() {
    let problem = constant(&[2.0, 4.0, 3.0], ObservationModel::Poisson, &[]);
    let resampler = ParametricResampler::new(&problem, &[3.0]).unwrap();
    let mut generator = Generator::from_seed(3);
    let mut total = 0.0;
    let replicates = 4000;
    for _ in 0..replicates {
        let Perturbation::Values(values) = resampler.replicate(&mut generator) else {
            panic!("a parametric replicate draws values");
        };
        assert_eq!(values.len(), 3);
        assert!(values.iter().all(|v| v.fract() == 0.0 && *v >= 0.0));
        total += values.iter().sum::<f64>();
    }
    let mean = total / (3.0 * replicates as f64);
    assert!((mean - 3.0).abs() < 0.05, "{mean}");
    let error = ParametricResampler::new(&problem, &[-1.0]).unwrap_err();
    assert!(error.contains("The fit has no likelihood"), "{error}");
}

#[test]
fn the_log_score_of_one_draw_is_the_log_density() {
    let problem = constant(
        &[1.0, 3.0],
        ObservationModel::NegativeBinomial {
            overdispersion: Argument::Parameter("k".to_string()),
        },
        &[("k", 0.0, 2.0)],
    );
    let noise = DistributionNoise::new(&problem).unwrap();
    let sample = ParameterSample::new(
        problem.parameter_names(),
        problem.parameter_bounds(),
        vec![vec![2.5, 0.4]],
        &[0.0],
        None,
    )
    .unwrap();

    let scores = log_score(&problem, &sample, problem.observed_data(), &noise).unwrap();

    let likelihood = noise.likelihood();
    for ((obs, &term), score) in problem
        .observed_data()
        .iter()
        .zip(likelihood.problem_terms())
        .zip(scores)
    {
        let expected = likelihood
            .noise(term, &[2.5, 0.4])
            .log_density(obs.value, 2.5);
        assert!(
            (score - expected).abs() < 1e-12,
            "{score} against {expected}"
        );
    }
}

#[test]
fn the_log_score_of_two_draws_is_their_mixture() {
    let problem = constant(&[1.0], sd(1.0), &[]);
    let noise = DistributionNoise::new(&problem).unwrap();
    let sample = ParameterSample::new(
        problem.parameter_names(),
        problem.parameter_bounds(),
        vec![vec![0.0], vec![2.0]],
        &[0.25f64.ln(), 0.75f64.ln()],
        None,
    )
    .unwrap();

    let score = log_score(&problem, &sample, problem.observed_data(), &noise).unwrap()[0];

    // Both draws are one away from the observation.
    let density = (-0.5f64).exp() / (2.0 * std::f64::consts::PI).sqrt();
    assert!((score - density.ln()).abs() < 1e-12, "{score}");
}
