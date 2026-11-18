//! The likelihood of observation models: their log densities and tails, the
//! meaning of a weight, the pointwise values, the noise parameters and the
//! penalty of the objective.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use commol_calibration::{
    Argument, CalibrationParameter, CalibrationParameterType, CalibrationProblem, Center,
    DiscrepancyConfig, LossConfig, Noise, ObservationModel, ObservedDataPoint,
    OptimizationObjective, ReplicateSummary, Replicates, Tails, Violation, Workspace,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};

/// One output `x = a + b·t` at every position `t`; it records the id of
/// every parameter it is given.
#[derive(Clone, Debug)]
struct LineEngine {
    parameters: HashMap<String, f64>,
    received: Arc<Mutex<Vec<String>>>,
}

impl LineEngine {
    fn new() -> Self {
        Self {
            parameters: HashMap::from([("a".to_string(), 0.0), ("b".to_string(), 0.0)]),
            received: Arc::default(),
        }
    }
}

#[derive(Clone, Debug)]
struct NoPlan;

impl ValuePlan for NoPlan {
    fn infeasibility(&self, excess: f64) -> String {
        format!("{excess}")
    }
}

impl SimulationEngine for LineEngine {
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
        let (a, b) = (self.parameters["a"], self.parameters["b"]);
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
        self.received.lock().unwrap().push(parameter_id.to_string());
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

fn line_parameters() -> Vec<CalibrationParameter> {
    vec![
        CalibrationParameter::new("a".to_string(), -10.0, 10.0),
        CalibrationParameter::new("b".to_string(), -10.0, 10.0),
    ]
}

fn noise(id: &str, min: f64, max: f64) -> CalibrationParameter {
    CalibrationParameter::with_type(id.to_string(), CalibrationParameterType::Noise, min, max)
}

fn models(model: ObservationModel) -> DiscrepancyConfig {
    DiscrepancyConfig::Likelihood(vec![("x".to_string(), model)])
}

fn normal(sd: Argument) -> DiscrepancyConfig {
    models(ObservationModel::Normal { sd })
}

fn problem(
    data: Vec<ObservedDataPoint>,
    parameters: Vec<CalibrationParameter>,
    config: DiscrepancyConfig,
) -> Result<CalibrationProblem<LineEngine>, String> {
    CalibrationProblem::new(LineEngine::new(), data, parameters, None, vec![], config)
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0),
        "{actual} differs from {expected}"
    );
}

#[test]
fn log_densities_equal_the_reference() {
    // Computed once with SciPy 1.16 (`scipy.stats`), which is not a
    // dependency of commol.
    let cases = [
        (Noise::Normal { sd: 1.5 }, 3.2, 2.0, -1.6444036413128371),
        (
            Noise::LogNormal {
                sigma: 0.4,
                center: Center::Mean,
            },
            5.0,
            4.0,
            -1.899260753462715,
        ),
        (
            Noise::LogNormal {
                sigma: 0.4,
                center: Center::Median,
            },
            5.0,
            4.0,
            -1.7676889778056095,
        ),
        (
            Noise::StudentT {
                df: 3.5,
                scale: 2.0,
            },
            -1.0,
            1.5,
            -2.5130900656783486,
        ),
        (Noise::Poisson, 6.0, 3.7, -2.429254294109028),
        (
            Noise::NegativeBinomial {
                overdispersion: 0.25,
                sum: None,
            },
            7.0,
            12.5,
            -2.824194492552487,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 0.01,
                sum: None,
            },
            1500.0,
            1800.0,
            -7.440755782369635,
        ),
        (Noise::Binomial { trials: 20 }, 9.0, 6.0, -2.727698487461354),
    ];
    for (noise, y, mean, expected) in cases {
        close(noise.log_density(y, mean), expected);
    }
}

#[test]
fn a_weight_of_two_is_the_observation_twice() {
    let mut weighted = observed(&[1.0, 3.0, 2.0]);
    weighted[1].weight = 2.0;
    let mut twice = observed(&[1.0, 3.0, 2.0]);
    twice.push(twice[1].clone());
    let sd = || normal(Argument::Value(0.7));
    let weighted = problem(weighted, line_parameters(), sd()).unwrap();
    let twice = problem(twice, line_parameters(), sd()).unwrap();
    let mut workspace = Workspace::default();
    for point in [[0.0, 1.0], [1.5, -0.5]] {
        close(
            weighted.evaluate(&point, &mut workspace).unwrap(),
            twice.evaluate(&point, &mut workspace).unwrap(),
        );
    }
}

#[test]
fn pointwise_values_sum_to_the_likelihood() {
    let mut data = observed(&[1.0, 3.0, 2.0, 5.0]);
    data[3].weight = 0.5;
    let mut parameters = line_parameters();
    parameters.push(noise("s", 0.1, 5.0));
    let problem = problem(
        data,
        parameters,
        models(ObservationModel::StudentT {
            df: Argument::Value(4.0),
            scale: Argument::Parameter("s".to_string()),
        }),
    )
    .unwrap();
    let mut workspace = Workspace::default();
    let point = [0.5, 1.0, 1.3];
    let pointwise = problem
        .pointwise_log_likelihood(&point, &mut workspace)
        .unwrap();
    assert_eq!(pointwise.len(), 4);
    close(
        pointwise.iter().sum::<f64>(),
        -problem.evaluate(&point, &mut workspace).unwrap(),
    );
}

#[test]
fn noise_parameters_never_reach_the_engine() {
    let mut parameters = line_parameters();
    parameters.push(noise("sd", 0.1, 5.0));
    let problem = problem(
        observed(&[1.0, 3.0, 2.0]),
        parameters,
        normal(Argument::Parameter("sd".to_string())),
    )
    .unwrap();
    let mut workspace = Workspace::default();
    let received = problem.forward().base_engine().received.clone();

    problem.evaluate(&[1.0, 0.5, 2.0], &mut workspace).unwrap();

    let received = received.lock().unwrap();
    assert!(received.iter().all(|id| id == "a" || id == "b"));
    assert!(!received.is_empty());
}

#[test]
fn the_penalty_is_above_every_feasible_cost() {
    // Counts in thousands with a normal model of a small sd: a poor fit costs
    // more than the penalty of a loss.
    let data = observed(&[3000.0, 3500.0, 4000.0]);
    let mut parameters = line_parameters();
    parameters.push(noise("sd", 0.01, 100.0));
    let problem = problem(
        data,
        parameters,
        normal(Argument::Parameter("sd".to_string())),
    )
    .unwrap();
    let objective = OptimizationObjective::new(&problem);

    let poor = objective.score(&[0.0, 0.0, 0.01]).unwrap();

    assert!(poor.feasible && poor.cost > 1e10, "{}", poor.cost);
    assert!(objective.penalty(&Violation::NonFinite) > poor.cost);
}

#[test]
fn a_mean_outside_the_model_is_a_violation_with_its_distance() {
    let problem = problem(
        observed(&[1.0, 3.0]),
        line_parameters(),
        models(ObservationModel::Poisson),
    )
    .unwrap();
    let mut workspace = Workspace::default();

    // Means -1 and -3: distances 1 and 3.
    let error = problem.evaluate(&[-1.0, -2.0], &mut workspace).unwrap_err();

    assert_eq!(error, Violation::MeanOutsideModel { amount: 4.0 }.into());
}

#[test]
fn observed_values_outside_a_model_name_the_observation() {
    let cases = [
        (
            ObservationModel::Poisson,
            1.5,
            None,
            "has the value 1.5; its Poisson model needs a whole number",
        ),
        (
            ObservationModel::Binomial { trials: Some(4) },
            5.0,
            None,
            "its Binomial model needs a value of at most its 4 trials",
        ),
        (
            ObservationModel::LogNormal {
                sigma: Argument::Value(1.0),
                center: Center::Median,
            },
            0.0,
            None,
            "its LogNormal model needs a value above 0",
        ),
        (
            ObservationModel::Binomial { trials: Some(4) },
            1.0,
            Some(4),
            "give the trials in one place",
        ),
        (
            ObservationModel::Binomial { trials: None },
            1.0,
            None,
            "needs trials",
        ),
        (
            ObservationModel::Poisson,
            1.0,
            Some(3),
            "only a Binomial model reads trials",
        ),
    ];
    for (model, value, trials, message) in cases {
        let mut data = observed(&[2.0, value]);
        data[1].trials = trials;
        data[0].trials = match model {
            ObservationModel::Binomial { trials: None } => Some(4),
            _ => None,
        };
        let error = problem(data, line_parameters(), models(model)).unwrap_err();
        assert!(error.contains("Observation 'x' at 1"), "{error}");
        assert!(error.contains(message), "{error}");
    }
}

#[test]
fn trials_of_each_observation_feed_a_binomial_model() {
    let mut data = observed(&[2.0, 3.0]);
    data[0].trials = Some(4);
    data[1].trials = Some(10);
    let problem = problem(
        data,
        line_parameters(),
        models(ObservationModel::Binomial { trials: None }),
    )
    .unwrap();
    let mut workspace = Workspace::default();

    // Means 2 and 5.
    let cost = problem.evaluate(&[2.0, 3.0], &mut workspace).unwrap();

    let expected = Noise::Binomial { trials: 4 }.log_density(2.0, 2.0)
        + Noise::Binomial { trials: 10 }.log_density(3.0, 5.0);
    close(cost, -expected);
}

#[test]
fn models_and_their_arguments_are_checked() {
    let data = || observed(&[1.0, 2.0]);
    let with_noise = |kind| {
        let mut parameters = line_parameters();
        parameters.push(CalibrationParameter::with_type(
            "k".to_string(),
            kind,
            0.0,
            1.0,
        ));
        parameters
    };
    let k = || Argument::Parameter("k".to_string());
    let cases: Vec<(Vec<CalibrationParameter>, DiscrepancyConfig, &str)> = vec![
        (
            line_parameters(),
            DiscrepancyConfig::Likelihood(vec![
                ("x".to_string(), ObservationModel::Poisson),
                ("y".to_string(), ObservationModel::Poisson),
            ]),
            "names a series that no observation has",
        ),
        (
            line_parameters(),
            DiscrepancyConfig::Likelihood(vec![
                ("x".to_string(), ObservationModel::Poisson),
                ("x".to_string(), ObservationModel::Poisson),
            ]),
            "has two observation models",
        ),
        (
            line_parameters(),
            normal(Argument::Value(0.0)),
            "The observation model of 'x': sd is a finite number greater than 0, got 0",
        ),
        (
            line_parameters(),
            normal(k()),
            "names 'k' as sd, which is not a calibration parameter",
        ),
        (
            with_noise(CalibrationParameterType::Parameter),
            normal(k()),
            "a Parameter parameter; a noise argument must be a Noise parameter",
        ),
        (
            with_noise(CalibrationParameterType::Noise),
            normal(k()),
            "its bounds must be a finite number greater than 0; they are [0, 1]",
        ),
        (
            with_noise(CalibrationParameterType::Noise),
            models(ObservationModel::Poisson),
            "Noise parameter 'k' is named by no observation model",
        ),
        (
            with_noise(CalibrationParameterType::Noise),
            LossConfig::SumSquaredError.into(),
            "Noise parameter 'k' is named by no observation model",
        ),
        (
            line_parameters(),
            models(ObservationModel::NegativeBinomial {
                overdispersion: Argument::Value(-1.0),
                sum: None,
            }),
            "overdispersion is a finite number of at least 0, got -1",
        ),
    ];
    for (parameters, config, message) in cases {
        let error = problem(data(), parameters, config).unwrap_err();
        assert!(error.contains(message), "{error}");
    }

    let mut series = observed(&[1.0]);
    series.push(ObservedDataPoint::new(0.0, "y".to_string(), 1.0));
    series[1].outputs = vec!["x".to_string()];
    let error = problem(series, line_parameters(), models(ObservationModel::Poisson)).unwrap_err();
    assert!(
        error.contains("the series 'y' has no observation model"),
        "{error}"
    );
}

#[test]
fn a_likelihood_needs_one_deterministic_run_and_no_normalization() {
    let poisson = || {
        problem(
            observed(&[1.0, 2.0]),
            line_parameters(),
            models(ObservationModel::Poisson),
        )
        .unwrap()
    };
    let replicates = Replicates::new(2, 1, ReplicateSummary::Mean).unwrap();
    let error = poisson().with_replicates(Some(replicates)).unwrap_err();
    assert!(
        error.contains("A stochastic engine needs a loss"),
        "{error}"
    );
    let error = poisson().with_observation_normalization(true).unwrap_err();
    assert!(
        error.contains("normalize_observations needs a loss"),
        "{error}"
    );
    assert!(poisson().with_observation_normalization(false).is_ok());
}

#[test]
fn new_data_keep_the_models_and_their_checks() {
    let problem = problem(
        observed(&[1.0, 2.0]),
        line_parameters(),
        models(ObservationModel::Poisson),
    )
    .unwrap();
    let mut workspace = Workspace::default();
    let copy = problem.with_data(&[3.0, 0.0], &[1.0, 1.0]).unwrap();
    let cost = copy.evaluate(&[1.0, 1.0], &mut workspace).unwrap();
    close(
        cost,
        -(Noise::Poisson.log_density(3.0, 1.0) + Noise::Poisson.log_density(0.0, 2.0)),
    );
    let error = problem.with_data(&[0.5, 0.0], &[1.0, 1.0]).unwrap_err();
    assert!(error.contains("needs a whole number"), "{error}");
}

#[test]
fn a_loss_has_no_pointwise_log_likelihood() {
    let problem = problem(
        observed(&[1.0]),
        line_parameters(),
        LossConfig::SumSquaredError.into(),
    )
    .unwrap();
    let error = problem
        .pointwise_log_likelihood(&[0.0, 0.0], &mut Workspace::default())
        .unwrap_err();
    assert!(error.to_string().contains("needs observation models"));
}

#[test]
fn an_observation_with_trials_names_the_model_of_its_series() {
    let cases = [
        (normal(Argument::Value(1.0)), "Normal"),
        (
            models(ObservationModel::LogNormal {
                sigma: Argument::Value(1.0),
                center: Center::Mean,
            }),
            "LogNormal",
        ),
        (
            models(ObservationModel::StudentT {
                df: Argument::Value(3.0),
                scale: Argument::Value(1.0),
            }),
            "StudentT",
        ),
        (
            models(ObservationModel::NegativeBinomial {
                overdispersion: Argument::Value(0.5),
                sum: None,
            }),
            "NegativeBinomial",
        ),
    ];
    for (config, name) in cases {
        let mut data = observed(&[1.0]);
        data[0].trials = Some(2);
        let error = problem(data, line_parameters(), config).unwrap_err();
        assert!(
            error.contains(&format!("its series has a {name} model")),
            "{error}"
        );
    }
}

#[test]
fn a_binomial_mean_below_zero_is_a_violation() {
    let problem = problem(
        observed(&[1.0]),
        line_parameters(),
        models(ObservationModel::Binomial { trials: Some(3) }),
    )
    .unwrap();
    let error = problem
        .evaluate(&[-2.0, 0.0], &mut Workspace::default())
        .unwrap_err();
    assert_eq!(error, Violation::MeanOutsideModel { amount: 2.0 }.into());
}

#[test]
fn an_observation_of_weight_zero_has_no_term() {
    let problem = problem(
        observed(&[1.0, 2.0]),
        line_parameters(),
        models(ObservationModel::Poisson),
    )
    .unwrap()
    .with_data(&[1.0, 2.0], &[1.0, 0.0])
    .unwrap();
    // The mean of the second observation, -1, would be outside the model.
    let pointwise = problem
        .pointwise_log_likelihood(&[1.0, -2.0], &mut Workspace::default())
        .unwrap();
    assert_eq!(pointwise[1], 0.0);
    close(pointwise[0], Noise::Poisson.log_density(1.0, 1.0));
}

#[test]
fn the_penalty_of_every_model_is_finite() {
    let df = || noise("df", 1.0, 30.0);
    let mut binomial = observed(&[0.0, 4.0, 0.0]);
    for (obs, trials) in binomial.iter_mut().zip([4, 4, 0]) {
        obs.trials = Some(trials);
    }
    let cases = [
        (
            binomial,
            line_parameters(),
            models(ObservationModel::Binomial { trials: None }),
        ),
        (
            observed(&[1.0, 5.0]),
            [line_parameters(), vec![df()]].concat(),
            models(ObservationModel::StudentT {
                df: Argument::Parameter("df".to_string()),
                scale: Argument::Value(0.5),
            }),
        ),
    ];
    for (data, parameters, config) in cases {
        let base = problem(data, parameters, config)
            .unwrap()
            .discrepancy()
            .penalty_base();
        assert!(base.is_finite() && base >= 1e10, "{base}");
    }
}

/// The derivatives of `log p(y)` with respect to the arguments `(μ, noise
/// arguments…)`, by central differences of [`Noise::log_density`].
fn differenced_scores(noise_of: &impl Fn(&[f64]) -> Noise, arguments: &[f64], y: f64) -> Vec<f64> {
    (0..arguments.len())
        .map(|k| {
            let h = 1e-5 * arguments[k].abs().max(1.0);
            let mut up = arguments.to_vec();
            let mut down = arguments.to_vec();
            up[k] += h;
            down[k] -= h;
            (noise_of(&up).log_density(y, up[0]) - noise_of(&down).log_density(y, down[0]))
                / (2.0 * h)
        })
        .collect()
}

/// `E[s sᵀ]` of a count model, row-major: the sum over the counts of the
/// probability times the outer product of the differenced scores.
fn count_information(
    noise_of: &impl Fn(&[f64]) -> Noise,
    arguments: &[f64],
    last: u64,
) -> Vec<f64> {
    let k = arguments.len();
    let mut information = vec![0.0; k * k];
    for y in 0..=last {
        let y = y as f64;
        let probability = noise_of(arguments).log_density(y, arguments[0]).exp();
        let scores = differenced_scores(noise_of, arguments, y);
        for a in 0..k {
            for b in 0..k {
                information[a * k + b] += probability * scores[a] * scores[b];
            }
        }
    }
    information
}

fn close_to(actual: f64, expected: f64, relative: f64) {
    assert!(
        (actual - expected).abs() <= relative * expected.abs().max(1e-12),
        "{actual} differs from {expected}"
    );
}

#[test]
fn fisher_information_of_each_model_has_its_closed_form() {
    let mean = 3.5;
    let poisson = Noise::Poisson.information(mean);
    assert_eq!(poisson.len(), 1);
    close(poisson[0], 1.0 / mean);

    // The dispersion has no closed form: its oracle is the variance of the
    // differenced score over the counts.
    let phi = 0.4;
    let negative_binomial = |a: &[f64]| Noise::NegativeBinomial {
        overdispersion: a[1],
        sum: None,
    };
    let information = negative_binomial(&[mean, phi]).information(mean);
    let expected = count_information(&negative_binomial, &[mean, phi], 400);
    close(information[0], 1.0 / (mean + phi * mean * mean));
    close_to(information[0], expected[0], 1e-7);
    assert_eq!((information[1], information[2]), (0.0, 0.0));
    assert!(expected[1].abs() < 1e-7, "{}", expected[1]);
    close_to(information[3], expected[3], 1e-7);

    // Of the mean of a binomial, not of its probability: n/(p(1−p)) / n².
    let binomial = Noise::Binomial { trials: 12 }.information(3.0);
    let p = 3.0 / 12.0;
    close(binomial[0], 1.0 / (12.0 * p * (1.0 - p)));
}

/// `∫ p(y) s(y) s(y)ᵀ dy` of a continuous model, row-major, on `y = c + w·tan x`.
fn continuous_information(
    noise_of: &impl Fn(&[f64]) -> Noise,
    arguments: &[f64],
    center: f64,
    width: f64,
    support_above_zero: bool,
) -> Vec<f64> {
    let k = arguments.len();
    let mut information = vec![0.0; k * k];
    let steps = 200_000;
    let half = std::f64::consts::FRAC_PI_2;
    let (low, high) = if support_above_zero {
        ((-center / width).atan(), half)
    } else {
        (-half, half)
    };
    let dx = (high - low) / steps as f64;
    for step in 0..steps {
        let x = low + (step as f64 + 0.5) * dx;
        let y = center + width * x.tan();
        let jacobian = width / (x.cos() * x.cos());
        let density = noise_of(arguments).log_density(y, arguments[0]).exp();
        if density == 0.0 || !density.is_finite() {
            continue;
        }
        let scores = differenced_scores(noise_of, arguments, y);
        for a in 0..k {
            for b in 0..k {
                information[a * k + b] += density * jacobian * dx * scores[a] * scores[b];
            }
        }
    }
    information
}

#[test]
fn fisher_information_of_continuous_models_is_the_variance_of_their_scores() {
    let normal = |a: &[f64]| Noise::Normal { sd: a[1] };
    let information = normal(&[2.0, 1.5]).information(2.0);
    assert_eq!(information, vec![1.0 / 2.25, 0.0, 0.0, 2.0 / 2.25]);

    let student = |a: &[f64]| Noise::StudentT {
        df: a[1],
        scale: a[2],
    };
    let arguments = [1.0, 4.5, 2.0];
    let information = student(&arguments).information(1.0);
    let expected = continuous_information(&student, &arguments, 1.0, 2.0, false);
    for (actual, expected) in information.iter().zip(&expected) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "{information:?} {expected:?}"
        );
    }

    for center in [Center::Mean, Center::Median] {
        let log_normal = |a: &[f64]| Noise::LogNormal {
            sigma: a[1],
            center,
        };
        let arguments = [4.0, 0.3];
        let information = log_normal(&arguments).information(4.0);
        let expected = continuous_information(&log_normal, &arguments, 4.0, 1.0, true);
        for (actual, expected) in information.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "{information:?} {expected:?}"
            );
        }
    }
}

/// The derivative of the discrepancy of one run with respect to the
/// prediction `i`, or to the parameter value `j`, by central differences.
fn differenced(f: impl Fn(&[f64]) -> f64, at: &[f64], index: usize) -> f64 {
    let h = 1e-6 * at[index].abs().max(1.0);
    let mut up = at.to_vec();
    let mut down = at.to_vec();
    up[index] += h;
    down[index] -= h;
    (f(&up) - f(&down)) / (2.0 * h)
}

/// Each term's gradient with respect to its mean equals the difference of
/// the discrepancy in that prediction, and the gradients of the noise
/// arguments sum to the difference in their parameter.
fn assert_scores(problem: &CalibrationProblem<LineEngine>, values: &[f64], predictions: &[f64]) {
    let discrepancy = problem.discrepancy();
    // Every observation has one part, whose mean is its prediction.
    let terms = problem
        .term_derivatives(values, predictions, predictions)
        .unwrap();
    for (i, term) in terms.iter().enumerate() {
        let expected = differenced(
            |p| discrepancy.of_run(values, p, p).unwrap(),
            predictions,
            i,
        );
        close_to(term.gradient[0], expected, 1e-6);
    }
    for j in 0..values.len() {
        let summed: f64 = terms
            .iter()
            .flat_map(|term| {
                term.parameters
                    .iter()
                    .zip(&term.gradient[1..])
                    .filter(move |(parameter, _)| **parameter == Some(j))
                    .map(|(_, gradient)| *gradient)
            })
            .sum();
        let expected = differenced(
            |v| discrepancy.of_run(v, predictions, predictions).unwrap(),
            values,
            j,
        );
        assert!(
            (summed - expected).abs() <= 1e-6 * expected.abs().max(1e-3),
            "parameter {j}: {summed} against {expected}"
        );
    }
}

#[test]
fn scores_equal_the_differences_of_the_terms() {
    let data = observed(&[3.0, 4.0, 1.0, 6.0]);
    let weights = [1.0, 2.0, 0.5, 1.0];
    let values = [3.0, 4.0, 1.0, 6.0];
    let predictions = [2.5, 4.4, 1.5, 5.0];
    for loss in [
        LossConfig::SumSquaredError,
        LossConfig::RootMeanSquaredError,
    ] {
        let problem = problem(data.clone(), line_parameters(), loss.into())
            .unwrap()
            .with_observation_normalization(true)
            .unwrap()
            .with_data(&values, &weights)
            .unwrap();
        assert_scores(&problem, &[0.0, 0.0], &predictions);
    }

    let parameter = |id: &str| Argument::Parameter(id.to_string());
    let cases = [
        (
            ObservationModel::Normal { sd: parameter("s") },
            vec![noise("s", 0.1, 5.0)],
            vec![0.7],
        ),
        (
            ObservationModel::LogNormal {
                sigma: parameter("s"),
                center: Center::Mean,
            },
            vec![noise("s", 0.1, 5.0)],
            vec![0.4],
        ),
        (
            ObservationModel::LogNormal {
                sigma: parameter("s"),
                center: Center::Median,
            },
            vec![noise("s", 0.1, 5.0)],
            vec![0.4],
        ),
        (
            ObservationModel::StudentT {
                df: parameter("df"),
                scale: parameter("s"),
            },
            vec![noise("df", 1.0, 30.0), noise("s", 0.1, 5.0)],
            vec![3.5, 0.8],
        ),
        (ObservationModel::Poisson, vec![], vec![]),
        (
            ObservationModel::NegativeBinomial {
                overdispersion: parameter("s"),
                sum: None,
            },
            vec![noise("s", 0.0, 5.0)],
            vec![0.3],
        ),
        (
            ObservationModel::Binomial { trials: Some(8) },
            vec![],
            vec![],
        ),
    ];
    for (model, noise_parameters, noise_values) in cases {
        let problem = problem(
            data.clone(),
            [line_parameters(), noise_parameters].concat(),
            models(model),
        )
        .unwrap()
        .with_data(&values, &weights)
        .unwrap();
        let values = [vec![0.0, 0.0], noise_values].concat();
        assert_scores(&problem, &values, &predictions);
    }
}

#[test]
fn terms_skip_a_weight_of_zero_and_name_a_mean_outside_their_model() {
    let problem = problem(
        observed(&[1.0, 2.0]),
        line_parameters(),
        models(ObservationModel::Poisson),
    )
    .unwrap()
    .with_data(&[1.0, 2.0], &[1.0, 0.0])
    .unwrap();
    // The second mean, −1, is outside the model, but its weight is 0.
    let terms = problem
        .term_derivatives(&[0.0, 0.0], &[1.5, -1.0], &[1.5, -1.0])
        .unwrap();
    assert_eq!(
        (terms[1].gradient.clone(), terms[1].curvature.clone()),
        (vec![0.0], vec![0.0])
    );

    let error = problem
        .term_derivatives(&[0.0, 0.0], &[-1.0, 2.0], &[-1.0, 2.0])
        .unwrap_err();
    assert_eq!(
        error,
        "Observation 'x' at 0 has a mean of a part outside the means its observation model \
         allows (its prediction is -1)"
    );

    let rmse = self::problem(
        observed(&[1.0, 2.0]),
        line_parameters(),
        LossConfig::RootMeanSquaredError.into(),
    )
    .unwrap();
    let error = rmse
        .term_derivatives(&[0.0, 0.0], &[1.0, 2.0], &[1.0, 2.0])
        .unwrap_err();
    assert!(error.contains("no derivative at a perfect fit"), "{error}");
}

/// `actual` within `tolerance` of `expected`, relative; exactly 0 where
/// `expected` is 0.
fn within(actual: f64, expected: f64, tolerance: f64) -> bool {
    if expected == 0.0 {
        actual == 0.0
    } else {
        (actual - expected).abs() <= tolerance * expected.abs()
    }
}

fn assert_tails(tails: Tails, expected: [f64; 3], tolerance: f64, case: &str) {
    let actual = [tails.below, tails.at, tails.above];
    for (side, (a, e)) in ["below", "at", "above"]
        .iter()
        .zip(actual.iter().zip(expected))
    {
        assert!(
            within(*a, e, tolerance),
            "{case}, {side}: {a:e} against {e:e}"
        );
    }
}

#[test]
fn the_tails_of_each_model_match_reference_values() {
    // `P(Y < y)`, `p(y)` and `P(Y > y)`, computed once with mpmath at 80
    // digits (the counts by summing their probabilities), which agrees with
    // SciPy 1.18 (`scipy.stats`) within 2e-11; neither is a dependency of
    // commol. The tolerance is 1e-10 where an argument of the incomplete
    // gamma or beta function is at least 10⁴, and 1e-12 elsewhere.
    let cases: [(Noise, f64, f64, [f64; 3], f64); 76] = [
        (
            Noise::Normal { sd: 2.0 },
            1.0,
            -20.0,
            [4.3190063178092304e-26, 0.0, 1.0],
            1e-12,
        ),
        (
            Noise::Normal { sd: 2.0 },
            1.0,
            -3.0,
            [0.02275013194817921, 0.0, 0.9772498680518208],
            1e-12,
        ),
        (
            Noise::Normal { sd: 2.0 },
            1.0,
            0.5,
            [0.4012936743170763, 0.0, 0.5987063256829237],
            1e-12,
        ),
        (Noise::Normal { sd: 2.0 }, 1.0, 1.0, [0.5, 0.0, 0.5], 1e-12),
        (
            Noise::Normal { sd: 2.0 },
            1.0,
            4.0,
            [0.9331927987311419, 0.0, 0.06680720126885807],
            1e-12,
        ),
        (
            Noise::Normal { sd: 2.0 },
            1.0,
            30.0,
            [1.0, 0.0, 6.057494764415221e-48],
            1e-12,
        ),
        (
            Noise::LogNormal {
                sigma: 0.5,
                center: Center::Median,
            },
            3.0,
            0.1,
            [5.144708852200758e-12, 0.0, 0.9999999999948553],
            1e-12,
        ),
        (
            Noise::LogNormal {
                sigma: 0.5,
                center: Center::Median,
            },
            3.0,
            2.0,
            [0.20870287338447127, 0.0, 0.7912971266155288],
            1e-12,
        ),
        (
            Noise::LogNormal {
                sigma: 0.5,
                center: Center::Median,
            },
            3.0,
            3.0,
            [0.49999999999999994, 0.0, 0.5000000000000001],
            1e-12,
        ),
        (
            Noise::LogNormal {
                sigma: 0.5,
                center: Center::Median,
            },
            3.0,
            10.0,
            [0.9919787145440149, 0.0, 0.008021285455985122],
            1e-12,
        ),
        (
            Noise::LogNormal {
                sigma: 0.5,
                center: Center::Median,
            },
            3.0,
            60.0,
            [0.9999999989602016, 0.0, 1.0397983214543171e-09],
            1e-12,
        ),
        (
            Noise::LogNormal {
                sigma: 0.5,
                center: Center::Mean,
            },
            3.0,
            0.1,
            [2.8310809929348707e-11, 0.0, 0.9999999999716892],
            1e-12,
        ),
        (
            Noise::LogNormal {
                sigma: 0.5,
                center: Center::Mean,
            },
            3.0,
            2.0,
            [0.28742255518746956, 0.0, 0.7125774448125304],
            1e-12,
        ),
        (
            Noise::LogNormal {
                sigma: 0.5,
                center: Center::Mean,
            },
            3.0,
            3.0,
            [0.5987063256829237, 0.0, 0.40129367431707635],
            1e-12,
        ),
        (
            Noise::LogNormal {
                sigma: 0.5,
                center: Center::Mean,
            },
            3.0,
            10.0,
            [0.9960690709909918, 0.0, 0.003930929009008187],
            1e-12,
        ),
        (
            Noise::LogNormal {
                sigma: 0.5,
                center: Center::Mean,
            },
            3.0,
            60.0,
            [0.9999999997832536, 0.0, 2.1674634974062028e-10],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 2.5,
                scale: 1.5,
            },
            0.0,
            -50.0,
            [0.00011188919592181435, 0.0, 0.9998881108040781],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 2.5,
                scale: 1.5,
            },
            0.0,
            -1.0,
            [0.2806398085244487, 0.0, 0.7193601914755513],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 2.5,
                scale: 1.5,
            },
            0.0,
            0.0,
            [0.5, 0.0, 0.5],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 2.5,
                scale: 1.5,
            },
            0.0,
            0.3,
            [0.5716951579854872, 0.0, 0.42830484201451274],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 2.5,
                scale: 1.5,
            },
            0.0,
            4.0,
            [0.9539216036033992, 0.0, 0.04607839639660085],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 2.5,
                scale: 1.5,
            },
            0.0,
            1000.0,
            [0.9999999373155509, 0.0, 6.26844491205593e-08],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 300.0,
                scale: 1.5,
            },
            0.0,
            -12.0,
            [1.3675076950769211e-14, 0.0, 0.9999999999999863],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 300.0,
                scale: 1.5,
            },
            0.0,
            -1.0,
            [0.25274872810610227, 0.0, 0.7472512718938977],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 300.0,
                scale: 1.5,
            },
            0.0,
            0.0,
            [0.5, 0.0, 0.5],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 300.0,
                scale: 1.5,
            },
            0.0,
            2.0,
            [0.9082830108828986, 0.0, 0.09171698911710131],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 300.0,
                scale: 1.5,
            },
            0.0,
            9.0,
            [0.9999999971617805, 0.0, 2.8382195461795766e-09],
            1e-12,
        ),
        (
            Noise::StudentT {
                df: 30000.0,
                scale: 1.5,
            },
            0.0,
            -10.0,
            [1.3310619244694363e-11, 0.0, 0.9999999999866894],
            1e-10,
        ),
        (
            Noise::StudentT {
                df: 30000.0,
                scale: 1.5,
            },
            0.0,
            -1.0,
            [0.2524951010029407, 0.0, 0.7475048989970593],
            1e-10,
        ),
        (
            Noise::StudentT {
                df: 30000.0,
                scale: 1.5,
            },
            0.0,
            0.5,
            [0.6305574950604846, 0.0, 0.36944250493951536],
            1e-10,
        ),
        (
            Noise::StudentT {
                df: 30000.0,
                scale: 1.5,
            },
            0.0,
            3.0,
            [0.9772453687333819, 0.0, 0.022754631266618107],
            1e-10,
        ),
        (
            Noise::StudentT {
                df: 30000.0,
                scale: 1.5,
            },
            0.0,
            12.0,
            [0.9999999999999993, 0.0, 6.443470672712701e-16],
            1e-10,
        ),
        (
            Noise::Poisson,
            0.5,
            0.0,
            [0.0, 0.6065306597126334, 0.3934693402873666],
            1e-12,
        ),
        (
            Noise::Poisson,
            0.5,
            1.0,
            [0.6065306597126334, 0.3032653298563167, 0.09020401043104986],
            1e-12,
        ),
        (
            Noise::Poisson,
            0.5,
            3.0,
            [
                0.9856123220330293,
                0.012636055410679864,
                0.0017516225562908237,
            ],
            1e-12,
        ),
        (
            Noise::Poisson,
            0.5,
            15.0,
            [1.0, 1.415478290768798e-17, 4.557180167512403e-19],
            1e-12,
        ),
        (
            Noise::Poisson,
            7.3,
            0.0,
            [0.0, 0.0006755387751938444, 0.9993244612248061],
            1e-12,
        ),
        (
            Noise::Poisson,
            7.3,
            3.0,
            [0.02360670249914889, 0.04379934461826396, 0.9325939528825872],
            1e-12,
        ),
        (
            Noise::Poisson,
            7.3,
            7.0,
            [0.4060322851591354, 0.148074326679942, 0.4458933881609226],
            1e-12,
        ),
        (
            Noise::Poisson,
            7.3,
            12.0,
            [0.9319188378164291, 0.03229889312845009, 0.03578226905512077],
            1e-12,
        ),
        (
            Noise::Poisson,
            7.3,
            40.0,
            [1.0, 2.82435718144656e-17, 6.080520772269776e-18],
            1e-12,
        ),
        (
            Noise::Poisson,
            12000.0,
            11500.0,
            [
                2.1153759378382907e-06,
                9.604671775171454e-08,
                0.9999977885773444,
            ],
            1e-10,
        ),
        (
            Noise::Poisson,
            12000.0,
            12000.0,
            [
                0.49878605673743615,
                0.0036418028115884865,
                0.49757214045097536,
            ],
            1e-10,
        ),
        (
            Noise::Poisson,
            12000.0,
            12500.0,
            [
                0.9999970536644868,
                1.2305693965010473e-07,
                2.8232785736286355e-06,
            ],
            1e-10,
        ),
        (
            Noise::Poisson,
            1e6,
            1e6,
            [
                0.4998670192391274,
                0.00039894224715624404,
                0.49973403851371634,
            ],
            1e-10,
        ),
        (
            Noise::Poisson,
            1e8,
            1e8,
            [
                0.49998670192398587,
                3.989422800689808e-05,
                0.4999734038480072,
            ],
            1e-10,
        ),
        (
            Noise::Poisson,
            1e8,
            1.0003e8,
            [
                0.9986492893409539,
                4.4331779627623997e-07,
                0.0013502673412498889,
            ],
            1e-10,
        ),
        (
            // Just above the overdispersion that reads the Poisson: the
            // continued fraction of a size 1/φ = 5e7 keeps about 1e-10.
            Noise::NegativeBinomial {
                overdispersion: 2e-8,
                sum: None,
            },
            50.0,
            30.0,
            [
                0.0009168331278127365,
                0.00067720096278685,
                0.9984059659094005,
            ],
            1e-9,
        ),
        (
            // Just above the overdispersion that reads the Poisson: the
            // continued fraction of a size 1/φ = 5e7 keeps about 1e-10.
            Noise::NegativeBinomial {
                overdispersion: 2e-8,
                sum: None,
            },
            50.0,
            50.0,
            [
                0.48119171269044353,
                0.05632497816270869,
                0.46248330914684777,
            ],
            1e-9,
        ),
        (
            // Just above the overdispersion that reads the Poisson: the
            // continued fraction of a size 1/φ = 5e7 keeps about 1e-10.
            Noise::NegativeBinomial {
                overdispersion: 2e-8,
                sum: None,
            },
            50.0,
            80.0,
            [
                0.999943349127538,
                2.229215686500456e-05,
                3.435871559702552e-05,
            ],
            1e-9,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 0.4,
                sum: None,
            },
            6.0,
            0.0,
            [0.0, 0.04691402634486509, 0.9530859736551349],
            1e-12,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 0.4,
                sum: None,
            },
            6.0,
            2.0,
            [0.12970348460050937, 0.1022693307863841, 0.7680271846131065],
            1e-12,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 0.4,
                sum: None,
            },
            6.0,
            6.0,
            [0.5418033004857533, 0.08509852553240649, 0.3730981739818402],
            1e-12,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 0.4,
                sum: None,
            },
            6.0,
            20.0,
            [
                0.9869152211325204,
                0.0032604404028230708,
                0.009824338464656518,
            ],
            1e-12,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 0.4,
                sum: None,
            },
            6.0,
            80.0,
            [
                0.9999999999272868,
                2.0463616868659733e-11,
                5.2249585217809263e-11,
            ],
            1e-12,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 3.0,
                sum: None,
            },
            0.7,
            0.0,
            [0.0, 0.6858241418122343, 0.3141758581877657],
            1e-12,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 3.0,
                sum: None,
            },
            0.7,
            1.0,
            [0.6858241418122343, 0.15486351589308517, 0.15931234229468055],
            1e-12,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 3.0,
                sum: None,
            },
            0.7,
            5.0,
            [
                0.9682772570894236,
                0.012212782907660308,
                0.01950996000291612,
            ],
            1e-12,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 3.0,
                sum: None,
            },
            0.7,
            200.0,
            [1.0, 1.1104751916371348e-36, 2.3084248806510554e-36],
            1e-12,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 1e-06,
                sum: None,
            },
            50.0,
            30.0,
            [
                0.000917042194911576,
                0.0006773237447348764,
                0.9984056340603535,
            ],
            1e-10,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 1e-06,
                sum: None,
            },
            50.0,
            50.0,
            [0.4811930926122808, 0.05632359825260052, 0.46248330913511865],
            1e-10,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 1e-06,
                sum: None,
            },
            50.0,
            80.0,
            [
                0.9999433237816304,
                2.2301115086997338e-05,
                3.4375103282554935e-05,
            ],
            1e-10,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 0.01,
                sum: None,
            },
            20000.0,
            15000.0,
            [
                0.003426500736122175,
                6.244536921770753e-06,
                0.9965672547269561,
            ],
            1e-10,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 0.01,
                sum: None,
            },
            20000.0,
            20000.0,
            [
                0.5131994344789724,
                0.00019880857819716492,
                0.4866017569428304,
            ],
            1e-10,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 0.01,
                sum: None,
            },
            20000.0,
            26000.0,
            [
                0.9971941499608203,
                3.6118033675646626e-06,
                0.0028022382358120624,
            ],
            1e-10,
        ),
        (
            Noise::Binomial { trials: 10 },
            3.0,
            0.0,
            [0.0, 0.0282475249, 0.9717524751],
            1e-12,
        ),
        (
            Noise::Binomial { trials: 10 },
            3.0,
            1.0,
            [0.0282475249, 0.121060821, 0.8506916541],
            1e-12,
        ),
        (
            Noise::Binomial { trials: 10 },
            3.0,
            3.0,
            [0.3827827864, 0.266827932, 0.3503892816],
            1e-12,
        ),
        (
            Noise::Binomial { trials: 10 },
            3.0,
            9.0,
            [0.9998563141, 0.000137781, 5.9049e-06],
            1e-12,
        ),
        (
            Noise::Binomial { trials: 10 },
            3.0,
            10.0,
            [0.9999940951, 5.9049e-06, 0.0],
            1e-12,
        ),
        (
            Noise::Binomial { trials: 5000 },
            50.0,
            20.0,
            [
                4.34145110603901e-07,
                6.918626393231277e-07,
                0.9999988739922501,
            ],
            1e-12,
        ),
        (
            Noise::Binomial { trials: 5000 },
            50.0,
            50.0,
            [
                0.48090841319493793,
                0.056608751771437434,
                0.46248283503362464,
            ],
            1e-12,
        ),
        (
            Noise::Binomial { trials: 5000 },
            50.0,
            100.0,
            [
                0.9999999997515169,
                1.2782565822688696e-10,
                1.206574684523916e-10,
            ],
            1e-12,
        ),
        (
            Noise::Binomial { trials: 40000 },
            20000.0,
            19800.0,
            [
                0.0224808492025122,
                0.0005399152894001629,
                0.9769792355080876,
            ],
            1e-10,
        ),
        (
            Noise::Binomial { trials: 40000 },
            20000.0,
            20000.0,
            [
                0.49800530106490015,
                0.003989397870199723,
                0.49800530106490015,
            ],
            1e-10,
        ),
        (
            Noise::Binomial { trials: 40000 },
            20000.0,
            20300.0,
            [
                0.9986279995146161,
                4.4315714105398564e-05,
                0.0013276847712784473,
            ],
            1e-10,
        ),
    ];
    for (noise, mean, y, expected, tolerance) in cases {
        assert_tails(
            noise.tails(y, mean),
            expected,
            tolerance,
            &format!("{noise:?} at {mean}, y = {y}"),
        );
    }
}

#[test]
fn the_tails_at_the_edges_of_each_model() {
    let negative_binomial = |overdispersion| Noise::NegativeBinomial {
        overdispersion,
        sum: None,
    };
    // No overdispersion, or one too small for the incomplete beta function:
    // the Poisson.
    for phi in [0.0, 1e-9, 1e-8] {
        for (mean, y) in [(0.5, 0.0), (7.3, 3.0), (7.3, 40.0), (12000.0, 12500.0)] {
            let poisson = Noise::Poisson.tails(y, mean);
            let expected = [poisson.below, poisson.at, poisson.above];
            let case = format!("φ = {phi}, μ = {mean}, y = {y}");
            assert_tails(
                negative_binomial(phi).tails(y, mean),
                expected,
                1e-12,
                &case,
            );
        }
    }
    // A mass at one count: no mean, no trials, or a probability of 1.
    let point = |y: f64, at: f64| -> [f64; 3] {
        [
            f64::from(u8::from(at < y)),
            f64::from(u8::from(at == y)),
            f64::from(u8::from(at > y)),
        ]
    };
    let cases = [
        (Noise::Poisson, 0.0, 0.0, 0.0),
        (Noise::Poisson, 0.0, 2.0, 0.0),
        (negative_binomial(0.5), 0.0, 0.0, 0.0),
        (negative_binomial(0.5), 0.0, 3.0, 0.0),
        (Noise::Binomial { trials: 0 }, 0.0, 0.0, 0.0),
        (Noise::Binomial { trials: 5 }, 0.0, 1.0, 0.0),
        (Noise::Binomial { trials: 5 }, 5.0, 5.0, 5.0),
        (Noise::Binomial { trials: 5 }, 5.0, 2.0, 5.0),
        // A mean so small that φ μ or μ/n underflows.
        (negative_binomial(0.1), 5e-324, 0.0, 0.0),
        (negative_binomial(0.1), 5e-324, 1.0, 0.0),
        (Noise::Binomial { trials: 10 }, 5e-324, 0.0, 0.0),
        (Noise::Binomial { trials: 10 }, 5e-324, 1.0, 0.0),
    ];
    for (noise, mean, y, mass) in cases {
        let tails = noise.tails(y, mean);
        assert_eq!(
            [tails.below, tails.at, tails.above],
            point(y, mass),
            "{noise:?} at {mean}, y = {y}"
        );
    }
}

#[test]
fn the_tails_read_the_nearest_mean_and_refuse_values_outside_the_support() {
    // x = a: a mean of −2 reads the Poisson of mean 0, a mass at 0; a
    // binomial mean of 7 of 4 trials reads 4.
    let data = observed(&[0.0, 3.0]);
    let poisson = problem(
        data.clone(),
        line_parameters(),
        models(ObservationModel::Poisson),
    )
    .unwrap();
    let likelihood = poisson.discrepancy().likelihood().unwrap();
    let tails = likelihood
        .tails(&data, &[-2.0, 0.0], &[-2.0, -2.0])
        .unwrap();
    assert_eq!(
        (tails[0].below, tails[0].at, tails[0].above),
        (0.0, 1.0, 0.0)
    );
    assert_eq!(
        (tails[1].below, tails[1].at, tails[1].above),
        (1.0, 0.0, 0.0)
    );
    let binomial = problem(
        data.clone(),
        line_parameters(),
        models(ObservationModel::Binomial { trials: Some(4) }),
    )
    .unwrap();
    let likelihood = binomial.discrepancy().likelihood().unwrap();
    let tails = likelihood.tails(&data, &[7.0, 0.0], &[7.0, 7.0]).unwrap();
    assert_eq!(
        (tails[1].below, tails[1].at, tails[1].above),
        (0.0, 0.0, 1.0)
    );

    // A held-out value outside the support has the message of the problem.
    let cases = [
        (
            ObservationModel::Poisson,
            1.5,
            "its Poisson model needs a whole number, at least 0",
        ),
        (
            ObservationModel::Binomial { trials: Some(4) },
            5.0,
            "its Binomial model needs a value of at most its 4 trials",
        ),
        (
            ObservationModel::LogNormal {
                sigma: Argument::Value(1.0),
                center: Center::Median,
            },
            -1.0,
            "its LogNormal model needs a value above 0",
        ),
    ];
    for (model, value, message) in cases {
        let problem = problem(observed(&[2.0, 1.0]), line_parameters(), models(model)).unwrap();
        let held_out = observed(&[2.0, value]);
        let error = problem
            .discrepancy()
            .likelihood()
            .unwrap()
            .tails(&held_out, &[1.0, 0.0], &[1.0, 1.0])
            .unwrap_err();
        assert!(
            error.contains("Observation 'x' at 1 has the value"),
            "{error}"
        );
        assert!(error.contains(message), "{error}");
    }
}
