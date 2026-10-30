//! The likelihood of observation models: their log densities, the meaning
//! of a weight, the pointwise values, the noise parameters and the penalty
//! of the objective.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use commol_calibration::{
    Argument, CalibrationParameter, CalibrationParameterType, CalibrationProblem, Center,
    DiscrepancyConfig, LossConfig, Noise, ObservationModel, ObservedDataPoint,
    OptimizationObjective, ReplicateSummary, Replicates, Violation, Workspace,
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
    CalibrationProblem::new(LineEngine::new(), data, parameters, vec![], config)
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
            },
            7.0,
            12.5,
            -2.824194492552487,
        ),
        (
            Noise::NegativeBinomial {
                overdispersion: 0.01,
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
    let terms = problem.term_derivatives(values, predictions).unwrap();
    for (i, term) in terms.iter().enumerate() {
        let expected = differenced(|p| discrepancy.of_run(values, p).unwrap(), predictions, i);
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
        let expected = differenced(|v| discrepancy.of_run(v, predictions).unwrap(), values, j);
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
    let terms = problem.term_derivatives(&[0.0, 0.0], &[1.5, -1.0]).unwrap();
    assert_eq!(
        (terms[1].gradient.clone(), terms[1].curvature.clone()),
        (vec![0.0], vec![0.0])
    );

    let error = problem
        .term_derivatives(&[0.0, 0.0], &[-1.0, 2.0])
        .unwrap_err();
    assert_eq!(
        error,
        "Observation 'x' at 0 has the mean -1, outside the means its observation model allows"
    );

    let rmse = self::problem(
        observed(&[1.0, 2.0]),
        line_parameters(),
        LossConfig::RootMeanSquaredError.into(),
    )
    .unwrap();
    let error = rmse.term_derivatives(&[0.0, 0.0], &[1.0, 2.0]).unwrap_err();
    assert!(error.contains("no derivative at a perfect fit"), "{error}");
}
