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
            "has sd=0; it must be finite and above 0",
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
            "its bounds must be finite and above 0; they are [0, 1]",
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
            "it must be finite and at least 0",
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
