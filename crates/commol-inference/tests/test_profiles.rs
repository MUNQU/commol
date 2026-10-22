//! Profiles: subproblems that hold one quantity fixed, the tracer and the
//! thresholds, on engines whose output is a closed form of the parameters.

use std::collections::HashMap;

use commol_calibration::{
    Argument, CalibrationConstraint, CalibrationParameter, CalibrationProblem, ConstraintSettings,
    DiscrepancyConfig, LossConfig, NelderMeadConfig, ObservationModel, ObservedDataPoint,
    OptimizationConfig, ParticleSwarmConfig, Subproblem, Workspace, optimize, optimize_subproblem,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};
use commol_inference::profiles::{
    ProfileEnd, ProfileTracer, ProfiledQuantity, beale, bootstrap_thresholds, chi_square,
};
use commol_inference::resampling::{Perturbation, ReplicateRunner, Resampler};
use commol_random::{Generator, replicate_seed};

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
        // An engine that fails past z = 0.9, for the refits that fail.
        if self.parameters.get("z").is_some_and(|&z| z > 0.9) {
            return Err(SimulationError::Failed("z above 0.9".to_string()));
        }
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

/// Noisy values of `1 + 0.5 t` at positions 0, 1, …, 9.
const VALUES: [f64; 10] = [1.3, 1.2, 2.1, 2.4, 3.2, 3.4, 4.1, 4.4, 4.9, 5.6];

/// The problem of fitting the line `a + b t` to `values` at 0, 1, 2, ….
fn line_problem(values: &[f64]) -> CalibrationProblem<ClosedForm> {
    let parameters = [("a", -10.0, 10.0), ("b", -5.0, 5.0)];
    CalibrationProblem::new(
        ClosedForm {
            formula: line,
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

fn config() -> OptimizationConfig {
    OptimizationConfig::NelderMead(NelderMeadConfig {
        sd_tolerance: 1e-14,
        max_iterations: 4000,
        ..NelderMeadConfig::default()
    })
}

/// The least-squares line through `values` at 0, 1, 2, …: `(a, b)`.
fn least_squares(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean_t = (n - 1.0) / 2.0;
    let mean_y = values.iter().sum::<f64>() / n;
    let (mut sty, mut stt) = (0.0, 0.0);
    for (t, y) in values.iter().enumerate() {
        let dt = t as f64 - mean_t;
        sty += dt * (y - mean_y);
        stt += dt * dt;
    }
    let b = sty / stt;
    (mean_y - b * mean_t, b)
}

#[test]
fn fixing_a_parameter_at_the_fit_keeps_the_loss() {
    let problem = line_problem(&VALUES);
    let fit = optimize(&problem, &config()).unwrap();

    for index in 0..2 {
        let subproblem =
            Subproblem::fix_parameter(&problem, index, fit.best_parameters[index]).unwrap();
        let refit = optimize_subproblem(&subproblem, &config(), &fit.best_parameters, None)
            .unwrap()
            .fit;

        assert!(
            (refit.final_loss - fit.final_loss).abs() < 1e-9,
            "{} against {}",
            refit.final_loss,
            fit.final_loss
        );
        assert_eq!(refit.best_parameters[index], fit.best_parameters[index]);
        assert_eq!(refit.parameter_names, problem.parameter_names());
    }
}

#[test]
fn fixing_a_prediction_at_the_fit_keeps_the_parameters() {
    let problem = line_problem(&VALUES);
    let (a, b) = least_squares(&VALUES);
    // The value at 15, past the data, is held at its fitted value; the
    // refit starts away from the fit.
    let spec = ObservedDataPoint::new(15.0, "x".to_string(), 0.0);
    let subproblem = Subproblem::fix_prediction(&problem, &spec, a + 15.0 * b).unwrap();
    let refit = optimize_subproblem(&subproblem, &config(), &[a + 1.0, b - 0.1], None).unwrap();

    assert!(refit.fit.converged, "{}", refit.fit.termination_reason);
    assert!(
        (refit.fit.best_parameters[0] - a).abs() < 1e-5,
        "{:?}",
        refit.fit
    );
    assert!(
        (refit.fit.best_parameters[1] - b).abs() < 1e-5,
        "{:?}",
        refit.fit
    );
    let prediction = refit.prediction.expect("a prediction subproblem");
    assert!((prediction - (a + 15.0 * b)).abs() < 1e-6, "{prediction}");
    // At the fit the constraint does not bind: the slope of the profile is 0.
    let multiplier = refit.multiplier.expect("a prediction subproblem");
    assert!(multiplier.abs() < 1e-4, "{multiplier}");
}

#[test]
fn a_problem_observing_more_positions_scores_as_the_problem() {
    let base = line_problem(&VALUES);
    // x(2) = a + 2b must stay at least 2: a constraint at a position.
    let problem = CalibrationProblem::new(
        base.forward().base_engine().clone(),
        base.observed_data().to_vec(),
        base.parameters().to_vec(),
        vec![
            CalibrationConstraint::new("x2".to_string(), "x - 2".to_string())
                .with_positions(vec![2.0]),
        ],
        LossConfig::SumSquaredError,
    )
    .unwrap();
    let mut windowed = ObservedDataPoint::new(20.0, "x".to_string(), 0.0);
    windowed.window = Some(5.5);
    let (observing, operator) = problem.observing(&[windowed]).unwrap();

    assert_eq!(
        observing.forward().schedule().len(),
        problem.forward().schedule().len() + 2
    );
    for values in [[1.0, 0.5], [0.0, 0.5]] {
        let mut workspace = Workspace::default();
        let expected = problem.evaluate(&values, &mut workspace);
        assert_eq!(observing.evaluate(&values, &mut workspace), expected);
    }
    let mut workspace = Workspace::default();
    observing.evaluate(&[1.0, 0.5], &mut workspace).unwrap();
    let mut out = [0.0];
    operator.apply(&workspace.rows[0], &[1.0, 0.5], &mut out);
    // x(20) − x(14.5) = 0.5 · 5.5.
    assert!((out[0] - 2.75).abs() < 1e-12, "{}", out[0]);
}

#[test]
fn a_subproblem_refuses_what_it_cannot_hold() {
    let problem = line_problem(&VALUES);
    let spec = ObservedDataPoint::new(15.0, "x".to_string(), 0.0);
    let errors = [
        Subproblem::fix_parameter(&problem, 2, 0.0).map(|_| ()),
        Subproblem::fix_parameter(&problem, 1, 6.0).map(|_| ()),
        Subproblem::fix_prediction(&problem, &spec, f64::NAN).map(|_| ()),
        Subproblem::fix_prediction(
            &problem,
            &ObservedDataPoint::new(15.0, "y".to_string(), 0.0),
            1.0,
        )
        .map(|_| ()),
        Subproblem::fix_parameter(&problem, 0, 1.0)
            .unwrap()
            .with_settings(ConstraintSettings::for_value(1.0))
            .map(|_| ()),
    ]
    .map(Result::unwrap_err);

    assert_eq!(errors[0], "Parameter 2 does not exist; the problem has 2");
    assert_eq!(
        errors[1],
        "'b' cannot be held at 6, outside its bounds [-5, 5]"
    );
    assert_eq!(errors[2], "A prediction is held at a finite value, got NaN");
    assert!(
        errors[3].contains("Invalid observed output name 'y'"),
        "{}",
        errors[3]
    );
    assert_eq!(
        errors[4],
        "Constraint settings apply to a held prediction, not a parameter"
    );

    let held = Subproblem::fix_prediction(&problem, &spec, 1.0).unwrap();
    let base = ConstraintSettings::for_value(1.0);
    let invalid = [
        ConstraintSettings {
            tolerance: 0.0,
            ..base.clone()
        },
        ConstraintSettings {
            max_outer_iterations: 0,
            ..base.clone()
        },
        ConstraintSettings {
            penalty: Some(-1.0),
            ..base.clone()
        },
        ConstraintSettings {
            multiplier: Some(f64::INFINITY),
            ..base.clone()
        },
    ]
    .map(|settings| held.clone().with_settings(settings).unwrap_err());
    assert_eq!(
        invalid,
        [
            "The tolerance of a held prediction is finite and positive, got 0",
            "A held prediction needs at least one outer iteration",
            "The penalty of a held prediction is finite and positive, got -1",
            "The multiplier of a held prediction is finite, got inf",
        ]
    );
    assert!(
        optimize_subproblem(&held, &config(), &[0.0], None)
            .unwrap_err()
            .contains("Expected 2 start values, got 1")
    );
}

#[test]
fn a_problem_of_one_parameter_held_is_scored_once() {
    let problem = CalibrationProblem::new(
        ClosedForm {
            formula: |p, _| p["a"],
            parameters: HashMap::from([("a".to_string(), 0.0)]),
        },
        vec![
            ObservedDataPoint::new(0.0, "x".to_string(), 1.0),
            ObservedDataPoint::new(1.0, "x".to_string(), 3.0),
        ],
        vec![CalibrationParameter::new("a".to_string(), 0.0, 5.0)],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap();
    let held = Subproblem::fix_parameter(&problem, 0, 1.5).unwrap();
    let result = optimize_subproblem(&held, &config(), &[4.0], None).unwrap();

    // (1 − 1.5)² + (3 − 1.5)².
    assert_eq!(result.fit.final_loss, 2.5);
    assert_eq!(result.fit.best_parameters, vec![1.5]);
    assert_eq!(result.evaluations, 1);
}

#[test]
fn a_swarm_fits_a_subproblem_and_repeats_from_its_seed() {
    let problem = line_problem(&VALUES);
    let (a, b) = least_squares(&VALUES);
    let swarm = OptimizationConfig::ParticleSwarm(ParticleSwarmConfig {
        num_particles: 20,
        max_iterations: 150,
        ..ParticleSwarmConfig::default()
    });
    let fixed = Subproblem::fix_parameter(&problem, 1, b).unwrap();
    let refit = optimize_subproblem(&fixed, &swarm, &[0.0, 0.0], Some(4)).unwrap();
    assert!(
        (refit.fit.best_parameters[0] - a).abs() < 1e-3,
        "{:?}",
        refit.fit
    );
    assert_eq!(refit.fit.best_parameters[1], b);

    let spec = ObservedDataPoint::new(15.0, "x".to_string(), 0.0);
    let held = Subproblem::fix_prediction(&problem, &spec, a + 15.0 * b + 1.0)
        .unwrap()
        .with_settings(ConstraintSettings {
            tolerance: 1e-6,
            max_outer_iterations: 6,
            multiplier: None,
            penalty: None,
        })
        .unwrap();
    let first = optimize_subproblem(&held, &swarm, &[0.0, 0.0], Some(4)).unwrap();
    let again = optimize_subproblem(&held, &swarm, &[0.0, 0.0], Some(4)).unwrap();
    // A swarm without stop rules runs every outer iteration to its limit.
    assert!(!first.fit.converged);
    let prediction = first.prediction.unwrap();
    assert!(
        (prediction - (a + 15.0 * b + 1.0)).abs() < 1e-2,
        "{prediction}"
    );
    assert_eq!(first.fit.best_parameters, again.fit.best_parameters);
    assert_eq!(first.evaluations, again.evaluations);
}

#[test]
fn profile_of_a_linear_model_is_a_parabola() {
    let problem = line_problem(&VALUES);
    let fit = optimize(&problem, &config()).unwrap();
    let (_, b) = least_squares(&VALUES);
    // Σ (t − 4.5)² over t = 0, …, 9.
    let stt: f64 = 82.5;
    let config = config();
    let tracer = ProfileTracer::new(&problem, &config, &fit.best_parameters, 7).unwrap();
    let profile = tracer
        .trace(&ProfiledQuantity::Parameter(1), 1.0, 100)
        .unwrap();

    assert!(profile.points.len() > 10, "{}", profile.points.len());
    for point in &profile.points {
        let expected = (point.value - b).powi(2) * stt;
        assert!(
            (point.statistic - expected).abs() < 1e-8,
            "{} at {}: {expected}",
            point.statistic,
            point.value
        );
    }
    assert_eq!(profile.ends, [ProfileEnd::Crossed, ProfileEnd::Crossed]);
    let half_width = (1.0 / stt).sqrt();
    let [lower, upper] = profile.interval();
    assert!(
        (lower.unwrap() - (b - half_width)).abs() < 1e-7,
        "{lower:?}"
    );
    assert!(
        (upper.unwrap() - (b + half_width)).abs() < 1e-7,
        "{upper:?}"
    );
}

/// The model of `abc_builder` (`tests/support/models.py`): A → B at
/// `k1 * A * B / total`, B → C at `k2 * B`, from 990 in A and 10 in B.
fn abc_engine() -> commol_difference::DifferenceEquations {
    let json = r#"{"name":"ABC","parameters":[{"id":"k1","value":0.3},{"id":"k2","value":0.1}],
        "engine":{"type":"DifferenceEquations"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"},{"id":"C","name":"C"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k1 * A * B / total",
        "stratified_rates":null,"per_compartment":null},
        {"id":"bc","source":["B"],"target":["C"],"rate":"k2 * B",
        "stratified_rates":null,"per_compartment":null}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":990.0,"B":10.0,"C":0.0}}}"#;
    let model: commol_core::Model = serde_json::from_str(json).expect("valid model JSON");
    commol_difference::DifferenceEquations::try_from_model(&model).unwrap()
}

/// `k1` and `k2` of the ABC model fitted to B at 4, 8, …, 24 (the truth
/// with a fixed draw of noise), with a normal model of sd 10.
fn abc_problem() -> CalibrationProblem<commol_difference::DifferenceEquations> {
    let mut engine = abc_engine();
    let positions: Vec<f64> = (1..=6).map(|i| f64::from(4 * i)).collect();
    let mut rows = Vec::new();
    engine
        .run_at_positions_into_buffer(&positions, &mut rows)
        .unwrap();
    let noise = [6.0, -9.0, 4.0, 11.0, -7.0, 3.0];
    CalibrationProblem::new(
        abc_engine(),
        positions
            .iter()
            .zip(&rows)
            .zip(noise)
            .map(|((&at, row), e)| ObservedDataPoint::new(at, "B".to_string(), row[1] + e))
            .collect(),
        vec![
            CalibrationParameter::new("k1".to_string(), 0.05, 1.0),
            CalibrationParameter::new("k2".to_string(), 0.01, 0.5),
        ],
        vec![],
        DiscrepancyConfig::Likelihood(vec![(
            "B".to_string(),
            ObservationModel::Normal {
                sd: Argument::Value(10.0),
            },
        )]),
    )
    .unwrap()
}

#[test]
fn the_multiplier_of_a_prediction_profile_is_minus_its_slope() {
    let problem = abc_problem();
    let config = config();
    let fit = optimize(&problem, &config).unwrap();
    let tracer = ProfileTracer::new(&problem, &config, &fit.best_parameters, 3).unwrap();
    let spec = ObservedDataPoint::new(30.0, "B".to_string(), 0.0);
    let profile = tracer
        .trace(&ProfiledQuantity::Prediction(spec), 1.92, 100)
        .unwrap();

    assert_eq!(profile.ends, [ProfileEnd::Crossed, ProfileEnd::Crossed]);
    let points = &profile.points;
    for window in points.windows(3) {
        let [before, point, after] = [&window[0], &window[1], &window[2]];
        let Some(multiplier) = point.multiplier else {
            continue;
        };
        // The slope of the quadratic through the three points, at the middle.
        let d1 = (point.statistic - before.statistic) / (point.value - before.value);
        let d2 = ((after.statistic - point.statistic) / (after.value - point.value) - d1)
            / (after.value - before.value);
        let slope = d1 + d2 * (point.value - before.value);
        assert!(
            (multiplier + slope).abs() <= 5e-2 * slope.abs().max(1e-3),
            "{multiplier} against {slope} at {}",
            point.value
        );
    }
}

#[test]
fn a_held_prediction_takes_a_warm_start() {
    let problem = line_problem(&VALUES);
    let (a, b) = least_squares(&VALUES);
    let spec = ObservedDataPoint::new(15.0, "x".to_string(), 0.0);
    // Held one above the fit: the multiplier is −dS/dc = −2 (c − ĉ) / h,
    // with h = 1/n + (15 − 4.5)² / Stt.
    let h = 0.1 + 10.5_f64.powi(2) / 82.5;
    let held = Subproblem::fix_prediction(&problem, &spec, a + 15.0 * b + 1.0)
        .unwrap()
        .with_settings(ConstraintSettings {
            tolerance: 1e-9,
            max_outer_iterations: 50,
            multiplier: Some(-1.0),
            penalty: Some(10.0),
        })
        .unwrap();
    let refit = optimize_subproblem(&held, &config(), &[a, b], None).unwrap();

    assert!(refit.fit.converged, "{}", refit.fit.termination_reason);
    let multiplier = refit.multiplier.unwrap();
    assert!((multiplier + 2.0 / h).abs() < 1e-4, "{multiplier}");
    // The loss rises by (c − ĉ)² / h.
    let fit = optimize(&problem, &config()).unwrap();
    assert!((refit.fit.final_loss - fit.final_loss - 1.0 / h).abs() < 1e-6);
}

#[test]
fn an_invalid_optimizer_fails_a_subproblem() {
    let problem = line_problem(&VALUES);
    let config = OptimizationConfig::NelderMead(NelderMeadConfig {
        initial_simplex_size: 0.0,
        ..NelderMeadConfig::default()
    });
    let spec = ObservedDataPoint::new(15.0, "x".to_string(), 0.0);
    let subproblems = [
        Subproblem::fix_parameter(&problem, 0, 1.0).unwrap(),
        Subproblem::fix_prediction(&problem, &spec, 8.0).unwrap(),
    ];
    for subproblem in &subproblems {
        let error = optimize_subproblem(subproblem, &config, &[1.0, 0.5], None).unwrap_err();
        assert!(
            error.contains("initial_simplex_size must be in (0, 1]"),
            "{error}"
        );
    }
}

#[test]
fn a_tracer_refuses_a_fit_it_cannot_start_from() {
    let config = config();
    let problem = line_problem(&VALUES);
    assert_eq!(
        ProfileTracer::new(&problem, &config, &[1.0], 0).err(),
        Some("The fit has 1 values for 2 parameters".to_string())
    );
    let tracer = ProfileTracer::new(&problem, &config, &[1.0, 0.5], 0).unwrap();
    assert_eq!(
        tracer.center(&ProfiledQuantity::Parameter(2)),
        Err("Parameter 2 does not exist; the problem has 2".to_string())
    );
    // a ≥ 20 cannot hold inside the bounds of a.
    let constrained = CalibrationProblem::new(
        problem.forward().base_engine().clone(),
        problem.observed_data().to_vec(),
        problem.parameters().to_vec(),
        vec![CalibrationConstraint::new(
            "a20".to_string(),
            "a - 20".to_string(),
        )],
        LossConfig::SumSquaredError,
    )
    .unwrap();
    assert_eq!(
        ProfileTracer::new(&constrained, &config, &[1.0, 0.5], 0).err(),
        Some("The fit a profile starts from is not feasible".to_string())
    );
}

#[test]
fn a_side_that_starts_at_a_bound_ends_there() {
    let problem = line_problem(&VALUES);
    let config = config();
    // The slope held at its lower bound, −5, from a fit there.
    let tracer = ProfileTracer::new(&problem, &config, &[20.0 / 2.0, -5.0], 0).unwrap();
    let profile = tracer
        .trace(&ProfiledQuantity::Parameter(1), 1.0, 3)
        .unwrap();

    assert_eq!(profile.ends[0], ProfileEnd::Bound);
    assert_eq!(profile.points[0].value, -5.0);
    assert_eq!(profile.interval()[0], None);
}

#[test]
fn the_tolerance_follows_the_optimizer() {
    let problem = line_problem(&VALUES);
    let fit = optimize(&problem, &config()).unwrap();
    let nelder_mead = config();
    let swarm = OptimizationConfig::ParticleSwarm(ParticleSwarmConfig::default());
    // Nelder-Mead: its sd_tolerance, 1e-14; a swarm: 1e-6 of the objective
    // at the fit, at least 1e-6. Both above the noise floor of this line.
    let scale = fit.final_loss.max(1.0);
    for (config, expected) in [(&nelder_mead, 1e-14), (&swarm, 1e-6 * scale)] {
        let tracer = ProfileTracer::new(&problem, config, &fit.best_parameters, 0).unwrap();
        assert!(tracer.noise_floor() < 1e-14, "{}", tracer.noise_floor());
        assert!((tracer.tolerance() - expected).abs() < 1e-18);
    }
}

#[test]
fn thresholds_refuse_what_they_cannot_read() {
    let problem = line_problem(&VALUES);
    assert_eq!(
        chi_square(&problem, 0.9).unwrap_err(),
        "The chi-square threshold reads −log L and needs observation models; the problem \
         has a loss. With a loss, use Threshold.beale (a sum of squares of independent \
         Gaussian errors) or Threshold.bootstrap_calibrated"
    );
    assert_eq!(
        beale(&problem, 1.0, 1.0).unwrap_err(),
        "A level is strictly between 0 and 1, got 1"
    );
    assert_eq!(
        beale(&line_problem(&VALUES[..2]), 1.0, 0.9).unwrap_err(),
        "Beale's threshold needs more observations than parameters: 2 observations, \
         2 parameters"
    );
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[1.0, 0.5], 1, 0).unwrap();
    let quantity = [ProfiledQuantity::Parameter(0)];
    assert_eq!(
        bootstrap_thresholds(&runner, &ZeroWeights, 2, &quantity, &[], 0.9).unwrap_err(),
        "One center per quantity: 1 quantities, 0 centers"
    );
    let error = bootstrap_thresholds(&runner, &ZeroWeights, 2, &quantity, &[1.0], 0.9).unwrap_err();
    assert!(error.starts_with("Every replicate failed (2)"), "{error}");
}

/// A resampler that gives every observation weight 0, which no problem
/// accepts.
struct ZeroWeights;

impl Resampler for ZeroWeights {
    fn replicate(&self, _: &mut Generator) -> Perturbation {
        Perturbation::Weights(vec![0.0; VALUES.len()])
    }
}

/// `VALUES` plus a uniform draw in `[−1, 1]` at each position.
struct Noisy;

impl Resampler for Noisy {
    fn replicate(&self, generator: &mut Generator) -> Perturbation {
        Perturbation::Values(
            VALUES
                .iter()
                .map(|&value| value + 2.0 * generator.uniform() - 1.0)
                .collect(),
        )
    }
}

#[test]
fn a_bootstrap_threshold_of_a_prediction_reads_the_rise_of_each_replicate() {
    let problem = line_problem(&VALUES);
    let config = config();
    let fit = optimize(&problem, &config).unwrap();
    let (a, b) = least_squares(&VALUES);
    let center = a + 15.0 * b;
    let runner = ReplicateRunner::new(&problem, &config, &fit.best_parameters, 1, 9).unwrap();
    let quantity = [ProfiledQuantity::Prediction(ObservedDataPoint::new(
        15.0,
        "x".to_string(),
        0.0,
    ))];
    let run = bootstrap_thresholds(&runner, &Noisy, 8, &quantity, &[center], 0.9).unwrap();

    // On a line, the sum of squares of a replicate held at the prediction
    // `c` rises by (c − ĉ_b)² / h above its fit.
    let h = 0.1 + 10.5_f64.powi(2) / 82.5;
    for (b_index, rise) in run.replicates.iter().zip(&run.rises[0]) {
        let mut generator = Generator::from_seed(replicate_seed(9, *b_index as u64));
        let Perturbation::Values(values) = Noisy.replicate(&mut generator) else {
            unreachable!("Noisy draws values")
        };
        let (a_b, b_b) = least_squares(&values);
        let expected = (center - (a_b + 15.0 * b_b)).powi(2) / h;
        assert!(
            (rise - expected).abs() < 1e-6 * expected.max(1.0),
            "{rise} against {expected}"
        );
    }
    let weights = vec![1.0; 8];
    assert_eq!(
        run.thresholds[0],
        commol_inference::statistics::quantiles(&run.rises[0], &weights, &[0.9]).unwrap()[0]
    );
}

fn curved_line(p: &HashMap<String, f64>, t: f64) -> f64 {
    p["a"] + p["b"] * t + p["z"] * t * t / 100.0
}

/// The data of the problem, unchanged.
struct Unchanged;

impl Resampler for Unchanged {
    fn replicate(&self, _: &mut Generator) -> Perturbation {
        Perturbation::Values(VALUES.to_vec())
    }
}

#[test]
fn thresholds_fail_when_every_refit_fails() {
    // The fit of each replicate keeps z near 0; held at 0.95, every run of
    // the refit fails.
    let parameters = [("a", -10.0, 10.0), ("b", -5.0, 5.0), ("z", 0.0, 1.0)];
    let problem = CalibrationProblem::new(
        ClosedForm {
            formula: curved_line,
            parameters: parameters
                .iter()
                .map(|(id, min, _)| (id.to_string(), *min))
                .collect(),
        },
        VALUES
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
    .unwrap();
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[1.0, 0.5, 0.1], 1, 0).unwrap();
    let perturbation = Unchanged.replicate(&mut Generator::from_seed(0));
    assert!(runner.fit_one(0, &perturbation).is_ok());

    let quantity = [ProfiledQuantity::Parameter(2)];
    let error = bootstrap_thresholds(&runner, &Unchanged, 2, &quantity, &[0.95], 0.9).unwrap_err();

    assert!(error.starts_with("Every replicate failed (2)"), "{error}");
    assert!(error.contains("z above 0.9"), "{error}");
}
