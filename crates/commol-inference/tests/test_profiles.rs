//! Profiles: subproblems that hold one quantity fixed, the tracer and the
//! thresholds, on engines whose output is a closed form of the parameters.

use std::collections::HashMap;

use commol_calibration::{
    CalibrationConstraint, CalibrationParameter, CalibrationProblem, ConstraintSettings,
    LossConfig, NelderMeadConfig, ObservedDataPoint, OptimizationConfig, ParticleSwarmConfig,
    Subproblem, Workspace, optimize, optimize_subproblem,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};

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
