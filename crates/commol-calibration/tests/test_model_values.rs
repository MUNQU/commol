//! The calibration writes values only through `SimulationEngine`: parameters
//! by id, and the values of the model kind through `plan_values` and
//! `apply_values`. The engine here is of a kind that is not compartmental.

use std::collections::HashMap;

use argmin::core::CostFunction;
use commol_calibration::{
    CalibrationConstraint, CalibrationParameter, CalibrationParameterType, CalibrationProblem,
    EvaluationError, ImprovementStop, LossConfig, MutationApplication, MutationStrategy,
    NelderMeadConfig, ObservedDataPoint, OptimizationConfig, OptimizationObjective,
    ParticleSwarmConfig, Violation, Workspace, check_model_values, configured_engine, optimize,
    optimize_from, optimize_with_history,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};

/// An engine with two outputs: `x = start * k`, its state, and `y = 2`,
/// which is not. Its kind has one value, `start`, which is feasible up to 1.
/// It keeps the remainder of the last plan it applied.
#[derive(Clone)]
struct StartEngine {
    start: f64,
    parameters: HashMap<String, f64>,
    remainder: Option<String>,
}

impl StartEngine {
    fn new() -> Self {
        Self {
            start: 0.5,
            parameters: HashMap::from([("k".to_string(), 1.0)]),
            remainder: None,
        }
    }
}

/// The ids and the remainder the engine was asked to resolve.
#[derive(Clone, Debug, PartialEq)]
struct StartPlan {
    ids: Vec<String>,
    remainder: Option<String>,
}

impl ValuePlan for StartPlan {
    fn infeasibility(&self, excess: f64) -> String {
        format!("start is {excess} above its limit")
    }
}

impl SimulationEngine for StartEngine {
    fn check_values(&self) -> Result<(), String> {
        Ok(())
    }

    type ValuePlan = StartPlan;

    fn check_position(&self, _position: f64) -> Result<(), String> {
        Ok(())
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        let row = vec![self.start * self.parameters["k"], 2.0];
        buffer.clear();
        buffer.extend(positions.iter().map(|_| row.clone()));
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        vec!["x".to_string(), "y".to_string()]
    }

    fn state_outputs(&self) -> usize {
        1
    }

    fn output_sums(&self) -> Vec<(String, Vec<usize>)> {
        vec![("xy".to_string(), vec![0, 1])]
    }

    fn reset(&mut self) {}

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        if value.is_nan() {
            return Err(format!("{parameter_id} is not a number"));
        }
        self.parameters.insert(parameter_id.to_string(), value);
        Ok(())
    }

    fn current_position(&self) -> f64 {
        0.0
    }

    fn plan_values(
        &self,
        values: &[ModelValue],
        remainder: Option<&str>,
    ) -> Result<StartPlan, String> {
        if let Some(value) = values.iter().find(|value| value.id != "start") {
            return Err(format!("this kind has no value '{}'", value.id));
        }
        if remainder == Some("nowhere") {
            return Err("this kind has no remainder 'nowhere'".to_string());
        }
        Ok(StartPlan {
            ids: values.iter().map(|value| value.id.clone()).collect(),
            remainder: remainder.map(str::to_string),
        })
    }

    fn apply_values(&mut self, plan: &StartPlan, values: &[f64]) -> Result<f64, String> {
        assert_eq!(values.len(), plan.ids.len());
        self.remainder.clone_from(&plan.remainder);
        if let Some(&start) = values.first() {
            self.start = start;
        }
        Ok((self.start - 1.0).max(0.0))
    }
}

fn parameter(id: &str, parameter_type: CalibrationParameterType) -> CalibrationParameter {
    CalibrationParameter::with_type_and_guess(id.to_string(), parameter_type, 0.0, 2.0, 0.5)
}

fn start_and_k() -> Vec<CalibrationParameter> {
    vec![
        parameter("k", CalibrationParameterType::Parameter),
        parameter("start", CalibrationParameterType::InitialCondition),
        factor(0.5),
    ]
}

/// The factor `f`, from `min_bound` to 2.
fn factor(min_bound: f64) -> CalibrationParameter {
    CalibrationParameter::with_type_and_guess(
        "f".to_string(),
        CalibrationParameterType::Factor,
        min_bound,
        2.0,
        1.0,
    )
}

fn problem(
    constraints: Vec<CalibrationConstraint>,
) -> Result<CalibrationProblem<StartEngine>, String> {
    CalibrationProblem::new(
        StartEngine::new(),
        vec![ObservedDataPoint::new(1.0, "x".to_string(), 0.5)],
        start_and_k(),
        constraints,
        LossConfig::SumSquaredError,
    )
}

#[test]
fn only_values_of_the_kind_reach_the_engine_and_parameters_are_set_by_id() {
    let (engine, applied, plan) =
        configured_engine(&StartEngine::new(), &start_and_k(), None, &[0.3, 0.8, 1.5]).unwrap();

    assert_eq!(plan.ids, vec!["start".to_string()]);
    assert_eq!(engine.start, 0.8);
    assert_eq!(engine.parameters["k"], 0.3);
    assert!(!engine.parameters.contains_key("f"));
    assert_eq!(applied, vec![0.3, 0.8, 1.5]);
}

#[test]
fn values_are_clamped_to_their_bounds_before_they_reach_the_engine() {
    let (engine, applied, _) =
        configured_engine(&StartEngine::new(), &start_and_k(), None, &[3.0, -1.0, 0.5]).unwrap();

    assert_eq!(applied, vec![2.0, 0.0, 0.5]);
    assert_eq!(engine.start, 0.0);
}

#[test]
fn the_remainder_reaches_the_engine() {
    let (_, _, plan) = configured_engine(
        &StartEngine::new(),
        &start_and_k(),
        Some("rest"),
        &[1.0, 0.5, 1.0],
    )
    .unwrap();

    assert_eq!(plan.remainder.as_deref(), Some("rest"));
    let problem = problem(vec![])
        .unwrap()
        .with_initial_condition_remainder("rest")
        .unwrap();
    let engine = problem.forward().configure(&[1.0, 0.5, 1.0]).unwrap();
    assert_eq!(engine.remainder.as_deref(), Some("rest"));
}

#[test]
fn a_value_the_engine_does_not_resolve_stops_the_problem() {
    let parameters = vec![parameter(
        "other",
        CalibrationParameterType::InitialCondition,
    )];

    let error = check_model_values(&StartEngine::new(), &parameters, None).unwrap_err();
    assert!(error.contains("no value 'other'"), "{error}");

    let error = CalibrationProblem::new(
        StartEngine::new(),
        vec![ObservedDataPoint::new(1.0, "x".to_string(), 0.5)],
        parameters,
        vec![],
        LossConfig::SumSquaredError,
    )
    .err()
    .expect("the engine rejects the value");
    assert!(error.contains("no value 'other'"), "{error}");
}

#[test]
fn a_factor_needs_positive_bounds() {
    let message = "Factor parameter 'f' must have positive bounds (got min=0, max=2)";

    let error = check_model_values(&StartEngine::new(), &[factor(0.0)], None).unwrap_err();
    assert_eq!(error, message);
    let error = CalibrationProblem::new(
        StartEngine::new(),
        vec![ObservedDataPoint::new(1.0, "x".to_string(), 0.5)],
        vec![factor(0.0)],
        vec![],
        LossConfig::SumSquaredError,
    )
    .err()
    .expect("a factor at 0 is rejected");
    assert_eq!(error, message);
}

#[test]
fn an_infeasible_state_is_penalized_and_its_reason_comes_from_the_engine() {
    let problem = problem(vec![]).unwrap();

    let objective = OptimizationObjective::new(&problem);
    let feasible = objective.cost(&vec![1.0, 0.5, 1.0]).unwrap();
    let infeasible = objective.cost(&vec![1.0, 1.5, 1.0]).unwrap();
    assert_eq!(feasible, 0.0);
    assert!(infeasible > 1e9, "{infeasible}");

    let error = problem.configured_engine(&[1.0, 1.5, 1.0]).err().unwrap();
    assert!(error.contains("start is 0.5 above its limit"), "{error}");
}

#[test]
fn a_constraint_reads_the_sums_and_the_state_total_the_engine_names() {
    // At k = 1 and start = 0.5: x = 0.5, y = 2, xy = 2.5 and total = x = 0.5.
    let penalty = |expression: &str| {
        let problem = problem(vec![
            CalibrationConstraint::new("c".to_string(), expression.to_string())
                .with_positions(vec![1.0]),
        ])
        .unwrap();
        OptimizationObjective::new(&problem)
            .cost(&vec![1.0, 0.5, 1.0])
            .unwrap()
    };

    assert_eq!(penalty("2.5 - xy"), 0.0);
    assert_eq!(penalty("0.5 - total"), 0.0);
    assert!(penalty("xy - 3") > 0.0);
    assert!(penalty("total - 1") > 0.0);

    let error = problem(vec![
        CalibrationConstraint::new("c".to_string(), "zz - 1".to_string()).with_positions(vec![1.0]),
    ])
    .err()
    .unwrap();
    assert!(error.contains("unknown variable 'zz'"), "{error}");
}

#[test]
fn every_entry_takes_one_value_per_parameter() {
    let problem = problem(vec![]).unwrap();
    let expected = "Expected 3 parameters, got 2";
    let config = OptimizationConfig::NelderMead(NelderMeadConfig::default());

    let errors = [
        configured_engine(&StartEngine::new(), &start_and_k(), None, &[0.3, 0.8])
            .err()
            .unwrap(),
        problem.configured_engine(&[0.3, 0.8]).err().unwrap(),
        problem
            .forward()
            .configure(&[0.3, 0.8])
            .err()
            .unwrap()
            .to_string(),
        OptimizationObjective::new(&problem)
            .score(&[0.3, 0.8, 1.0, 1.0])
            .err()
            .unwrap()
            .to_string(),
        optimize_from(&problem, &config, &[0.3, 0.8], 0).unwrap_err(),
    ];

    assert_eq!(errors[0], expected);
    assert_eq!(errors[1], expected);
    assert!(errors[2].ends_with(expected), "{}", errors[2]);
    assert_eq!(errors[3], "Expected 3 parameters, got 4");
    assert_eq!(errors[4], "Expected 3 start values, got 2");
}

#[test]
fn a_history_reports_an_optimizer_it_cannot_build() {
    let problem = problem(vec![]).unwrap();
    let config = OptimizationConfig::NelderMead(NelderMeadConfig {
        initial_simplex_size: 0.0,
        ..NelderMeadConfig::default()
    });

    let error = optimize_with_history(&problem, &config, 0).err().unwrap();

    assert_eq!(
        error,
        "initial_simplex_size is a number greater than 0 and at most 1, got 0"
    );
}

#[test]
fn a_problem_has_observations_and_parameters_of_the_model() {
    let new = |observed: Vec<ObservedDataPoint>, parameters: Vec<CalibrationParameter>| {
        CalibrationProblem::new(
            StartEngine::new(),
            observed,
            parameters,
            vec![],
            LossConfig::SumSquaredError,
        )
        .err()
        .unwrap()
    };
    let at_one = || vec![ObservedDataPoint::new(1.0, "x".to_string(), 0.5)];
    let mut no_outputs = at_one();
    no_outputs[0].outputs.clear();

    assert_eq!(
        new(vec![], start_and_k()),
        "A calibration problem needs observed_data: at least one observation to fit"
    );
    assert_eq!(
        new(at_one(), vec![]),
        "A calibration problem needs parameters: at least one value to calibrate"
    );
    assert_eq!(
        new(no_outputs, start_and_k()),
        "Observation 'x' at 1 has an empty list of outputs"
    );
    assert_eq!(
        new(
            vec![ObservedDataPoint::new(1.0, "z".to_string(), 0.5)],
            start_and_k()
        ),
        "Invalid observed output name 'z' for observation 'z' (available: x, y)"
    );
    assert_eq!(
        new(
            vec![ObservedDataPoint::with_factor(
                1.0,
                "x".to_string(),
                0.5,
                "g".to_string()
            )],
            start_and_k()
        ),
        "Invalid factor_id 'g' referenced in observed data (not found in parameters)"
    );
}

#[test]
fn a_constraint_is_a_formula_that_reads_outputs_only_at_positions() {
    let error = |expression: &str| {
        problem(vec![CalibrationConstraint::new(
            "c".to_string(),
            expression.to_string(),
        )])
        .err()
        .unwrap()
    };

    assert_eq!(
        error("k +"),
        "Constraint 'c': Formula 'k +': the operator '+' needs two operands"
    );
    assert_eq!(
        error("1 - (k > 1)"),
        "Constraint 'c': Formula '1 - (k > 1)': the comparison '>' is only allowed in the \
         condition of if(condition, a, b)"
    );
    assert_eq!(
        error("1 - x"),
        "Constraint 'c' references output 'x' but has no positions (`at`)"
    );
}

#[test]
fn an_engine_that_rejects_a_value_or_a_remainder_names_it() {
    let problem = problem(vec![]).unwrap();

    assert_eq!(
        problem
            .configured_engine(&[f64::NAN, 0.5, 1.0])
            .err()
            .unwrap(),
        "Failed to set parameter 'k': k is not a number"
    );
    assert_eq!(
        problem
            .with_initial_condition_remainder("nowhere")
            .err()
            .unwrap(),
        "this kind has no remainder 'nowhere'"
    );
}

#[test]
fn a_loss_or_a_run_that_is_not_finite_is_infeasible() {
    let mut workspace = Workspace::default();
    let infinite = CalibrationProblem::new(
        StartEngine::new(),
        vec![ObservedDataPoint::new(1.0, "x".to_string(), f64::INFINITY)],
        start_and_k(),
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap();

    assert!(matches!(
        infinite.evaluate(&[1.0, 0.5, 1.0], &mut workspace),
        Err(EvaluationError::Violation(Violation::NonFinite))
    ));
    let finite = problem(vec![]).unwrap();
    assert!(matches!(
        finite.evaluate(&[f64::INFINITY, 0.5, 1.0], &mut workspace),
        Err(EvaluationError::Violation(Violation::NonFinite))
    ));
}

#[test]
fn a_swarm_with_mutation_a_target_and_an_observer_runs() {
    let problem = problem(vec![]).unwrap();
    for (strategy, application) in [
        (
            MutationStrategy::Cauchy(0.1),
            MutationApplication::BelowAverage,
        ),
        (
            MutationStrategy::Gaussian(0.1),
            MutationApplication::AllParticles,
        ),
    ] {
        let config = OptimizationConfig::ParticleSwarm(ParticleSwarmConfig {
            particles: 5,
            max_iterations: 20,
            target_cost: Some(1e-8),
            mutation_strategy: strategy,
            mutation_probability: 0.5,
            mutation_application: application,
            ..ParticleSwarmConfig::default()
        });

        let result = optimize(&problem, &config, 3).unwrap();

        assert!(result.final_loss < 1e-3, "{}", result.final_loss);
    }
}

#[test]
fn a_particle_swarm_with_a_stop_rule_converges() {
    let problem = problem(vec![]).unwrap();
    let swarm = |improvement_stop, radius_stop| {
        OptimizationConfig::ParticleSwarm(ParticleSwarmConfig {
            particles: 10,
            max_iterations: 1000,
            improvement_stop,
            radius_stop,
            ..ParticleSwarmConfig::default()
        })
    };
    let without = optimize(&problem, &swarm(None, None), 1).unwrap();
    assert!(!without.converged);
    assert_eq!(without.iterations, 1000);

    let improvement = Some(ImprovementStop {
        window: 20,
        tol_abs: 1e-12,
        tol_rel: 1e-8,
    });
    let config = swarm(improvement, None);

    let result = optimize(&problem, &config, 1).unwrap();

    assert!(result.converged, "{result:?}");
    assert_eq!(result.termination_reason, "Terminated(SolverConverged)");
    assert!(result.iterations < 1000, "{}", result.iterations);
    assert!(result.final_loss < 1e-6, "{}", result.final_loss);
    // The values of this problem lie on a curve of zero loss, so the swarm
    // never gathers around one point: the radius rule does not stop it.
    let spread = optimize(&problem, &swarm(None, Some(1e-4)), 1).unwrap();
    assert!(!spread.converged);
}
