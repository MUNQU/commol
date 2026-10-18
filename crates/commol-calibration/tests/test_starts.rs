//! A multi-start optimization runs every start, keeps the reason of each
//! start that fails, and repeats with its seed.

use std::collections::HashMap;

use commol_calibration::{
    optimize, optimize_starts, CalibrationParameter, CalibrationProblem, LossConfig,
    NelderMeadConfig, ObservedDataPoint, OptimizationConfig, ParticleSwarmConfig,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};

/// One output `x = k` at every position; a run with `k > 2/3` fails.
#[derive(Clone)]
struct LineEngine {
    parameters: HashMap<String, f64>,
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
        let k = self.parameters["k"];
        if k > 2.0 / 3.0 {
            return Err(SimulationError::Failed(format!("k = {k} is too large")));
        }
        buffer.clear();
        buffer.extend(positions.iter().map(|_| vec![k]));
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

    fn get_parameters(&self) -> &HashMap<String, f64> {
        &self.parameters
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

fn problem() -> CalibrationProblem<LineEngine> {
    problem_within(1.0)
}

/// The problem with `k` between 0 and `max`.
fn problem_within(max: f64) -> CalibrationProblem<LineEngine> {
    CalibrationProblem::new(
        LineEngine {
            parameters: HashMap::from([("k".to_string(), 0.0)]),
        },
        vec![ObservedDataPoint::new(1.0, "x".to_string(), 0.2)],
        vec![CalibrationParameter::with_initial_guess(
            "k".to_string(),
            0.0,
            max,
            0.1,
        )],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

fn config() -> OptimizationConfig {
    OptimizationConfig::NelderMead(NelderMeadConfig {
        sd_tolerance: 1e-12,
        ..NelderMeadConfig::default()
    })
}

#[test]
fn a_start_that_fails_keeps_its_reason_and_the_others_finish() {
    let outcomes = optimize_starts(&problem(), &config(), 4, 11);

    assert_eq!(outcomes.len(), 4);
    assert_eq!(outcomes[0].start, vec![0.1]);
    for outcome in &outcomes {
        if outcome.start[0] > 2.0 / 3.0 {
            let error = outcome.result.as_ref().unwrap_err();
            assert!(error.contains("too large"), "{error}");
        } else {
            let result = outcome.result.as_ref().unwrap();
            assert!((result.best_parameters[0] - 0.2).abs() < 1e-6, "{result:?}");
        }
    }
    // One of the three strata of the Latin hypercube is above 2/3.
    assert_eq!(outcomes.iter().filter(|o| o.result.is_err()).count(), 1);
}

#[test]
fn one_seed_gives_one_set_of_outcomes() {
    let first = optimize_starts(&problem(), &config(), 4, 11);
    let again = optimize_starts(&problem(), &config(), 4, 11);
    let starts = |outcomes: &[commol_calibration::StartOutcome]| {
        outcomes.iter().map(|o| o.start.clone()).collect::<Vec<_>>()
    };
    assert_eq!(starts(&first), starts(&again));
}

#[test]
fn the_first_start_of_a_particle_swarm_runs_its_seed_and_the_others_their_own() {
    let config = OptimizationConfig::ParticleSwarm(ParticleSwarmConfig {
        num_particles: 4,
        max_iterations: 3,
        seed: Some(5),
        ..ParticleSwarmConfig::default()
    });
    // Below 2/3, where every run succeeds.
    let problem = problem_within(0.6);
    let outcomes = optimize_starts(&problem, &config, 3, 11);
    let result = |index: usize| outcomes[index].result.clone().unwrap();

    let alone = optimize(&problem, &config).unwrap();
    assert_eq!(result(0).best_parameters, alone.best_parameters);
    assert_ne!(result(1).best_parameters, result(2).best_parameters);
    assert_ne!(result(1).best_parameters, alone.best_parameters);
}
