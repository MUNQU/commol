//! A multi-start optimization runs every start, keeps the reason of each
//! start that fails, chooses the first start of lowest loss, and repeats
//! with its seed.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, ConstraintSettings, LossConfig, NelderMeadConfig,
    ObservedDataPoint, OptimizationConfig, ParticleSwarmConfig, Starts, Subproblem, optimize,
    optimize_from, optimize_starts, optimize_subproblem, optimize_with_history, start_points,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};

/// One output `x = k` at every position, or `x = 0.4` when it is flat; a
/// run with `k > 2/3` fails, and so does every run after the first `limit`
/// runs of the engine and its clones, which `runs` counts.
#[derive(Clone)]
struct LineEngine {
    parameters: HashMap<String, f64>,
    flat: bool,
    runs: Arc<AtomicUsize>,
    limit: usize,
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
        let run = self.runs.fetch_add(1, Ordering::SeqCst) + 1;
        if run > self.limit {
            return Err(SimulationError::Failed(format!(
                "run {run} is over the limit"
            )));
        }
        let k = self.parameters["k"];
        if k > 2.0 / 3.0 {
            return Err(SimulationError::Failed(format!("k = {k} is too large")));
        }
        buffer.clear();
        let x = if self.flat { 0.4 } else { k };
        buffer.extend(positions.iter().map(|_| vec![x]));
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

fn problem() -> CalibrationProblem<LineEngine> {
    problem_within(1.0)
}

/// The problem with `k` between 0 and `max`.
fn problem_within(max: f64) -> CalibrationProblem<LineEngine> {
    line_problem(false, 0.0, max, 0.1)
}

/// The problem of `k` between `min` and `max` from `guess`, on the line or
/// on the flat engine.
fn line_problem(flat: bool, min: f64, max: f64, guess: f64) -> CalibrationProblem<LineEngine> {
    limited_problem(flat, min, max, guess, usize::MAX).0
}

/// [`line_problem`] on an engine whose runs fail after the first `limit`,
/// with the counter of its runs.
fn limited_problem(
    flat: bool,
    min: f64,
    max: f64,
    guess: f64,
    limit: usize,
) -> (CalibrationProblem<LineEngine>, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let problem = CalibrationProblem::new(
        LineEngine {
            parameters: HashMap::from([("k".to_string(), 0.0)]),
            flat,
            runs: Arc::clone(&runs),
            limit,
        },
        vec![ObservedDataPoint::new(1.0, "x".to_string(), 0.2)],
        vec![CalibrationParameter::with_initial_guess(
            "k".to_string(),
            min,
            max,
            guess,
        )],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap();
    (problem, runs)
}

fn config() -> OptimizationConfig {
    OptimizationConfig::NelderMead(NelderMeadConfig {
        sd_tolerance: 1e-12,
        ..NelderMeadConfig::default()
    })
}

fn swarm() -> OptimizationConfig {
    OptimizationConfig::ParticleSwarm(ParticleSwarmConfig {
        particles: 4,
        max_iterations: 3,
        ..ParticleSwarmConfig::default()
    })
}

/// The `count` starts of `problem` seeded with `seed`.
fn run(
    problem: &CalibrationProblem<LineEngine>,
    config: &OptimizationConfig,
    count: usize,
    seed: u64,
) -> Result<Starts, String> {
    optimize_starts(
        problem,
        config,
        start_points(problem.parameters(), count, seed),
        seed,
    )
}

#[test]
fn a_start_that_fails_keeps_its_reason_and_the_others_finish() {
    let starts = run(&problem(), &config(), 4, 11).unwrap();
    let outcomes = starts.outcomes();

    assert_eq!(outcomes.len(), 4);
    assert_eq!(outcomes[0].start, vec![0.1]);
    for outcome in outcomes {
        if outcome.start[0] > 2.0 / 3.0 {
            let error = outcome.result.as_ref().unwrap_err();
            assert!(error.contains("too large"), "{error}");
        } else {
            let result = outcome.result.as_ref().unwrap();
            assert!((result.best_parameters[0] - 0.2).abs() < 1e-5, "{result:?}");
        }
    }
    // One of the three strata of the Latin hypercube is above 2/3.
    assert_eq!(outcomes.iter().filter(|o| o.result.is_err()).count(), 1);
    assert!(outcomes[starts.best_index()].result.is_ok());
}

#[test]
fn one_seed_gives_one_set_of_outcomes() {
    let first = run(&problem(), &config(), 4, 11).unwrap();
    let again = run(&problem(), &config(), 4, 11).unwrap();
    assert_eq!(format!("{first:?}"), format!("{again:?}"));
}

#[test]
fn the_best_start_is_the_first_of_lowest_loss() {
    // Every run that succeeds gives `x = 0.4`, so every start that ends ends
    // at one loss; the guess, above 2/3, fails.
    let problem = line_problem(true, 0.0, 1.0, 0.9);
    let starts = run(&problem, &config(), 4, 11).unwrap();
    let outcomes = starts.outcomes();
    let ended: Vec<usize> = (0..outcomes.len())
        .filter(|&index| outcomes[index].result.is_ok())
        .collect();

    assert!(outcomes[0].result.is_err());
    assert_eq!(ended.len(), 2, "{outcomes:?}");
    let loss = |index: usize| outcomes[index].result.as_ref().unwrap().final_loss;
    assert_eq!(loss(ended[0]).to_bits(), loss(ended[1]).to_bits());
    assert_eq!(starts.best_index(), ended[0]);
    assert_eq!(
        format!("{:?}", starts.best()),
        format!("{:?}", outcomes[ended[0]].result.as_ref().unwrap())
    );
}

#[test]
fn every_failed_start_gives_its_reason() {
    // Above 2/3, where every run fails.
    let problem = line_problem(false, 0.7, 1.0, 0.8);
    let error = run(&problem, &config(), 3, 11).unwrap_err();

    assert!(
        error.starts_with("Every start failed: Start 0 failed: "),
        "{error}"
    );
    for index in 0..3 {
        let reason = format!("Start {index} failed: ");
        assert_eq!(error.matches(&reason).count(), 1, "{error}");
    }
    assert_eq!(error.matches("too large").count(), 3, "{error}");
}

#[test]
fn no_start_is_an_error() {
    let error = optimize_starts(&problem(), &config(), Vec::new(), 11).unwrap_err();
    assert_eq!(error, "A multi-start optimization needs at least one start");
}

#[test]
fn a_swarm_repeats_its_fit_from_its_seed() {
    // Below 2/3, where every run succeeds.
    let problem = problem_within(0.6);
    let first = optimize_with_history(&problem, &swarm(), 5).unwrap();
    let again = optimize_with_history(&problem, &swarm(), 5).unwrap();
    let other = optimize_with_history(&problem, &swarm(), 6).unwrap();
    let population = |run: &commol_calibration::CalibrationResultWithHistory| {
        run.evaluations[..4]
            .iter()
            .map(|evaluation| evaluation.parameters.clone())
            .collect::<Vec<_>>()
    };

    assert_eq!(format!("{first:?}"), format!("{again:?}"));
    assert_ne!(population(&first), population(&other));
}

#[test]
fn the_calibration_from_the_guess_is_the_first_start() {
    // Below 2/3, where every run succeeds.
    let problem = problem_within(0.6);
    let starts = run(&problem, &swarm(), 3, 11).unwrap();
    let result = |index: usize| starts.outcomes()[index].result.clone().unwrap();

    let alone = optimize(&problem, &swarm(), 11).unwrap();
    assert_eq!(format!("{:?}", result(0)), format!("{alone:?}"));
    assert_ne!(result(1).best_parameters, result(2).best_parameters);
    assert_ne!(result(1).best_parameters, alone.best_parameters);
}

/// The swarm of [`swarm`], polished by the Nelder-Mead of [`config`].
fn polished_swarm() -> OptimizationConfig {
    let OptimizationConfig::ParticleSwarm(swarm) = swarm() else {
        unreachable!("`swarm` is a particle swarm")
    };
    let OptimizationConfig::NelderMead(polish) = config() else {
        unreachable!("`config` is Nelder-Mead")
    };
    OptimizationConfig::ParticleSwarm(ParticleSwarmConfig {
        polish: Some(Box::new(polish)),
        ..swarm
    })
}

#[test]
fn a_polished_start_counts_both_stages() {
    // Below 2/3, where every run succeeds.
    let problem = problem_within(0.6);
    let swarm_alone = run(&problem, &swarm(), 3, 11).unwrap();
    let polished = run(&problem, &polished_swarm(), 3, 11).unwrap();

    for (alone, outcome) in swarm_alone.outcomes().iter().zip(polished.outcomes()) {
        let alone = alone.result.as_ref().unwrap();
        let by_hand = optimize_from(&problem, &config(), &alone.best_parameters, 0).unwrap();
        let result = outcome.result.as_ref().unwrap();
        assert_eq!(result.iterations, alone.iterations + by_hand.iterations);
        assert_eq!(result.termination_reason, by_hand.termination_reason);
        assert_eq!(result.termination_reason, "Terminated(SolverConverged)");
        assert!(result.converged);
        assert_eq!(result.best_parameters, by_hand.best_parameters);
        assert_eq!(result.final_loss.to_bits(), by_hand.final_loss.to_bits());
        assert!((result.best_parameters[0] - 0.2).abs() < 1e-5, "{result:?}");
    }
    let best = polished.best();
    let alone = swarm_alone.outcomes()[polished.best_index()]
        .result
        .as_ref()
        .unwrap();
    assert_eq!(alone.iterations, 3);
    assert!(best.iterations > 3, "{best:?}");
}

#[test]
fn a_swarm_with_no_polish_is_unchanged() {
    // Below 2/3, where every run succeeds. The values are those of the swarm
    // alone from 3 starts with the seed 11.
    let problem = problem_within(0.6);
    let starts = run(&problem, &swarm(), 3, 11).unwrap();
    let expected: [(f64, f64, f64); 3] = [
        (0.1, 0.19840354483680456, 2.548669088093423e-6),
        (
            0.052852272125482304,
            0.19216470775023914,
            6.139180463916279e-5,
        ),
        (
            0.5242938534109123,
            0.20840671260648147,
            7.067281684797429e-5,
        ),
    ];

    for (outcome, (start, end, loss)) in starts.outcomes().iter().zip(expected) {
        let result = outcome.result.as_ref().unwrap();
        assert_eq!(outcome.start, vec![start]);
        assert_eq!(result.best_parameters, vec![end]);
        assert_eq!(result.final_loss.to_bits(), loss.to_bits());
        assert_eq!(result.iterations, 3);
        assert_eq!(result.termination_reason, "Terminated(MaxItersReached)");
    }
}

#[test]
fn a_polish_outside_its_ranges_names_its_setting() {
    let OptimizationConfig::ParticleSwarm(swarm) = polished_swarm() else {
        unreachable!("`polished_swarm` is a particle swarm")
    };
    let mut polish = swarm.polish.clone().unwrap();
    polish.sd_tolerance = -1.0;
    let config = OptimizationConfig::ParticleSwarm(ParticleSwarmConfig {
        polish: Some(polish),
        ..swarm
    });

    let error = run(&problem_within(0.6), &config, 1, 11).unwrap_err();
    assert!(error.contains("sd_tolerance"), "{error}");
}

#[test]
fn a_polish_that_fails_gives_its_reason() {
    // Below 2/3, so only the limit fails. The limit is the runs of the swarm
    // alone, so the first run of Nelder-Mead fails.
    let (problem, runs) = limited_problem(false, 0.0, 0.6, 0.1, usize::MAX);
    optimize_from(&problem, &swarm(), &[0.1], 11).unwrap();
    let limit = runs.load(Ordering::SeqCst);
    let (limited, _) = limited_problem(false, 0.0, 0.6, 0.1, limit);

    let error = optimize_from(&limited, &polished_swarm(), &[0.1], 11).unwrap_err();
    assert!(
        error.contains(&format!("run {} is over the limit", limit + 1)),
        "{error}"
    );
}

#[test]
fn a_polish_of_a_held_prediction_that_fails_gives_its_reason() {
    // The limit is the runs of the constrained swarm alone, without the run
    // of its prediction at the end, so the first run of the constrained
    // Nelder-Mead fails.
    let spec = ObservedDataPoint::new(1.0, "x".to_string(), 0.0);
    let held = |problem: &CalibrationProblem<LineEngine>, config: &OptimizationConfig| {
        let held = Subproblem::fix_prediction(problem, &spec, 0.3)
            .unwrap()
            .with_settings(ConstraintSettings {
                tolerance: 1e-6,
                max_outer_iterations: 2,
                multiplier: None,
                penalty: None,
            })
            .unwrap();
        optimize_subproblem(&held, config, &[0.1], 11).map(|_| ())
    };
    let (problem, runs) = limited_problem(false, 0.0, 0.6, 0.1, usize::MAX);
    held(&problem, &swarm()).unwrap();
    let limit = runs.load(Ordering::SeqCst) - 1;
    let (limited, _) = limited_problem(false, 0.0, 0.6, 0.1, limit);

    let error = held(&limited, &polished_swarm()).unwrap_err();
    assert!(
        error.contains(&format!("run {} is over the limit", limit + 1)),
        "{error}"
    );
}
