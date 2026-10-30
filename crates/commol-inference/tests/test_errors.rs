//! Every method returns one error, `MethodError`, of two kinds: `Invalid`
//! when a setting, an input, the data or the model does not fit the method,
//! `Failed` when a computation that could start fails.

use std::collections::HashMap;

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, LossConfig, NelderMeadConfig, ObservedDataPoint,
    OptimizationConfig,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};
use commol_inference::conformal::{
    ConformalSettings, NonconformityScore, Pooling, Predictor, adaptive_conformal, enbpi,
    jackknife_plus, rolling_conformal, split_conformal,
};
use commol_inference::design::InputSpace;
use commol_inference::global_sensitivity::ForwardAnalysis;
use commol_inference::identifiability::structural_rank;
use commol_inference::local::delta_band;
use commol_inference::profiles::{ProfiledQuantity, bootstrap_thresholds};
use commol_inference::resampling::{
    BayesianBootstrap, BlockScheme, Perturbation, ReplicateRunner, Resampler, ResidualScale,
    Residuals, block_jackknife, bootstrap,
};
use commol_inference::sensitivity::{Design, FiniteDifferences};
use commol_inference::smc::{RandomWalk, SmcSampler, bootstrap_coverage_eta, gibbs_posterior};
use commol_inference::{ErrorKind, MethodError, Priors};
use commol_random::Generator;

type Formula = fn(&HashMap<String, f64>, f64) -> f64;

/// One output `x = formula(parameters, t)` at every position `t`; a run
/// whose output is not a number fails.
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
        buffer.clear();
        for &t in positions {
            let value = (self.formula)(&self.parameters, t);
            if value.is_nan() {
                return Err(SimulationError::Failed(
                    "the run is not a number".to_string(),
                ));
            }
            buffer.push(vec![value]);
        }
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

/// A run that is never a number.
fn broken(_: &HashMap<String, f64>, _: f64) -> f64 {
    f64::NAN
}

/// The problem of fitting `formula` with `a` and `b` to `values` at
/// positions 0, 1, 2, ….
fn problem(formula: Formula, values: &[f64]) -> CalibrationProblem<ClosedForm> {
    CalibrationProblem::new(
        ClosedForm {
            formula,
            parameters: HashMap::from([("a".to_string(), 0.0), ("b".to_string(), 0.0)]),
        },
        values
            .iter()
            .enumerate()
            .map(|(t, &value)| ObservedDataPoint::new(t as f64, "x".to_string(), value))
            .collect(),
        parameters(),
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

fn parameters() -> Vec<CalibrationParameter> {
    vec![
        CalibrationParameter::new("a".to_string(), -50.0, 50.0),
        CalibrationParameter::new("b".to_string(), -10.0, 10.0),
    ]
}

fn config() -> OptimizationConfig {
    OptimizationConfig::NelderMead(NelderMeadConfig {
        sd_tolerance: 1e-12,
        max_iterations: 2000,
        ..NelderMeadConfig::default()
    })
}

/// An optimizer whose every start fails: its simplex has no size.
fn failing_config() -> OptimizationConfig {
    OptimizationConfig::NelderMead(NelderMeadConfig {
        initial_simplex_size: 0.0,
        ..NelderMeadConfig::default()
    })
}

const VALUES: [f64; 8] = [1.0, 2.2, 2.9, 4.1, 5.0, 5.8, 7.2, 8.0];
const FIT: [f64; 2] = [1.0, 1.0];

fn settings() -> ConformalSettings {
    ConformalSettings::new(NonconformityScore::Absolute, Pooling::Series, 0.5).unwrap()
}

/// The spread score, which a fit alone does not give.
fn spread() -> ConformalSettings {
    ConformalSettings::new(NonconformityScore::Spread, Pooling::Series, 0.5).unwrap()
}

/// A resampler whose every replicate has a value that is not finite.
struct NotFinite(Vec<f64>);

impl Resampler for NotFinite {
    fn replicate(&self, _: &mut Generator) -> Perturbation {
        let mut values = self.0.clone();
        values[0] = f64::NAN;
        Perturbation::Values(values)
    }
}

fn kind_of<T>(result: Result<T, MethodError>) -> (ErrorKind, String) {
    match result {
        Ok(_) => panic!("the method returned a result"),
        Err(error) => (error.kind(), error.to_string()),
    }
}

#[test]
fn a_setting_out_of_range_is_invalid_in_every_method() {
    let problem = problem(line, &VALUES);
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &FIT, 1, 4).unwrap();
    let one_block = BlockScheme::new(problem.observed_data(), VALUES.len()).unwrap();
    let blocks = BlockScheme::new(problem.observed_data(), 2).unwrap();
    let resampler = BayesianBootstrap::new(problem.observed_data(), &blocks);
    let priors =
        Priors::uniform_on_bounds(&problem.parameter_names(), &problem.parameter_bounds()).unwrap();
    let mut sampler = SmcSampler::new(20, RandomWalk::new(), 0.5, 0.5, 0.01).unwrap();
    let analysis = ForwardAnalysis::new(
        problem.forward().base_engine().clone(),
        parameters(),
        None,
        None,
        InputSpace::Independent(priors.clone()),
        Some(3),
    )
    .unwrap();
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let held = [ProfiledQuantity::Parameter(0)];

    let errors = [
        // needs nine.
        kind_of(bootstrap_coverage_eta(
            &problem,
            &priors,
            &FIT,
            &mut sampler,
            &blocks,
            1,
            0.9,
            5,
        )),
        kind_of(SmcSampler::new(1, RandomWalk::new(), 0.5, 0.5, 0.01)),
        kind_of(analysis.sobol(8, 1, 0.9, &[])),
        kind_of(bootstrap_thresholds(
            &runner,
            &resampler,
            2,
            &held,
            &[1.0, 2.0],
            0.5,
        )),
        kind_of(block_jackknife(&runner, &one_block)),
        kind_of(ReplicateRunner::new(&problem, &config, &FIT, 0, 4)),
        kind_of(delta_band(
            &problem,
            &problem.parameter_names(),
            &[("a".to_string(), FIT[0]), ("b".to_string(), FIT[1])],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            None,
            &[1.0],
            1.5,
        )),
        kind_of(adaptive_conformal(
            &problem,
            Predictor::Fit(&FIT),
            problem.observed_data(),
            -1.0,
            settings(),
        )),
        kind_of(structural_rank(&source, 0, 7)),
        kind_of(split_conformal(
            &problem,
            Predictor::Fit(&FIT),
            problem.observed_data(),
            &[8.0],
            spread(),
        )),
        kind_of(adaptive_conformal(
            &problem,
            Predictor::Fit(&FIT),
            problem.observed_data(),
            0.01,
            spread(),
        )),
    ];

    let expected = [
        "replicates times parameters, got 2; use more replicates",
        "particles is a whole number of at least 2, got 1",
        "replicates is a whole number of at least 2, got 1",
        "One center per quantity: 1 quantities, 2 centers",
        "a jackknife needs at least two",
        "starts is a whole number of at least 1, got 0",
        "strictly between 0 and 1",
        "step is a finite number greater than 0, got -1",
        "points is a whole number of at least 1, got 0",
        "give a sample",
        "give a sample",
    ];
    for ((kind, message), expected) in errors.iter().zip(expected) {
        assert_eq!(*kind, ErrorKind::Invalid, "{message}");
        assert!(message.contains(expected), "{message}");
    }
}

#[test]
fn a_computation_that_fails_is_failed_in_every_method() {
    let problem = problem(line, &VALUES);
    let failing = failing_config();
    let runner = ReplicateRunner::new(&problem, &failing, &FIT, 1, 4).unwrap();
    let blocks = BlockScheme::new(problem.observed_data(), 2).unwrap();
    let resampler = BayesianBootstrap::new(problem.observed_data(), &blocks);
    let config = config();
    let fitting = ReplicateRunner::new(&problem, &config, &FIT, 1, 4).unwrap();
    let held = [ProfiledQuantity::Parameter(0)];
    let priors =
        Priors::uniform_on_bounds(&problem.parameter_names(), &problem.parameter_bounds()).unwrap();
    // The first move of a step accepts less than every proposal, so the
    // sampler stops before the temperature 1.
    let mut stopping = SmcSampler::new(50, RandomWalk::new(), 0.5, 0.5, 0.01)
        .unwrap()
        .with_min_acceptance(0.999)
        .unwrap();
    let no_run = self::problem(broken, &VALUES);
    let source = FiniteDifferences::new(&no_run, Design::Observations).unwrap();

    let errors = [
        kind_of(bootstrap(&fitting, &NotFinite(VALUES.to_vec()), 3)),
        kind_of(block_jackknife(&runner, &blocks)),
        kind_of(rolling_conformal(&runner, &[3.0], &[8.0], settings())),
        kind_of(jackknife_plus(&runner, &blocks, &[8.0], settings())),
        kind_of(enbpi(&runner, &blocks, 4, &[8.0], settings())),
        kind_of(bootstrap_thresholds(
            &runner,
            &resampler,
            2,
            &held,
            &[1.0],
            0.5,
        )),
        kind_of(gibbs_posterior(&problem, &priors, 1.0, &mut stopping, 3)),
        kind_of(structural_rank(&source, 2, 7)),
        kind_of(Residuals::new(&no_run, &FIT, ResidualScale::Additive)),
    ];

    let expected = [
        "Every replicate failed",
        "Every block failed",
        "Every origin failed",
        "Every block failed",
        "Every replicate failed",
        "Every replicate failed",
        "The Gibbs posterior stopped at the temperature",
        "the run is not a number",
        "The fit has no run: ",
    ];
    for ((kind, message), expected) in errors.iter().zip(expected) {
        assert_eq!(*kind, ErrorKind::Failed, "{message}");
        assert!(message.contains(expected), "{message}");
    }
}
