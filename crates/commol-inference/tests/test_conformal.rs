//! The conformal quantile, the scores, the working level of adaptive
//! conformal inference, and what each variant rejects, on an engine whose
//! output is a parameter: `x = c` at every position.

use std::collections::HashMap;

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, LossConfig, NelderMeadConfig, ObservedDataPoint,
    OptimizationConfig,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};
use commol_inference::conformal::{
    AdaptiveLevel, ConformalSettings, NonconformityScore, Pooling, Predictor, adaptive_conformal,
    conformal_quantile, enbpi, jackknife_plus, rolling_conformal, scales, split_conformal,
};
use commol_inference::resampling::{BlockScheme, ReplicateRunner};
use commol_inference::settings::conformal::ScoreAndPooling;
use commol_inference::{MethodError, PredictiveError};

/// One output `x = c` at every position; a run with `c` above `cap`
/// fails.
#[derive(Clone, Debug)]
struct Constant {
    parameters: HashMap<String, f64>,
    cap: f64,
}

#[derive(Clone, Debug)]
struct NoPlan;

impl ValuePlan for NoPlan {
    fn infeasibility(&self, excess: f64) -> String {
        format!("{excess}")
    }
}

impl SimulationEngine for Constant {
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
        let c = self.parameters["c"];
        if c > self.cap {
            return Err(SimulationError::Failed(format!("c = {c} is above the cap")));
        }
        buffer.clear();
        buffer.extend(positions.iter().map(|_| vec![c]));
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

/// Observations of `x` at positions 1, 2, …, with weight 0 at the
/// positions of `unweighted`.
fn observations(values: &[f64], unweighted: &[f64]) -> Vec<ObservedDataPoint> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let at = (i + 1) as f64;
            let mut point = ObservedDataPoint::new(at, "x".to_string(), value);
            if unweighted.contains(&at) {
                point.weight = 0.0;
            }
            point
        })
        .collect()
}

fn problem(observed: Vec<ObservedDataPoint>) -> CalibrationProblem<Constant> {
    capped_problem(observed, f64::INFINITY)
}

fn capped_problem(observed: Vec<ObservedDataPoint>, cap: f64) -> CalibrationProblem<Constant> {
    CalibrationProblem::new(
        Constant {
            parameters: HashMap::from([("c".to_string(), 0.0)]),
            cap,
        },
        observed,
        vec![CalibrationParameter::new("c".to_string(), -50.0, 50.0)],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

fn config() -> OptimizationConfig {
    OptimizationConfig::NelderMead(NelderMeadConfig {
        sd_tolerance: 1e-12,
        max_iterations: 500,
        ..NelderMeadConfig::default()
    })
}

/// Ten values around 10.
const VALUES: [f64; 10] = [9.0, 11.0, 10.5, 9.5, 10.0, 8.5, 11.5, 10.2, 9.8, 10.1];

fn settings(score: NonconformityScore) -> ConformalSettings {
    ConformalSettings::new(score, Pooling::Series, 0.5).unwrap()
}

fn invalid(result: Result<impl std::fmt::Debug, MethodError>) -> String {
    match result {
        Err(MethodError::Invalid(message)) => message,
        other => panic!("not rejected as invalid: {other:?}"),
    }
}

#[test]
fn conformal_quantile_is_the_ceiling_order_statistic() {
    // ⌈(19 + 1) · 0.9⌉ = 18: the 18th smallest of 1, …, 19, given in any order.
    let scores: Vec<f64> = (1..=19).rev().map(f64::from).collect();

    assert_eq!(conformal_quantile(&scores, 0.9), 18.0);
}

#[test]
fn adaptive_level_follows_the_update_rule() {
    // α = 0.1 and γ = 0.05: a miss moves α_t by γ (α − 1), a cover by γ α.
    let mut level = AdaptiveLevel::new(0.9, 0.05).unwrap();
    assert!((level.alpha() - 0.1).abs() < 1e-12);

    level.update(false);
    assert!((level.alpha() - (0.1 + 0.05 * (0.1 - 1.0))).abs() < 1e-12);

    level.update(true);
    assert!((level.alpha() - (0.1 + 0.05 * (0.1 - 1.0) + 0.05 * 0.1)).abs() < 1e-12);
    assert!((level.level() - (1.0 - level.alpha())).abs() < 1e-12);
}

#[test]
fn an_adaptive_level_or_step_out_of_range_is_rejected() {
    for (level, step) in [(1.0, 0.05), (0.9, 0.0), (0.9, f64::INFINITY)] {
        assert!(AdaptiveLevel::new(level, step).is_err(), "{level} {step}");
    }
    let problem = problem(observations(&VALUES, &[]));
    let stream = observations(&VALUES, &[]);
    let message = invalid(adaptive_conformal(
        &problem,
        Predictor::Fit(&[10.0]),
        &stream,
        -1.0,
        settings(NonconformityScore::Absolute),
    ));
    assert!(message.contains("step"), "{message}");
}

#[test]
fn scores_and_poolings_name_their_valid_values() {
    let read = |text: &str| commol_core::json::read::<ScoreAndPooling>(text);
    assert_eq!(
        read(r#"{"score": "other", "by": "all"}"#),
        Err(
            "score: unknown variant `other`, expected one of `absolute`, `normalized`, \
             `spread`"
                .to_string()
        )
    );
    assert_eq!(
        read(r#"{"score": "absolute", "by": "other"}"#),
        Err("by: unknown variant `other`, expected `series` or `all`".to_string())
    );
    assert_eq!(
        read(r#"{"by": "all"}"#),
        Err("missing field `score`".to_string())
    );
    let pair = read(r#"{"score": "normalized", "by": "series"}"#).unwrap();
    assert!(pair.check().unwrap_err().to_string().contains("by=\"all\""));
    assert_eq!(
        read(r#"{"score": "normalized", "by": "all"}"#).map(|pair| pair.check().is_ok()),
        Ok(true)
    );
    let pair = ConformalSettings::new(NonconformityScore::Normalized, Pooling::Series, 0.9);
    assert!(pair.unwrap_err().to_string().contains("by=\"all\""));
    let level = ConformalSettings::new(NonconformityScore::Absolute, Pooling::All, 1.0);
    assert!(
        level
            .unwrap_err()
            .to_string()
            .contains("strictly between 0 and 1")
    );
}

#[test]
fn a_score_without_its_scale_is_rejected() {
    let observed = observations(&VALUES, &[]);
    let mut other = observed[0].clone();
    other.label = "y".to_string();

    let normalized = scales(NonconformityScore::Normalized, &observed, &[other], None);
    assert!(normalized.unwrap_err().to_string().contains("none of 'y'"));
    let spread = scales(NonconformityScore::Spread, &observed, &observed, None);
    assert!(spread.unwrap_err().to_string().contains("give a sample"));
    let flat = scales(
        NonconformityScore::Spread,
        &observed[..1],
        &observed[..1],
        Some(&[0.0]),
    );
    assert!(flat.unwrap_err().to_string().contains("positive spread"));
}

#[test]
fn a_conformal_error_says_why() {
    let error = MethodError::from(PredictiveError::Mismatch("names differ".to_string()));
    assert_eq!(error.to_string(), "names differ");
    assert_eq!(
        MethodError::Failed("no fit".to_string()).to_string(),
        "no fit"
    );
}

#[test]
fn split_conformal_needs_held_out_observations() {
    let problem = problem(observations(&VALUES, &[]));

    let message = invalid(split_conformal(
        &problem,
        Predictor::Fit(&[10.0]),
        &[],
        &[11.0],
        settings(NonconformityScore::Absolute),
    ));

    assert!(message.contains("calibration_data is empty"), "{message}");
}

#[test]
fn rolling_conformal_rejects_what_it_cannot_score() {
    let problem = problem(observations(&VALUES, &[]));
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[10.0], 1, 1).unwrap();
    let absolute = settings(NonconformityScore::Absolute);

    let spread = invalid(rolling_conformal(
        &runner,
        &[5.0],
        &[11.0],
        settings(NonconformityScore::Spread),
    ));
    assert!(spread.contains("once per origin"), "{spread}");
    let inside = invalid(rolling_conformal(&runner, &[5.0], &[10.0], absolute));
    assert!(inside.contains("ahead of the data"), "{inside}");
    let last = invalid(rolling_conformal(&runner, &[10.0], &[11.0], absolute));
    assert!(last.contains("origin 10"), "{last}");
}

#[test]
fn a_rolling_origin_with_no_weight_is_a_failed_refit() {
    // The origin at 2 has only observations of weight 0 to fit.
    let problem = problem(observations(&VALUES, &[1.0, 2.0]));
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[10.0], 1, 1).unwrap();

    let run = rolling_conformal(
        &runner,
        &[2.0, 5.0],
        &[11.0],
        settings(NonconformityScore::Absolute),
    )
    .unwrap();

    assert_eq!(run.failed.len(), 1);
    assert_eq!(run.failed[0].0, 0);
    // Only the origin at 5 scores: the 5 observations after it.
    assert_eq!(run.scores.len(), 5);
}

#[test]
fn jackknife_plus_needs_two_blocks_and_two_refits() {
    let config = config();
    let all = problem(observations(&VALUES, &[]));
    let runner = ReplicateRunner::new(&all, &config, &[10.0], 1, 1).unwrap();
    let one_block = BlockScheme::new(all.observed_data(), 10).unwrap();
    let absolute = settings(NonconformityScore::Absolute);
    let message = invalid(jackknife_plus(&runner, &one_block, &[11.0], absolute));
    assert!(message.contains("at least two"), "{message}");

    // The first block has weight 0, so leaving out the second leaves nothing.
    let half = problem(observations(&VALUES, &[1.0, 2.0, 3.0, 4.0, 5.0]));
    let runner = ReplicateRunner::new(&half, &config, &[10.0], 1, 1).unwrap();
    let two_blocks = BlockScheme::new(half.observed_data(), 5).unwrap();
    match jackknife_plus(&runner, &two_blocks, &[11.0], absolute) {
        Err(MethodError::Failed(message)) => assert!(
            message.starts_with("Only 1 of 2 blocks fitted, and the method needs 2"),
            "{message}"
        ),
        other => panic!("not a failure: {other:?}"),
    }
}

#[test]
fn jackknife_plus_with_one_fitted_block_says_how_many_fitted() {
    // The fit is the mean, 20. Leaving out 100 moves it down to 0; leaving
    // out a 0 moves it up to 25, past the cap 22, where the run fails.
    let problem = capped_problem(observations(&[0.0, 0.0, 0.0, 0.0, 100.0], &[]), 22.0);
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[20.0], 1, 1).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 1).unwrap();

    let message = failed(jackknife_plus(
        &runner,
        &scheme,
        &[6.0],
        settings(NonconformityScore::Absolute),
    ));

    assert!(
        message.starts_with(
            "Only 1 of 5 blocks fitted, and the method needs 2; the first reasons: block 0: "
        ),
        "{message}"
    );
}

#[test]
fn jackknife_plus_with_the_spread_score_reads_the_spread_of_its_refits() {
    let problem = problem(observations(&VALUES, &[]));
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[10.0], 1, 1).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 1).unwrap();

    let run = jackknife_plus(
        &runner,
        &scheme,
        &[11.0],
        settings(NonconformityScore::Spread),
    )
    .unwrap();

    let intervals = run.run.intervals;
    assert!(intervals.lower[0] < intervals.prediction[0]);
    assert!(intervals.prediction[0] < intervals.upper[0]);
    assert!((run.bound - (1.0 - 2.0 * 0.5)).abs() < 1e-12);
}

#[test]
fn an_enbpi_replicate_with_no_weight_is_a_failed_refit() {
    // Three blocks, the first of weight 0: a replicate that draws only it
    // has nothing to fit, with probability 1/27.
    let problem = problem(observations(&VALUES[..9], &[1.0, 2.0, 3.0]));
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[10.0], 1, 3).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 3).unwrap();

    let run = enbpi(
        &runner,
        &scheme,
        200,
        &[10.0],
        settings(NonconformityScore::Spread),
    )
    .unwrap();

    assert!(!run.failed.is_empty());
    assert_eq!(run.scores.len(), 9);
}

fn failed(result: Result<impl std::fmt::Debug, MethodError>) -> String {
    match result {
        Err(MethodError::Failed(message)) => message,
        other => panic!("not a failure: {other:?}"),
    }
}

#[test]
fn a_rolling_conformal_with_no_fit_names_its_origins() {
    // The only origin, at 2, has only observations of weight 0 to fit.
    let problem = problem(observations(&VALUES, &[1.0, 2.0]));
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[10.0], 1, 1).unwrap();

    let message = failed(rolling_conformal(
        &runner,
        &[2.0],
        &[11.0],
        settings(NonconformityScore::Absolute),
    ));

    assert!(
        message.starts_with(
            "Every origin failed: 0 of 1 origins fitted, and the method needs 1; the first \
             reasons: origin 0: "
        ),
        "{message}"
    );
}

/// Nine observations of which only the first block of three has weight:
/// a refit without it has nothing to fit.
fn weighted_first_block() -> CalibrationProblem<Constant> {
    let unweighted: Vec<f64> = (4..=9).map(f64::from).collect();
    problem(observations(&VALUES[..9], &unweighted))
}

#[test]
fn enbpi_fails_when_every_replicate_fails() {
    // Neither replicate of seed 1 draws a copy of the first block.
    let problem = weighted_first_block();
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[10.0], 1, 1).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 3).unwrap();

    let message = failed(enbpi(
        &runner,
        &scheme,
        2,
        &[10.0],
        settings(NonconformityScore::Absolute),
    ));

    assert!(
        message.starts_with(
            "Every replicate failed: 0 of 2 replicates fitted, and the method needs 1; the \
             first reasons: replicate 0: "
        ),
        "{message}"
    );
}

#[test]
fn jackknife_plus_of_a_series_with_no_scored_observation_has_no_prediction() {
    // Series y holds the only weight, all in the first block, whose refit
    // fails: x is scored by the two other refits, y by none.
    let mut observed = observations(&VALUES[..9], &(1..=9).map(f64::from).collect::<Vec<_>>());
    for (i, &value) in VALUES[..3].iter().enumerate() {
        let mut point = ObservedDataPoint::new((i + 1) as f64, "x".to_string(), value);
        point.label = "y".to_string();
        observed.push(point);
    }
    let problem = problem(observed);
    let config = config();
    let runner = ReplicateRunner::new(&problem, &config, &[10.0], 1, 1).unwrap();
    let scheme = BlockScheme::new(problem.observed_data(), 3).unwrap();

    let run = jackknife_plus(
        &runner,
        &scheme,
        &[10.0],
        settings(NonconformityScore::Absolute),
    )
    .unwrap();

    assert_eq!(run.run.failed.len(), 1);
    let intervals = run.run.intervals;
    let labels: Vec<&str> = intervals.targets.iter().map(|t| t.label.as_str()).collect();
    assert_eq!(labels, ["x", "y"]);
    assert!(intervals.prediction[0].is_finite());
    assert!(intervals.prediction[1].is_nan());
    assert_eq!(intervals.lower[1], f64::NEG_INFINITY);
    assert_eq!(intervals.upper[1], f64::INFINITY);
}

#[test]
fn a_conformal_method_needs_its_lists_and_a_score_it_can_scale() {
    use commol_inference::settings::conformal::{
        AdaptiveConformalSettings, RollingConformalSettings, SplitConformalSettings,
    };

    let rolling = |score: &str, origins: &str, positions: &str| {
        commol_core::json::read::<RollingConformalSettings>(&format!(
            r#"{{"score": "{score}", "by": "all", "confidence_level": 0.9,
            "origins": {origins}, "positions": {positions}, "starts": 1}}"#
        ))
    };
    let split = commol_core::json::read::<SplitConformalSettings>(
        r#"{"score": "spread", "by": "all", "confidence_level": 0.9,
        "calibration_data": [], "positions": [1.0], "starts": 1}"#,
    );
    let adaptive = commol_core::json::read::<AdaptiveConformalSettings>(
        r#"{"score": "spread", "by": "all", "confidence_level": 0.9, "step": 0.05,
        "stream": [], "starts": 1}"#,
    );

    assert_eq!(
        rolling("absolute", "[]", "[5.0]").unwrap_err(),
        "origins is empty: give at least one origin"
    );
    assert_eq!(
        rolling("absolute", "[2.0]", "[]").unwrap_err(),
        "positions is empty: give at least one position"
    );
    assert!(
        rolling("spread", "[2.0]", "[5.0]")
            .unwrap_err()
            .starts_with("RollingConformal fits once per origin")
    );
    assert_eq!(
        split.unwrap_err(),
        "calibration_data is empty: give at least one held-out observation"
    );
    assert_eq!(
        adaptive.unwrap_err(),
        "stream is empty: give at least one observation that arrives"
    );
    let with_spread = commol_core::json::read::<SplitConformalSettings>(
        r#"{"score": "spread", "by": "all", "confidence_level": 0.9,
        "calibration_data": [{"at": 1.0, "output": "B", "outputs": null, "label": "B",
        "value": 2.0, "weight": 1.0, "factor_id": null, "scale": 1.0, "window": null,
        "trials": null}], "positions": [1.0], "starts": 1}"#,
    )
    .unwrap();
    assert!(with_spread.check_predictor(true).is_ok());
    assert!(
        with_spread
            .check_predictor(false)
            .unwrap_err()
            .to_string()
            .contains("give a sample")
    );
}
