//! A calibration of a stochastic engine runs the replicates its settings
//! name, and the engine says which settings it needs.

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, DiscrepancyConfig, LossConfig, ObservationModel,
    ObservedDataPoint, ReplicateSummary,
};
use commol_core::{Model, SimulationEngine};
use commol_difference::DifferenceEquations;
use commol_stochastic::StochasticSteps;

/// A → B at the proportional rate `k`, from A = 1000, on `engine`.
fn model(engine: &str) -> Model {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"ENGINE"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],
        "target":["B"],"rate":"k","stratified_rates":null,"per_compartment":null}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#;
    serde_json::from_str(&json.replace("ENGINE", engine)).expect("valid model JSON")
}

fn problem<E: SimulationEngine>(
    engine: E,
    discrepancy: impl Into<DiscrepancyConfig>,
) -> Result<CalibrationProblem<E>, String> {
    CalibrationProblem::new(
        engine,
        vec![ObservedDataPoint::new(1.0, "A".to_string(), 900.0)],
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        vec![],
        discrepancy,
    )
}

fn stochastic() -> CalibrationProblem<StochasticSteps> {
    let engine = StochasticSteps::try_from_model(&model("StochasticSteps")).unwrap();
    problem(engine, LossConfig::SumSquaredError).unwrap()
}

#[test]
fn only_a_stochastic_engine_is_stochastic() {
    let steps = StochasticSteps::try_from_model(&model("StochasticSteps")).unwrap();
    let difference = DifferenceEquations::try_from_model(&model("DifferenceEquations")).unwrap();

    assert!(steps.is_stochastic());
    assert!(!difference.is_stochastic());
}

#[test]
fn a_stochastic_engine_names_each_setting_it_needs() {
    let error = |count, seed, summary| {
        stochastic()
            .with_replicate_settings(count, seed, summary)
            .err()
            .unwrap()
    };

    let no_summary = error(Some(2), Some(1), None);
    let no_count = error(None, Some(1), Some(ReplicateSummary::Mean));
    let no_seed = error(Some(2), None, Some(ReplicateSummary::Mean));
    let no_replicate = error(Some(0), Some(1), Some(ReplicateSummary::Mean));

    assert!(no_summary.contains("set replicate_summary"), "{no_summary}");
    assert!(no_summary.contains("Quantile(level)"), "{no_summary}");
    assert_eq!(
        no_count,
        "The StochasticSteps engine is stochastic: set replicates, how many replicates each evaluation runs."
    );
    assert_eq!(
        no_seed,
        "The StochasticSteps engine is stochastic: set seed, so the replicates and the calibration are \
         reproducible."
    );
    assert_eq!(no_replicate, "replicates is at least 1, got 0");
}

#[test]
fn a_stochastic_engine_runs_the_replicates_of_its_settings() {
    let problem = stochastic()
        .with_replicate_settings(Some(3), Some(7), Some(ReplicateSummary::Median))
        .unwrap();

    let replicates = problem.forward().replicates().unwrap();

    assert_eq!(replicates.seeds.len(), 3);
    assert_eq!(replicates.summary, ReplicateSummary::Median);
}

#[test]
fn a_deterministic_engine_reads_no_replicate_setting() {
    let engine = DifferenceEquations::try_from_model(&model("DifferenceEquations")).unwrap();
    let problem = problem(engine, LossConfig::SumSquaredError)
        .unwrap()
        .with_replicate_settings(Some(3), None, None)
        .unwrap();

    assert!(problem.forward().replicates().is_none());
}

#[test]
fn a_stochastic_engine_needs_a_loss() {
    let engine = StochasticSteps::try_from_model(&model("StochasticSteps")).unwrap();
    let models = DiscrepancyConfig::Likelihood(vec![("A".to_string(), ObservationModel::Poisson)]);

    let error = problem(engine, models).err().unwrap();

    assert!(
        error.starts_with("A stochastic engine needs a loss"),
        "{error}"
    );
}
