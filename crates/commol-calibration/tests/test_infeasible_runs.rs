//! A run whose state leaves the values the model allows is a point the
//! calibration avoids; any other error of a run stops the calibration.

use argmin::core::CostFunction;
use commol_calibration::{
    optimize, CalibrationParameter, CalibrationParameterType, CalibrationProblem, LossConfig,
    NelderMeadConfig, ObservedDataPoint, OptimizationConfig,
};
use commol_core::Model;
use commol_difference::DifferenceEquations;

/// A → B at the proportional rate `rate`, in discrete steps from A = 1000.
fn decay_engine(rate: &str) -> DifferenceEquations {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":0.1}}],
        "engine":{{"type":"DifferenceEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"ab","source":["A"],
        "target":["B"],"rate":"{rate}","stratified_rates":null,"per_compartment":null}}]}},
        "structure":{{"kind":"aggregate","total":1000,"bin_fractions":{{"A":1.0,"B":0.0}}}}}}"#
    );
    let model: Model = serde_json::from_str(&json).expect("valid model JSON");
    DifferenceEquations::try_from_model(&model).unwrap()
}

/// `k` between 0 and 3, from `guess`: above 1 a step moves more than `A`
/// holds. The data are the run at `k = 0.2`.
fn problem(rate: &str, guess: f64) -> CalibrationProblem<DifferenceEquations> {
    let observed = (1..=5)
        .map(|step| {
            let amount = 1000.0 * 0.8_f64.powi(step);
            ObservedDataPoint::new(f64::from(step), "A".to_string(), amount)
        })
        .collect();
    CalibrationProblem::new(
        decay_engine(rate),
        observed,
        vec![CalibrationParameter::with_type_and_guess(
            "k".to_string(),
            CalibrationParameterType::Parameter,
            0.0,
            3.0,
            guess,
        )],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

#[test]
fn an_infeasible_run_gets_the_base_penalty_and_no_history() {
    let problem = problem("k", 0.5);

    let infeasible = problem.cost(&vec![2.0]).unwrap();

    assert!(infeasible > 1e9, "{infeasible}");
    assert!(problem.get_evaluations().is_empty());
}

#[test]
fn a_calibration_whose_bounds_include_infeasible_rates_finds_the_rate_of_the_data() {
    let config = OptimizationConfig::NelderMead(
        NelderMeadConfig::new()
            .with_initial_simplex_size(0.5)
            .with_sd_tolerance(1e-12),
    );

    let result = optimize(problem("k", 0.5), config).unwrap();

    assert!(
        (result.best_parameters[0] - 0.2).abs() < 1e-4,
        "{:?}",
        result.best_parameters
    );
}

#[test]
fn a_rate_that_cannot_be_evaluated_stops_the_calibration_with_its_message() {
    let problem = problem("k * missing", 0.5);

    let error = problem.cost(&vec![0.5]).unwrap_err().to_string();

    assert!(error.contains("missing"), "{error}");
}
