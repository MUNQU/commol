//! The Jacobian of the predicted observations by finite differences, its
//! estimated error, and its scaling.

use commol_calibration::{
    CalibrationParameter, CalibrationParameterType, CalibrationProblem, LossConfig,
    ObservedDataPoint,
};
use commol_core::{Model, SimulationEngine};
use commol_difference::DifferenceEquations;
use commol_differential::DifferentialEquations;
use commol_inference::sensitivity::{
    Design, FiniteDifferences, SensitivityError, SensitivitySource, observation_scales,
};

/// A → B at the proportional rate `k`, from A = 1000, with the parameters
/// `extra` that no rate reads.
fn decay(engine: &str, settings: &str, extra: &[&str]) -> Model {
    let parameters: Vec<String> = std::iter::once(r#"{"id":"k","value":0.2}"#.to_string())
        .chain(
            extra
                .iter()
                .map(|id| format!(r#"{{"id":"{id}","value":1.0}}"#)),
        )
        .collect();
    let json = format!(
        r#"{{"name":"m","parameters":[{}],
        "engine":{{"type":"{engine}"{settings}}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"k",
        "stratified_rates":null,"per_compartment":null}}]}},
        "structure":{{"kind":"aggregate","total":1000,"bin_fractions":{{"A":1.0,"B":0.0}}}}}}"#,
        parameters.join(",")
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

fn ode_settings(rtol: f64) -> String {
    format!(r#","settings":{{"method":"tsit45","rtol":{rtol},"atol":1e-10}}"#)
}

fn factor(id: &str, min: f64, max: f64) -> CalibrationParameter {
    CalibrationParameter::with_type(id.to_string(), CalibrationParameterType::Factor, min, max)
}

/// A observed through the factor `f` at `positions`.
fn observed(positions: &[f64], values: &[f64]) -> Vec<ObservedDataPoint> {
    positions
        .iter()
        .zip(values)
        .map(|(&at, &value)| {
            ObservedDataPoint::with_factor(at, "A".to_string(), value, "f".to_string())
        })
        .collect()
}

fn problem<E: SimulationEngine>(
    engine: E,
    observed_data: Vec<ObservedDataPoint>,
    parameters: Vec<CalibrationParameter>,
) -> CalibrationProblem<E> {
    CalibrationProblem::new(
        engine,
        observed_data,
        parameters,
        vec![],
        LossConfig::SumSquaredError,
    )
    .expect("a valid problem")
}

fn difference(extra: &[&str]) -> DifferenceEquations {
    DifferenceEquations::try_from_model(&decay("DifferenceEquations", "", extra)).unwrap()
}

fn assert_within(actual: f64, expected: f64, error: f64, what: &str) {
    let difference = (actual - expected).abs();
    assert!(
        difference <= error + 1e-12 * expected.abs(),
        "{what}: got {actual}, expected {expected}, difference {difference} above the \
         estimated error {error}"
    );
}

#[test]
fn jacobian_of_decay_matches_the_closed_form() {
    // y_t = f · 1000 (1 − k)^t in discrete steps.
    let positions = [1.0, 2.0, 3.0];
    let problem = problem(
        difference(&[]),
        observed(&positions, &[400.0, 320.0, 256.0]),
        vec![
            CalibrationParameter::new("k".to_string(), 0.0, 1.0),
            factor("f", 0.1, 2.0),
        ],
    );
    let (k, f) = (0.2, 0.5);
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let sensitivities = source.jacobian(&[k, f]).unwrap();

    assert_eq!(sensitivities.jacobian.shape(), (3, 2));
    for (i, &t) in positions.iter().enumerate() {
        let d_k = -t * f * 1000.0 * (1.0 - k).powf(t - 1.0);
        let d_f = 1000.0 * (1.0 - k).powf(t);
        assert_within(
            sensitivities.jacobian[(i, 0)],
            d_k,
            sensitivities.errors[(i, 0)],
            &format!("∂y{t}/∂k"),
        );
        assert_within(
            sensitivities.jacobian[(i, 1)],
            d_f,
            sensitivities.errors[(i, 1)],
            &format!("∂y{t}/∂f"),
        );
        assert!((sensitivities.predictions[i] - f * d_f).abs() < 1e-9);
    }
}

#[test]
fn an_unread_parameter_gives_a_zero_column() {
    let problem = problem(
        difference(&["u"]),
        vec![
            ObservedDataPoint::new(1.0, "A".to_string(), 800.0),
            ObservedDataPoint::new(2.0, "A".to_string(), 640.0),
        ],
        vec![
            CalibrationParameter::new("k".to_string(), 0.0, 1.0),
            CalibrationParameter::new("u".to_string(), 0.0, 2.0),
        ],
    );
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let sensitivities = source.jacobian(&[0.2, 1.0]).unwrap();

    assert!(sensitivities.jacobian.column(1).iter().all(|&d| d == 0.0));
    assert!(sensitivities.jacobian.column(0).iter().all(|&d| d != 0.0));
}

#[test]
fn a_step_at_a_bound_stays_inside_the_bounds() {
    // Above k = 1 a step leaves A below zero, so a difference that crossed
    // the upper bound would fail. A_1 = 1000 (1 − k) and A_2 = 1000 (1 − k)²:
    // the one-sided difference of second order is exact for both.
    let problem = problem(
        difference(&[]),
        vec![
            ObservedDataPoint::new(1.0, "A".to_string(), 0.0),
            ObservedDataPoint::new(2.0, "A".to_string(), 0.0),
        ],
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
    );
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let sensitivities = source.jacobian(&[1.0]).unwrap();

    assert_within(
        sensitivities.jacobian[(0, 0)],
        -1000.0,
        sensitivities.errors[(0, 0)],
        "∂A1/∂k at k = 1",
    );
    assert_within(
        sensitivities.jacobian[(1, 0)],
        0.0,
        sensitivities.errors[(1, 0)],
        "∂A2/∂k at k = 1",
    );
}

#[test]
fn the_ode_step_follows_the_solver_tolerance() {
    let engine = DifferentialEquations::try_from_model(&decay(
        "DifferentialEquations",
        &ode_settings(1e-6),
        &[],
    ))
    .unwrap();
    assert_eq!(engine.output_accuracy(), 1e-6);
    let positions = [1.0, 2.0, 3.0];
    let problem = problem(
        engine,
        observed(&positions, &[400.0, 320.0, 256.0]),
        vec![
            CalibrationParameter::new("k".to_string(), 0.0, 1.0),
            factor("f", 0.1, 2.0),
        ],
    );
    let (k, f) = (0.2, 0.5);
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();

    // h_j = η^{1/3} · max(|θ_j|, w_j).
    let steps = source.steps(&[k, f]);
    assert!(
        (steps[0] - 1e-2 * 1.0).abs() < 1e-12,
        "step of k: {}",
        steps[0]
    );
    assert!(
        (steps[1] - 1e-2 * 1.9).abs() < 1e-12,
        "step of f: {}",
        steps[1]
    );

    // y_t = f · 1000 e^{−k t}.
    let sensitivities = source.jacobian(&[k, f]).unwrap();
    for (i, &t) in positions.iter().enumerate() {
        let d_f = 1000.0 * (-k * t).exp();
        assert_within(
            sensitivities.jacobian[(i, 0)],
            -t * f * d_f,
            sensitivities.errors[(i, 0)],
            &format!("∂y{t}/∂k"),
        );
        assert_within(
            sensitivities.jacobian[(i, 1)],
            d_f,
            sensitivities.errors[(i, 1)],
            &format!("∂y{t}/∂f"),
        );
    }
}

#[test]
fn an_infeasible_difference_names_the_parameter() {
    // k + h passes 1, where a step leaves A below zero; the bound at 2 lets
    // the central difference try it.
    let problem = problem(
        difference(&[]),
        vec![ObservedDataPoint::new(1.0, "A".to_string(), 0.0)],
        vec![CalibrationParameter::new("k".to_string(), 0.0, 2.0)],
    );
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let error = source.jacobian(&[0.999_999]).unwrap_err();

    assert!(
        matches!(&error, SensitivityError::Infeasible { parameter, .. } if parameter == "k"),
        "{error:?}"
    );
    assert!(error.to_string().contains("'k'"), "{error}");
}

#[test]
fn the_scaled_jacobian_does_not_depend_on_units() {
    // The same decay with k per day at days 7, 14, 21, and per week at
    // weeks 1, 2, 3.
    let settings = ode_settings(1e-10);
    let model = decay("DifferentialEquations", &settings, &[]);
    let days = problem(
        DifferentialEquations::try_from_model(&model).unwrap(),
        observed(&[7.0, 14.0, 21.0], &[200.0, 80.0, 30.0]),
        vec![
            CalibrationParameter::new("k".to_string(), 0.0, 1.0),
            factor("f", 0.1, 2.0),
        ],
    );
    let weeks = problem(
        DifferentialEquations::try_from_model(&model).unwrap(),
        observed(&[1.0, 2.0, 3.0], &[200.0, 80.0, 30.0]),
        vec![
            CalibrationParameter::new("k".to_string(), 0.0, 7.0),
            factor("f", 0.1, 2.0),
        ],
    );
    let per_day = FiniteDifferences::new(&days, Design::Observations)
        .unwrap()
        .scaled_jacobian(&[0.1, 0.5])
        .unwrap();
    let per_week = FiniteDifferences::new(&weeks, Design::Observations)
        .unwrap()
        .scaled_jacobian(&[0.7, 0.5])
        .unwrap();

    for i in 0..3 {
        for j in 0..2 {
            let error = per_day.errors[(i, j)] + per_week.errors[(i, j)];
            assert_within(
                per_week.jacobian[(i, j)],
                per_day.jacobian[(i, j)],
                error,
                &format!("S[{i}, {j}]"),
            );
        }
    }
}

#[test]
fn a_zero_series_has_scale_one() {
    let observed_data = vec![
        ObservedDataPoint::new(1.0, "A".to_string(), 3.0),
        ObservedDataPoint::new(2.0, "A".to_string(), 4.0),
        ObservedDataPoint::new(1.0, "B".to_string(), 0.0),
        ObservedDataPoint::new(2.0, "B".to_string(), 0.0),
    ];
    let rms = 12.5_f64.sqrt();

    assert_eq!(observation_scales(&observed_data), [rms, rms, 1.0, 1.0]);
}
