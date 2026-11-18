//! An observation splits its window into parts: the boundaries of the parts,
//! the means the operator reads for them, and the density of their sum
//! under each observation model.

use commol_calibration::{
    Argument, CalibrationParameter, CalibrationProblem, DiscrepancyConfig, EvaluationError, Noise,
    ObservationModel, ObservedDataPoint, Sum, Violation, Workspace,
};
use commol_core::{Model, SimulationEngine};
use commol_difference::DifferenceEquations;
use commol_differential::DifferentialEquations;

/// A → B → C at the rates `k` and `k2`, from A = 1000, run by the given
/// engine: B rises, then falls.
fn chain_model(engine: &str) -> Model {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":0.1}},{{"id":"k2","value":0.2}}],
        "engine":{{"type":"{engine}"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}},
        {{"id":"C","name":"C"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"k",
        "stratified_rates":null,"per_compartment":null}},
        {{"id":"bc","source":["B"],"target":["C"],"rate":"k2",
        "stratified_rates":null,"per_compartment":null}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":1000.0,"B":0.0,"C":0.0}}}}}}"#
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

fn difference() -> DifferenceEquations {
    DifferenceEquations::try_from_model(&chain_model("DifferenceEquations")).unwrap()
}

fn differential() -> DifferentialEquations {
    DifferentialEquations::try_from_model(&chain_model("DifferentialEquations")).unwrap()
}

/// An observation of `output` at `at` over `window`, in `parts` parts.
fn split(output: &str, at: f64, window: f64, parts: usize, value: f64) -> ObservedDataPoint {
    ObservedDataPoint {
        parts,
        ..ObservedDataPoint::with_window(at, output.to_string(), value, window)
    }
}

fn poisson(labels: &[&str]) -> DiscrepancyConfig {
    DiscrepancyConfig::Likelihood(
        labels
            .iter()
            .map(|label| (label.to_string(), ObservationModel::Poisson))
            .collect(),
    )
}

fn problem<E: SimulationEngine>(
    engine: E,
    observed: Vec<ObservedDataPoint>,
) -> Result<CalibrationProblem<E>, String> {
    let labels: Vec<String> = observed.iter().map(|obs| obs.label.clone()).collect();
    let mut labels: Vec<&str> = labels.iter().map(String::as_str).collect();
    labels.dedup();
    CalibrationProblem::new(
        engine,
        observed,
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        None,
        vec![],
        poisson(&labels),
    )
}

/// The predictions and the means of the parts of `problem` at `k`.
fn means<E: SimulationEngine>(problem: &CalibrationProblem<E>, k: f64) -> (Vec<f64>, Vec<f64>) {
    let mut workspace = Workspace::default();
    problem.predict_once(&[k], None, &mut workspace).unwrap();
    (workspace.predictions, workspace.parts)
}

fn relative(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1.0)
}

#[test]
fn boundaries_come_from_the_start_and_the_end() {
    // The window starts at 0 within the tolerance of the axis.
    let observation = split("B", 1000.0, 1000.0 + 1e-7, 10, 0.0);
    let schedule = problem(difference(), vec![observation])
        .unwrap()
        .forward()
        .schedule()
        .to_vec();
    let steps: Vec<f64> = (0..=10).map(|k| f64::from(k) * 100.0).collect();
    assert_eq!(schedule, steps);

    let whole = problem(differential(), vec![split("B", 0.7, 0.7, 1, 0.0)]).unwrap();
    let parted = problem(differential(), vec![split("B", 0.7, 0.7, 7, 0.0)]).unwrap();
    assert_eq!(parted.forward().schedule().len(), 8);
    let (window, _) = means(&whole, 0.1);
    let (total, parts) = means(&parted, 0.1);
    assert_eq!(parts.len(), 7);
    // The parts share their boundary rows: they telescope to the increment
    // of the rows of the run at the start and the end of the window.
    let rows = parted
        .forward()
        .run_once_at(&[0.1], parted.forward().schedule(), None)
        .unwrap();
    let increment = rows[7][1] - rows[0][1];
    assert!(relative(parts.iter().sum(), increment) < 1e-12);
    assert!(relative(total[0], window[0]) < 1e-9, "{total:?} {window:?}");
}

#[test]
fn a_boundary_the_engine_refuses_is_named() {
    let error = problem(difference(), vec![split("B", 10.0, 10.0, 4, 0.0)])
        .err()
        .expect("a boundary at 2.5 is refused");
    assert!(
        error.contains(
            "Observation 'B' at 10 with the window 10 and parts=4 has a boundary of its parts \
             at 2.5"
        ),
        "{error}"
    );
    assert!(error.contains("2.5 is not one"), "{error}");
}

#[test]
fn the_parts_sum_to_the_window() {
    let observed = vec![
        split("B", 20.0, 20.0, 4, 0.0),
        ObservedDataPoint::new(3.0, "A".to_string(), 0.0),
        split("C", 12.0, 6.0, 3, 0.0),
    ];
    let problem = problem(difference(), observed).unwrap();
    let (predictions, parts) = means(&problem, 0.1);

    assert_eq!(parts.len(), 4 + 1 + 3);
    assert_eq!(predictions[0], parts[..4].iter().sum::<f64>());
    assert_eq!(predictions[1], parts[4]);
    assert_eq!(predictions[2], parts[5..].iter().sum::<f64>());
    // The rows of the run at the boundaries.
    let rows = problem
        .forward()
        .run_once_at(&[0.1], &[0.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0], None)
        .unwrap();
    let b = |row: usize| rows[row][1];
    let c = |row: usize| rows[row][2];
    let expected = [b(1) - b(0), b(4) - b(1), b(6) - b(4), b(7) - b(6)];
    for (part, expected) in parts[..4].iter().zip(expected) {
        assert!(relative(*part, expected) < 1e-12, "{part} {expected}");
    }
    let expected = [c(3) - c(2), c(4) - c(3), c(5) - c(4)];
    for (part, expected) in parts[5..].iter().zip(expected) {
        assert!(relative(*part, expected) < 1e-12, "{part} {expected}");
    }
}

#[test]
fn a_negative_part_is_outside_the_support() {
    // B rises and then falls: the last part of its window has a negative
    // increment, though the total is positive.
    let problem = problem(difference(), vec![split("B", 40.0, 40.0, 2, 3.0)]).unwrap();
    let (predictions, parts) = means(&problem, 0.1);
    assert!(predictions[0] > 0.0 && parts[1] < 0.0, "{parts:?}");

    let error = problem
        .evaluate(&[0.1], &mut Workspace::default())
        .unwrap_err();
    assert_eq!(
        error,
        EvaluationError::Violation(Violation::MeanOutsideModel { amount: -parts[1] })
    );
}

/// The log of `Σ exp(values)`.
fn log_sum(values: &[f64]) -> f64 {
    let largest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if largest == f64::NEG_INFINITY {
        return largest;
    }
    largest + values.iter().map(|v| (v - largest).exp()).sum::<f64>().ln()
}

/// `log p(n)` of the sum of independent negative binomials of means `parts`
/// and overdispersion `phi`, for `n` up to `last`, by direct convolution of
/// their log probabilities.
fn convolution(parts: &[f64], phi: f64, last: usize) -> Vec<f64> {
    let single = Noise::NegativeBinomial {
        overdispersion: phi,
        sum: None,
    };
    let mut total: Vec<f64> = (0..=last)
        .map(|n| if n == 0 { 0.0 } else { f64::NEG_INFINITY })
        .collect();
    for &mean in parts {
        let part: Vec<f64> = (0..=last)
            .map(|n| single.log_density(n as f64, mean))
            .collect();
        total = (0..=last)
            .map(|n| {
                let terms: Vec<f64> = (0..=n).map(|m| total[m] + part[n - m]).collect();
                log_sum(&terms)
            })
            .collect();
    }
    total
}

#[test]
fn the_exact_sum_is_the_convolution_of_its_parts() {
    let cases: [(&[f64], f64); 6] = [
        (&[3.0, 10.0, 7.5, 0.5], 0.3),
        (&[3.0, 10.0, 7.5, 0.5], 1.0),
        (&[12.0, 0.0, 4.0], 2.0),
        (&[3.0, 10.0, 7.5, 0.5], 0.0),
        (&[3.0, 10.0, 7.5, 0.5], 1e-12),
        // log p(0) is about −787: p(0) is below the smallest number.
        (&[300.0, 300.0, 300.0], 1e-3),
    ];
    for (parts, phi) in cases {
        let mean: f64 = parts.iter().sum();
        let last = (1.5 * mean).ceil() as usize;
        let exact = Noise::NegativeBinomial {
            overdispersion: phi,
            sum: Some(Sum::Exact),
        };
        let expected = convolution(parts, phi, last);
        for (n, expected) in expected.iter().enumerate() {
            let actual = exact.sum_log_density(n as f64, parts);
            assert!(
                relative(actual, *expected) < 1e-12,
                "φ = {phi}, n = {n}: {actual} against {expected}"
            );
        }
        if phi == 0.0 {
            for n in 0..=last {
                let poisson = Noise::Poisson.log_density(n as f64, mean);
                let actual = exact.sum_log_density(n as f64, parts);
                assert!(relative(actual, poisson) < 1e-12, "{n}");
            }
        }
    }
    let exact = Noise::NegativeBinomial {
        overdispersion: 1e-3,
        sum: Some(Sum::Exact),
    };
    assert!(exact.sum_log_density(0.0, &[300.0, 300.0, 300.0]) < -745.0);
}

#[test]
fn the_moments_of_a_sum() {
    let moments = |phi: f64| Noise::NegativeBinomial {
        overdispersion: phi,
        sum: Some(Sum::Moments),
    };
    let single = |phi: f64| Noise::NegativeBinomial {
        overdispersion: phi,
        sum: None,
    };
    let parts = [3.0, 10.0, 7.5, 0.5];
    let (mean, squares) = (21.0, 9.0 + 100.0 + 56.25 + 0.25);
    for y in [0.0, 4.0, 21.0, 60.0] {
        let expected = single(0.8 * squares / (mean * mean)).log_density(y, mean);
        assert!(relative(moments(0.8).sum_log_density(y, &parts), expected) < 1e-14);
        // Equal means: NB(Kμ, φ/K).
        let equal = moments(0.8).sum_log_density(y, &[5.0; 4]);
        assert!(relative(equal, single(0.2).log_density(y, 20.0)) < 1e-14);
    }
    // With no mean the share is 1: the sum has no mass above 0.
    assert_eq!(moments(0.8).sum_log_density(0.0, &[0.0, 0.0]), 0.0);
    assert_eq!(
        moments(0.8).sum_log_density(1.0, &[0.0, 0.0]),
        f64::NEG_INFINITY
    );
}

#[test]
fn poisson_and_normal_sums_are_exact() {
    let parts = [3.0, 10.0, 7.5, 0.5];
    let mean: f64 = parts.iter().sum();
    for y in [0.0, 5.0, 21.0, 40.0] {
        let expected = Noise::Poisson.log_density(y, mean);
        assert_eq!(Noise::Poisson.sum_log_density(y, &parts), expected);
        // The density of N(M, K sd²), K = 4 parts.
        let sd = 1.5;
        let variance = 4.0 * sd * sd;
        let normal = -0.5 * (2.0 * std::f64::consts::PI * variance).ln()
            - (y - mean) * (y - mean) / (2.0 * variance);
        let actual = Noise::Normal { sd }.sum_log_density(y, &parts);
        assert!(relative(actual, normal) < 1e-14, "{actual} {normal}");
    }

    // Through a problem: its pointwise log-likelihood reads the sum.
    let observed = vec![split("B", 20.0, 20.0, 4, 300.0)];
    let normal = DiscrepancyConfig::Likelihood(vec![(
        "B".to_string(),
        ObservationModel::Normal {
            sd: Argument::Value(2.0),
        },
    )]);
    let problem = CalibrationProblem::new(
        difference(),
        observed,
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        None,
        vec![],
        normal,
    )
    .unwrap();
    let (predictions, _) = means(&problem, 0.1);
    let pointwise = problem
        .pointwise_log_likelihood(&[0.1], &mut Workspace::default())
        .unwrap();
    let expected = Noise::Normal { sd: 4.0 }.log_density(300.0, predictions[0]);
    assert!(relative(pointwise[0], expected) < 1e-12);
}

#[test]
fn a_sum_of_no_mean_or_no_dispersion_has_finite_information() {
    let nb = |phi: f64, sum: Sum| Noise::NegativeBinomial {
        overdispersion: phi,
        sum: Some(sum),
    };
    for sum in [Sum::Exact, Sum::Moments] {
        // Parts of mean 0: the mass at 0, whose scores are −1 for each mean
        // and 0 for φ.
        let information = nb(0.5, sum).sum_information(&[0.0, 0.0]);
        assert_eq!(
            information,
            vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "{sum:?}"
        );
        assert_eq!(
            nb(0.5, sum).sum_scores(0.0, &[0.0, 0.0]),
            vec![-1.0, -1.0, 0.0]
        );

        // At φ = 0 the sum is Poisson(M): 1/M for each pair of means, and
        // for φ the variance of its score, (Σμ²)²/(2M²).
        let parts = [1.0, 2.0];
        let information = nb(0.0, sum).sum_information(&parts);
        for entry in [0, 1, 3, 4] {
            assert!(
                relative(information[entry], 1.0 / 3.0) < 1e-9,
                "{sum:?} {information:?}"
            );
        }
        assert!(
            relative(information[8], 25.0 / 18.0) < 1e-9,
            "{sum:?} {information:?}"
        );
    }
}

#[test]
fn parts_need_a_window_in_the_engine() {
    let observation = ObservedDataPoint {
        parts: 3,
        ..ObservedDataPoint::new(6.0, "B".to_string(), 0.0)
    };
    let error = problem(difference(), vec![observation])
        .err()
        .expect("parts without a window are refused");
    assert!(
        error.contains("Observation 'B' at 6 has parts=3 and no window"),
        "{error}"
    );
}
