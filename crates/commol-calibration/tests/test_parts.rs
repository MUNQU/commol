//! An observation splits its window into parts: the boundaries of the parts,
//! the means the operator reads for them, and the density and the tails of
//! their sum under each observation model.

use commol_calibration::{
    Argument, CalibrationParameter, CalibrationProblem, DerivativesError, DiscrepancyConfig,
    EvaluationError, MAX_COUNTS, Noise, ObservationModel, ObservedDataPoint, Sum, Tails,
    UnendedSum, Violation, Workspace,
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
fn each_part_has_its_share_of_the_background() {
    let without = split("B", 12.0, 12.0, 3, 0.0);
    let with = ObservedDataPoint {
        background: 6.0,
        ..without.clone()
    };
    let (gain, increments) = means(&problem(difference(), vec![without]).unwrap(), 0.1);
    let (total, parts) = means(&problem(difference(), vec![with.clone()]).unwrap(), 0.1);

    // A third of the background in each part, and the whole of it once.
    assert_eq!(with.part_background(), 2.0);
    assert_eq!(parts.len(), 3);
    for (part, increment) in parts.iter().zip(&increments) {
        assert_eq!(*part, increment + 2.0);
    }
    assert!(relative(total[0], gain[0] + 6.0) < 1e-12, "{total:?}");
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
    let cases: [(&[f64], f64); 7] = [
        (&[3.0, 10.0, 7.5, 0.5], 0.3),
        (&[3.0, 10.0, 7.5, 0.5], 1.0),
        (&[12.0, 0.0, 4.0], 2.0),
        (&[3.0, 10.0, 7.5, 0.5], 0.0),
        (&[3.0, 10.0, 7.5, 0.5], 1e-12),
        (&[3.0, 10.0, 7.5, 0.5], 1e-100),
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
            let actual = exact.sum_log_density(n as f64, parts).unwrap();
            assert!(
                relative(actual, *expected) < 1e-12,
                "φ = {phi}, n = {n}: {actual} against {expected}"
            );
        }
        if phi == 0.0 {
            for n in 0..=last {
                let poisson = Noise::Poisson.log_density(n as f64, mean);
                let actual = exact.sum_log_density(n as f64, parts).unwrap();
                assert!(relative(actual, poisson) < 1e-12, "{n}");
            }
        }
    }
    let exact = Noise::NegativeBinomial {
        overdispersion: 1e-3,
        sum: Some(Sum::Exact),
    };
    assert!(exact.sum_log_density(0.0, &[300.0, 300.0, 300.0]).unwrap() < -745.0);
    // Near φ = 0 the tilt to a total far above the mean is that of φ = 0.
    let limit = Noise::NegativeBinomial {
        overdispersion: 1e-100,
        sum: Some(Sum::Exact),
    };
    let density = limit.sum_log_density(1500.0, &[50.0, 50.0]).unwrap();
    let poisson = Noise::Poisson.log_density(1500.0, 100.0);
    assert!(relative(density, poisson) < 1e-12, "{density}");
}

#[test]
fn the_tails_of_a_sum() {
    let parts = [3.0, 10.0, 7.5, 0.5];
    let mean: f64 = parts.iter().sum();
    // Every reading but the exact one is that of its model of the total.
    let phi = 0.8;
    let squares: f64 = parts.iter().map(|mu| mu * mu).sum();
    let matched = Noise::NegativeBinomial {
        overdispersion: phi * squares / (mean * mean),
        sum: None,
    };
    let moments = Noise::NegativeBinomial {
        overdispersion: phi,
        sum: Some(Sum::Moments),
    };
    for y in [0.0, 4.0, 21.0, 60.0] {
        assert_eq!(
            moments.sum_tails(y, &parts).unwrap(),
            matched.tails(y, mean)
        );
        assert_eq!(
            Noise::Poisson.sum_tails(y, &parts).unwrap(),
            Noise::Poisson.tails(y, mean)
        );
        // N(M, K sd²), K = 4 parts: the sd doubles.
        assert_eq!(
            Noise::Normal { sd: 1.5 }.sum_tails(y, &parts).unwrap(),
            Noise::Normal { sd: 3.0 }.tails(y, mean)
        );
    }
    // One part is the model itself.
    assert_eq!(
        moments.sum_tails(4.0, &[5.0]).unwrap(),
        moments.tails(4.0, 5.0)
    );
}

/// The exact tails of the total `y` of a sum of parts of means `parts`.
fn exact_tails(phi: f64, y: f64, parts: &[f64]) -> Tails {
    let exact = Noise::NegativeBinomial {
        overdispersion: phi,
        sum: Some(Sum::Exact),
    };
    exact.sum_tails(y, parts).unwrap()
}

#[test]
fn an_exact_sum_far_above_its_mean_is_in_its_upper_tail() {
    // Totals so far above the mean that no number holds the mass from them
    // on, up to the last count of the exact reading.
    let outlier = Tails {
        below: 1.0,
        at: 0.0,
        above: 0.0,
    };
    let last = (MAX_COUNTS - 1) as f64;
    for (phi, y, parts) in [
        (0.0, 1500.0, &[50.0, 50.0][..]),
        (0.0, 2790.0, &[500.0, 500.0]),
        (0.0, 4000.0, &[500.0, 500.0]),
        (0.0, 99_999.0, &[50.0, 50.0]),
        (0.0, 99_999.0, &[5e3, 5e3]),
        (1e-100, 1500.0, &[50.0, 50.0]),
        (1e-9, 1500.0, &[50.0, 50.0]),
        (0.01, 99_999.0, &[50.0, 50.0]),
        (0.01, 99_999.0, &[500.0, 500.0]),
        (0.01, last, &[500.0, 500.0]),
        // Parts of mean 0, and of a mean near the smallest number.
        (0.0, 1500.0, &[0.0, 100.0, 0.0]),
        (0.5, 7.0, &[0.0, 0.0]),
        (0.0, 7.0, &[1e-310, 1e-310]),
        (0.5, 7.0, &[1e-310, 1e-310]),
    ] {
        let case = format!("φ = {phi}, y = {y}, {parts:?}");
        assert_eq!(exact_tails(phi, y, parts), outlier, "{case}");
    }
}

/// Asserts that `tails` split the mass 1, each in `[0, 1]`, and that each
/// is `expected` to the relative accuracy `accuracy`.
fn assert_tails(tails: Tails, expected: Tails, accuracy: f64, case: &str) {
    for (actual, expected) in [
        (tails.below, expected.below),
        (tails.at, expected.at),
        (tails.above, expected.above),
    ] {
        assert!((0.0..=1.0).contains(&actual), "{case}: {tails:?}");
        assert!(
            (actual - expected).abs() <= accuracy * expected,
            "{case}: {tails:?} against {expected:e}"
        );
    }
    let mass = tails.below + tails.at + tails.above;
    assert!(
        (mass - 1.0).abs() < 1e-15,
        "{case}: {tails:?} of mass {mass}"
    );
}

#[test]
fn the_tails_of_an_exact_sum_split_its_mass() {
    // Totals far below the mean.
    let floor = Tails {
        below: 0.0,
        at: 0.0,
        above: 1.0,
    };
    for (phi, y, parts) in [
        (0.0, 0.0, &[500.0, 500.0]),
        (0.0, 40.0, &[500.0, 500.0]),
        (0.0, 99_999.0, &[2e5, 2e5]),
        (1e-3, 10.0, &[5e5, 5e5]),
        (0.0, 99_999.0, &[1e100, 1e100]),
        (1e-100, 10.0, &[500.0, 500.0]),
    ] {
        let case = format!("φ = {phi}, y = {y}, {parts:?}");
        assert_eq!(exact_tails(phi, y, parts), floor, "{case}");
    }

    // The sum of K equal parts is NB(M, φ/K), whose tails are incomplete
    // beta functions (at φ = 0 those of the Poisson, incomplete gamma ones):
    // from far below the mean to far above it. These references, not the
    // exact tails, limit the accuracy: at M = 100, φ = 1e-5 and y = 30 they
    // err by 3e-12.
    for (mean, phi) in [
        (20.0f64, 0.0),
        (100.0, 0.0),
        (2e3, 0.0),
        (1e3, 1e-3),
        (100.0, 1e-100),
        (100.0, 1e-5),
        (100.0, 0.01),
        (2e3, 0.01),
        (100.0, 0.5),
        (2e3, 0.5),
        (100.0, 30.0),
        // All but 1 % of the mass is at 0, so P(Y > 0) is summed.
        (1.0, 1e4),
    ] {
        let parts = [mean / 4.0; 4];
        let total = Noise::NegativeBinomial {
            overdispersion: phi / 4.0,
            sum: None,
        };
        for share in [
            0.0, 0.3, 0.9, 1.0, 1.02, 1.1, 1.4, 2.0, 3.0, 6.0, 15.0, 60.0,
        ] {
            let y = (share * mean).round();
            let expected = total.tails(y, mean);
            // The smallest numbers hold fewer digits.
            if [expected.below, expected.at, expected.above]
                .iter()
                .any(|&tail| tail > 0.0 && tail < 1e-290)
            {
                continue;
            }
            let case = format!("M = {mean}, φ = {phi}, y = {y}");
            assert_tails(exact_tails(phi, y, &parts), expected, 1e-11, &case);
        }
    }

    // Far above a mean of heavy tail, where `P(Y > y)` is 1.1 %: the sum
    // above `y` would not end. The reference is NB(M, φ/K) to 60 digits.
    let heavy = Tails {
        below: 0.988_878_190_461_087_3,
        at: 4.286_569_937_261_549e-7,
        above: 0.011_121_380_881_919,
    };
    let tails = exact_tails(1e3, 8906.0, &[250.0; 4]);
    assert_tails(tails, heavy, 1e-12, "φ = 1000, M = 1000, y = 8906");

    // Parts of unequal means, against their convolution up to a count far
    // above the mean, to the accuracy of its log densities.
    for (parts, phi, last, accuracy) in [
        (&[3.0, 10.0, 7.5, 0.5][..], 0.3, 1500usize, 1e-12),
        (&[3.0, 10.0, 7.5, 0.5], 2.0, 1500, 1e-12),
        (&[90.0, 0.5, 0.0, 9.5], 0.0, 1100, 1e-11),
        (&[90.0, 0.5, 0.0, 9.5], 1e-12, 1100, 1e-11),
        (&[90.0, 0.5, 0.0, 9.5], 0.02, 1200, 1e-11),
        (&[30.0, 1e-3, 0.5], 0.5, 1100, 1e-11),
        (&[1.0, 1e-6], 20.0, 1000, 1e-11),
    ] {
        let probabilities: Vec<f64> = convolution(parts, phi, last)
            .into_iter()
            .map(f64::exp)
            .collect();
        for y in [0usize, 1, 4, 21, 60, 150, 300] {
            let expected = Tails {
                below: probabilities[..y].iter().sum(),
                at: probabilities[y],
                above: probabilities[y + 1..].iter().sum(),
            };
            let case = format!("φ = {phi}, y = {y}, {parts:?}");
            assert_tails(exact_tails(phi, y as f64, parts), expected, accuracy, &case);
        }
    }
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
        assert!(relative(moments(0.8).sum_log_density(y, &parts).unwrap(), expected) < 1e-14);
        // Equal means: NB(Kμ, φ/K).
        let equal = moments(0.8).sum_log_density(y, &[5.0; 4]).unwrap();
        assert!(relative(equal, single(0.2).log_density(y, 20.0)) < 1e-14);
    }
    // With no mean the share is 1: the sum has no mass above 0.
    assert_eq!(moments(0.8).sum_log_density(0.0, &[0.0, 0.0]).unwrap(), 0.0);
    assert_eq!(
        moments(0.8).sum_log_density(1.0, &[0.0, 0.0]).unwrap(),
        f64::NEG_INFINITY
    );
}

#[test]
fn poisson_and_normal_sums_are_exact() {
    let parts = [3.0, 10.0, 7.5, 0.5];
    let mean: f64 = parts.iter().sum();
    for y in [0.0, 5.0, 21.0, 40.0] {
        let expected = Noise::Poisson.log_density(y, mean);
        assert_eq!(Noise::Poisson.sum_log_density(y, &parts).unwrap(), expected);
        // The density of N(M, K sd²), K = 4 parts.
        let sd = 1.5;
        let variance = 4.0 * sd * sd;
        let normal = -0.5 * (2.0 * std::f64::consts::PI * variance).ln()
            - (y - mean) * (y - mean) / (2.0 * variance);
        let actual = Noise::Normal { sd }.sum_log_density(y, &parts).unwrap();
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
fn an_exact_sum_that_does_not_end_fails() {
    let exact = |phi: f64| Noise::NegativeBinomial {
        overdispersion: phi,
        sum: Some(Sum::Exact),
    };
    let unended = |mean: f64, phi: f64| UnendedSum {
        mean,
        overdispersion: phi,
    };
    // At φ = 10⁶ the mass spreads over counts far above the mean: the tail
    // above 0 and the information do not reach a negligible term within the
    // counts, and fail at once.
    let heavy = [2.0; 7];
    assert_eq!(exact(1e6).sum_information(&heavy), Err(unended(14.0, 1e6)));
    assert_eq!(exact(1e6).sum_tails(0.0, &heavy), Err(unended(14.0, 1e6)));
    // A mean at the counts leaves no count above it.
    let large = [2e5; 7];
    assert_eq!(
        exact(0.01).sum_information(&large),
        Err(unended(1.4e6, 0.01))
    );
    // A total at the counts.
    let y = MAX_COUNTS as f64;
    let parts = [1.0, 2.0];
    assert_eq!(
        exact(0.5).sum_log_density(y, &parts),
        Err(unended(3.0, 0.5))
    );
    assert_eq!(exact(0.5).sum_scores(y, &parts), Err(unended(3.0, 0.5)));
    assert_eq!(exact(0.5).sum_tails(y, &parts), Err(unended(3.0, 0.5)));
    // The moments read the same sums.
    let moments = Noise::NegativeBinomial {
        overdispersion: 1e6,
        sum: Some(Sum::Moments),
    };
    assert!(moments.sum_tails(0.0, &heavy).is_ok());
    assert!(moments.sum_log_density(y, &parts).is_ok());

    // Through a problem, the error names the observation, its total, φ and
    // the counts, and points to the moments; it is a failure, which stops a
    // fit, and the reason the terms have no derivatives.
    let model = |value: f64| {
        CalibrationProblem::new(
            difference(),
            vec![split("B", 5.0, 5.0, 5, value)],
            vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
            None,
            vec![],
            DiscrepancyConfig::Likelihood(vec![(
                "B".to_string(),
                ObservationModel::NegativeBinomial {
                    overdispersion: Argument::Value(1e6),
                    sum: Some(Sum::Exact),
                },
            )]),
        )
        .unwrap()
    };
    let says_why = |message: &str, total: &str| {
        for part in [
            &format!("Observation 'B' at 5 has the total {total} in 5 parts")[..],
            "sum=\"exact\"",
            "the overdispersion 1000000",
            &format!("does not end within {MAX_COUNTS} counts"),
            "Read the sum with sum=\"moments\"",
        ] {
            assert!(message.contains(part), "{part:?} in {message}");
        }
    };
    let problem = model(200.0);
    assert!(problem.evaluate(&[0.1], &mut Workspace::default()).is_ok());
    let (predictions, parts) = means(&problem, 0.1);
    let Err(DerivativesError::Failed(error)) =
        problem.term_derivatives(&[0.1], &predictions, &parts)
    else {
        panic!("an information that does not end fails");
    };
    says_why(&error, "200");
    let mean: f64 = parts.iter().sum();
    assert!(
        error.contains(&format!("the mean {mean} of the total")),
        "{error}"
    );

    let problem = model(y);
    let total = MAX_COUNTS.to_string();
    let Err(EvaluationError::Failed(error)) = problem.evaluate(&[0.1], &mut Workspace::default())
    else {
        panic!("a total at the counts fails");
    };
    says_why(&error, &total);
    let Err(EvaluationError::Failed(error)) =
        problem.pointwise_log_likelihood(&[0.1], &mut Workspace::default())
    else {
        panic!("a total at the counts fails");
    };
    says_why(&error, &total);
    let tails = problem
        .discrepancy()
        .likelihood()
        .unwrap()
        .tails(problem.observed_data(), &[0.1], &parts)
        .unwrap_err();
    says_why(&tails, &total);
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
        let information = nb(0.5, sum).sum_information(&[0.0, 0.0]).unwrap();
        assert_eq!(
            information,
            vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "{sum:?}"
        );
        assert_eq!(
            nb(0.5, sum).sum_scores(0.0, &[0.0, 0.0]).unwrap(),
            vec![-1.0, -1.0, 0.0]
        );

        // At φ = 0, and at its limit, the sum is Poisson(M): 1/M for each
        // pair of means, and for φ the variance of its score, (Σμ²)²/(2M²).
        let parts = [1.0, 2.0];
        for phi in [0.0, 1e-100] {
            let information = nb(phi, sum).sum_information(&parts).unwrap();
            for entry in [0, 1, 3, 4] {
                assert!(
                    relative(information[entry], 1.0 / 3.0) < 1e-9,
                    "{sum:?} {phi} {information:?}"
                );
            }
            assert!(
                relative(information[8], 25.0 / 18.0) < 1e-9,
                "{sum:?} {phi} {information:?}"
            );
        }
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

#[test]
fn an_exact_sum_of_a_large_total_is_cheap() {
    // Seven equal parts of a total of mean 10⁵: their sum is NB(M, φ/7). The
    // exact reading gives its density, tails, scores and information in a
    // time that grows about linearly with the total.
    let (k, phi, y) = (7usize, 0.1, 100_000.0);
    let parts = vec![y / k as f64; k];
    let mean: f64 = parts.iter().sum();
    let exact = Noise::NegativeBinomial {
        overdispersion: phi,
        sum: Some(Sum::Exact),
    };
    let started = std::time::Instant::now();
    let density = exact.sum_log_density(y, &parts).unwrap();
    let tails = [y, 2.0 * y].map(|y| exact.sum_tails(y, &parts).unwrap());
    let scores = exact.sum_scores(y, &parts).unwrap();
    let information = exact.sum_information(&parts).unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    // About 0.07 s in a release build and 0.8 s in a debug one; the limit
    // leaves room for a loaded machine.
    assert!(elapsed < 20.0, "{elapsed} s");
    eprintln!("an exact sum of a total of 10⁵: {elapsed:.3} s");

    // NB(M, φ/7) at 40 digits: log p(y), and (P(Y < y), p(y), P(Y > y)) at
    // y and 2y, the upper tail summed far above the mean. The terms carry
    // log p(0) ≈ −508 apart, which holds log p(y) to about 1e-12.
    assert!(
        relative(density, -10.309_156_723_382_754) < 1e-12,
        "{density}"
    );
    let expected = [
        (
            0.515_878_799_712_644_2,
            3.332_653_193_603_074_6e-5,
            0.484_087_873_755_419_8,
        ),
        (
            0.999_999_999_977_965_6,
            7.918_429_551_339_245e-15,
            2.202_648_420_775_554_5e-11,
        ),
    ];
    for (tails, (below, at, above)) in tails.iter().zip(expected) {
        for (actual, expected) in [(tails.below, below), (tails.at, at), (tails.above, above)] {
            assert!(
                (actual - expected).abs() <= 1e-11 * expected,
                "{tails:?} against {expected:e}"
            );
        }
    }
    // Moving every part by the same amount moves M by K times it, and φ/K
    // is the overdispersion of the total: the scores and the information
    // are those of NB(M, φ/K), as far as its own reading of ln Γ allows.
    let total = Noise::NegativeBinomial {
        overdispersion: phi / k as f64,
        sum: None,
    };
    let [score_mean, score_phi] = total.scores(y, mean)[..] else {
        panic!("a negative binomial has two scores");
    };
    for score in &scores[..k] {
        assert!(relative(*score, score_mean) < 1e-10, "{scores:?}");
    }
    assert!(
        relative(scores[k], score_phi / k as f64) < 1e-10,
        "{scores:?}"
    );
    let expected = total.information(mean);
    let size = k + 1;
    let means: f64 = (0..k)
        .flat_map(|a| (0..k).map(move |b| a * size + b))
        .map(|entry| information[entry])
        .sum();
    let cross: f64 = (0..k).map(|a| information[a * size + k]).sum();
    let close = |actual: f64, expected: f64| (actual - expected).abs() <= 1e-9 * expected.abs();
    assert!(
        close(means, (k * k) as f64 * expected[0]),
        "{information:?}"
    );
    let dispersion = information[size * size - 1];
    assert!(
        close(dispersion, expected[3] / (k * k) as f64),
        "{information:?} {expected:?}"
    );
    // The mean and φ of a negative binomial are orthogonal.
    assert_eq!(expected[1], 0.0);
    assert!(
        cross.abs() <= 1e-12 * (means * dispersion).sqrt(),
        "{information:?}"
    );
}
