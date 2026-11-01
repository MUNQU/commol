//! The series of a set of observations, the schedule a run reports at, and
//! the mean over replicates.

use commol_calibration::{
    CalibrationParameter, CalibrationParameterType, ObservationOperator, ObservedDataPoint,
    SeriesIndex, replicate_mean, schedule_of,
};

fn point(at: f64, label: &str) -> ObservedDataPoint {
    ObservedDataPoint::new(at, label.to_string(), 0.0)
}

#[test]
fn a_series_index_keeps_the_order_of_first_appearance() {
    let observations = [
        point(2.0, "B"),
        point(1.0, "A"),
        point(1.0, "B"),
        point(3.0, "A"),
    ];

    let index = SeriesIndex::new(&observations);

    assert_eq!(index.labels(), ["B", "A"]);
    assert_eq!(index.indices(), [vec![2, 0], vec![1, 3]]);
    assert_eq!(index.series_of(), [0, 1, 0, 1]);
    assert_eq!(index.series("A"), Some(1));
    assert_eq!(index.series("C"), None);
    assert_eq!(index.len(), 2);
}

#[test]
fn a_series_index_of_labels_keeps_the_order_of_the_labels() {
    let labels = ["y", "x", "y"].map(String::from);

    let index = SeriesIndex::of_labels(&labels);

    assert_eq!(index.labels(), ["y", "x"]);
    assert_eq!(index.indices(), [vec![0, 2], vec![1]]);
    assert!(!index.is_empty());
    assert!(SeriesIndex::of_labels(&[]).is_empty());
}

#[test]
fn the_schedule_holds_each_position_and_window_start_once() {
    let mut first = point(7.0, "A");
    first.window = Some(7.0);
    let mut second = point(14.0, "A");
    second.window = Some(7.0);

    assert_eq!(schedule_of(&[first.clone(), second], []), [0.0, 7.0, 14.0]);
    assert_eq!(schedule_of(&[first], [21.0, 0.0]), [0.0, 7.0, 21.0]);
}

#[test]
fn the_mean_over_replicates_is_the_mean_of_each_observation() {
    let predictions = [1.0, 10.0, 3.0, 30.0];

    assert_eq!(replicate_mean(predictions.chunks(2), 2), [2.0, 20.0]);
    assert!(replicate_mean(std::iter::empty::<&[f64]>(), 0).is_empty());
}

#[test]
fn weighted_sum_over_a_window() {
    let mut observation = ObservedDataPoint::of_sum(
        2.0,
        "seen".to_string(),
        vec!["A".to_string(), "B".to_string()],
        0.0,
    );
    observation.coefficients = Some(vec![1.0, 0.6]);
    observation.window = Some(1.0);
    observation.scale = 0.5;
    observation.factor_id = Some("f".to_string());
    let parameters = [CalibrationParameter::with_type(
        "f".to_string(),
        CalibrationParameterType::Factor,
        0.1,
        10.0,
    )];
    let names = ["A".to_string(), "B".to_string()];
    let operator =
        ObservationOperator::new(&[observation], &parameters, &names, &[1.0, 2.0]).unwrap();
    let rows = [vec![10.0, 20.0], vec![30.0, 70.0]];
    let mut out = [0.0];

    operator.apply(&rows, &[4.0], &mut out);

    // A + 0.6 B grows from 22 to 72; the gain 50 times the scale 0.5 times
    // the factor 4.
    assert!((out[0] - 100.0).abs() < 1e-12, "{}", out[0]);
}

#[test]
fn an_operator_refuses_invalid_coefficients() {
    let names = ["A".to_string(), "B".to_string()];
    let cases: [(Vec<f64>, &str); 4] = [
        (vec![1.0], "has 1 coefficients for 2 outputs"),
        (vec![1.0, -0.5], "has the coefficient -0.5 for output 'B'"),
        (
            vec![f64::NAN, 1.0],
            "has the coefficient NaN for output 'A'",
        ),
        (vec![0.0, 0.0], "has every coefficient 0"),
    ];
    for (coefficients, message) in cases {
        let mut observation =
            ObservedDataPoint::of_sum(1.0, "seen".to_string(), names.to_vec(), 0.0);
        observation.coefficients = Some(coefficients);

        let error = ObservationOperator::new(&[observation], &[], &names, &[1.0]).unwrap_err();

        assert!(error.starts_with("Observation 'seen' "), "{error}");
        assert!(error.contains(message), "{error}");
    }
}
