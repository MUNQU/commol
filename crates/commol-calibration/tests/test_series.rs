//! The series of a set of observations, the schedule a run reports at, and
//! the mean over replicates.

use commol_calibration::{ObservedDataPoint, SeriesIndex, replicate_mean, schedule_of};

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
