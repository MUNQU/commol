//! The summary of the replicates of a stochastic engine: the one quantile of
//! commol and the one check of its level.

use commol_calibration::{ReplicateSummary, Replicates};
use commol_statistics::statistics::{check_levels, quantiles};

fn summary_of_one_to_five(summary: ReplicateSummary) -> f64 {
    summary.summarize(&[4.0, 1.0, 5.0, 3.0, 2.0])
}

#[test]
fn a_quantile_of_replicates_is_the_one_quantile_of_commol() {
    // Hazen's positions (i − ½)/5: 0.25 is 3/4 of the way from 1 to 2, and
    // 0.1 is the midpoint of the smallest value.
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    let quarter = summary_of_one_to_five(ReplicateSummary::Quantile(0.25));
    let tenth = summary_of_one_to_five(ReplicateSummary::Quantile(0.1));
    let median = summary_of_one_to_five(ReplicateSummary::Median);
    assert!(close(quarter, 1.75), "{quarter}");
    assert!(close(tenth, 1.0), "{tenth}");
    assert!(close(median, 3.0), "{median}");

    let values = [1.0, 2.0, 3.0, 4.0, 5.0];
    let one = quantiles(&values, &[1.0; 5], &[0.25, 0.1, 0.5]).unwrap();
    assert_eq!(one, vec![quarter, tenth, median]);
}

#[test]
fn a_quantile_level_outside_0_and_1_is_rejected() {
    let error = Replicates::new(3, 1, ReplicateSummary::Quantile(1.5)).unwrap_err();
    assert_eq!(Err(error), check_levels(&[1.5]));
}
