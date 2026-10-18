//! Scores of a predictive against observations: empirical coverage, the
//! interval score, the weighted interval score, the CRPS of a weighted
//! sample, and PIT values.
//!
//! Every score takes the draws of a predictive at a set of observations
//! ([`ObservationDraws`]) and the observed values, and returns one value per
//! observation; [`by_series`] averages them per series. Which observations
//! to score, for example the ones held out of a calibration, is the caller's
//! choice.

use commol_calibration::weighted_crps;
use commol_random::Generator;

use crate::predictive::ObservationDraws;
use crate::statistics;

fn column(draws: &ObservationDraws, observation: usize) -> Vec<f64> {
    draws.values.iter().map(|row| row[observation]).collect()
}

fn check(draws: &ObservationDraws, observed: &[f64]) -> Result<(), String> {
    match draws.values.first() {
        Some(row) if row.len() == observed.len() => Ok(()),
        Some(row) => Err(format!(
            "The draws hold {} observations; {} observed values were given",
            row.len(),
            observed.len()
        )),
        None => Err("A score needs at least one draw".to_string()),
    }
}

/// The central interval of `confidence_level` of each observation.
fn intervals(
    draws: &ObservationDraws,
    observed: &[f64],
    confidence_level: f64,
) -> Result<Vec<(f64, f64)>, String> {
    check(draws, observed)?;
    let levels = statistics::interval_levels(confidence_level)?;
    Ok((0..observed.len())
        .map(|i| {
            let q = statistics::quantiles(&column(draws, i), &draws.weights, &levels);
            (q[0], q[1])
        })
        .collect())
}

/// Whether each observation lies inside the central interval of
/// `confidence_level`: 1 or 0. Their mean is the empirical coverage.
///
/// # Errors
///
/// Returns a message when the shapes differ or the level is not strictly
/// between 0 and 1.
pub fn coverage(
    draws: &ObservationDraws,
    observed: &[f64],
    confidence_level: f64,
) -> Result<Vec<f64>, String> {
    Ok(intervals(draws, observed, confidence_level)?
        .into_iter()
        .zip(observed)
        .map(|((lower, upper), &y)| f64::from(u8::from(lower <= y && y <= upper)))
        .collect())
}

fn interval_score_of(lower: f64, upper: f64, alpha: f64, y: f64) -> f64 {
    (upper - lower) + 2.0 / alpha * (lower - y).max(0.0) + 2.0 / alpha * (y - upper).max(0.0)
}

/// The interval score of the central interval of `confidence_level`, with
/// `α = 1 − c`: `(u − l) + (2/α)(l − y)₊ + (2/α)(y − u)₊`. Lower is better.
///
/// # Errors
///
/// As [`coverage`].
pub fn interval_score(
    draws: &ObservationDraws,
    observed: &[f64],
    confidence_level: f64,
) -> Result<Vec<f64>, String> {
    let alpha = 1.0 - confidence_level;
    Ok(intervals(draws, observed, confidence_level)?
        .into_iter()
        .zip(observed)
        .map(|((lower, upper), &y)| interval_score_of(lower, upper, alpha, y))
        .collect())
}

/// The weighted interval score of Bracher et al. (2021) over the central
/// intervals of `confidence_levels`:
/// `(½|y − m| + Σₖ (αₖ/2) ISₐₖ) / (K + ½)`, with `m` the median and
/// `αₖ = 1 − cₖ`. Lower is better.
///
/// # Errors
///
/// As [`coverage`], for any of the levels.
pub fn weighted_interval_score(
    draws: &ObservationDraws,
    observed: &[f64],
    confidence_levels: &[f64],
) -> Result<Vec<f64>, String> {
    check(draws, observed)?;
    let mut levels = vec![0.5];
    for &c in confidence_levels {
        levels.extend(statistics::interval_levels(c)?);
    }
    let k = confidence_levels.len() as f64;
    Ok(observed
        .iter()
        .enumerate()
        .map(|(i, &y)| {
            let q = statistics::quantiles(&column(draws, i), &draws.weights, &levels);
            let mut total = 0.5 * (y - q[0]).abs();
            for (index, &c) in confidence_levels.iter().enumerate() {
                let alpha = 1.0 - c;
                let (lower, upper) = (q[1 + 2 * index], q[2 + 2 * index]);
                total += alpha / 2.0 * interval_score_of(lower, upper, alpha, y);
            }
            total / (k + 0.5)
        })
        .collect())
}

/// The continuous ranked probability score of each observation under the
/// weighted draws ([`weighted_crps`]). Lower is better.
///
/// # Errors
///
/// Returns a message when the shapes differ.
pub fn crps(draws: &ObservationDraws, observed: &[f64]) -> Result<Vec<f64>, String> {
    check(draws, observed)?;
    Ok(observed
        .iter()
        .enumerate()
        .map(|(i, &y)| weighted_crps(&column(draws, i), &draws.weights, y))
        .collect())
}

/// The probability integral transform of each observation: `F(y)` of the
/// weighted draws, randomized between `F(y⁻)` and `F(y)` where draws tie
/// with `y` (so a discrete predictive gives uniform values too), with the
/// generator seeded with `seed`. Calibrated predictions give uniform values.
///
/// # Errors
///
/// Returns a message when the shapes differ.
pub fn pit(draws: &ObservationDraws, observed: &[f64], seed: u64) -> Result<Vec<f64>, String> {
    check(draws, observed)?;
    let total: f64 = draws.weights.iter().sum();
    let mut generator = Generator::from_seed(seed);
    Ok(observed
        .iter()
        .enumerate()
        .map(|(i, &y)| {
            let (mut below, mut at) = (0.0, 0.0);
            for (row, w) in draws.values.iter().zip(&draws.weights) {
                if row[i] < y {
                    below += w;
                } else if row[i] == y {
                    at += w;
                }
            }
            (below + generator.uniform() * at) / total
        })
        .collect())
}

/// The mean of `values` per series, named by `labels`, in the order each
/// series first appears.
///
/// # Panics
///
/// Panics when `labels` and `values` differ in length.
pub fn by_series(labels: &[String], values: &[f64]) -> Vec<(String, f64)> {
    assert_eq!(labels.len(), values.len(), "one label per value");
    let mut series: Vec<(String, f64, usize)> = Vec::new();
    for (label, &value) in labels.iter().zip(values) {
        match series.iter_mut().find(|(name, _, _)| name == label) {
            Some(entry) => {
                entry.1 += value;
                entry.2 += 1;
            }
            None => series.push((label.clone(), value, 1)),
        }
    }
    series
        .into_iter()
        .map(|(name, sum, count)| (name, sum / count as f64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Draws 0, 1, …, 99 of one observation, with equal weights.
    fn uniform_draws() -> ObservationDraws {
        ObservationDraws {
            values: (0..100).map(|i| vec![f64::from(i)]).collect(),
            weights: vec![0.01; 100],
        }
    }

    #[test]
    fn coverage_counts_an_observation_inside_its_interval() {
        let draws = uniform_draws();
        assert_eq!(coverage(&draws, &[50.0], 0.9).unwrap(), vec![1.0]);
        assert_eq!(coverage(&draws, &[99.0], 0.9).unwrap(), vec![0.0]);
    }

    #[test]
    fn the_interval_score_adds_the_penalty_of_a_miss() {
        let draws = uniform_draws();
        // Hazen 5 % and 95 % of 0..99: 4.5 and 94.5.
        let inside = interval_score(&draws, &[50.0], 0.9).unwrap()[0];
        assert!((inside - 90.0).abs() < 1e-9, "{inside}");
        let above = interval_score(&draws, &[104.5], 0.9).unwrap()[0];
        assert!((above - (90.0 + 2.0 / 0.1 * 10.0)).abs() < 1e-9, "{above}");
    }

    #[test]
    fn the_weighted_interval_score_of_no_interval_is_the_absolute_error() {
        let draws = uniform_draws();
        // ½|y − m| / ½ with the median 49.5.
        let wis = weighted_interval_score(&draws, &[60.0], &[]).unwrap()[0];
        assert!((wis - 10.5).abs() < 1e-9, "{wis}");
    }

    #[test]
    fn the_sample_crps_is_its_closed_form() {
        let draws = ObservationDraws {
            values: vec![vec![1.0], vec![4.0], vec![2.0], vec![8.0]],
            weights: vec![0.25; 4],
        };
        assert!((crps(&draws, &[3.0]).unwrap()[0] - 0.8125).abs() < 1e-12);
    }

    #[test]
    fn the_pit_of_a_tie_lies_between_the_two_sides() {
        // Weights of any scale: the PIT divides by their sum.
        let draws = ObservationDraws {
            values: vec![vec![0.0], vec![1.0], vec![1.0], vec![2.0]],
            weights: vec![2.0; 4],
        };
        let value = pit(&draws, &[1.0], 5).unwrap()[0];
        assert!(value > 0.25 && value < 0.75, "{value}");
        assert_eq!(pit(&draws, &[3.0], 5).unwrap(), vec![1.0]);
        assert_eq!(pit(&draws, &[0.5], 5).unwrap(), vec![0.25]);
    }

    #[test]
    fn a_score_needs_one_value_per_observation_and_a_draw() {
        let draws = uniform_draws();
        let error = coverage(&draws, &[1.0, 2.0], 0.9).unwrap_err();
        assert!(error.contains("1 observations; 2 observed"), "{error}");
        let empty = ObservationDraws {
            values: vec![],
            weights: vec![],
        };
        assert!(crps(&empty, &[]).unwrap_err().contains("at least one draw"));
    }

    #[test]
    fn series_average_their_observations() {
        let labels = ["A", "B", "A"].map(String::from);
        assert_eq!(
            by_series(&labels, &[1.0, 5.0, 3.0]),
            vec![("A".to_string(), 2.0), ("B".to_string(), 5.0)]
        );
    }
}
