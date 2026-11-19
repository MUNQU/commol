//! Scores of a predictive against observations: empirical coverage, the
//! interval score, the weighted interval score, the CRPS of a weighted
//! sample, PIT values, quantile residuals, and the log score of observation
//! models.
//!
//! Coverage, the interval scores and the CRPS take the draws of a
//! predictive at a set of observations ([`ObservationDraws`]) and the
//! observed values; the log score reads the density of the observation
//! models at each draw's predicted mean. The PIT and the quantile residuals
//! read [`ObservationBounds`]: the tails of each draw at each observed
//! value, from the distribution functions of the observation models
//! ([`observation_bounds`]) or from the draws ([`ObservationBounds::of_draws`]).
//! Each returns one value per observation; [`by_series`] averages them per
//! series. Which observations to score, for example the ones held out of a
//! calibration, is the caller's choice.

use commol_calibration::likelihood::parts_of;
use commol_calibration::{
    CalibrationProblem, EvaluationError, ObservedDataPoint, SeriesIndex, Tails, weighted_crps,
};
use commol_core::{CONFIDENCE_LEVEL, SimulationEngine};
use commol_random::Generator;
use commol_statistics::numeric::log_sum_exp;
use commol_statistics::special::{normal_quantile, normal_quantile_sf};
use commol_statistics::statistics;
use rayon::prelude::*;

use crate::predictive::{
    DistributionNoise, ObservationDraws, ObservationNoise, PredictiveError, check_names,
    observe_sets,
};
use crate::sample::ParameterSample;
use commol_random::Stream;

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
    CONFIDENCE_LEVEL.check(confidence_level)?;
    let levels = statistics::interval_levels(confidence_level);
    (0..observed.len())
        .map(|i| {
            let q = statistics::quantiles(&column(draws, i), &draws.weights, &levels)?;
            Ok((q[0], q[1]))
        })
        .collect()
}

/// Whether each observation lies inside the central interval of
/// `confidence_level`: 1 or 0. Their mean is the empirical coverage.
///
/// # Errors
///
/// Returns a message when the shapes differ, the level is not strictly
/// between 0 and 1, or a draw is NaN.
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
        CONFIDENCE_LEVEL.check(c)?;
        levels.extend(statistics::interval_levels(c));
    }
    let k = confidence_levels.len() as f64;
    observed
        .iter()
        .enumerate()
        .map(|(i, &y)| {
            let q = statistics::quantiles(&column(draws, i), &draws.weights, &levels)?;
            let mut total = 0.5 * (y - q[0]).abs();
            for (index, &c) in confidence_levels.iter().enumerate() {
                let alpha = 1.0 - c;
                let (lower, upper) = (q[1 + 2 * index], q[2 + 2 * index]);
                total += alpha / 2.0 * interval_score_of(lower, upper, alpha, y);
            }
            Ok(total / (k + 0.5))
        })
        .collect()
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

/// The tails of every draw of a sample at a set of observed values, with the
/// weights of the draws: the one primitive of the PIT, the quantile
/// residuals and every reader of the distribution of an observation under
/// each draw, which adds its own uniform numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationBounds {
    /// `[draw][observation]`: `P(Y < y)`, `p(y)` and `P(Y > y)` under the
    /// draw, its replicates mixed with equal shares.
    pub tails: Vec<Vec<Tails>>,
    /// The weight of every draw.
    pub weights: Vec<f64>,
    /// The seed of the draws; the place `v` of each observed value within
    /// its mass ([`Self::places`]) comes from the stream
    /// [`Stream::PitTies`] of it.
    pub seed: u64,
}

impl ObservationBounds {
    /// The bounds of `draws` at the values `observed`: each draw is a mass
    /// at its value, so its tails are 0 or 1.
    ///
    /// # Errors
    ///
    /// Returns a message when the shapes differ or there is no draw.
    pub fn of_draws(draws: &ObservationDraws, observed: &[f64]) -> Result<Self, String> {
        check(draws, observed)?;
        let tails = draws
            .values
            .iter()
            .map(|row| {
                row.iter()
                    .zip(observed)
                    .map(|(&x, &y)| Tails::point_mass(y, x))
                    .collect()
            })
            .collect();
        Ok(Self {
            tails,
            weights: draws.weights.clone(),
            seed: draws.seed,
        })
    }

    /// The tails of the predictive at each observed value: those of the
    /// mixture of the draws, `Σ_d w_d T_d / Σ_d w_d`.
    pub fn mixture(&self) -> Vec<Tails> {
        let total: f64 = self.weights.iter().sum();
        let count = self.tails.first().map_or(0, Vec::len);
        (0..count)
            .map(|i| {
                let mut mixed = Tails::default();
                for (row, &w) in self.tails.iter().zip(&self.weights) {
                    mixed.add_weighted(w / total, row[i]);
                }
                mixed
            })
            .collect()
    }

    /// The place `v` of each observed value within the mass `p(y)` that the
    /// predictive gives it: one uniform number per observation, in order,
    /// from the stream [`Stream::PitTies`] of the seed. The PIT and the
    /// quantile residuals read the same places.
    pub fn places(&self) -> Vec<f64> {
        let count = self.tails.first().map_or(0, Vec::len);
        let mut generator = Generator::from_seed(Stream::PitTies.seed(self.seed));
        (0..count).map(|_| generator.uniform()).collect()
    }

    /// The randomized probability integral transform of each observation,
    /// `F(y⁻) + v · p(y)` of the mixture with the places `v` of
    /// [`Self::places`]: uniform for data the predictive did not see, also
    /// for a discrete model, and `F(y)` whatever the seed for a model with
    /// no atoms.
    pub fn pit(&self) -> Vec<f64> {
        self.mixture()
            .iter()
            .zip(self.places())
            .map(|(tails, v)| tails.pit(v))
            .collect()
    }

    /// The randomized quantile residual of each observation (Dunn and Smyth
    /// 1996): [`quantile_residual`] of the mixture with the places of
    /// [`Self::places`], from the PIT `u` of [`Self::pit`] read from the
    /// nearer tail. Standard normal for data the predictive did not see.
    pub fn quantile_residuals(&self) -> Vec<f64> {
        self.mixture()
            .iter()
            .zip(self.places())
            .map(|(tails, v)| quantile_residual(tails, v))
            .collect()
    }
}

/// The randomized quantile residual of an observed value with `tails` and
/// the place `v` within its mass: `Φ⁻¹(u)` of its PIT `u` up to `½`, and
/// `−Φ⁻¹(P(Y > y) + (1 − v) · p(y))` above, so a residual far in the upper
/// tail stays finite.
pub fn quantile_residual(tails: &Tails, v: f64) -> f64 {
    let u = tails.pit(v);
    if u <= 0.5 {
        normal_quantile(u)
    } else {
        normal_quantile_sf(tails.above + (1.0 - v) * tails.at)
    }
}

/// The tails of the observed value of each of `observations` under every
/// draw of `sample`, from the distribution functions of the observation
/// models of `noise` ([`commol_calibration::Likelihood::tails`]) at each
/// draw's predicted means and noise parameters, each mean the nearest the
/// model allows. A replicate of a draw is seeded from `seed` as in
/// [`crate::Predictive::new`], and the tails of a draw mix its replicates
/// with equal shares; the bounds keep `seed`. The models run once for every
/// reader of the bounds.
///
/// # Errors
///
/// Returns an error, before any run, when the sample's parameters are not
/// the problem's, an observation names a series with no model, or an
/// observed value is outside the support of its model; and when an
/// observation names an unknown output or factor, a draw has no run, or the
/// exact sum of an observation does not end at a draw
/// ([`commol_calibration::UnendedSum`]).
pub fn observation_bounds<E>(
    problem: &CalibrationProblem<E>,
    sample: &ParameterSample,
    observations: &[ObservedDataPoint],
    noise: &DistributionNoise,
    seed: u64,
) -> Result<ObservationBounds, PredictiveError>
where
    E: SimulationEngine + Send + Sync,
{
    check_names(problem.forward(), sample)?;
    let likelihood = noise.likelihood();
    let terms = likelihood
        .check_values(observations)
        .map_err(PredictiveError::Mismatch)?;
    let runs = observe_sets(problem.forward(), sample.values(), observations, Some(seed))?;
    let tails = runs
        .par_iter()
        .zip(sample.values())
        .enumerate()
        .map(|(draw, (replicates, values))| {
            let share = 1.0 / replicates.len() as f64;
            let mut mixed = vec![Tails::default(); observations.len()];
            for means in replicates {
                let tails = likelihood
                    .tails_of(observations, &terms, values, &means.parts)
                    .map_err(|message| PredictiveError::Draw {
                        draw,
                        error: EvaluationError::Failed(message),
                    })?;
                for (sum, tails) in mixed.iter_mut().zip(tails) {
                    sum.add_weighted(share, tails);
                }
            }
            Ok(mixed)
        })
        .collect::<Result<_, PredictiveError>>()?;
    Ok(ObservationBounds {
        tails,
        weights: sample.weights().to_vec(),
        seed,
    })
}

/// The log score of each of `observations`: the log density of its
/// observed value under the predictive, the mixture over the draws of
/// `sample` of the observation models of `noise`,
/// `log Σ_d w_d · p(y | μ_d, θ_d)`, where `μ_d` is the draw's predicted mean,
/// or the means of the parts of an observation of several parts, each the
/// nearest the model allows ([`commol_calibration::Noise::sum_log_density`]),
/// and `θ_d` its noise parameters. A
/// replicate of a draw is a draw with an equal share of its weight, seeded
/// from `seed` as in [`crate::Predictive::new`]. Higher
/// is better; an observation the predictive gives no mass scores `−∞`.
///
/// # Errors
///
/// Returns an error when the sample's parameters are not the problem's, an
/// observation names an unknown output or factor or a series with no model,
/// a draw has no run, or the exact sum of an observation does not end at a
/// draw ([`commol_calibration::UnendedSum`]).
pub fn log_score<E>(
    problem: &CalibrationProblem<E>,
    sample: &ParameterSample,
    observations: &[ObservedDataPoint],
    noise: &DistributionNoise,
    seed: u64,
) -> Result<Vec<f64>, PredictiveError>
where
    E: SimulationEngine + Send + Sync,
{
    check_names(problem.forward(), sample)?;
    noise
        .check(observations)
        .map_err(PredictiveError::Mismatch)?;
    let likelihood = noise.likelihood();
    let terms = likelihood
        .terms(observations)
        .expect("checked by the noise");
    let runs = observe_sets(problem.forward(), sample.values(), observations, Some(seed))?;
    // Each replicate of a draw: the index of the draw, the log of its share
    // of the weight, its parameter values and the means of the parts of each
    // observation.
    type Draw<'a> = (usize, f64, &'a [f64], Vec<&'a [f64]>);
    let mut draws: Vec<Draw> = Vec::new();
    for (draw, ((replicates, &weight), values)) in runs
        .iter()
        .zip(sample.weights())
        .zip(sample.values())
        .enumerate()
    {
        let share = libm::log(weight / replicates.len() as f64);
        for means in replicates {
            draws.push((
                draw,
                share,
                values,
                parts_of(&terms, &means.parts).collect(),
            ));
        }
    }
    observations
        .iter()
        .zip(&terms)
        .enumerate()
        .map(|(i, (obs, &term))| {
            let logs: Vec<f64> = draws
                .iter()
                .map(|&(draw, share, values, ref parts)| {
                    let model = likelihood.noise(term, values);
                    let nearest: Vec<f64> = parts[i]
                        .iter()
                        .map(|&mean| model.nearest_mean(mean))
                        .collect();
                    model
                        .sum_log_density(obs.value, &nearest)
                        .map(|log_density| share + log_density)
                        .map_err(|error| PredictiveError::Draw {
                            draw,
                            error: EvaluationError::Failed(
                                error.message(&obs.label, obs.at, obs.value, obs.parts),
                            ),
                        })
                })
                .collect::<Result<_, _>>()?;
            Ok(log_sum_exp(&logs))
        })
        .collect()
}

/// The mean of `values` per series, named by `labels`, in the order each
/// series first appears.
///
/// # Panics
///
/// Panics when `labels` and `values` differ in length.
pub fn by_series(labels: &[String], values: &[f64]) -> Vec<(String, f64)> {
    assert_eq!(labels.len(), values.len(), "one label per value");
    let index = SeriesIndex::of_labels(labels);
    index
        .labels()
        .iter()
        .zip(index.indices())
        .map(|(name, members)| {
            let sum: f64 = members.iter().map(|&i| values[i]).sum();
            (name.clone(), sum / members.len() as f64)
        })
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
            seed: 0,
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
            seed: 0,
        };
        assert!((crps(&draws, &[3.0]).unwrap()[0] - 0.8125).abs() < 1e-12);
    }

    #[test]
    fn the_pit_of_a_tie_lies_between_the_two_sides() {
        // Weights of any scale: the PIT divides by their sum.
        let draws = ObservationDraws {
            values: vec![vec![0.0], vec![1.0], vec![1.0], vec![2.0]],
            weights: vec![2.0; 4],
            seed: 5,
        };
        let pit = |y: f64| ObservationBounds::of_draws(&draws, &[y]).unwrap().pit()[0];
        let value = pit(1.0);
        assert!(value > 0.25 && value < 0.75, "{value}");
        assert_eq!(pit(3.0), 1.0);
        assert_eq!(pit(0.5), 0.25);
    }

    #[test]
    fn a_score_needs_one_value_per_observation_and_a_draw() {
        let draws = uniform_draws();
        let error = coverage(&draws, &[1.0, 2.0], 0.9).unwrap_err();
        assert!(error.contains("1 observations; 2 observed"), "{error}");
        let empty = ObservationDraws {
            values: vec![],
            weights: vec![],
            seed: 0,
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
