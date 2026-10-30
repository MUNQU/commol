//! Starts of a multi-start optimization.
//!
//! Start 0 is a point the caller gives, the initial guess of a calibration
//! or the fit of a replicate. Start `i > 0` is row `i − 1` of a Latin
//! hypercube of `count − 1` rows over the bounds, so every start but the
//! first covers its own stratum of each parameter. Uniform draws leave
//! clusters and gaps with few starts; small moves around the guess put every
//! start in the same basin.
//!
//! One seed gives every random number of the starts, in two streams of the
//! registry of `commol-random` ([`Stream`]): `Stream::StartPoints` draws the
//! hypercube, and start `i` takes `Stream::StartSeeds.replicate(seed, i)` for
//! a particle swarm, so no start draws from the stream of another or of
//! another method, and start 0 takes a seed of the same rule as the others. [`optimize_starts`] runs the starts and
//! chooses the best of them, for a calibration and for every fit of a
//! replicate.

use argmin::core::observers::Observe;
use commol_core::SimulationEngine;
use commol_random::{Generator, Stream, latin_hypercube};
use rayon::prelude::*;

use crate::calibration_problem::CalibrationProblem;
use crate::objective::OptimizationObjective;
use crate::optimization::{
    NelderMeadState, NoObserver, OptimizationConfig, ParticleSwarmState, SearchSpace, run,
};
use crate::types::{CalibrationParameter, CalibrationResult};

/// The `count` start points of a multi-start optimization seeded with
/// `seed`, each with one value per parameter: the initial guess, then the
/// rows of a Latin hypercube drawn from `Stream::StartPoints.seed(seed)`.
pub fn start_points(parameters: &[CalibrationParameter], count: usize, seed: u64) -> Vec<Vec<f64>> {
    if count == 0 {
        return Vec::new();
    }
    let mut points = vec![parameters.iter().map(|p| p.initial_value()).collect()];
    let rows = count - 1;
    if rows == 0 {
        return points;
    }
    let mut generator = Generator::from_seed(Stream::StartPoints.seed(seed));
    let unit = latin_hypercube(&mut generator, rows, parameters.len());
    points.extend(unit.into_iter().map(|point| {
        point
            .into_iter()
            .zip(parameters)
            .map(|(fraction, parameter)| {
                parameter.min_bound + fraction * (parameter.max_bound - parameter.min_bound)
            })
            .collect()
    }));
    points
}

/// The seed of the particle swarm of start `index` of a multi-start
/// optimization seeded with `seed`: `Stream::StartSeeds.replicate(seed, index)`.
pub(crate) fn start_seed(seed: u64, index: usize) -> u64 {
    Stream::StartSeeds.replicate(seed, index as u64)
}

/// The outcome of one start: where it began and what the optimizer
/// returned, or why it failed.
#[derive(Debug, Clone)]
pub struct StartOutcome {
    /// The start point, in problem order.
    pub start: Vec<f64>,
    /// The result of the optimization, or the reason it failed.
    pub result: Result<CalibrationResult, String>,
}

/// The outcomes of the starts of a multi-start optimization, in the order
/// of the starts, and the best of them: the start of lowest loss among those
/// that did not fail, the first on a tie, a loss that is NaN counting as
/// `+∞`.
#[derive(Debug, Clone)]
pub struct Starts {
    outcomes: Vec<StartOutcome>,
    best: usize,
}

impl Starts {
    /// The starts of `outcomes`, with their best.
    ///
    /// # Errors
    ///
    /// Returns a message when there is no outcome, or one that lists the
    /// reason of each start when every start failed.
    fn new(outcomes: Vec<StartOutcome>) -> Result<Self, String> {
        if outcomes.is_empty() {
            return Err("A multi-start optimization needs at least one start".to_string());
        }
        let loss = |result: &CalibrationResult| {
            if result.final_loss.is_nan() {
                f64::INFINITY
            } else {
                result.final_loss
            }
        };
        let mut best: Option<(usize, f64)> = None;
        for (index, outcome) in outcomes.iter().enumerate() {
            if let Ok(result) = &outcome.result
                && best.is_none_or(|(_, kept)| loss(result) < kept)
            {
                best = Some((index, loss(result)));
            }
        }
        match best {
            Some((best, _)) => Ok(Self { outcomes, best }),
            None => {
                let reasons: Vec<String> = outcomes
                    .iter()
                    .enumerate()
                    .filter_map(|(index, outcome)| {
                        outcome
                            .result
                            .as_ref()
                            .err()
                            .map(|reason| format!("Start {index} failed: {reason}"))
                    })
                    .collect();
                Err(format!("Every start failed: {}", reasons.join("; ")))
            }
        }
    }

    /// Every outcome, in the order of the starts.
    pub fn outcomes(&self) -> &[StartOutcome] {
        &self.outcomes
    }

    /// The index of the best start.
    pub fn best_index(&self) -> usize {
        self.best
    }

    /// The result of the best start.
    pub fn best(&self) -> &CalibrationResult {
        self.outcomes[self.best]
            .result
            .as_ref()
            .expect("`Starts::new` chooses a start that did not fail")
    }

    /// The result of the best start, with the outcomes dropped.
    pub fn into_best(self) -> CalibrationResult {
        self.best().clone()
    }
}

/// Run the starts of `config` on `problem` from `points`, in parallel,
/// seeded with `seed`, and choose the best ([`Starts`]).
///
/// The points are the start points of [`start_points`] of the same seed, or
/// any points of a caller with a start of its own, such as a fit. A particle
/// swarm, which draws its own population, takes the seed of start `i` from
/// `Stream::StartSeeds`; the calibration from the initial guess ([`crate::optimize`])
/// is start 0.
///
/// # Errors
///
/// Returns the message of [`Starts`] when there is no point or every start
/// fails.
pub fn optimize_starts<E>(
    problem: &CalibrationProblem<E>,
    config: &OptimizationConfig,
    points: Vec<Vec<f64>>,
    seed: u64,
) -> Result<Starts, String>
where
    E: SimulationEngine + Send + Sync,
{
    optimize_starts_with_observer(problem, config, points, seed, || NoObserver)
}

/// [`optimize_starts`], with an observer of every iteration of each start
/// from `observer`.
///
/// # Errors
///
/// As [`optimize_starts`].
pub fn optimize_starts_with_observer<E, O, F>(
    problem: &CalibrationProblem<E>,
    config: &OptimizationConfig,
    points: Vec<Vec<f64>>,
    seed: u64,
    observer: F,
) -> Result<Starts, String>
where
    E: SimulationEngine + Send + Sync,
    O: Observe<NelderMeadState> + Observe<ParticleSwarmState> + Clone + 'static,
    F: Fn() -> O + Sync,
{
    let space = SearchSpace::of(problem);
    let outcomes = points
        .into_par_iter()
        .enumerate()
        .map(|(index, start)| {
            let result = run(
                &space,
                OptimizationObjective::new(problem),
                config,
                &start,
                start_seed(seed, index),
                Some(observer()),
            )
            .map(|(result, _)| result);
            StartOutcome { start, result }
        })
        .collect();
    Starts::new(outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters() -> Vec<CalibrationParameter> {
        vec![
            CalibrationParameter::with_initial_guess("k1".to_string(), 0.0, 1.0, 0.3),
            CalibrationParameter::new("k2".to_string(), -2.0, 2.0),
        ]
    }

    #[test]
    fn no_starts_are_no_points() {
        assert!(start_points(&parameters(), 0, 7).is_empty());
    }

    #[test]
    fn the_first_start_is_the_initial_guess() {
        let points = start_points(&parameters(), 5, 7);
        assert_eq!(points[0], vec![0.3, 0.0]);
        assert_eq!(points.len(), 5);
    }

    #[test]
    fn the_other_starts_fill_one_stratum_of_each_parameter() {
        let points = start_points(&parameters(), 5, 7);
        for (index, (min, max)) in [(0.0, 1.0), (-2.0, 2.0)].into_iter().enumerate() {
            let mut strata: Vec<usize> = points[1..]
                .iter()
                .map(|point| (((point[index] - min) / (max - min)) * 4.0) as usize)
                .collect();
            strata.sort_unstable();
            assert_eq!(strata, vec![0, 1, 2, 3]);
        }
    }

    #[test]
    fn every_order_of_the_strata_is_about_as_likely() {
        // Two starts after the first: each parameter puts them in its strata
        // in one of two orders.
        let in_order = (0..400)
            .filter(|&seed| start_points(&parameters(), 3, seed)[1][0] < 0.5)
            .count();
        assert!((150..250).contains(&in_order), "{in_order}");
    }

    #[test]
    fn each_parameter_orders_its_strata_on_its_own() {
        let points = start_points(&parameters(), 9, 7);
        let strata = |index: usize, (min, max): (f64, f64)| {
            points[1..]
                .iter()
                .map(|point| (((point[index] - min) / (max - min)) * 8.0) as usize)
                .collect::<Vec<_>>()
        };
        assert_ne!(strata(0, (0.0, 1.0)), strata(1, (-2.0, 2.0)));
    }

    #[test]
    fn the_hypercube_comes_from_the_stream_of_the_start_points() {
        let unit = latin_hypercube(
            &mut Generator::from_seed(Stream::StartPoints.seed(11)),
            3,
            2,
        );
        let points = start_points(&parameters(), 4, 11);
        for (point, row) in points[1..].iter().zip(unit) {
            assert_eq!(point, &vec![row[0], -2.0 + 4.0 * row[1]]);
        }
    }

    #[test]
    fn start_i_takes_seed_i_of_the_stream_of_the_start_seeds() {
        for index in 0..3 {
            assert_eq!(
                start_seed(11, index),
                Stream::StartSeeds.replicate(11, index as u64)
            );
        }
        assert_ne!(start_seed(11, 0), Stream::StartPoints.seed(11));
    }

    fn outcome(result: Result<f64, &str>) -> StartOutcome {
        StartOutcome {
            start: vec![0.0],
            result: result
                .map(|final_loss| CalibrationResult {
                    best_parameters: vec![final_loss],
                    parameter_names: vec!["k".to_string()],
                    final_loss,
                    iterations: 1,
                    converged: true,
                    termination_reason: String::new(),
                })
                .map_err(str::to_string),
        }
    }

    #[test]
    fn a_loss_that_is_not_a_number_is_the_worst() {
        let starts = Starts::new(vec![
            outcome(Ok(f64::NAN)),
            outcome(Err("stopped")),
            outcome(Ok(3.0)),
            outcome(Ok(2.0)),
            outcome(Ok(2.0)),
        ])
        .unwrap();
        assert_eq!(starts.best_index(), 3);
        assert_eq!(starts.into_best().final_loss, 2.0);
        let only_nan = Starts::new(vec![outcome(Err("stopped")), outcome(Ok(f64::NAN))]);
        assert_eq!(only_nan.unwrap().best_index(), 1);
    }

    #[test]
    fn one_seed_gives_one_set_of_starts() {
        assert_eq!(
            start_points(&parameters(), 6, 1),
            start_points(&parameters(), 6, 1)
        );
        assert_ne!(
            start_points(&parameters(), 6, 1),
            start_points(&parameters(), 6, 2)
        );
    }
}
