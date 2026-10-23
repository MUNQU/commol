//! Starts of a multi-start optimization.
//!
//! Start 0 is the initial guess. Start `i > 0` is row `i − 1` of a Latin
//! hypercube of `count − 1` rows over the bounds, drawn with the generator of
//! the run seed, so every start but the first covers its own stratum of each
//! parameter. Uniform draws leave clusters and gaps with few starts; small
//! moves around the guess put every start in the same basin.

use commol_core::SimulationEngine;
use commol_random::{Generator, latin_hypercube, replicate_seed};
use rayon::prelude::*;

use crate::calibration_problem::CalibrationProblem;
use crate::optimization::{OptimizationConfig, optimize_from};
use crate::types::{CalibrationParameter, CalibrationResult};

/// The `count` start points of a multi-start optimization seeded with
/// `seed`, each with one value per parameter.
pub fn start_points(parameters: &[CalibrationParameter], count: usize, seed: u64) -> Vec<Vec<f64>> {
    if count == 0 {
        return Vec::new();
    }
    let mut points = vec![parameters.iter().map(|p| p.initial_value()).collect()];
    let rows = count - 1;
    if rows == 0 {
        return points;
    }
    let unit = latin_hypercube(&mut Generator::from_seed(seed), rows, parameters.len());
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

/// The outcome of one start: where it began and what the optimizer
/// returned, or why it failed.
#[derive(Debug, Clone)]
pub struct StartOutcome {
    /// The start point, in problem order.
    pub start: Vec<f64>,
    /// The result of the optimization, or the reason it failed.
    pub result: Result<CalibrationResult, String>,
}

/// Run `count` starts of `config` on `problem` in parallel, seeded with
/// `seed`, in the order of the starts.
///
/// Start 0 runs the configuration as given. A particle swarm, which draws its
/// own population, takes the seed of start `i > 0` from
/// [`commol_random::replicate_seed`] of `seed` and `i`.
pub fn optimize_starts<E>(
    problem: &CalibrationProblem<E>,
    config: &OptimizationConfig,
    count: usize,
    seed: u64,
) -> Vec<StartOutcome>
where
    E: SimulationEngine + Send + Sync,
{
    start_points(problem.parameters(), count, seed)
        .into_par_iter()
        .enumerate()
        .map(|(index, start)| {
            let start_seed = (index > 0).then(|| replicate_seed(seed, index as u64));
            let result = optimize_from(problem, config, &start, start_seed);
            StartOutcome { start, result }
        })
        .collect()
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
    fn a_seed_gives_the_starts_it_always_gave() {
        // The starts of seed 11, drawn before the Latin hypercube moved to
        // commol-random.
        assert_eq!(
            start_points(&parameters(), 4, 11),
            vec![
                vec![0.3, 0.0],
                vec![0.32140129672335693, -0.6392004253419341],
                vec![0.5344058815793623, -0.9480498987687109],
                vec![0.7541233761814915, 1.1916370017242173],
            ]
        );
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
