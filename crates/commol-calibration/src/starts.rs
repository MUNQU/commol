//! Starts of a multi-start optimization.
//!
//! Start 0 is the initial guess. Start `i > 0` is row `i − 1` of a Latin
//! hypercube of `count − 1` rows over the bounds, drawn with the generator of
//! the run seed, so every start but the first covers its own stratum of each
//! parameter. Uniform draws leave clusters and gaps with few starts; small
//! moves around the guess put every start in the same basin.

use commol_core::SimulationEngine;
use commol_random::{replicate_seed, Generator};
use rayon::prelude::*;

use crate::calibration_problem::CalibrationProblem;
use crate::optimization::{optimize_from, OptimizationConfig};
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
    let mut generator = Generator::from_seed(seed);
    // One permutation of the strata per parameter, then one uniform draw
    // inside each stratum.
    let columns: Vec<Vec<f64>> = parameters
        .iter()
        .map(|parameter| {
            let mut strata: Vec<usize> = (0..rows).collect();
            for i in (1..rows).rev() {
                let j = (generator.uniform() * (i + 1) as f64) as usize;
                strata.swap(i, j.min(i));
            }
            let width = parameter.max_bound - parameter.min_bound;
            strata
                .into_iter()
                .map(|stratum| {
                    let fraction = (stratum as f64 + generator.uniform()) / rows as f64;
                    parameter.min_bound + fraction * width
                })
                .collect()
        })
        .collect();
    points.extend((0..rows).map(|row| columns.iter().map(|column| column[row]).collect()));
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
