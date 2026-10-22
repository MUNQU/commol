//! The residuals of a fit, on the scale a method names, and the residual of
//! a series at any position of the data.

use std::collections::HashMap;

use commol_calibration::{CalibrationProblem, ObservedDataPoint, Workspace};
use commol_core::SimulationEngine;
use commol_random::Generator;

use crate::error::MethodError;
use crate::resampling::blocks::{distinct_positions, position_indices};

/// The scale of the residuals of a fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualScale {
    /// `r = y − ŷ`, and a synthetic value is `ŷ + r`.
    Additive,
    /// `r = log(y / ŷ)`, and a synthetic value is `ŷ · exp(r)`; it needs
    /// positive observed and fitted values.
    Multiplicative,
}

impl ResidualScale {
    /// The scale of a name: `"additive"` or `"multiplicative"`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`], which lists the names, when `name` is none
    /// of them.
    pub fn from_name(name: &str) -> Result<Self, MethodError> {
        match name {
            "additive" => Ok(Self::Additive),
            "multiplicative" => Ok(Self::Multiplicative),
            _ => Err(MethodError::Invalid(format!(
                "residual_scale is '{name}'; it must be \"additive\" or \"multiplicative\""
            ))),
        }
    }

    /// The value an observation shows when the model gives `fitted` and the
    /// residual is `residual`.
    pub fn apply(self, fitted: f64, residual: f64) -> f64 {
        match self {
            Self::Additive => fitted + residual,
            Self::Multiplicative => fitted * residual.exp(),
        }
    }
}

/// The fitted values and the residuals of a fit, one per observation of the
/// problem.
#[derive(Debug, Clone, PartialEq)]
pub struct Residuals {
    scale: ResidualScale,
    fitted: Vec<f64>,
    residuals: Vec<f64>,
}

impl Residuals {
    /// The residuals of `problem` at the parameter values `fit`.
    ///
    /// The fitted value of an observation is its prediction; with a
    /// stochastic engine, the mean of the predictions of the replicates.
    ///
    /// # Errors
    ///
    /// As [`MethodError::of_point`] when the fit has no run;
    /// [`MethodError::Invalid`], which names the observation, when on the
    /// multiplicative scale an observed or fitted value is not positive.
    pub fn new<E: SimulationEngine>(
        problem: &CalibrationProblem<E>,
        fit: &[f64],
        scale: ResidualScale,
    ) -> Result<Self, MethodError> {
        let mut workspace = Workspace::default();
        let applied = problem.applied_parameter_values(fit);
        let predictions = problem
            .forward()
            .run_observed(&applied, &mut workspace)
            .map_err(|error| MethodError::of_point("The fit has no run", error))?;
        let observed = problem.observed_data();
        let n = observed.len();
        let replicates = predictions.len() / n.max(1);
        let fitted: Vec<f64> = (0..n)
            .map(|i| {
                (0..replicates).map(|r| predictions[r * n + i]).sum::<f64>() / replicates as f64
            })
            .collect();
        let residuals = observed
            .iter()
            .zip(&fitted)
            .map(|(obs, &fit)| match scale {
                ResidualScale::Additive => Ok(obs.value - fit),
                ResidualScale::Multiplicative => {
                    if obs.value > 0.0 && fit > 0.0 {
                        Ok((obs.value / fit).ln())
                    } else {
                        Err(MethodError::Invalid(format!(
                            "Observation '{}' at {} has the observed value {} and the \
                             fitted value {fit}; residual_scale=\"multiplicative\" needs both \
                             positive. Use residual_scale=\"additive\" for this data",
                            obs.label, obs.at, obs.value
                        )))
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            scale,
            fitted,
            residuals,
        })
    }

    /// The scale.
    pub fn scale(&self) -> ResidualScale {
        self.scale
    }

    /// The fitted value of each observation.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// The residual of each observation.
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// The values the observations show with the residuals `residuals`.
    pub fn values(&self, residuals: &[f64]) -> Vec<f64> {
        self.fitted
            .iter()
            .zip(residuals)
            .map(|(&fitted, &residual)| self.scale.apply(fitted, residual))
            .collect()
    }
}

/// The residuals of each series by the index of their position among the
/// distinct positions of the data.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SeriesResiduals {
    positions: usize,
    series: HashMap<String, usize>,
    /// Per series: `(position index, residual)`, ascending.
    residuals: Vec<Vec<(usize, f64)>>,
}

impl SeriesResiduals {
    pub(crate) fn new(observations: &[ObservedDataPoint], residuals: &[f64]) -> Self {
        let positions = distinct_positions(observations);
        let mut series = HashMap::new();
        let mut by_series: Vec<Vec<(usize, f64)>> = Vec::new();
        for ((obs, &residual), position) in observations
            .iter()
            .zip(residuals)
            .zip(position_indices(observations, &positions))
        {
            let next = by_series.len();
            let index = *series.entry(obs.label.clone()).or_insert(next);
            if index == by_series.len() {
                by_series.push(Vec::new());
            }
            by_series[index].push((position, residual));
        }
        for points in &mut by_series {
            points.sort_by_key(|(position, _)| *position);
            points.dedup_by_key(|(position, _)| *position);
        }
        Self {
            positions: positions.len(),
            series,
            residuals: by_series,
        }
    }

    /// The number of distinct positions of the data.
    pub(crate) fn positions(&self) -> usize {
        self.positions
    }

    /// The index of the series `label`, if it has residuals.
    pub(crate) fn series(&self, label: &str) -> Option<usize> {
        self.series.get(label).copied()
    }

    /// The residual of `series` at `position`, or at the nearest position
    /// where the series has one (the earlier one on a tie).
    pub(crate) fn at(&self, series: usize, position: usize) -> f64 {
        let points = &self.residuals[series];
        let after = points.partition_point(|(p, _)| *p < position);
        match (after.checked_sub(1).map(|i| points[i]), points.get(after)) {
            (_, Some(&(p, residual))) if p == position => residual,
            (Some((before, early)), Some(&(next, late))) => {
                if position - before <= next - position {
                    early
                } else {
                    late
                }
            }
            (Some((_, early)), None) => early,
            (None, Some(&(_, late))) => late,
            // coverage: invariant, new gives a series only with the residual that names it
            (None, None) => unreachable!("a series has at least one residual"),
        }
    }
}

/// How the blocks of residuals of a synthetic series are drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualBlocks {
    /// Blocks of `length` consecutive positions with a uniform start
    /// (Künsch 1989).
    Moving,
    /// Blocks with a uniform start and a geometric length of mean `length`,
    /// wrapped around the end (Politis and Romano 1994).
    Stationary,
}

impl ResidualBlocks {
    /// The blocks of a name: `"moving"` or `"stationary"`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`], which lists the names, when `name` is none
    /// of them.
    pub fn from_name(name: &str) -> Result<Self, MethodError> {
        match name {
            "moving" => Ok(Self::Moving),
            "stationary" => Ok(Self::Stationary),
            _ => Err(MethodError::Invalid(format!(
                "scheme is '{name}'; it must be \"moving\" or \"stationary\""
            ))),
        }
    }
}

/// The source position of each of `count` synthetic positions, drawn from
/// `positions` positions by joining blocks of mean `length`.
pub(crate) fn source_positions(
    positions: usize,
    length: usize,
    count: usize,
    blocks: ResidualBlocks,
    generator: &mut Generator,
) -> Vec<usize> {
    let mut source = Vec::with_capacity(count);
    match blocks {
        ResidualBlocks::Moving => {
            let starts = positions - length + 1;
            while source.len() < count {
                let start = ((generator.uniform() * starts as f64) as usize).min(starts - 1);
                source.extend(start..start + length);
            }
            source.truncate(count);
        }
        ResidualBlocks::Stationary => {
            let draw = |generator: &mut Generator| {
                ((generator.uniform() * positions as f64) as usize).min(positions - 1)
            };
            let renewal = 1.0 / length as f64;
            let mut position = draw(generator);
            source.push(position);
            while source.len() < count {
                position = if generator.uniform() < renewal {
                    draw(generator)
                } else {
                    (position + 1) % positions
                };
                source.push(position);
            }
            source.truncate(count);
        }
    }
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_residual_comes_from_the_nearest_position() {
        let observations = vec![
            ObservedDataPoint::new(0.0, "x".to_string(), 0.0),
            ObservedDataPoint::new(3.0, "x".to_string(), 0.0),
            ObservedDataPoint::new(1.0, "y".to_string(), 0.0),
            ObservedDataPoint::new(2.0, "y".to_string(), 0.0),
        ];
        let grid = SeriesResiduals::new(&observations, &[10.0, 13.0, 21.0, 22.0]);
        let x = grid.series("x").unwrap();
        assert_eq!(
            (0..4).map(|p| grid.at(x, p)).collect::<Vec<_>>(),
            [10.0, 10.0, 13.0, 13.0]
        );
        let y = grid.series("y").unwrap();
        assert_eq!(grid.at(y, 0), 21.0);
        assert_eq!(grid.at(y, 3), 22.0);
        assert_eq!(grid.series("z"), None);
    }

    #[test]
    fn the_nearest_position_is_the_earlier_one_on_a_tie() {
        // x has residuals at the positions 2 and 6 of 0, …, 8.
        let mut observations = vec![
            ObservedDataPoint::new(2.0, "x".to_string(), 0.0),
            ObservedDataPoint::new(6.0, "x".to_string(), 0.0),
        ];
        observations.extend(
            [0.0, 1.0, 3.0, 4.0, 5.0, 7.0, 8.0]
                .map(|at| ObservedDataPoint::new(at, "y".to_string(), 0.0)),
        );
        let mut residuals = vec![20.0, 60.0];
        residuals.extend([0.0; 7]);
        let grid = SeriesResiduals::new(&observations, &residuals);
        let x = grid.series("x").unwrap();
        assert_eq!(
            (2..=6).map(|p| grid.at(x, p)).collect::<Vec<_>>(),
            [20.0, 20.0, 20.0, 60.0, 60.0]
        );
    }

    /// How often each of `positions` positions is a source, over many draws.
    fn coverage(positions: usize, length: usize, blocks: ResidualBlocks) -> Vec<usize> {
        let mut generator = Generator::from_seed(8);
        let mut counts = vec![0; positions];
        for _ in 0..200 {
            for p in source_positions(positions, length, positions, blocks, &mut generator) {
                counts[p] += 1;
            }
        }
        counts
    }

    #[test]
    fn every_position_is_a_source_of_both_kinds_of_block() {
        // A stationary block of mean length 1 starts anew at every position.
        for (length, blocks) in [
            (3, ResidualBlocks::Moving),
            (3, ResidualBlocks::Stationary),
            (1, ResidualBlocks::Stationary),
        ] {
            let counts = coverage(10, length, blocks);
            assert!(
                counts.iter().all(|&count| count > 0),
                "{blocks:?} {counts:?}"
            );
        }
    }

    #[test]
    fn a_stationary_block_has_the_mean_length() {
        let mut generator = Generator::from_seed(9);
        let source = source_positions(1000, 4, 20_000, ResidualBlocks::Stationary, &mut generator);
        // A block ends where the next source is not the next position.
        let ends = source
            .windows(2)
            .filter(|w| w[1] != (w[0] + 1) % 1000)
            .count();
        let mean = source.len() as f64 / (ends + 1) as f64;
        assert!((mean - 4.0).abs() < 0.2, "{mean}");
    }

    #[test]
    fn stationary_blocks_wrap_around_the_end() {
        let mut generator = Generator::from_seed(4);
        let source = source_positions(5, 50, 200, ResidualBlocks::Stationary, &mut generator);
        let wraps = source.windows(2).filter(|w| w[0] == 4 && w[1] == 0).count();
        assert!(wraps > 0);
        assert!(source.iter().all(|&p| p < 5));
    }
}
