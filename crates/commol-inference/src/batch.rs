//! The batch runner: every point of a design through a forward model, each
//! run reduced to its quantities of interest.
//!
//! The points run in parallel. With a stochastic engine each point runs the
//! replicates of the forward model, the same seeds at every point (common
//! random numbers), so two points differ only by their values.

use commol_calibration::ForwardModel;
use commol_core::SimulationEngine;
use rayon::prelude::*;

use crate::predictive::PredictiveError;
use crate::quantity::Quantities;

/// A forward model and the quantities each of its runs is reduced to.
#[derive(Debug, Clone, Copy)]
pub struct BatchRunner<'a, E: SimulationEngine> {
    forward: &'a ForwardModel<E>,
    quantities: &'a Quantities,
}

impl<'a, E: SimulationEngine + Send + Sync> BatchRunner<'a, E> {
    /// The runner of `quantities` on `forward`.
    ///
    /// # Errors
    ///
    /// [`PredictiveError::Mismatch`] when the engine does not accept a
    /// position the quantities read.
    pub fn new(
        forward: &'a ForwardModel<E>,
        quantities: &'a Quantities,
    ) -> Result<Self, PredictiveError> {
        for &position in quantities.schedule() {
            forward
                .base_engine()
                .check_position(position)
                .map_err(|reason| {
                    PredictiveError::Mismatch(format!(
                        "A quantity reads the position {position}: {reason}"
                    ))
                })?;
        }
        Ok(Self {
            forward,
            quantities,
        })
    }

    /// The quantities of every replicate of every point,
    /// `[point][replicate][quantity]`.
    ///
    /// # Errors
    ///
    /// [`PredictiveError::Draw`] with the index of the first point that has
    /// no run.
    pub fn replicates(&self, points: &[Vec<f64>]) -> Result<Vec<Vec<Vec<f64>>>, PredictiveError> {
        points
            .par_iter()
            .enumerate()
            .map(|(draw, values)| {
                let runs = self
                    .forward
                    .run_at(values, self.quantities.schedule())
                    .map_err(|error| PredictiveError::Draw { draw, error })?;
                Ok(runs
                    .iter()
                    .map(|rows| self.quantities.evaluate(rows))
                    .collect())
            })
            .collect()
    }

    /// The quantities of every point, each the mean of its replicates',
    /// `[point][quantity]`.
    ///
    /// # Errors
    ///
    /// As [`Self::replicates`].
    pub fn means(&self, points: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, PredictiveError> {
        Ok(self
            .replicates(points)?
            .into_iter()
            .map(|replicates| {
                let count = replicates.len() as f64;
                let mut mean = vec![0.0; self.quantities.len()];
                for values in &replicates {
                    for (sum, value) in mean.iter_mut().zip(values) {
                        *sum += value / count;
                    }
                }
                mean
            })
            .collect())
    }
}
