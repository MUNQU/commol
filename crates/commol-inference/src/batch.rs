//! The batch runner: every point of a design through a forward model, each
//! run reduced to its quantities of interest.
//!
//! The points run in parallel. With a stochastic engine each point runs the
//! replicates of the forward model, the same seeds at every point (common
//! random numbers), so two points differ only by their values, and
//! [`Quantities::combine`] combines the replicates of each point.

use commol_calibration::ForwardModel;
use commol_core::SimulationEngine;
use rayon::prelude::*;

use crate::predictive::{NanRun, PredictiveError};
use crate::quantity::{Quantities, Term, Undefined};

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
        Ok(self
            .terms(points)?
            .into_iter()
            .map(|runs| {
                runs.into_iter()
                    .map(|terms| terms.into_iter().map(Term::value).collect())
                    .collect()
            })
            .collect())
    }

    /// The [`Quantities::terms`] of every replicate of every point,
    /// `[point][replicate][quantity]`.
    fn terms(&self, points: &[Vec<f64>]) -> Result<Vec<Vec<Vec<Term>>>, PredictiveError> {
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
                    .map(|rows| self.quantities.terms(rows))
                    .collect())
            })
            .collect()
    }

    /// The quantities of every point, each its replicates combined by
    /// [`Quantities::combine`] (the mean of each quantity, and the rule of
    /// each ratio), `[point][quantity]`. Every method that reduces
    /// quantities across the points of a design reads them here.
    ///
    /// # Errors
    ///
    /// As [`Self::replicates`]; [`PredictiveError::NotANumber`] with the
    /// first quantity that has no value at a point, every point where it
    /// has none, and the first such point, with the replicate that is NaN
    /// when one is.
    pub fn means(&self, points: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, PredictiveError> {
        let combined: Vec<Vec<Result<f64, Undefined>>> = self
            .terms(points)?
            .iter()
            .map(|runs| self.quantities.combine(runs))
            .collect();
        let first =
            (0..self.quantities.len()).find(|&q| combined.iter().any(|values| values[q].is_err()));
        if let Some(quantity) = first {
            let points_nan: Vec<usize> = (0..combined.len())
                .filter(|&point| combined[point][quantity].is_err())
                .collect();
            let draw = points_nan[0];
            let replicate = match combined[draw][quantity] {
                Err(Undefined::Replicate(replicate)) => Some(replicate),
                _ => None,
            };
            return Err(PredictiveError::NotANumber {
                quantity,
                points: points_nan,
                run: Some(Box::new(NanRun {
                    draw,
                    replicate,
                    seed: replicate
                        .and_then(|replicate| Some(self.forward.replicates()?.seeds[replicate])),
                    inputs: points[draw].clone(),
                })),
            });
        }
        Ok(combined
            .into_iter()
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| value.expect("checked"))
                    .collect()
            })
            .collect())
    }
}
