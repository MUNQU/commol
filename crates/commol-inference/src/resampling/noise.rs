//! Residual noise: the observation noise of a fit with no noise model.

use commol_calibration::ObservedDataPoint;
use commol_random::Generator;

use crate::error::MethodError;
use crate::predictive::ObservationNoise;
use crate::resampling::blocks::{
    check_length, distinct_positions, one_per_position, position_indices,
};
use crate::resampling::residuals::{
    ResidualBlocks, ResidualScale, Residuals, SeriesResiduals, source_positions,
};

/// Observation noise made of the residuals of a fit: each draw adds moving
/// blocks of residuals, drawn with replacement across every series at once,
/// to the predictions, on the scale of the residuals.
///
/// The positions of the observations it serves need not be those of the
/// fit: the residual of a series at a drawn position is the one at the
/// nearest position where the series has one. It serves only series the fit
/// has residuals of, each with one observation per position: a position of
/// a series draws one residual.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualNoise {
    scale: ResidualScale,
    length: usize,
    grid: SeriesResiduals,
}

impl ResidualNoise {
    /// The noise of `residuals`, the residuals of `observations`, drawn in
    /// blocks of `length` positions.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when `length` is 0 or larger than the number
    /// of distinct positions of `observations`, or when a series has two
    /// observations at one position.
    pub fn new(
        observations: &[ObservedDataPoint],
        residuals: &Residuals,
        length: usize,
    ) -> Result<Self, MethodError> {
        check_length(length, distinct_positions(observations).len())
            .map_err(MethodError::Invalid)?;
        Ok(Self {
            scale: residuals.scale(),
            length,
            grid: SeriesResiduals::new(observations, residuals.residuals())
                .map_err(MethodError::Invalid)?,
        })
    }
}

impl ObservationNoise for ResidualNoise {
    fn check(&self, observations: &[ObservedDataPoint]) -> Result<(), String> {
        match observations
            .iter()
            .find(|obs| self.grid.series(&obs.label).is_none())
        {
            Some(obs) => Err(format!(
                "Observation '{}' at {}: the fit has no residuals of the series '{}', so \
                 residual noise cannot add any",
                obs.label, obs.at, obs.label
            )),
            None => one_per_position(
                observations,
                "Residual noise draws one residual per position of a series",
                "",
            ),
        }
    }

    fn observe(
        &self,
        observations: &[ObservedDataPoint],
        _values: &[f64],
        predicted: &[f64],
        generator: &mut Generator,
        out: &mut [f64],
    ) {
        let positions = distinct_positions(observations);
        let source = source_positions(
            self.grid.positions(),
            self.length,
            positions.len(),
            ResidualBlocks::Moving,
            generator,
        );
        for (((obs, position), &predicted), shown) in observations
            .iter()
            .zip(position_indices(observations, &positions))
            .zip(predicted)
            .zip(out.iter_mut())
        {
            let series = self
                .grid
                .series(&obs.label)
                .expect("checked before the draws");
            *shown = self
                .scale
                .apply(predicted, self.grid.at(series, source[position]));
        }
    }
}
