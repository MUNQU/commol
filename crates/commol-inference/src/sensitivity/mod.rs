//! Sensitivities: the Jacobian of the predicted observations with respect to
//! the parameters, with an estimated error of each entry.
//!
//! [`SensitivitySource`] is the one port every reader of derivatives asks;
//! only a source knows how its derivatives are made. [`FiniteDifferences`]
//! is the first source.
//!
//! The scaled Jacobian `S_ij = J_ij · θ̃_j / s_i` is the change of
//! observation `i`, relative to the scale `s_i` of its series, for a relative
//! change of parameter `j`, so it does not depend on the units of either.

mod finite_differences;

pub use finite_differences::FiniteDifferences;

use std::fmt;

use commol_calibration::discrepancy::series_rms;
use commol_calibration::{CalibrationParameter, ObservedDataPoint};

/// A dense matrix of `f64`, row-major in meaning: `(observation, parameter)`.
pub type Matrix = nalgebra::DMatrix<f64>;

/// Which observations a source differentiates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Design {
    /// The positions and outputs of the problem's observations, their values
    /// ignored: can this design determine the parameters?
    Observations,
    /// Every series of the observations, with its outputs, factor and window
    /// and the scale of its first observation, at every unit position from 0
    /// up to the last observation: can denser data of the same series
    /// determine them?
    Dense,
}

/// The Jacobian of the predicted observations at one point.
#[derive(Debug, Clone)]
pub struct Sensitivities {
    /// The predicted observations at the point.
    pub predictions: Vec<f64>,
    /// `jacobian[(i, j)]`: the derivative of observation `i` with respect to
    /// parameter `j`.
    pub jacobian: Matrix,
    /// The estimated error of each entry of `jacobian`.
    pub errors: Matrix,
    /// The series (label) of each observation.
    pub labels: Vec<String>,
}

impl Sensitivities {
    /// The scaled sensitivities `S_ij = J_ij · θ̃_j / s_i`, with
    /// `θ̃_j = |θ_j|`, or the width of its bounds when `θ_j = 0`; the errors
    /// scale in the same way.
    ///
    /// # Panics
    ///
    /// Panics when `values` or `parameters` do not have one entry per column,
    /// or `scales` one per row.
    pub fn scaled(
        &self,
        values: &[f64],
        parameters: &[CalibrationParameter],
        scales: &[f64],
    ) -> Self {
        let columns = self.jacobian.ncols();
        assert_eq!(values.len(), columns, "one value per parameter");
        assert_eq!(parameters.len(), columns, "one parameter per column");
        assert_eq!(scales.len(), self.jacobian.nrows(), "one scale per row");
        let magnitudes: Vec<f64> = values
            .iter()
            .zip(parameters)
            .map(|(&value, parameter)| parameter_magnitude(value, parameter))
            .collect();
        let scale = |matrix: &Matrix| {
            Matrix::from_fn(matrix.nrows(), columns, |i, j| {
                matrix[(i, j)] * magnitudes[j] / scales[i]
            })
        };
        Self {
            predictions: self.predictions.clone(),
            jacobian: scale(&self.jacobian),
            errors: scale(&self.errors),
            labels: self.labels.clone(),
        }
    }
}

/// `|θ_j|`, or the width of the bounds of the parameter when `θ_j = 0`.
fn parameter_magnitude(value: f64, parameter: &CalibrationParameter) -> f64 {
    if value == 0.0 {
        parameter.max_bound - parameter.min_bound
    } else {
        value.abs()
    }
}

/// Why a source could not differentiate at a point.
#[derive(Debug, Clone, PartialEq)]
pub enum SensitivityError {
    /// A point a difference of `parameter` runs is infeasible.
    Infeasible {
        /// The parameter whose difference left the feasible values.
        parameter: String,
        /// Why the point is infeasible.
        reason: String,
    },
    /// The point itself is infeasible.
    InfeasiblePoint(String),
    /// The engine is stochastic: its runs are piecewise constant in the
    /// parameters, so differences are zero or jumps.
    Stochastic,
    /// Any other failure, such as an engine error.
    Failed(String),
}

impl fmt::Display for SensitivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Infeasible { parameter, reason } => write!(
                f,
                "A finite difference of '{parameter}' runs an infeasible point: {reason}"
            ),
            Self::InfeasiblePoint(reason) => write!(f, "The point is infeasible: {reason}"),
            Self::Stochastic => f.write_str(
                "Derivatives need a deterministic engine: the runs of a stochastic engine \
                 change in jumps with its parameters",
            ),
            Self::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for SensitivityError {}

/// The port every reader of derivatives asks.
pub trait SensitivitySource: Sync {
    /// The parameters, in problem order: one column each.
    fn parameters(&self) -> &[CalibrationParameter];

    /// The Jacobian of the predicted observations at `values`.
    ///
    /// # Errors
    ///
    /// A [`SensitivityError`] when the point, or a point the source needs
    /// around it, cannot run.
    fn jacobian(&self, values: &[f64]) -> Result<Sensitivities, SensitivityError>;

    /// The scale `s_i` of each row of `sensitivities`.
    fn scales(&self, sensitivities: &Sensitivities) -> Vec<f64>;

    /// The scaled Jacobian at `values` ([`Sensitivities::scaled`]).
    ///
    /// # Errors
    ///
    /// As [`Self::jacobian`].
    fn scaled_jacobian(&self, values: &[f64]) -> Result<Sensitivities, SensitivityError> {
        let sensitivities = self.jacobian(values)?;
        let scales = self.scales(&sensitivities);
        Ok(sensitivities.scaled(values, self.parameters(), &scales))
    }
}

/// The root mean square of the observed values of the series of each
/// observation, or 1 for a series whose values are all zero: the scale of
/// each row of the sensitivities of the design of the observations.
pub fn observation_rms(observed_data: &[ObservedDataPoint]) -> Vec<f64> {
    let labels: Vec<String> = observed_data.iter().map(|obs| obs.label.clone()).collect();
    let values: Vec<f64> = observed_data.iter().map(|obs| obs.value).collect();
    series_rms(&labels, &values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stochastic_engine_has_no_derivatives() {
        assert_eq!(
            SensitivityError::Stochastic.to_string(),
            "Derivatives need a deterministic engine: the runs of a stochastic engine change \
             in jumps with its parameters"
        );
    }

    #[test]
    fn scaling_multiplies_by_the_magnitude_and_divides_by_the_scale() {
        let jacobian = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let sensitivities = Sensitivities {
            predictions: vec![0.0, 0.0],
            errors: jacobian.clone() * 0.5,
            jacobian,
            labels: vec!["A".to_string(), "B".to_string()],
        };
        let parameters = [
            CalibrationParameter::new("k1".to_string(), 0.0, 10.0),
            CalibrationParameter::new("k2".to_string(), 1.0, 4.0),
        ];

        // θ̃ = (|2|, the width 3 of [1, 4] for k2 = 0) and s = (2, 4).
        let scaled = sensitivities.scaled(&[2.0, 0.0], &parameters, &[2.0, 4.0]);

        let expected = Matrix::from_row_slice(2, 2, &[1.0, 3.0, 1.5, 3.0]);
        assert_eq!(scaled.jacobian, expected);
        assert_eq!(scaled.errors, expected * 0.5);
    }
}
