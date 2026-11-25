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
//!
//! Beside the Jacobian of the observations, the sensitivities hold the
//! Jacobian of the means of the parts of each observation
//! ([`PartSensitivities`]), from the same runs: the curvature of a sum of
//! parts reads it, and the structural rank and the collinearity, which ask
//! what the observations determine, do not.

mod finite_differences;

pub use finite_differences::FiniteDifferences;

use std::fmt;

use commol_calibration::discrepancy::series_rms;
use commol_calibration::{CalibrationParameter, ObservedDataPoint};
use commol_core::{Accuracy, AccuracySettings};

/// A dense matrix of `f64`, row-major in meaning: `(observation, parameter)`.
pub type Matrix = nalgebra::DMatrix<f64>;

/// Which observations a source differentiates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Design {
    /// The positions and outputs of the problem's observations, their values
    /// ignored: can this design determine the parameters?
    Observations,
    /// Every series of the observations, with its outputs, factor and window
    /// and the scale and the background of its first observation, at every
    /// unit position from 0 up to the last observation: can denser data of the
    /// same series determine them?
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
    /// The Jacobian of the means of the parts of the observations.
    pub parts: PartSensitivities,
}

/// The Jacobian of the means of the parts of each observation
/// ([`commol_calibration::ObservationOperator::apply_parts`]) at one point,
/// one row per part, observation after observation.
#[derive(Debug, Clone)]
pub struct PartSensitivities {
    /// The number of parts of each observation.
    pub counts: Vec<usize>,
    /// The mean of each part at the point.
    pub means: Vec<f64>,
    /// `jacobian[(k, j)]`: the derivative of part `k` with respect to
    /// parameter `j`.
    pub jacobian: Matrix,
    /// The estimated error of each entry of `jacobian`.
    pub errors: Matrix,
}

impl PartSensitivities {
    /// The parts of observations of one part each, which are the
    /// observations: their predictions, Jacobian and errors.
    pub fn one_per_observation(predictions: &[f64], jacobian: &Matrix, errors: &Matrix) -> Self {
        Self {
            counts: vec![1; predictions.len()],
            means: predictions.to_vec(),
            jacobian: jacobian.clone(),
            errors: errors.clone(),
        }
    }

    fn columns(&self, columns: &[usize]) -> Self {
        Self {
            counts: self.counts.clone(),
            means: self.means.clone(),
            jacobian: self.jacobian.select_columns(columns),
            errors: self.errors.select_columns(columns),
        }
    }

    /// The observation of each part.
    fn observation_of(&self) -> Vec<usize> {
        self.counts
            .iter()
            .enumerate()
            .flat_map(|(observation, &count)| std::iter::repeat_n(observation, count))
            .collect()
    }
}

/// A prediction or a derivative that is not finite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NotFinite {
    /// Its observation.
    pub observation: usize,
    /// Its part, from 0, of an observation of several parts; `None` for the
    /// whole observation.
    pub part: Option<usize>,
    /// The parameter of a derivative; `None` for a prediction.
    pub column: Option<usize>,
    /// The value.
    pub value: f64,
}

impl Sensitivities {
    /// The first prediction or derivative that is not finite, observation by
    /// observation: in each, its parts before the whole, so that it names the
    /// part, and in each of these the prediction before the derivatives.
    pub(crate) fn first_not_finite(&self) -> Option<NotFinite> {
        let parts = &self.parts;
        let in_row = |prediction: f64, jacobian: &Matrix, row: usize| {
            std::iter::once((None, prediction))
                .chain((0..jacobian.ncols()).map(|j| (Some(j), jacobian[(row, j)])))
                .find(|(_, value)| !value.is_finite())
        };
        let mut first = 0;
        for (observation, &count) in parts.counts.iter().enumerate() {
            let found = (0..count)
                .find_map(|t| {
                    let (column, value) =
                        in_row(parts.means[first + t], &parts.jacobian, first + t)?;
                    Some(((count > 1).then_some(t), column, value))
                })
                .or_else(|| {
                    let prediction = self.predictions[observation];
                    let (column, value) = in_row(prediction, &self.jacobian, observation)?;
                    Some((None, column, value))
                });
            if let Some((part, column, value)) = found {
                return Some(NotFinite {
                    observation,
                    part,
                    column,
                    value,
                });
            }
            first += count;
        }
        None
    }

    /// The same sensitivities restricted to the parameters `columns`, by
    /// index, in that order.
    ///
    /// # Panics
    ///
    /// Panics when an index of `columns` is not a column.
    pub fn columns(&self, columns: &[usize]) -> Self {
        Self {
            predictions: self.predictions.clone(),
            jacobian: self.jacobian.select_columns(columns),
            errors: self.errors.select_columns(columns),
            labels: self.labels.clone(),
            parts: self.parts.columns(columns),
        }
    }

    /// The scaled sensitivities `S_ij = J_ij · θ̃_j / s_i`, with
    /// `θ̃_j = |θ_j|`, or the width of its bounds when `θ_j = 0`; the errors
    /// and the parts, each by the scale of its observation, scale in the
    /// same way.
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
        let scale = |matrix: &Matrix, row_scales: &[f64]| {
            Matrix::from_fn(matrix.nrows(), columns, |i, j| {
                matrix[(i, j)] * magnitudes[j] / row_scales[i]
            })
        };
        let part_scales: Vec<f64> = self
            .parts
            .observation_of()
            .into_iter()
            .map(|observation| scales[observation])
            .collect();
        Self {
            predictions: self.predictions.clone(),
            jacobian: scale(&self.jacobian, scales),
            errors: scale(&self.errors, scales),
            labels: self.labels.clone(),
            parts: PartSensitivities {
                jacobian: scale(&self.parts.jacobian, &part_scales),
                errors: scale(&self.parts.errors, &part_scales),
                ..self.parts.clone()
            },
        }
    }
}

/// `|θ_j|`, or when `θ_j = 0` the width of the bounds of the parameter, 1
/// for an infinite bound ([`CalibrationParameter::width_scale`]).
fn parameter_magnitude(value: f64, parameter: &CalibrationParameter) -> f64 {
    if value == 0.0 {
        parameter.width_scale()
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

    /// The Jacobian of the predicted observations at `values`. Its entries
    /// and the predictions may be not finite.
    ///
    /// # Errors
    ///
    /// A [`SensitivityError`] when the point, or a point the source needs
    /// around it, cannot run.
    fn jacobian(&self, values: &[f64]) -> Result<Sensitivities, SensitivityError>;

    /// The scale `s_i` of each row of `sensitivities`.
    fn scales(&self, sensitivities: &Sensitivities) -> Vec<f64>;

    /// The accuracy of the runs the derivatives come from:
    /// [`Accuracy::EXACT`] by default.
    fn accuracy(&self) -> Accuracy {
        Accuracy::EXACT
    }

    /// The settings that set [`Self::accuracy`], or `None` when no setting
    /// sets it (the default).
    fn accuracy_settings(&self) -> Option<AccuracySettings> {
        None
    }

    /// The same source on runs at `accuracy`, or `None` when no setting sets
    /// the accuracy of its runs (the default).
    fn at_accuracy(&self, _accuracy: Accuracy) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }

    /// The Jacobian at `values` of the parameters `columns`, by index, in
    /// that order: one column each. By default the columns of
    /// [`Self::jacobian`]; a source that differentiates one parameter at a
    /// time differentiates only these.
    ///
    /// # Errors
    ///
    /// As [`Self::jacobian`].
    fn jacobian_of(
        &self,
        values: &[f64],
        columns: &[usize],
    ) -> Result<Sensitivities, SensitivityError> {
        Ok(self.jacobian(values)?.columns(columns))
    }

    /// The scaled Jacobian at `values` of the parameters `columns`, by
    /// index, in that order ([`Self::jacobian_of`],
    /// [`Sensitivities::scaled`]).
    ///
    /// # Errors
    ///
    /// As [`Self::jacobian`].
    fn scaled_jacobian(
        &self,
        values: &[f64],
        columns: &[usize],
    ) -> Result<Sensitivities, SensitivityError> {
        let sensitivities = self.jacobian_of(values, columns)?;
        let scales = self.scales(&sensitivities);
        let parameters: Vec<CalibrationParameter> = columns
            .iter()
            .map(|&j| self.parameters()[j].clone())
            .collect();
        let values: Vec<f64> = columns.iter().map(|&j| values[j]).collect();
        Ok(sensitivities.scaled(&values, &parameters, &scales))
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
        let errors = jacobian.clone() * 0.5;
        let sensitivities = Sensitivities {
            predictions: vec![0.0, 0.0],
            parts: PartSensitivities::one_per_observation(&[0.0, 0.0], &jacobian, &errors),
            errors,
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
        assert_eq!(scaled.errors, expected.clone() * 0.5);
        assert_eq!(scaled.parts.jacobian, expected);
    }

    /// Observations of 1, 3 and 2 parts and two parameters, every entry 1.
    fn of_parts() -> Sensitivities {
        let ones = |rows: usize| Matrix::from_element(rows, 2, 1.0);
        Sensitivities {
            predictions: vec![1.0; 3],
            jacobian: ones(3),
            errors: ones(3),
            labels: vec!["A".to_string(), "B".to_string(), "B".to_string()],
            parts: PartSensitivities {
                counts: vec![1, 3, 2],
                means: vec![1.0; 6],
                jacobian: ones(6),
                errors: ones(6),
            },
        }
    }

    #[test]
    fn the_first_entry_that_is_not_finite_names_its_place() {
        assert_eq!(of_parts().first_not_finite(), None);
        // Each row: its index, its observation and its part.
        let rows_of_parts = [
            (0, 0, None),
            (1, 1, Some(0)),
            (2, 1, Some(1)),
            (3, 1, Some(2)),
            (4, 2, Some(0)),
            (5, 2, Some(1)),
        ];
        let rows_of_wholes = [(0, 0, None), (1, 1, None), (2, 2, None)];
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            for column in [None, Some(0), Some(1)] {
                let place = |changed: &Sensitivities| {
                    let found = changed.first_not_finite().unwrap();
                    assert_eq!(found.value.to_bits(), value.to_bits());
                    assert_eq!(found.column, column);
                    (found.observation, found.part)
                };
                for (row, observation, part) in rows_of_parts {
                    let mut changed = of_parts();
                    match column {
                        None => changed.parts.means[row] = value,
                        Some(j) => changed.parts.jacobian[(row, j)] = value,
                    }
                    assert_eq!(place(&changed), (observation, part));
                }
                for (row, observation, part) in rows_of_wholes {
                    let mut changed = of_parts();
                    match column {
                        None => changed.predictions[row] = value,
                        Some(j) => changed.jacobian[(row, j)] = value,
                    }
                    assert_eq!(place(&changed), (observation, part));
                }
            }
        }
    }

    #[test]
    fn a_part_that_is_not_finite_comes_before_its_whole() {
        let mut changed = of_parts();
        changed.predictions[2] = f64::INFINITY;
        changed.parts.jacobian[(5, 1)] = f64::NEG_INFINITY;

        assert_eq!(
            changed.first_not_finite(),
            Some(NotFinite {
                observation: 2,
                part: Some(1),
                column: Some(1),
                value: f64::NEG_INFINITY,
            })
        );
        // An earlier observation, and its prediction before its derivatives.
        changed.jacobian[(0, 0)] = f64::INFINITY;
        changed.predictions[0] = f64::NAN;
        let found = changed.first_not_finite().unwrap();
        assert_eq!((found.observation, found.column), (0, None));
    }
}
