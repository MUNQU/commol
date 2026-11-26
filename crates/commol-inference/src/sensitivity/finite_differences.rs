//! Finite differences of the forward model.
//!
//! The step of parameter `j` is `h_j = η^{1/3} · max(|θ_j|, w_j)`, at most
//! `w_j / 8`, where `w_j` is the width of its bounds (1 when a bound is
//! infinite, so the step stays finite) and `η` the output
//! relative accuracy of the engine ([`SimulationEngine::accuracy`]): the step
//! that balances the truncation error of a difference of second order with
//! the error of the runs. The difference is central when `θ_j ± 2h_j` is
//! inside the bounds, and one-sided of second order towards the inside
//! otherwise, so no run leaves the bounds. Each derivative is computed with
//! the steps `h` and `2h`; their difference estimates the truncation error.
//! Every run is finite, but its observation, a scale times its outputs plus
//! a background, or the quotient of a difference can overflow.

use std::collections::BTreeSet;

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, EvaluationError, ForwardModel, ObservedDataPoint,
    Workspace,
};
use commol_core::{Accuracy, AccuracySettings, SimulationEngine};
use rayon::prelude::*;

use super::{
    Design, Matrix, PartSensitivities, Sensitivities, SensitivityError, SensitivitySource,
    observation_rms,
};
use commol_calibration::discrepancy::series_rms;

/// Finite differences of the predicted observations of a problem.
#[derive(Debug, Clone)]
pub struct FiniteDifferences<E: SimulationEngine> {
    forward: ForwardModel<E>,
    observed_data: Vec<ObservedDataPoint>,
    design: Design,
    accuracy: f64,
}

/// How a column is differenced: the offsets of its runs, in steps, and the
/// coefficients of the derivative with steps `h` and `2h`.
struct Stencil {
    /// Offsets in units of `h`, the point itself excluded.
    offsets: &'static [f64],
    /// Coefficients of the runs at `[0, offsets...]` for the step `h`,
    /// divided by `h`.
    with_h: &'static [f64],
    /// The same for the step `2h`, divided by `h`.
    with_2h: &'static [f64],
    /// `Σ|c|` of `with_h`: how much the error of the runs grows.
    rounding: f64,
}

const CENTRAL: Stencil = Stencil {
    offsets: &[-2.0, -1.0, 1.0, 2.0],
    with_h: &[0.0, 0.0, -0.5, 0.5, 0.0],
    with_2h: &[0.0, -0.25, 0.0, 0.0, 0.25],
    rounding: 1.0,
};

/// Forward differences of second order; a backward one negates the offsets
/// and the derivative.
const ONE_SIDED: Stencil = Stencil {
    offsets: &[1.0, 2.0, 4.0],
    with_h: &[-1.5, 2.0, -0.5, 0.0],
    with_2h: &[-0.75, 0.0, 1.0, -0.25],
    rounding: 4.0,
};

impl<E: SimulationEngine + Send + Sync> FiniteDifferences<E> {
    /// Finite differences of `problem` on `design`.
    ///
    /// # Errors
    ///
    /// [`SensitivityError::Stochastic`] when the problem runs replicates of a
    /// stochastic engine; a failure when the dense design has a position the
    /// engine rejects.
    pub fn new(problem: &CalibrationProblem<E>, design: Design) -> Result<Self, SensitivityError> {
        if problem.forward().replicates().is_some() {
            return Err(SensitivityError::Stochastic);
        }
        let observed_data = match design {
            Design::Observations => problem.observed_data().to_vec(),
            Design::Dense => dense_observations(problem.observed_data()),
        };
        let forward = match design {
            Design::Observations => problem.forward().clone(),
            Design::Dense => problem
                .forward()
                .with_observations(&observed_data)
                .map_err(SensitivityError::Failed)?,
        };
        let accuracy = forward.base_engine().accuracy().relative;
        Ok(Self {
            forward,
            observed_data,
            design,
            accuracy,
        })
    }

    /// Finite differences of the predictions of `observed_data`, which need
    /// not be the problem's: the engine, parameters and values of `problem`
    /// observed there, their values ignored.
    ///
    /// # Errors
    ///
    /// [`SensitivityError::Stochastic`] when the problem runs replicates of a
    /// stochastic engine; a failure when the engine rejects a position or an
    /// observation names an unknown output or factor.
    pub fn of_observations(
        problem: &CalibrationProblem<E>,
        observed_data: Vec<ObservedDataPoint>,
    ) -> Result<Self, SensitivityError> {
        if problem.forward().replicates().is_some() {
            return Err(SensitivityError::Stochastic);
        }
        let forward = problem
            .forward()
            .with_observations(&observed_data)
            .map_err(SensitivityError::Failed)?;
        let accuracy = forward.base_engine().accuracy().relative;
        Ok(Self {
            forward,
            observed_data,
            design: Design::Observations,
            accuracy,
        })
    }

    /// The step `h_j` of each parameter at `values`.
    pub fn steps(&self, values: &[f64]) -> Vec<f64> {
        let factor = self.accuracy.cbrt();
        values
            .iter()
            .zip(self.forward.parameters())
            .map(|(&value, parameter)| {
                let width = parameter.max_bound - parameter.min_bound;
                (factor * value.abs().max(parameter.width_scale())).min(width / 8.0)
            })
            .collect()
    }

    /// The predicted observations of a run at `values`, then the means of
    /// their parts.
    fn run(&self, values: &[f64], workspace: &mut Workspace) -> Result<Vec<f64>, EvaluationError> {
        self.forward.run_observed(values, workspace)?;
        Ok(workspace
            .predictions
            .iter()
            .chain(&workspace.parts)
            .copied()
            .collect())
    }

    /// The background of every observation, then of every part.
    fn backgrounds(&self) -> Vec<f64> {
        let observations = self.observed_data.iter().map(|obs| obs.background);
        let parts = self
            .observed_data
            .iter()
            .flat_map(|obs| std::iter::repeat_n(obs.part_background(), obs.parts.max(1)));
        observations.chain(parts).collect()
    }

    /// The derivative of every observation, then of every part, with
    /// respect to parameter `j`, and its estimated error.
    fn column(
        &self,
        values: &[f64],
        center: &[f64],
        backgrounds: &[f64],
        j: usize,
        step: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), SensitivityError> {
        let parameter = &self.forward.parameters()[j];
        let rows = center.len();
        let value = values[j];
        let (stencil, direction) = if value - 2.0 * step >= parameter.min_bound
            && value + 2.0 * step <= parameter.max_bound
        {
            (&CENTRAL, 1.0)
        } else if value + 4.0 * step <= parameter.max_bound {
            (&ONE_SIDED, 1.0)
        } else {
            (&ONE_SIDED, -1.0)
        };
        let mut workspace = Workspace::default();
        let mut point = values.to_vec();
        let mut runs = vec![center.to_vec()];
        for &offset in stencil.offsets {
            point[j] = value + direction * offset * step;
            let predictions = self
                .run(&point, &mut workspace)
                .map_err(|error| match error {
                    EvaluationError::Violation(violation) => SensitivityError::Infeasible {
                        parameter: parameter.id.clone(),
                        reason: violation.to_string(),
                    },
                    EvaluationError::Failed(message) => SensitivityError::Failed(message),
                })?;
            runs.push(predictions);
        }
        let derivative = |coefficients: &[f64], i: usize| {
            direction
                * coefficients
                    .iter()
                    .zip(&runs)
                    .map(|(c, run)| c * run[i])
                    .sum::<f64>()
                / step
        };
        let mut column = Vec::with_capacity(rows);
        let mut errors = Vec::with_capacity(rows);
        for (i, &background) in backgrounds.iter().enumerate() {
            let with_h = derivative(stencil.with_h, i);
            let with_2h = derivative(stencil.with_2h, i);
            // The error of a run is a share of what it gives; the background
            // is exact, and only its sum is rounded.
            let size = runs
                .iter()
                .map(|run| (run[i] - background).abs())
                .fold(0.0, f64::max);
            let error = self.accuracy * size + f64::EPSILON * background;
            column.push(with_h);
            errors.push((with_h - with_2h).abs() / 3.0 + error * stencil.rounding / step);
        }
        Ok((column, errors))
    }
}

impl<E: SimulationEngine + Send + Sync> SensitivitySource for FiniteDifferences<E> {
    fn parameters(&self) -> &[CalibrationParameter] {
        self.forward.parameters()
    }

    fn jacobian(&self, values: &[f64]) -> Result<Sensitivities, SensitivityError> {
        let all: Vec<usize> = (0..values.len()).collect();
        self.jacobian_of(values, &all)
    }

    /// Differences of the parameters `columns` only.
    fn jacobian_of(
        &self,
        values: &[f64],
        columns: &[usize],
    ) -> Result<Sensitivities, SensitivityError> {
        let mut workspace = Workspace::default();
        let center = self
            .run(values, &mut workspace)
            .map_err(|error| match error {
                EvaluationError::Violation(violation) => {
                    SensitivityError::InfeasiblePoint(violation.to_string())
                }
                EvaluationError::Failed(message) => SensitivityError::Failed(message),
            })?;
        let steps = self.steps(values);
        let backgrounds = self.backgrounds();
        let columns: Vec<(Vec<f64>, Vec<f64>)> = columns
            .par_iter()
            .map(|&j| self.column(values, &center, &backgrounds, j, steps[j]))
            .collect::<Result<_, _>>()?;
        let rows = self.observed_data.len();
        let parts = center.len() - rows;
        let block = |first: usize, count: usize, errors: bool| {
            Matrix::from_fn(count, columns.len(), |i, j| {
                let (derivatives, error) = &columns[j];
                let column = if errors { error } else { derivatives };
                column[first + i]
            })
        };
        Ok(Sensitivities {
            predictions: center[..rows].to_vec(),
            jacobian: block(0, rows, false),
            errors: block(0, rows, true),
            labels: self
                .observed_data
                .iter()
                .map(|obs| obs.label.clone())
                .collect(),
            parts: PartSensitivities {
                counts: self.observed_data.iter().map(|obs| obs.parts).collect(),
                means: center[rows..].to_vec(),
                jacobian: block(rows, parts, false),
                errors: block(rows, parts, true),
            },
        })
    }

    fn scales(&self, sensitivities: &Sensitivities) -> Vec<f64> {
        match self.design {
            Design::Observations => observation_rms(&self.observed_data),
            Design::Dense => series_rms(&sensitivities.labels, &sensitivities.predictions),
        }
    }

    fn accuracy(&self) -> Accuracy {
        self.forward.base_engine().accuracy()
    }

    fn accuracy_settings(&self) -> Option<AccuracySettings> {
        self.forward.base_engine().accuracy_settings()
    }

    /// The same differences of runs of the engine at `accuracy`
    /// ([`SimulationEngine::at_accuracy`]), whose steps follow it.
    fn at_accuracy(&self, accuracy: Accuracy) -> Option<Self> {
        let forward = self.forward.at_accuracy(accuracy)?;
        Some(Self {
            accuracy: forward.base_engine().accuracy().relative,
            forward,
            observed_data: self.observed_data.clone(),
            design: self.design,
        })
    }
}

/// Every series of `observed_data` (its label, outputs, coefficients, factor
/// and window, with the scale and the background of its first observation) at
/// every unit position from 0 up to the last observation, and at the positions
/// of the observations, each of one part; a series with a window starts once a
/// whole window fits.
fn dense_observations(observed_data: &[ObservedDataPoint]) -> Vec<ObservedDataPoint> {
    let last = observed_data.iter().map(|obs| obs.at).fold(0.0, f64::max);
    let mut series: Vec<&ObservedDataPoint> = Vec::new();
    for obs in observed_data {
        let same = |other: &&ObservedDataPoint| {
            other.label == obs.label
                && other.outputs == obs.outputs
                && other.output_coefficients() == obs.output_coefficients()
                && other.factor_id == obs.factor_id
                && other.window == obs.window
        };
        if !series.iter().any(same) {
            series.push(obs);
        }
    }
    let mut positions: BTreeSet<u64> = (0..=last.floor() as u64)
        .map(|t| (t as f64).to_bits())
        .collect();
    // `−0.0` is the unit position `0.0`.
    positions.extend(observed_data.iter().map(|obs| (obs.at + 0.0).to_bits()));
    let mut positions: Vec<f64> = positions.into_iter().map(f64::from_bits).collect();
    positions.sort_by(f64::total_cmp);
    series
        .iter()
        .flat_map(|obs| {
            let start = obs.window.unwrap_or(0.0);
            positions
                .iter()
                .filter(move |&&at| at >= start)
                .map(move |&at| ObservedDataPoint {
                    at,
                    value: 0.0,
                    weight: 1.0,
                    parts: 1,
                    ..(*obs).clone()
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observations_with_other_coefficients_are_another_series() {
        let plain = ObservedDataPoint::of_sum(
            1.0,
            "seen".to_string(),
            vec!["A".to_string(), "B".to_string()],
            0.0,
        );
        let ones = ObservedDataPoint {
            coefficients: Some(vec![1.0, 1.0]),
            ..plain.clone()
        };
        let weighted = ObservedDataPoint {
            coefficients: Some(vec![1.0, 0.6]),
            ..plain.clone()
        };

        // Coefficients of 1 are the plain sum: one series at 0 and 1.
        assert_eq!(dense_observations(&[plain.clone(), ones]).len(), 2);
        // Other coefficients are a second series.
        let dense = dense_observations(&[plain, weighted]);
        assert_eq!(dense.len(), 4);
        assert_eq!(dense[3].coefficients, Some(vec![1.0, 0.6]));
    }

    #[test]
    fn the_dense_design_keeps_the_background_of_its_series() {
        let first = ObservedDataPoint {
            background: 2.0,
            ..ObservedDataPoint::new(1.0, "A".to_string(), 0.0)
        };
        let other = ObservedDataPoint {
            background: 5.0,
            ..ObservedDataPoint::new(2.0, "A".to_string(), 0.0)
        };

        // One series, at 0, 1 and 2, with the background of its first
        // observation.
        let backgrounds: Vec<f64> = dense_observations(&[first, other])
            .iter()
            .map(|obs| obs.background)
            .collect();
        assert_eq!(backgrounds, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn minus_zero_is_the_unit_position_zero() {
        let dense = dense_observations(&[
            ObservedDataPoint::new(-0.0, "A".to_string(), 0.0),
            ObservedDataPoint::new(1.0, "A".to_string(), 0.0),
        ]);

        let positions: Vec<u64> = dense.iter().map(|obs| obs.at.to_bits()).collect();
        assert_eq!(positions, [0.0f64.to_bits(), 1.0f64.to_bits()]);
    }
}
