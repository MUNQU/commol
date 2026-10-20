//! Finite differences of the forward model.
//!
//! The step of parameter `j` is `h_j = η^{1/3} · max(|θ_j|, w_j)`, at most
//! `w_j / 8`, where `w_j` is the width of its bounds (1 when a bound is
//! infinite, so the step stays finite) and `η` the output
//! accuracy of the engine ([`SimulationEngine::output_accuracy`]): the step
//! that balances the truncation error of a difference of second order with
//! the error of the runs. The difference is central when `θ_j ± 2h_j` is
//! inside the bounds, and one-sided of second order towards the inside
//! otherwise, so no run leaves the bounds. Each derivative is computed with
//! the steps `h` and `2h`; their difference estimates the truncation error.

use std::collections::BTreeSet;

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, EvaluationError, ForwardModel, ObservedDataPoint,
    Violation, Workspace,
};
use commol_core::SimulationEngine;
use rayon::prelude::*;

use super::{
    Design, Matrix, Sensitivities, SensitivityError, SensitivitySource, observation_scales,
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
        let accuracy = forward.base_engine().output_accuracy();
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
        let accuracy = forward.base_engine().output_accuracy();
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
                let scale = if width.is_finite() { width } else { 1.0 };
                (factor * value.abs().max(scale)).min(width / 8.0)
            })
            .collect()
    }

    fn run(&self, values: &[f64], workspace: &mut Workspace) -> Result<Vec<f64>, EvaluationError> {
        self.forward
            .run_observed(values, workspace)
            .map(<[f64]>::to_vec)
    }

    /// The derivative of every observation with respect to parameter `j`,
    /// and its estimated error.
    fn column(
        &self,
        values: &[f64],
        center: &[f64],
        j: usize,
        step: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), SensitivityError> {
        let parameter = &self.forward.parameters()[j];
        let rows = center.len();
        if step <= 0.0 {
            return Ok((vec![0.0; rows], vec![0.0; rows]));
        }
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
                        reason: reason(&violation),
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
        for i in 0..rows {
            let with_h = derivative(stencil.with_h, i);
            let with_2h = derivative(stencil.with_2h, i);
            let size = runs.iter().map(|run| run[i].abs()).fold(0.0, f64::max);
            column.push(with_h);
            errors.push(
                (with_h - with_2h).abs() / 3.0 + self.accuracy * size * stencil.rounding / step,
            );
        }
        Ok((column, errors))
    }
}

impl<E: SimulationEngine + Send + Sync> SensitivitySource for FiniteDifferences<E> {
    fn parameters(&self) -> &[CalibrationParameter] {
        self.forward.parameters()
    }

    fn jacobian(&self, values: &[f64]) -> Result<Sensitivities, SensitivityError> {
        let mut workspace = Workspace::default();
        let center = self
            .run(values, &mut workspace)
            .map_err(|error| match error {
                EvaluationError::Violation(violation) => {
                    SensitivityError::InfeasiblePoint(reason(&violation))
                }
                EvaluationError::Failed(message) => SensitivityError::Failed(message),
            })?;
        let steps = self.steps(values);
        let columns: Vec<(Vec<f64>, Vec<f64>)> = (0..values.len())
            .into_par_iter()
            .map(|j| self.column(values, &center, j, steps[j]))
            .collect::<Result<_, _>>()?;
        let rows = center.len();
        let jacobian = Matrix::from_fn(rows, columns.len(), |i, j| columns[j].0[i]);
        let errors = Matrix::from_fn(rows, columns.len(), |i, j| columns[j].1[i]);
        Ok(Sensitivities {
            predictions: center,
            jacobian,
            errors,
            labels: self
                .observed_data
                .iter()
                .map(|obs| obs.label.clone())
                .collect(),
        })
    }

    fn scales(&self, sensitivities: &Sensitivities) -> Vec<f64> {
        match self.design {
            Design::Observations => observation_scales(&self.observed_data),
            Design::Dense => series_rms(&sensitivities.labels, &sensitivities.predictions),
        }
    }
}

/// Why a run of the forward model is infeasible, in words.
fn reason(violation: &Violation) -> String {
    match violation {
        Violation::InfeasibleStart { excess } => {
            format!("the starting state is infeasible (excess {excess})")
        }
        Violation::InfeasibleRun => "the run leaves the values the model allows".to_string(),
        Violation::NonFinite => "a value of the run is not finite".to_string(),
        Violation::OutOfBounds { .. }
        | Violation::Constraint { .. }
        | Violation::MeanOutsideModel { .. } => {
            unreachable!(
                "a run of the forward model checks no bounds, no constraint and no observation \
                 model"
            )
        }
    }
}

/// Every series of `observed_data` (its label, outputs, factor and window)
/// at every unit position from 0 up to the last observation, and at the
/// positions of the observations; a series with a window starts once a whole
/// window fits.
fn dense_observations(observed_data: &[ObservedDataPoint]) -> Vec<ObservedDataPoint> {
    let last = observed_data.iter().map(|obs| obs.at).fold(0.0, f64::max);
    let mut series: Vec<&ObservedDataPoint> = Vec::new();
    for obs in observed_data {
        let same = |other: &&ObservedDataPoint| {
            other.label == obs.label
                && other.outputs == obs.outputs
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
    positions.extend(observed_data.iter().map(|obs| obs.at.to_bits()));
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
                    ..(*obs).clone()
                })
        })
        .collect()
}
