//! The observation operator: one predicted value per observation from the
//! rows of a run.
//!
//! This is the only code that turns a run into observed quantities. The
//! optimizer, every inference method, the predictive and the scores read
//! observations through it.

use std::collections::HashMap;

use crate::types::{CalibrationParameter, CalibrationParameterType, ObservedDataPoint};

/// How each observation reads a run: the outputs it sums, the row of its
/// position, the row its window starts at, and the factor that multiplies
/// it.
///
/// The rows are those of a run at the positions of the schedule the operator
/// was built with ([`ObservationOperator::new`]).
#[derive(Debug, Clone)]
pub struct ObservationOperator {
    /// Indices of the outputs each observation sums.
    outputs: Vec<Vec<usize>>,
    /// Row of each observation's position.
    rows: Vec<usize>,
    /// Row of the start of each observation's window, if it has one.
    window_rows: Vec<Option<usize>>,
    /// Index of the factor parameter of each observation, if it has one.
    factors: Vec<Option<usize>>,
}

impl ObservationOperator {
    /// The operator of `observed_data` on a run at `positions`.
    ///
    /// `positions` is sorted, deduplicated, and holds every observation's
    /// position and the start of every window. `output_names` are the
    /// outputs of a row, in order.
    ///
    /// # Errors
    ///
    /// Returns a message when an observation has no outputs or names an
    /// unknown output, or its factor is not a factor parameter.
    pub fn new(
        observed_data: &[ObservedDataPoint],
        parameters: &[CalibrationParameter],
        output_names: &[String],
        positions: &[f64],
    ) -> Result<Self, String> {
        let starts: Vec<Option<f64>> = observed_data
            .iter()
            .map(|obs| obs.window.map(|window| obs.at - window))
            .collect();
        Self::with_starts(observed_data, &starts, parameters, output_names, positions)
    }

    /// The operator of `observed_data` whose windows start at `starts`, one
    /// entry per observation and `None` for an observation without a window.
    ///
    /// An observation given by the start of its window reads it here, not at
    /// `at − window`, which can round away from the start. The `window` of
    /// each observation is not read. `positions` holds every start.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    ///
    /// # Panics
    ///
    /// Panics when `starts` does not hold one entry per observation.
    pub fn with_starts(
        observed_data: &[ObservedDataPoint],
        starts: &[Option<f64>],
        parameters: &[CalibrationParameter],
        output_names: &[String],
        positions: &[f64],
    ) -> Result<Self, String> {
        assert_eq!(
            starts.len(),
            observed_data.len(),
            "one start per observation"
        );
        let output_map: HashMap<&str, usize> = output_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();
        let parameter_map: HashMap<&str, usize> = parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| (parameter.id.as_str(), index))
            .collect();
        let row_of = |position: f64| {
            positions
                .binary_search_by(|candidate| candidate.total_cmp(&position))
                .expect("the schedule holds every observed position")
        };

        let mut outputs = Vec::with_capacity(observed_data.len());
        let mut factors = Vec::with_capacity(observed_data.len());
        for obs in observed_data {
            if obs.outputs.is_empty() {
                return Err(format!(
                    "Observation '{}' at {} has an empty list of outputs",
                    obs.label, obs.at
                ));
            }
            let indices = obs
                .outputs
                .iter()
                .map(|name| {
                    output_map.get(name.as_str()).copied().ok_or_else(|| {
                        format!(
                            "Invalid observed output name '{}' for observation '{}' \
                             (available: {})",
                            name,
                            obs.label,
                            output_names.join(", ")
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            outputs.push(indices);
            factors.push(match &obs.factor_id {
                None => None,
                Some(factor_id) => match parameter_map.get(factor_id.as_str()) {
                    Some(&index)
                        if parameters[index].parameter_type == CalibrationParameterType::Factor =>
                    {
                        Some(index)
                    }
                    Some(_) => {
                        return Err(format!(
                            "Parameter '{factor_id}' referenced as factor_id but is not a \
                             Factor parameter"
                        ));
                    }
                    None => {
                        return Err(format!(
                            "Invalid factor_id '{factor_id}' referenced in observed data (not \
                             found in parameters)"
                        ));
                    }
                },
            });
        }

        Ok(Self {
            outputs,
            rows: observed_data.iter().map(|obs| row_of(obs.at)).collect(),
            window_rows: starts.iter().map(|start| start.map(row_of)).collect(),
            factors,
        })
    }

    /// Number of observations.
    pub fn count(&self) -> usize {
        self.rows.len()
    }

    /// Write the prediction of each observation into `out`: the sum of its
    /// outputs, its increment over its window, times its factor.
    ///
    /// `rows` are the rows of a run at the schedule; `parameters` are the
    /// values of the calibration parameters, which hold the factors.
    ///
    /// # Panics
    ///
    /// Panics when `out` does not hold one value per observation.
    pub fn apply(&self, rows: &[Vec<f64>], parameters: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), self.count(), "one prediction per observation");
        let sum_at = |row: usize, outputs: &[usize]| -> f64 {
            rows.get(row)
                .map(|values| outputs.iter().filter_map(|&i| values.get(i)).sum())
                .unwrap_or(0.0)
        };
        for (index, slot) in out.iter_mut().enumerate() {
            let outputs = &self.outputs[index];
            let current = sum_at(self.rows[index], outputs);
            let predicted = match self.window_rows[index] {
                Some(start) => current - sum_at(start, outputs),
                None => current,
            };
            *slot = match self.factors[index] {
                Some(factor) => predicted * parameters[factor],
                None => predicted,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["A".to_string(), "B".to_string(), "C".to_string()]
    }

    #[test]
    fn an_observation_sums_its_outputs_over_its_window_times_its_factor() {
        let mut windowed = ObservedDataPoint::of_sum(
            2.0,
            "AB".to_string(),
            vec!["A".to_string(), "B".to_string()],
            0.0,
        );
        windowed.window = Some(1.0);
        windowed.factor_id = Some("f".to_string());
        let observed = vec![ObservedDataPoint::new(1.0, "C".to_string(), 0.0), windowed];
        let parameters = vec![
            CalibrationParameter::new("k".to_string(), 0.0, 1.0),
            CalibrationParameter::with_type(
                "f".to_string(),
                CalibrationParameterType::Factor,
                0.1,
                10.0,
            ),
        ];
        let operator =
            ObservationOperator::new(&observed, &parameters, &names(), &[1.0, 2.0]).unwrap();
        let rows = vec![vec![1.0, 2.0, 3.0], vec![4.0, 6.0, 9.0]];
        let mut out = [0.0; 2];

        operator.apply(&rows, &[0.5, 2.0], &mut out);

        // C at 1 is 3; (A + B) grows from 3 to 10 and is doubled.
        assert_eq!(out, [3.0, 14.0]);
    }

    #[test]
    fn a_factor_must_be_a_factor_parameter() {
        let observed = vec![ObservedDataPoint::with_factor(
            1.0,
            "A".to_string(),
            0.0,
            "k".to_string(),
        )];
        let parameters = vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)];
        let error = ObservationOperator::new(&observed, &parameters, &names(), &[1.0]).unwrap_err();
        assert!(error.contains("not a Factor parameter"), "{error}");
    }
}
