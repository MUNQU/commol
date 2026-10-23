//! The observation operator: one predicted value per observation from the
//! rows of a run.
//!
//! This is the only code that turns a run into observed quantities. The
//! optimizer, every inference method, the predictive and the scores read
//! observations through it.

use std::collections::HashMap;

use commol_core::{POSITION_TOLERANCE, same_position};

use crate::types::{CalibrationParameter, CalibrationParameterType, ObservedDataPoint};

/// The index of `at` in `positions`, ascending: the entry equal to it, or
/// else a neighbour within the tolerance of the axis, so a position computed
/// as `end − window` finds the entry of its start. None when no entry is the
/// same position.
pub fn index_at(positions: &[f64], at: f64) -> Option<usize> {
    match positions.binary_search_by(|candidate| candidate.total_cmp(&at)) {
        Ok(index) => Some(index),
        Err(index) => [index.checked_sub(1), Some(index)]
            .into_iter()
            .flatten()
            .find(|&candidate| {
                positions
                    .get(candidate)
                    .is_some_and(|&position| same_position(position, at))
            }),
    }
}

/// The start of the window of length `window` that ends at `at`, where an
/// observation with that window reads the start of its increment: `at −
/// window`, or 0 when `|at − window|` is at most [`POSITION_TOLERANCE`] times
/// the larger of `|at|`, `window` and 1.
///
/// The rounding of `at − window` grows with its operands, and
/// [`same_position`] measures a difference by the magnitude of the two
/// positions, which near 0 loses that scale; so a start at 0 is read at the
/// scale of the window, and a window has one verdict whatever its end.
pub fn window_start(at: f64, window: f64) -> f64 {
    let start = at - window;
    if start.abs() <= POSITION_TOLERANCE * at.abs().max(window).max(1.0) {
        0.0
    } else {
        start
    }
}

/// Why a window has no increment ([`window_increments`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowError {
    /// The window that ends at `end` starts before 0 ([`window_start`]) and
    /// before every position.
    Incomplete {
        /// The end of the window.
        end: f64,
        /// The length of the window.
        window: f64,
    },
    /// No position is this position, the end or the start of a window.
    Missing(f64),
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Incomplete { end, window } => write!(
                f,
                "Position {end} has no complete window of {window} before it"
            ),
            Self::Missing(at) => write!(f, "No position is {at}"),
        }
    }
}

impl std::error::Error for WindowError {}

/// The increment of `series` over the window of length `window` that ends at
/// each of `ends`, as [`ObservationOperator`] reads a windowed observation:
/// the value at the end minus the value at [`window_start`], each found with
/// [`index_at`]. `series` has one value per entry of `positions`, ascending.
///
/// # Errors
///
/// [`WindowError::Incomplete`] for the first window that starts before 0 at
/// no position, and [`WindowError::Missing`] for the first end or start that
/// is no position.
///
/// # Panics
///
/// Panics when `series` and `positions` differ in length.
pub fn window_increments(
    series: &[f64],
    positions: &[f64],
    window: f64,
    ends: &[f64],
) -> Result<Vec<f64>, WindowError> {
    assert_eq!(series.len(), positions.len(), "one value per position");
    ends.iter()
        .map(|&end| {
            let start = window_start(end, window);
            let start_row = index_at(positions, start);
            if start_row.is_none() && start < 0.0 {
                return Err(WindowError::Incomplete { end, window });
            }
            let end_row = index_at(positions, end).ok_or(WindowError::Missing(end))?;
            let start_row = start_row.ok_or(WindowError::Missing(start))?;
            Ok(series[end_row] - series[start_row])
        })
        .collect()
}

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
    /// position and the start of every window, each found with the tolerance
    /// of [`index_at`]. `output_names` are the outputs of a row, in order.
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
            .map(ObservedDataPoint::window_start)
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
            index_at(positions, position).expect("the schedule holds every observed position")
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
    fn a_window_finds_its_start_within_the_tolerance_of_the_axis() {
        let mut windowed = ObservedDataPoint::new(0.3, "C".to_string(), 0.0);
        windowed.window = Some(0.1);
        // 0.3 − 0.1 is 0.19999999999999998, not the position 0.2 of the run.
        let operator = ObservationOperator::new(&[windowed], &[], &names(), &[0.2, 0.3]).unwrap();
        let rows = vec![vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 4.0]];
        let mut out = [0.0];

        operator.apply(&rows, &[], &mut out);

        assert_eq!(out, [3.0]);
    }

    #[test]
    fn a_position_is_found_exactly_or_within_the_tolerance() {
        let positions = [0.0, 0.2, 1.0, 1e12];
        assert_eq!(index_at(&positions, 0.2), Some(1));
        assert_eq!(index_at(&positions, 0.3 - 0.1), Some(1));
        assert_eq!(index_at(&positions, 1.0 + 1e-10), Some(2));
        assert_eq!(index_at(&positions, 1e12 + 1.0), Some(3));
        assert_eq!(index_at(&positions, 0.5), None);
        assert_eq!(index_at(&positions, 2e12), None);
        assert_eq!(index_at(&[], 0.0), None);
    }

    #[test]
    fn an_increment_is_read_as_a_windowed_observation() {
        let positions = [0.0, 0.1, 0.2, 0.3];
        let series = [0.0, 1.0, 3.0, 6.0];
        let mut windowed = ObservedDataPoint::new(0.3, "C".to_string(), 0.0);
        windowed.window = Some(0.2);
        let operator = ObservationOperator::new(&[windowed], &[], &names(), &positions).unwrap();
        let rows: Vec<Vec<f64>> = series.iter().map(|&c| vec![0.0, 0.0, c]).collect();
        let mut out = [0.0];

        operator.apply(&rows, &[], &mut out);

        assert_eq!(
            window_increments(&series, &positions, 0.2, &[0.3]),
            Ok(out.to_vec())
        );
        assert_eq!(
            window_increments(&series, &positions, 0.1, &[0.1 + 0.2]),
            Ok(vec![3.0])
        );
    }

    #[test]
    fn a_window_without_its_end_or_start_has_no_increment() {
        let positions = [0.0, 1.0, 2.0];
        let series = [0.0, 1.0, 3.0];

        assert_eq!(
            window_increments(&series, &positions, 1.0, &[1.0 + 1e-10]),
            Ok(vec![1.0])
        );
        assert_eq!(
            window_increments(&series, &positions, 5.0, &[9.0, 2.0]),
            Err(WindowError::Missing(9.0))
        );
        assert_eq!(
            window_increments(&series, &positions, 0.5, &[2.0]),
            Err(WindowError::Missing(1.5))
        );
        let incomplete = window_increments(&series, &positions, 3.0, &[2.0]).unwrap_err();
        assert_eq!(
            incomplete.to_string(),
            "Position 2 has no complete window of 3 before it"
        );
        assert_eq!(WindowError::Missing(1.5).to_string(), "No position is 1.5");
    }

    #[test]
    fn a_window_has_one_verdict_whatever_its_end() {
        let positions: Vec<f64> = (0..=2000).map(f64::from).collect();
        let window = 1000.0 + 1e-7;

        // At 1000 the start rounds to 0, at 2000 to the position 1000.
        assert_eq!(window_start(1000.0, window), 0.0);
        for end in [1000.0, 2000.0] {
            assert_eq!(
                window_increments(&positions, &positions, window, &[end]),
                Ok(vec![1000.0]),
                "{end}"
            );
        }
    }

    #[test]
    fn a_window_that_starts_clearly_before_zero_is_incomplete() {
        let positions: Vec<f64> = (0..=1000).map(f64::from).collect();

        for (window, end) in [(1000.0 + 1e-5, 1000.0), (1.0 + 1.5e-9, 1.0)] {
            assert!(window_start(end, window) < 0.0, "{window}");
            assert_eq!(
                window_increments(&positions, &positions, window, &[end]),
                Err(WindowError::Incomplete { end, window })
            );
        }
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
