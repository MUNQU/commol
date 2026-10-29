//! The observation operator: one predicted value per observation from the
//! rows of a run.
//!
//! This is the only code that turns a run into observed quantities. The
//! optimizer, every inference method, the predictive and the scores read
//! observations through it.
//!
//! It is also the only code of the arithmetic of windows: the start of the
//! window of an observation ([`window_start`], [`check_window_end`]), the
//! increments of a series over windows ([`window_increments`]), and the grid
//! of windows that results, figures and reports read: window `w` of length
//! `window` covers `(w · window, (w + 1) · window]` ([`window_bounds`],
//! [`window_index`], [`window_ends`], [`windows_in`]).

use std::collections::HashMap;

use commol_core::{POSITION_TOLERANCE, same_position};

use crate::ranges::{AT, GRID_END, WINDOW, WINDOW_INDEX};
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

/// Check the window of length `window` that ends at `at`: a length of
/// [`WINDOW`] and a start ([`window_start`]) at 0 or after it, so an end is
/// at least its window. Returns the start.
///
/// # Errors
///
/// Returns the message of [`WINDOW`], or a message that names the window and
/// its end when the window starts before 0.
pub fn check_window_end(at: f64, window: f64) -> Result<f64, String> {
    WINDOW.check(window)?;
    let start = window_start(at, window);
    if start < 0.0 {
        Err(format!(
            "the window {window} that ends at {at} starts before 0; an end is at least \
             its window"
        ))
    } else {
        Ok(start)
    }
}

/// The number of windows of length `window` in `span`: `span / window`, or
/// the nearest whole number when the two are the same within the tolerance
/// of the axis ([`same_position`], a count of windows being a position in
/// units of the window), so `0.3 / 0.1` is 3 windows.
///
/// # Errors
///
/// Returns the message of [`WINDOW`] when `window` is not a finite, positive
/// length.
pub fn windows_in(span: f64, window: f64) -> Result<f64, String> {
    WINDOW.check(window)?;
    let count = span / window;
    let whole = count.round();
    Ok(if same_position(count, whole) {
        whole
    } else {
        count
    })
}

/// The positions that bound window `index` of the grid of length `window`:
/// it opens at `index · window` and closes at `(index + 1) · window`, where
/// the next window opens.
///
/// # Errors
///
/// Returns the message of [`WINDOW`], or of [`WINDOW_INDEX`] for a negative
/// index.
pub fn window_bounds(index: i64, window: f64) -> Result<(f64, f64), String> {
    WINDOW.check(window)?;
    WINDOW_INDEX.check(index)?;
    // An index of the grid is far below 2^53, where each is a number.
    let index = index as f64;
    Ok((index * window, (index + 1.0) * window))
}

/// The window of the grid of length `window` that holds the position `at`:
/// `max(ceil(at / window) − 1, 0)`, with the number of windows of
/// [`windows_in`]. Position 0 is in window 0, and the end of each window is
/// in that window.
///
/// # Errors
///
/// Returns the message of [`WINDOW`], or of [`AT`] when `at` is not a
/// position on the axis.
pub fn window_index(at: f64, window: f64) -> Result<usize, String> {
    let count = windows_in(at, window)?;
    AT.check(at)?;
    // A finite count of at least 0 below 2^64 windows.
    Ok((count.ceil() as usize).saturating_sub(1))
}

/// The end of each complete window of the grid of length `window` up to
/// `end`: `window`, `2 · window`, … with the number of windows of
/// [`windows_in`]. A trailing partial window is left out, and an `end` below
/// one window gives none.
///
/// # Errors
///
/// Returns the message of [`WINDOW`], or of [`GRID_END`] when `end` is not
/// finite.
pub fn window_ends(window: f64, end: f64) -> Result<Vec<f64>, String> {
    let count = windows_in(end, window)?;
    GRID_END.check(end)?;
    // A finite count, read as 0 below 0.
    let count = count.floor().max(0.0) as usize;
    (0..count)
        .map(|index| {
            let index = i64::try_from(index).expect("a count of windows fits an i64");
            window_bounds(index, window).map(|(_, end)| end)
        })
        .collect()
}

/// Why a window has no increment ([`window_increments`]).
#[derive(Debug, Clone, PartialEq)]
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
    /// The length of the window is outside the rule [`WINDOW`]: its message.
    Length(String),
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Incomplete { end, window } => write!(
                f,
                "Position {end} has no complete window of {window} before it"
            ),
            Self::Missing(at) => write!(f, "No position is {at}"),
            Self::Length(message) => f.write_str(message),
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
/// [`WindowError::Length`] when `window` is outside the rule [`WINDOW`],
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
    WINDOW.check(window).map_err(WindowError::Length)?;
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
/// position, the row its window starts at, and its scale and the factor that
/// multiply it.
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
    /// The known scale of each observation.
    scales: Vec<f64>,
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
            scales: observed_data.iter().map(|obs| obs.scale).collect(),
        })
    }

    /// Number of observations.
    pub fn count(&self) -> usize {
        self.rows.len()
    }

    /// Write the prediction of each observation into `out`: the sum of its
    /// outputs, or its increment over its window, times its scale and times
    /// its factor.
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
            let scaled = predicted * self.scales[index];
            *slot = match self.factors[index] {
                Some(factor) => scaled * parameters[factor],
                None => scaled,
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
    fn a_scale_and_a_factor_multiply() {
        let mut observation = ObservedDataPoint::new(1.0, "A".to_string(), 0.0);
        observation.scale = 0.25;
        observation.factor_id = Some("f".to_string());
        let parameters = [CalibrationParameter::with_type(
            "f".to_string(),
            CalibrationParameterType::Factor,
            0.1,
            10.0,
        )];
        let operator =
            ObservationOperator::new(&[observation], &parameters, &names(), &[1.0]).unwrap();
        let mut out = [0.0];

        operator.apply(&[vec![400.0, 0.0, 0.0]], &[3.0], &mut out);

        // 400 times the scale 0.25 times the factor 3.
        assert_eq!(out, [300.0]);
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

    #[test]
    fn the_ends_of_the_grid_are_its_complete_windows() {
        assert_eq!(window_ends(5.0, 20.0), Ok(vec![5.0, 10.0, 15.0, 20.0]));
        assert_eq!(window_ends(5.0, 23.0), Ok(vec![5.0, 10.0, 15.0, 20.0]));
        for end in [3.0, 0.0, -5.0] {
            assert_eq!(window_ends(5.0, end), Ok(vec![]), "{end}");
        }
        // 0.3 / 0.1 is 2.9999999999999996, which is 3 windows.
        let ends = window_ends(0.1, 0.3).unwrap();
        assert_eq!(ends.len(), 3);
        assert!(same_position(ends[2], 0.3));
        assert_eq!(
            window_ends(5.0, f64::INFINITY),
            Err("end is a finite number, got inf".to_string())
        );
    }

    #[test]
    fn the_index_of_a_position_inverts_the_end_of_its_window() {
        for window in [168.0, 2.5] {
            for index in 0..33 {
                let (start, end) = window_bounds(index, window).unwrap();
                assert_eq!(end - start, window);
                assert_eq!(window_bounds(index + 1, window).unwrap().0, end);
                let index = usize::try_from(index).unwrap();
                assert_eq!(window_index(end, window), Ok(index));
                assert_eq!(window_index(start + window / 2.0, window), Ok(index));
            }
        }
        // Window 1 of length 7 covers (7, 14].
        let indices: Vec<usize> = [0.0, 0.5, 7.0, 7.5, 14.0, 14.5]
            .into_iter()
            .map(|at| window_index(at, 7.0).unwrap())
            .collect();
        assert_eq!(indices, [0, 0, 0, 1, 1, 2]);
        // 0.1 + 0.2 is 0.30000000000000004, the end of window 2 of 0.1.
        assert_eq!(window_index(0.1 + 0.2, 0.1), Ok(2));
    }

    #[test]
    fn the_grid_has_the_rule_of_a_window_its_index_and_its_positions() {
        for window in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let message = WINDOW.check(window).unwrap_err();
            assert_eq!(window_bounds(0, window), Err(message.clone()));
            assert_eq!(window_index(1.0, window), Err(message.clone()));
            assert_eq!(window_ends(window, 10.0), Err(message.clone()));
            assert_eq!(windows_in(1.0, window), Err(message.clone()));
            assert_eq!(check_window_end(10.0, window), Err(message.clone()));
            assert_eq!(
                window_increments(&[0.0], &[0.0], window, &[]),
                Err(WindowError::Length(message.clone()))
            );
            assert_eq!(WindowError::Length(message.clone()).to_string(), message);
        }
        assert_eq!(
            window_bounds(-1, 7.0),
            Err("index is a whole number of at least 0, got -1".to_string())
        );
        assert_eq!(
            window_index(-1.0, 7.0),
            Err("at is a finite number of at least 0, got -1".to_string())
        );
    }

    #[test]
    fn a_window_ends_at_least_one_window_after_0() {
        assert_eq!(check_window_end(7.0, 7.0), Ok(0.0));
        assert_eq!(check_window_end(1000.0, 1000.0 + 1e-7), Ok(0.0));
        assert_eq!(check_window_end(10.0, 4.0), Ok(6.0));
        assert_eq!(
            check_window_end(3.0, 4.0),
            Err(
                "the window 4 that ends at 3 starts before 0; an end is at least its \
                 window"
                    .to_string()
            )
        );
    }

    #[test]
    fn a_count_of_windows_is_whole_within_the_tolerance_of_the_axis() {
        assert_eq!(windows_in(0.3, 0.1), Ok(3.0));
        assert_eq!(windows_in(7.5, 2.5), Ok(3.0));
        assert_eq!(windows_in(1.0, 4.0), Ok(0.25));
        assert_eq!(windows_in(-14.0, 7.0), Ok(-2.0));
    }
}
