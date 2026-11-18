//! The series of a dependence analysis on their grids: the observations of
//! each label, on the grid of its smallest spacing, and the positions the
//! series share.

use commol_calibration::{ObservedDataPoint, SeriesIndex};
use commol_core::same_position;

use crate::resampling::blocks::one_per_position;

/// What a dependence analysis reads of a series, for the message of a series
/// with two observations at one position.
const READER: &str = "A dependence analysis pairs the values of a series by position";

/// The most points of a grid per value: a position close to another sets
/// a step so small that the lags would read few pairs, and the grid would
/// fill the memory.
const POINTS_PER_VALUE: usize = 10;

/// The grid of ascending `positions` with no two at one position: the
/// point of each on the grid of the smallest spacing from the first, and the
/// number of points.
///
/// # Errors
///
/// Returns why a position is off that grid, why two positions are one
/// within the tolerance of the axis, or why the grid is too sparse: more
/// than [`POINTS_PER_VALUE`] points per value.
fn grid(positions: &[f64]) -> Result<(Vec<usize>, usize), String> {
    let Some(&first) = positions.first() else {
        return Ok((Vec::new(), 0));
    };
    let mut step = f64::INFINITY;
    for pair in positions.windows(2) {
        if same_position(pair[0], pair[1]) {
            return Err(format!(
                "its positions {} and {} are one position within the tolerance of the axis",
                pair[0], pair[1]
            ));
        }
        step = step.min(pair[1] - pair[0]);
    }
    if positions.len() == 1 {
        return Ok((vec![0], 1));
    }
    let mut points = Vec::with_capacity(positions.len());
    for &position in positions {
        let point = ((position - first) / step).round();
        if !same_position(first + point * step, position) {
            return Err(format!(
                "its position {position} is off the grid of its smallest spacing, {step} from \
                 {first}, so a lag would pair values at different distances"
            ));
        }
        points.push(point as usize);
    }
    let length = points.last().map_or(0, |&last| last + 1);
    if length > POINTS_PER_VALUE * positions.len() {
        return Err(format!(
            "its smallest spacing, {step} from {first}, gives a grid of {length} points for {} \
             values, so its lags would read few pairs; give the close positions labels of their own",
            positions.len()
        ));
    }
    Ok((points, length))
}

/// One series of a dependence analysis.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SeriesLayout {
    /// The label of the series.
    pub label: String,
    /// The observations of the series of weight above 0, by index, in
    /// position order.
    pub members: Vec<usize>,
    /// The point of each member on the grid of the series.
    pub points: Vec<usize>,
    /// The points of the grid.
    pub length: usize,
    /// Why the series has no statistics.
    pub reason: Option<String>,
}

impl SeriesLayout {
    /// The values of the members of the series on its grid; a point with
    /// no member is a gap.
    pub fn gridded(&self, values: &[f64]) -> Vec<Option<f64>> {
        let mut series = vec![None; self.length];
        for (&member, &point) in self.members.iter().zip(&self.points) {
            series[point] = Some(values[member]);
        }
        series
    }
}

/// The series of a set of observations, and the positions of the series
/// that have statistics, joined within the tolerance of the axis.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Layout {
    /// Each series, in order of first appearance of its label.
    pub series: Vec<SeriesLayout>,
    /// For each series with statistics, the shared position of each member.
    pub shared: Vec<Vec<usize>>,
    /// The point of each shared position on the grid of its smallest
    /// spacing, and the number of points; or why they are off it.
    pub shared_grid: Result<(Vec<usize>, usize), String>,
}

impl Layout {
    /// The layout of `observations`: an observation of weight 0 is left
    /// out; a series with two observations at one position, or with a
    /// position off its grid, has no statistics, with the reason.
    pub fn new(observations: &[ObservedDataPoint]) -> Self {
        let index = SeriesIndex::new(observations);
        let series: Vec<SeriesLayout> = index
            .labels()
            .iter()
            .zip(index.indices())
            .map(|(label, members)| {
                let members: Vec<usize> = members
                    .iter()
                    .copied()
                    .filter(|&i| observations[i].weight > 0.0)
                    .collect();
                let points: Vec<ObservedDataPoint> =
                    members.iter().map(|&i| observations[i].clone()).collect();
                let positions: Vec<f64> = points.iter().map(|obs| obs.at).collect();
                let layout = one_per_position(&points, READER, "").and_then(|()| {
                    grid(&positions).map_err(|reason| {
                        format!("The series '{label}' has no statistics: {reason}")
                    })
                });
                match layout {
                    Ok((grid, length)) => SeriesLayout {
                        label: label.clone(),
                        members,
                        points: grid,
                        length,
                        reason: None,
                    },
                    Err(reason) => SeriesLayout {
                        label: label.clone(),
                        members,
                        points: Vec::new(),
                        length: 0,
                        reason: Some(reason),
                    },
                }
            })
            .collect();
        let mut entries: Vec<(f64, usize, usize)> = Vec::new();
        for (s, layout) in series.iter().enumerate() {
            if layout.reason.is_none() {
                for (m, &member) in layout.members.iter().enumerate() {
                    entries.push((observations[member].at, s, m));
                }
            }
        }
        entries.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut positions: Vec<f64> = Vec::new();
        let mut shared: Vec<Vec<usize>> = series
            .iter()
            .map(|layout| vec![0; layout.points.len()])
            .collect();
        for (at, s, m) in entries {
            if positions
                .last()
                .is_none_or(|&last| !same_position(last, at))
            {
                positions.push(at);
            }
            shared[s][m] = positions.len() - 1;
        }
        Self {
            series,
            shared,
            shared_grid: grid(&positions)
                .map_err(|reason| format!("The positions of the series together: {reason}")),
        }
    }

    /// The number of shared positions.
    pub fn shared_positions(&self) -> usize {
        self.shared
            .iter()
            .flatten()
            .max()
            .map_or(0, |&last| last + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_with_a_spacing_that_is_no_multiple_are_off_the_grid() {
        // Spacings 3, 4 and 7: the smallest, 3, leaves 14 and 21 off its grid.
        let error = grid(&[7.0, 10.0, 14.0, 21.0]).unwrap_err();
        assert!(error.contains("14 is off the grid"), "{error}");
        assert_eq!(grid(&[0.5, 1.0, 2.0]), Ok((vec![0, 1, 3], 4)));
        assert_eq!(grid(&[3.0]), Ok((vec![0], 1)));
        let close = grid(&[1.0, 1.0 + 1e-12]).unwrap_err();
        assert!(close.contains("one position"), "{close}");
    }
}
