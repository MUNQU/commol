//! Blocks of observations.
//!
//! A block holds every observation, of every series, whose position falls in
//! one window of consecutive distinct positions, so that the dependence
//! across series and along the positions stays inside blocks. The
//! resamplers, the jackknife, the residual noise, the sandwich, the
//! calibration of `η` and the conformal methods read their blocks here.

use commol_calibration::{ObservedDataPoint, SeriesIndex, index_at};

use crate::error::MethodError;

/// The observations of a problem grouped into blocks of `length`
/// consecutive distinct positions.
///
/// The blocks do not overlap and cover every observation once; the last
/// block is shorter when `length` does not divide the number of positions.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockScheme {
    length: usize,
    positions: Vec<f64>,
    position_of: Vec<usize>,
    blocks: Vec<Vec<usize>>,
}

impl BlockScheme {
    /// The blocks of `observations` with `length` positions each.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when `length` is 0 or larger than the number
    /// of distinct positions.
    pub fn new(observations: &[ObservedDataPoint], length: usize) -> Result<Self, MethodError> {
        let positions = distinct_positions(observations);
        check_length(length, positions.len()).map_err(MethodError::Invalid)?;
        let position_of = position_indices(observations, &positions);
        let count = positions.len().div_ceil(length);
        let mut blocks = vec![Vec::new(); count];
        for (observation, &position) in position_of.iter().enumerate() {
            blocks[position / length].push(observation);
        }
        Ok(Self {
            length,
            positions,
            position_of,
            blocks,
        })
    }

    /// The number of positions of a block.
    pub fn length(&self) -> usize {
        self.length
    }

    /// The index of each observation's position among the distinct positions
    /// of the observations, ascending.
    pub fn position_of(&self) -> &[usize] {
        &self.position_of
    }

    /// The observations of each block, as indices in problem order.
    pub fn blocks(&self) -> &[Vec<usize>] {
        &self.blocks
    }

    /// The block of each observation.
    pub fn block_of(&self) -> Vec<usize> {
        self.position_of.iter().map(|p| p / self.length).collect()
    }

    /// The first and last position of each block.
    pub fn spans(&self) -> Vec<(f64, f64)> {
        (0..self.blocks.len())
            .map(|block| {
                let first = block * self.length;
                let last = ((block + 1) * self.length).min(self.positions.len()) - 1;
                (self.positions[first], self.positions[last])
            })
            .collect()
    }

    /// The number of blocks.
    pub fn count(&self) -> usize {
        self.blocks.len()
    }
}

/// The distinct positions of `observations`, ascending. `−0.0` is the
/// position `0.0`, and is listed as `0.0`.
pub fn distinct_positions(observations: &[ObservedDataPoint]) -> Vec<f64> {
    let mut positions: Vec<f64> = observations.iter().map(|obs| obs.at + 0.0).collect();
    positions.sort_by(f64::total_cmp);
    positions.dedup();
    positions
}

/// The index in `positions` of each observation's position, found as every
/// reader of the axis finds a position ([`index_at`]).
///
/// # Panics
///
/// Panics when an observation's position is not in `positions`.
pub fn position_indices(observations: &[ObservedDataPoint], positions: &[f64]) -> Vec<usize> {
    observations
        .iter()
        .map(|obs| index_at(positions, obs.at).expect("every position is listed"))
        .collect()
}

/// Refuse a series of `observations` with two observations at one position,
/// for a reader that reads one value per position of a series: `reader`
/// says what it reads.
///
/// A label may hold two observations at one position (an increment and a
/// total that end there), each read by the loss with its own window. A
/// reader that pairs a series by position cannot tell them apart: it would
/// keep one value of the two, or read them as consecutive positions.
pub(crate) fn one_per_position(
    observations: &[ObservedDataPoint],
    reader: &str,
) -> Result<(), String> {
    let series = SeriesIndex::new(observations);
    for (label, members) in series.labels().iter().zip(series.indices()) {
        if let Some(pair) = members
            .windows(2)
            .find(|pair| observations[pair[0]].at == observations[pair[1]].at)
        {
            return Err(format!(
                "{reader}, and the series '{label}' has two observations at {}. Give one of \
                 them a label of its own",
                observations[pair[0]].at
            ));
        }
    }
    Ok(())
}

pub(crate) fn check_length(length: usize, positions: usize) -> Result<(), String> {
    crate::ranges::BLOCK_LENGTH.check(length)?;
    if length > positions {
        return Err(format!(
            "block_length is {length}; it must be between 1 and the number of distinct \
             positions of the observations, {positions}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed(positions: &[f64], label: &str) -> Vec<ObservedDataPoint> {
        positions
            .iter()
            .map(|&at| ObservedDataPoint::new(at, label.to_string(), 0.0))
            .collect()
    }

    #[test]
    fn a_short_last_block_keeps_the_remaining_positions() {
        let scheme = BlockScheme::new(&observed(&[0.0, 1.0, 2.0, 3.0, 4.0], "x"), 2).unwrap();
        assert_eq!(scheme.blocks(), [vec![0, 1], vec![2, 3], vec![4]]);
        assert_eq!(scheme.spans(), [(0.0, 1.0), (2.0, 3.0), (4.0, 4.0)]);
        assert_eq!(scheme.block_of(), [0, 0, 1, 1, 2]);
    }

    #[test]
    fn minus_zero_is_the_position_zero() {
        let mut observations = observed(&[-0.0], "x");
        observations.extend(observed(&[0.0, 1.0], "y"));

        let positions = distinct_positions(&observations);
        let scheme = BlockScheme::new(&observations, 1).unwrap();

        let bits: Vec<u64> = positions.iter().map(|p| p.to_bits()).collect();
        assert_eq!(bits, [0.0f64.to_bits(), 1.0f64.to_bits()]);
        assert_eq!(position_indices(&observations, &positions), [0, 0, 1]);
        assert_eq!(scheme.blocks(), [vec![0, 1], vec![2]]);
    }

    #[test]
    fn a_length_outside_the_positions_is_rejected() {
        let observations = observed(&[0.0, 1.0, 2.0], "x");
        let zero = BlockScheme::new(&observations, 0).unwrap_err().to_string();
        let long = BlockScheme::new(&observations, 4).unwrap_err().to_string();
        assert!(
            zero.contains("block_length is a whole number of at least 1, got 0"),
            "{zero}"
        );
        assert!(
            long.contains("block_length") && long.contains("3"),
            "{long}"
        );
    }
}
