//! Blocks of observations, and the automatic block length.
//!
//! A block holds every observation, of every series, whose position falls in
//! one window of consecutive distinct positions, so that the dependence
//! across series and along the positions stays inside blocks. The
//! resamplers, the jackknife and the residual noise read their blocks here.

use std::collections::BTreeMap;

use commol_calibration::ObservedDataPoint;

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
    /// Returns a message when `length` is 0 or larger than the number of
    /// distinct positions.
    pub fn new(observations: &[ObservedDataPoint], length: usize) -> Result<Self, String> {
        let positions = distinct_positions(observations);
        check_length(length, positions.len())?;
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

    /// The distinct positions of the observations, ascending.
    pub fn positions(&self) -> &[f64] {
        &self.positions
    }

    /// The index in [`Self::positions`] of each observation's position.
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
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether there are no blocks; never true for a built scheme.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// The distinct positions of `observations`, ascending.
pub fn distinct_positions(observations: &[ObservedDataPoint]) -> Vec<f64> {
    let mut positions: Vec<f64> = observations.iter().map(|obs| obs.at).collect();
    positions.sort_by(f64::total_cmp);
    positions.dedup();
    positions
}

/// The index in `positions` of each observation's position.
///
/// # Panics
///
/// Panics when an observation's position is not in `positions`.
pub fn position_indices(observations: &[ObservedDataPoint], positions: &[f64]) -> Vec<usize> {
    observations
        .iter()
        .map(|obs| {
            positions
                .binary_search_by(|p| p.total_cmp(&obs.at))
                .expect("every position is listed")
        })
        .collect()
}

pub(crate) fn check_length(length: usize, positions: usize) -> Result<(), String> {
    if length == 0 || length > positions {
        return Err(format!(
            "block_length is {length}; it must be between 1 and the number of distinct \
             positions of the observations, {positions}, or \"auto\""
        ));
    }
    Ok(())
}

/// How the blocks of a method are drawn, which sets the constant of the
/// automatic block length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDrawing {
    /// Blocks that do not overlap: the Bayesian bootstrap, the jackknife and
    /// the wild bootstrap.
    NonOverlapping,
    /// Moving blocks (Künsch 1989), as the circular block bootstrap.
    Moving,
    /// Stationary blocks of geometric length (Politis and Romano 1994).
    Stationary,
}

/// The automatic block length of one series, in positions: the rule of
/// Politis and White (2004) with the correction of Patton, Politis and White
/// (2009).
///
/// The series is centered. The bandwidth `M` is twice the smallest lag `m`
/// after which `K_N = max(5, ⌈√log₁₀ n⌉)` autocorrelations in a row are
/// below `2 √(log₁₀ n / n)`, at most `⌈√n⌉ + K_N`. With the flat-top kernel
/// `λ`, `G = Σ λ(k/M) |k| R(k)` and `ĝ = Σ λ(k/M) R(k)` over `|k| ≤ M`, and
/// the length is `(2 G² / D)^{1/3} n^{1/3}`, with `D = (4/3) ĝ²` for moving
/// blocks and `D = 2 ĝ²` for stationary and non-overlapping blocks, at most
/// `min(3√n, n/3)` and at least 1.
pub fn automatic_length(series: &[f64], drawing: BlockDrawing) -> f64 {
    let n = series.len();
    if n < 4 {
        return 1.0;
    }
    let size = n as f64;
    let mean = series.iter().sum::<f64>() / size;
    let centered: Vec<f64> = series.iter().map(|value| value - mean).collect();
    let lags_in_a_row = 5usize.max(size.log10().sqrt().ceil() as usize);
    let max_bandwidth = (size.sqrt().ceil() as usize + lags_in_a_row).min(n - 1);
    let autocovariance: Vec<f64> = (0..=max_bandwidth)
        .map(|lag| {
            centered[lag..]
                .iter()
                .zip(&centered[..n - lag])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / size
        })
        .collect();
    if autocovariance[0] <= 0.0 {
        return 1.0;
    }
    let threshold = 2.0 * (size.log10() / size).sqrt();
    let small = |lag: usize| {
        lag > max_bandwidth || (autocovariance[lag] / autocovariance[0]).abs() < threshold
    };
    let first_quiet_lag = (0..=max_bandwidth.saturating_sub(lags_in_a_row))
        .find(|&m| (1..=lags_in_a_row).all(|k| small(m + k)));
    let bandwidth = match first_quiet_lag {
        Some(m) => (2 * m.max(1)).min(max_bandwidth),
        None => max_bandwidth,
    };
    let kernel = |t: f64| {
        if t <= 0.5 {
            1.0
        } else if t <= 1.0 {
            2.0 * (1.0 - t)
        } else {
            0.0
        }
    };
    let mut g = 0.0;
    let mut spectral = autocovariance[0];
    for (lag, &value) in autocovariance
        .iter()
        .enumerate()
        .take(bandwidth + 1)
        .skip(1)
    {
        let weight = kernel(lag as f64 / bandwidth as f64);
        g += 2.0 * weight * lag as f64 * value;
        spectral += 2.0 * weight * value;
    }
    let d = match drawing {
        BlockDrawing::Moving => 4.0 / 3.0 * spectral * spectral,
        BlockDrawing::NonOverlapping | BlockDrawing::Stationary => 2.0 * spectral * spectral,
    };
    if d <= 0.0 || g == 0.0 {
        return 1.0;
    }
    let length = (2.0 * g * g / d).cbrt() * size.cbrt();
    length.min((3.0 * size.sqrt()).min(size / 3.0)).max(1.0)
}

/// The block length `"auto"` chose, and the estimate of each series.
#[derive(Debug, Clone, PartialEq)]
pub struct AutomaticLength {
    /// The length used: the largest estimate, rounded, between 1 and the
    /// number of distinct positions.
    pub length: usize,
    /// The estimate of each series, by label, in order of first appearance.
    pub estimates: Vec<(String, f64)>,
}

/// The automatic block length of `residuals`, one per observation: the
/// [`automatic_length`] of each series (its observations in position order),
/// and the largest of them.
pub fn automatic_block_length(
    observations: &[ObservedDataPoint],
    residuals: &[f64],
    drawing: BlockDrawing,
) -> AutomaticLength {
    let mut order: Vec<&str> = Vec::new();
    let mut series: BTreeMap<&str, Vec<(f64, f64)>> = BTreeMap::new();
    for (obs, &residual) in observations.iter().zip(residuals) {
        if !series.contains_key(obs.label.as_str()) {
            order.push(&obs.label);
        }
        series
            .entry(&obs.label)
            .or_default()
            .push((obs.at, residual));
    }
    let estimates: Vec<(String, f64)> = order
        .iter()
        .map(|label| {
            let mut points = series[label].clone();
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
            let values: Vec<f64> = points.into_iter().map(|(_, r)| r).collect();
            (label.to_string(), automatic_length(&values, drawing))
        })
        .collect();
    let positions = distinct_positions(observations).len();
    let largest = estimates.iter().map(|(_, b)| *b).fold(1.0, f64::max);
    AutomaticLength {
        length: (largest.round() as usize).clamp(1, positions),
        estimates,
    }
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
    fn a_length_outside_the_positions_is_rejected() {
        let observations = observed(&[0.0, 1.0, 2.0], "x");
        for length in [0, 4] {
            let error = BlockScheme::new(&observations, length).unwrap_err();
            assert!(error.contains("block_length"), "{error}");
            assert!(error.contains("3"), "{error}");
        }
    }

    #[test]
    fn a_constant_or_short_series_has_length_one() {
        assert_eq!(automatic_length(&[2.0; 50], BlockDrawing::Moving), 1.0);
        assert_eq!(automatic_length(&[1.0, -1.0], BlockDrawing::Moving), 1.0);
    }
}
