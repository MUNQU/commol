//! The kernel density of one marginal of a sample, on the scale of its
//! parameter: its modes, told from noise by the standard error of the
//! density, and the shape of the density at a bound.

use commol_statistics::statistics::quantiles;

use crate::distribution::{BoundSide, KERNEL_REACH};

/// The standard errors by which a dip, or the density at a bound above its
/// reference, must exceed the noise of the kernel density.
pub(super) const STANDARD_ERRORS: f64 = 3.0;
/// The largest step of the grid, in bandwidths.
const STEP: f64 = 0.25;
/// The weighted Gaussian kernel density of one marginal, reflected at each
/// finite bound, on the points that a kernel reaches of an even grid of step
/// at most `h/4` over the draws ± 3 bandwidths cut at the bounds. Between
/// two runs of such points one point of density 0 stays, the valley of the
/// gap, so a range of any width keeps the step: the grid holds at most
/// `2·8/STEP + 2` points per kernel.
#[derive(Debug, Clone)]
pub(super) struct Density {
    /// Silverman's bandwidth.
    pub bandwidth: f64,
    /// The points of the grid, ascending.
    pub grid: Vec<f64>,
    /// The density at each point.
    pub density: Vec<f64>,
    /// The standard error of the density at each point.
    pub error: Vec<f64>,
    /// The grid points of the modes, ascending.
    pub modes: Vec<usize>,
    /// Whether the grid starts at the lower bound and ends at the upper.
    reaches: [bool; 2],
}

/// Silverman's bandwidth of the weighted `values` with the standard
/// deviation `deviation`: `0.9 · min(σ, IQR / 1.34) · n_eff^{−1/5}`, or `σ`
/// in place of the minimum when the IQR is 0.
pub(super) fn bandwidth(values: &[f64], weights: &[f64], deviation: f64, size: f64) -> f64 {
    let quartiles =
        quantiles(values, weights, &[0.25, 0.75]).expect("the values of a sample are finite");
    let spread = (quartiles[1] - quartiles[0]) / 1.34;
    let scale = if spread > 0.0 {
        deviation.min(spread)
    } else {
        deviation
    };
    0.9 * scale * size.powf(-0.2)
}

impl Density {
    /// The density of the weighted `values`, distinct, on `[lower, upper]`
    /// (either infinite), with the standard deviation `deviation` and the
    /// effective size `size`; None when the bandwidth is not positive, as for
    /// one value.
    pub fn new(
        values: &[f64],
        weights: &[f64],
        deviation: f64,
        size: f64,
        (lower, upper): (f64, f64),
    ) -> Option<Self> {
        let h = bandwidth(values, weights, deviation, size);
        if h.is_nan() || h <= 0.0 {
            return None;
        }
        let smallest = values.iter().copied().fold(f64::INFINITY, f64::min);
        let largest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let low = lower.max(smallest - 3.0 * h);
        let high = upper.min(largest + 3.0 * h);
        let reach = KERNEL_REACH * h;
        // Each draw, with its reflection at each bound within the reach.
        let kernels: Vec<[Option<f64>; 3]> = values
            .iter()
            .map(|&value| {
                let lower_image =
                    (lower.is_finite() && value - lower < reach).then_some(2.0 * lower - value);
                let upper_image =
                    (upper.is_finite() && upper - value < reach).then_some(2.0 * upper - value);
                [Some(value), lower_image, upper_image]
            })
            .collect();
        let lattice = Lattice::new(
            low,
            high,
            STEP * h,
            reach,
            kernels.iter().flatten().flatten().copied(),
        );
        let grid = lattice.grid();

        // The density adds the kernels of each draw; its variance adds the
        // square of their sum, `w² (Σ K)²`: each squared kernel and each
        // product of two, a Gaussian of bandwidth `h/√2` at their midpoint.
        let narrow = h / std::f64::consts::SQRT_2;
        let mut density = vec![0.0; grid.len()];
        let mut variance = vec![0.0; grid.len()];
        for (centers, &weight) in kernels.iter().zip(weights) {
            let centers: Vec<f64> = centers.iter().flatten().copied().collect();
            for (i, &center) in centers.iter().enumerate() {
                lattice.add(&mut density, h, center, weight);
                lattice.add(&mut variance, narrow, center, weight * weight);
                for &other in &centers[i + 1..] {
                    let d = (center - other) / (2.0 * h);
                    let cross = 2.0 * weight * weight * (-d * d).exp();
                    lattice.add(&mut variance, narrow, 0.5 * (center + other), cross);
                }
            }
        }
        let norm = 1.0 / (h * (2.0 * std::f64::consts::PI).sqrt());
        density.iter_mut().for_each(|f| *f *= norm);
        let error: Vec<f64> = variance.iter().map(|v| norm * v.max(0.0).sqrt()).collect();
        let modes = modes(&density, &error);
        Some(Self {
            bandwidth: h,
            grid,
            density,
            error,
            modes,
            reaches: [low == lower, high == upper],
        })
    }

    /// Whether the bound of `side` sets that side of the density: the grid
    /// reaches the bound, (a) the density there is not more than
    /// [`STANDARD_ERRORS`] standard errors below its highest point between
    /// the bound and the nearest mode, and (b) it exceeds `smoothed`, the
    /// reference smoothed by the same reflected kernel, by more than
    /// [`STANDARD_ERRORS`] standard errors.
    pub fn at_bound(&self, side: BoundSide, smoothed: f64) -> bool {
        let last = self.grid.len() - 1;
        let (bound, between) = match side {
            BoundSide::Lower => (0, 0..=self.modes[0]),
            BoundSide::Upper => (last, self.modes[self.modes.len() - 1]..=last),
        };
        let highest = between
            .max_by(|&a, &b| self.density[a].total_cmp(&self.density[b]))
            .expect("the range holds the mode");
        let (f, se) = (self.density[bound], self.error[bound]);
        let top = f
            >= self.density[highest]
                - STANDARD_ERRORS * self.error[bound].hypot(self.error[highest]);
        top && f - smoothed > STANDARD_ERRORS * se
    }

    /// Whether the grid reaches the bound of `side`.
    pub fn reaches(&self, side: BoundSide) -> bool {
        match side {
            BoundSide::Lower => self.reaches[0],
            BoundSide::Upper => self.reaches[1],
        }
    }
}

/// The points of an even grid from `low` to `high` that a kernel reaches:
/// runs of consecutive indices, each with the offset of its first point
/// among the points kept.
struct Lattice {
    low: f64,
    high: f64,
    step: f64,
    last: f64,
    runs: Vec<(f64, f64, usize)>,
}

impl Lattice {
    /// The grid of step at most `step` whose points lie within `reach` of
    /// one of `centers`, with one point of density 0 after each run that a
    /// gap follows.
    fn new(low: f64, high: f64, step: f64, reach: f64, centers: impl Iterator<Item = f64>) -> Self {
        let last = ((high - low) / step).ceil().max(1.0);
        let step = (high - low) / last;
        let mut spans: Vec<(f64, f64)> = centers
            .filter_map(|center| {
                let first = ((center - reach - low) / step).ceil().max(0.0);
                let end = ((center + reach - low) / step).floor().min(last);
                (first <= end).then_some((first, end))
            })
            .collect();
        spans.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut runs: Vec<(f64, f64, usize)> = Vec::new();
        let mut offset = 0;
        for (first, end) in spans {
            match runs.last_mut() {
                Some(run) if first <= run.1 + 2.0 => run.1 = run.1.max(end),
                _ => {
                    if let Some(run) = runs.last_mut() {
                        // The point after the run, in the gap: the valley.
                        run.1 += 1.0;
                        offset += (run.1 - run.0) as usize + 1;
                    }
                    runs.push((first, end, offset));
                }
            }
        }
        Self {
            low,
            high,
            step,
            last,
            runs,
        }
    }

    /// The kept points of the grid, ascending; the last is `high`.
    fn grid(&self) -> Vec<f64> {
        let mut grid: Vec<f64> = self
            .runs
            .iter()
            .flat_map(|&(first, end, _)| {
                (first as u64..=end as u64).map(|g| self.low + g as f64 * self.step)
            })
            .collect();
        if let (Some(point), Some(&(_, end, _))) = (grid.last_mut(), self.runs.last())
            && end == self.last
        {
            *point = self.high;
        }
        grid
    }

    /// Add `weight` times the kernel of bandwidth `h` at `center` to the
    /// run that holds the center, the run that its reach lies in.
    fn add(&self, values: &mut [f64], h: f64, center: f64, weight: f64) {
        let index = ((center - self.low) / self.step)
            .round()
            .clamp(0.0, self.last);
        let r = self
            .runs
            .partition_point(|run| run.0 <= index)
            .saturating_sub(1);
        let (first, end, offset) = self.runs[r];
        let run = &mut values[offset..=offset + (end - first) as usize];
        add_kernel(
            run,
            self.low + first * self.step,
            self.step,
            h,
            center,
            weight,
        );
    }
}

/// Add `weight` times the unnormalized Gaussian kernel of bandwidth `h` at
/// `center` to the grid points within [`KERNEL_REACH`] bandwidths of it, by
/// the recurrence of a Gaussian on an even grid: two exponentials per
/// center.
fn add_kernel(density: &mut [f64], low: f64, step: f64, h: f64, center: f64, weight: f64) {
    let reach = KERNEL_REACH * h;
    let first = ((center - reach - low) / step).ceil().max(0.0);
    let last = ((center + reach - low) / step).floor();
    if last < 0.0 || first >= density.len() as f64 {
        return;
    }
    let (first, last) = (first as usize, (last as usize).min(density.len() - 1));
    let u = (low + first as f64 * step - center) / h;
    let d = step / h;
    let mut value = (-0.5 * u * u).exp();
    let mut ratio = (-(u * d + 0.5 * d * d)).exp();
    let shrink = (-d * d).exp();
    for f in &mut density[first..=last] {
        *f += weight * value;
        value *= ratio;
        ratio *= shrink;
    }
}

/// The modes of `density`, with standard errors `error`: the local maxima,
/// the ends of the grid among them, less the lower peak of the shallowest
/// dip, one at a time, while the depth of that dip,
/// `f(lower peak) − f(valley) − 3·hypot(se_peak, se_valley)`, is not
/// positive.
fn modes(density: &[f64], error: &[f64]) -> Vec<usize> {
    let n = density.len();
    let mut peaks: Vec<usize> = (0..n)
        .filter(|&g| {
            density[g] > 0.0
                && (g == 0 || density[g] > density[g - 1])
                && (g + 1 == n || density[g] >= density[g + 1])
        })
        .collect();
    let lowest = |a: usize, b: usize| if density[b] < density[a] { b } else { a };
    let mut valleys: Vec<usize> = peaks
        .windows(2)
        .map(|pair| (pair[0]..=pair[1]).fold(pair[0], lowest))
        .collect();
    while peaks.len() > 1 {
        let lower_peak = |i: usize| {
            if density[peaks[i]] <= density[peaks[i + 1]] {
                peaks[i]
            } else {
                peaks[i + 1]
            }
        };
        let depth = |i: usize| {
            let (peak, valley) = (lower_peak(i), valleys[i]);
            density[peak] - density[valley] - STANDARD_ERRORS * error[peak].hypot(error[valley])
        };
        let (i, shallowest) = (0..valleys.len())
            .map(|i| (i, depth(i)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("two peaks have a valley");
        if shallowest > 0.0 {
            break;
        }
        if lower_peak(i) == peaks[i] {
            peaks.remove(i);
            if i > 0 {
                valleys[i - 1] = lowest(valleys[i - 1], valleys[i]);
            }
        } else {
            peaks.remove(i + 1);
            if i + 1 < valleys.len() {
                valleys[i + 1] = lowest(valleys[i], valleys[i + 1]);
            }
        }
        valleys.remove(i);
    }
    peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * b.abs().max(1.0)
    }

    #[test]
    fn the_bandwidth_takes_the_smaller_of_the_deviation_and_the_spread() {
        // Hazen quartiles of 0, 1, 2, 3, 10: 0.75 and 4.75.
        let values = [0.0, 1.0, 2.0, 3.0, 10.0];
        let weights = [0.2; 5];
        let factor = 0.9 * 5f64.powf(-0.2);
        assert!(close(
            bandwidth(&values, &weights, 5.0, 5.0),
            factor * 4.0 / 1.34
        ));
        assert!(close(bandwidth(&values, &weights, 1.0, 5.0), factor));
    }

    #[test]
    fn without_a_spread_the_bandwidth_takes_the_deviation() {
        // Both quartiles of five ones and a nine are 1.
        let values = [1.0, 1.0, 1.0, 1.0, 1.0, 9.0];
        let weights = [1.0 / 6.0; 6];
        let expected = 0.9 * 2.0 * 6f64.powf(-0.2);
        assert!(close(bandwidth(&values, &weights, 2.0, 6.0), expected));
    }

    #[test]
    fn one_value_has_no_density() {
        let infinite = (f64::NEG_INFINITY, f64::INFINITY);
        assert!(Density::new(&[0.3], &[1.0], f64::NAN, 1.0, infinite).is_none());
    }

    #[test]
    fn the_recurrence_gives_the_kernel_at_every_point() {
        let (low, step, h, center) = (-1.0, 0.01, 0.1, 0.123);
        let mut density = vec![0.0; 300];
        add_kernel(&mut density, low, step, h, center, 2.0);
        for (g, f) in density.iter().enumerate() {
            let z = (low + g as f64 * step - center) / h;
            let expected = if z.abs() <= KERNEL_REACH {
                2.0 * (-0.5 * z * z).exp()
            } else {
                0.0
            };
            assert!((f - expected).abs() < 1e-12, "{g}: {f} {expected}");
        }
        // A center beyond the grid adds nothing.
        let mut beyond = vec![0.0; 10];
        add_kernel(&mut beyond, 0.0, 0.01, 0.001, 5.0, 1.0);
        add_kernel(&mut beyond, 0.0, 0.01, 0.001, -5.0, 1.0);
        assert!(beyond.iter().all(|&f| f == 0.0));
    }

    #[test]
    fn a_dip_within_the_noise_joins_its_peaks() {
        // Peaks of 1.0 and 0.9 over a valley of 0.8: a dip of 0.1.
        let density = [0.5, 1.0, 0.8, 0.9, 0.4];
        assert_eq!(modes(&density, &[0.01; 5]), vec![1, 3]);
        assert_eq!(modes(&density, &[0.1; 5]), vec![1]);
        // The ends are candidates, and the lower of two peaks goes.
        let rising = [1.0, 0.2, 0.3, 0.2, 2.0];
        assert_eq!(modes(&rising, &[0.01; 5]), vec![0, 2, 4]);
        assert_eq!(modes(&rising, &[0.04; 5]), vec![0, 4]);
        assert_eq!(modes(&rising, &[1.0; 5]), vec![4]);
    }
}
