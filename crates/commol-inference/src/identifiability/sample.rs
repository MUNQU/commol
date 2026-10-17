//! Practical identifiability from a parameter sample: how much the sample
//! contracts each parameter against a reference, which parameters move
//! together, and which marginals have several modes.

use super::IdentifiabilityError;
use crate::sample::ParameterSample;
use crate::statistics::{effective_size, quantiles};

/// Points of the grid a kernel density is evaluated on.
const GRID: usize = 512;
/// A local maximum of a density counts as a mode above this share of the
/// highest.
const MODE_HEIGHT: f64 = 0.1;

/// The result of [`sample_identifiability`].
#[derive(Debug, Clone, PartialEq)]
pub struct SampleIdentifiability {
    /// `1 − Var(sample) / Var(reference)` of each parameter, with the
    /// uniform distribution on its bounds as the reference: near 0 when the
    /// sample spreads over the bounds, near 1 when it sits on one value.
    pub contraction: Vec<f64>,
    /// The weighted correlation matrix.
    pub correlation: Vec<Vec<f64>>,
    /// The pairs `(j, k, r)`, `j < k`, whose absolute correlation `r` is
    /// above the threshold.
    pub correlated_pairs: Vec<(usize, usize, f64)>,
    /// Whether the marginal of each parameter has two or more modes.
    pub multimodal: Vec<bool>,
    /// The positions of the modes of each marginal.
    pub modes: Vec<Vec<f64>>,
}

/// The practical identifiability of `sample`, with the pairs whose absolute
/// correlation is above `correlation_threshold`.
///
/// The modes are the local maxima, above a tenth of the highest, of a
/// weighted Gaussian kernel density with Silverman's bandwidth
/// `0.9 · min(σ, IQR / 1.34) · n_eff^{−1/5}`, on a grid of 512 points.
///
/// # Errors
///
/// [`IdentifiabilityError::Invalid`] when `correlation_threshold` is not
/// strictly between 0 and 1.
pub fn sample_identifiability(
    sample: &ParameterSample,
    correlation_threshold: f64,
) -> Result<SampleIdentifiability, IdentifiabilityError> {
    if !(correlation_threshold > 0.0 && correlation_threshold < 1.0) {
        return Err(IdentifiabilityError::Invalid(format!(
            "correlation_threshold must be strictly between 0 and 1, got {correlation_threshold}"
        )));
    }
    let covariance = sample.covariance();
    let contraction = sample
        .bounds()
        .iter()
        .enumerate()
        .map(|(j, (min, max))| 1.0 - covariance[j][j] / ((max - min).powi(2) / 12.0))
        .collect();
    let correlation = sample.correlation();
    let mut correlated_pairs = Vec::new();
    for (j, row) in correlation.iter().enumerate() {
        for (k, &r) in row.iter().enumerate().skip(j + 1) {
            if r.abs() > correlation_threshold {
                correlated_pairs.push((j, k, r));
            }
        }
    }
    let weights = sample.weights();
    let modes: Vec<Vec<f64>> = (0..sample.names().len())
        .map(|j| {
            let values: Vec<f64> = sample.values().iter().map(|draw| draw[j]).collect();
            density_modes(&values, weights, covariance[j][j].sqrt())
        })
        .collect();
    Ok(SampleIdentifiability {
        contraction,
        correlation,
        correlated_pairs,
        multimodal: modes.iter().map(|m| m.len() >= 2).collect(),
        modes,
    })
}

/// The modes of the weighted Gaussian kernel density of `values`.
fn density_modes(values: &[f64], weights: &[f64], deviation: f64) -> Vec<f64> {
    let quartiles = quantiles(values, weights, &[0.25, 0.75]);
    let spread = (quartiles[1] - quartiles[0]) / 1.34;
    let scale = if spread > 0.0 {
        deviation.min(spread)
    } else {
        deviation
    };
    let bandwidth = 0.9 * scale * effective_size(weights).powf(-0.2);
    if bandwidth.is_nan() || bandwidth <= 0.0 {
        return vec![values[0]];
    }
    let low = values.iter().copied().fold(f64::INFINITY, f64::min) - 3.0 * bandwidth;
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 3.0 * bandwidth;
    let grid: Vec<f64> = (0..GRID)
        .map(|g| low + (high - low) * g as f64 / (GRID - 1) as f64)
        .collect();
    let density: Vec<f64> = grid
        .iter()
        .map(|&x| {
            values
                .iter()
                .zip(weights)
                .map(|(&v, &w)| w * (-0.5 * ((x - v) / bandwidth).powi(2)).exp())
                .sum()
        })
        .collect();
    let highest = density.iter().copied().fold(0.0, f64::max);
    (1..GRID - 1)
        .filter(|&g| {
            density[g] > density[g - 1]
                && density[g] >= density[g + 1]
                && density[g] >= MODE_HEIGHT * highest
        })
        .map(|g| grid[g])
        .collect()
}
