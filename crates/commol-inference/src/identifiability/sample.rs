//! Practical identifiability from a parameter sample: how much the sample
//! contracts each parameter against a reference, which parameters move
//! together, and which marginals have several modes.

use super::IdentifiabilityError;
use crate::priors::Priors;
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
    /// `1 − Var(sample) / Var(reference)` of each parameter ([`Reference`]):
    /// near 0 when the sample spreads as the reference, near 1 when it sits
    /// on one value.
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

/// The distribution the contraction of each parameter compares with.
#[derive(Debug, Clone, Copy)]
pub enum Reference<'a> {
    /// The uniform distribution on the bounds of each parameter.
    Bounds,
    /// The prior of each parameter, truncated to its bounds.
    Prior(&'a Priors),
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
/// strictly between 0 and 1, or the priors are not those of the sample's
/// parameters and bounds.
pub fn sample_identifiability(
    sample: &ParameterSample,
    reference: Reference<'_>,
    correlation_threshold: f64,
) -> Result<SampleIdentifiability, IdentifiabilityError> {
    if !(correlation_threshold > 0.0 && correlation_threshold < 1.0) {
        return Err(IdentifiabilityError::Invalid(format!(
            "correlation_threshold must be strictly between 0 and 1, got {correlation_threshold}"
        )));
    }
    let variances: Vec<f64> = match reference {
        Reference::Bounds => sample
            .bounds()
            .iter()
            .map(|(min, max)| (max - min).powi(2) / 12.0)
            .collect(),
        Reference::Prior(priors) => {
            if priors.names() != sample.names() || priors.bounds() != sample.bounds() {
                return Err(IdentifiabilityError::Invalid(format!(
                    "The priors of {:?} on {:?} are not those of the sample's parameters {:?} \
                     on {:?}",
                    priors.names(),
                    priors.bounds(),
                    sample.names(),
                    sample.bounds()
                )));
            }
            priors
                .distributions()
                .iter()
                .map(|d| d.variance())
                .collect()
        }
    };
    let covariance = sample.covariance();
    let contraction = variances
        .iter()
        .enumerate()
        .map(|(j, variance)| 1.0 - covariance[j][j] / variance)
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

/// Silverman's bandwidth of the weighted `values` with the standard
/// deviation `deviation`: `0.9 · min(σ, IQR / 1.34) · n_eff^{−1/5}`, or `σ`
/// in place of the minimum when the IQR is 0.
fn bandwidth(values: &[f64], weights: &[f64], deviation: f64) -> f64 {
    let quartiles =
        quantiles(values, weights, &[0.25, 0.75]).expect("the values of a sample are finite");
    let spread = (quartiles[1] - quartiles[0]) / 1.34;
    let scale = if spread > 0.0 {
        deviation.min(spread)
    } else {
        deviation
    };
    0.9 * scale * effective_size(weights).powf(-0.2)
}

/// The modes of the weighted Gaussian kernel density of `values`.
fn density_modes(values: &[f64], weights: &[f64], deviation: f64) -> Vec<f64> {
    let bandwidth = bandwidth(values, weights, deviation);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * b.abs().max(1.0)
    }

    #[test]
    fn priors_of_other_parameters_are_no_reference() {
        let names = vec!["k".to_string()];
        let sample = ParameterSample::new(
            names.clone(),
            vec![(0.0, 1.0)],
            vec![vec![0.2], vec![0.7]],
            &[0.0, 0.0],
            None,
        )
        .unwrap();
        let wider = Priors::uniform_on_bounds(&names, &[(0.0, 2.0)]).unwrap();
        let error = sample_identifiability(&sample, Reference::Prior(&wider), 0.9).unwrap_err();
        assert!(
            error.to_string().contains("are not those of the sample"),
            "{error}"
        );

        let same = Priors::uniform_on_bounds(&names, &[(0.0, 1.0)]).unwrap();
        let by_prior = sample_identifiability(&sample, Reference::Prior(&same), 0.9).unwrap();
        let by_bounds = sample_identifiability(&sample, Reference::Bounds, 0.9).unwrap();
        assert!(close(by_prior.contraction[0], by_bounds.contraction[0]));
    }

    #[test]
    fn the_bandwidth_takes_the_smaller_of_the_deviation_and_the_spread() {
        // Hazen quartiles of 0, 1, 2, 3, 10: 0.75 and 4.75.
        let values = [0.0, 1.0, 2.0, 3.0, 10.0];
        let weights = [0.2; 5];
        let factor = 0.9 * 5f64.powf(-0.2);
        assert!(close(
            bandwidth(&values, &weights, 5.0),
            factor * 4.0 / 1.34
        ));
        assert!(close(bandwidth(&values, &weights, 1.0), factor));
    }

    #[test]
    fn without_a_spread_the_bandwidth_takes_the_deviation() {
        // Both quartiles of five ones and a nine are 1.
        let values = [1.0, 1.0, 1.0, 1.0, 1.0, 9.0];
        let weights = [1.0 / 6.0; 6];
        let expected = 0.9 * 2.0 * 6f64.powf(-0.2);
        assert!(close(bandwidth(&values, &weights, 2.0), expected));
    }

    #[test]
    fn a_sample_without_a_bandwidth_has_its_one_value_as_mode() {
        // One draw has no deviation; equal draws have a deviation of 0.
        assert_eq!(density_modes(&[0.3], &[1.0], f64::NAN), vec![0.3]);
        assert_eq!(density_modes(&[0.3, 0.3], &[0.5, 0.5], 0.0), vec![0.3]);
    }

    #[test]
    fn a_mode_holds_a_tenth_of_the_highest_density() {
        let modes = |weight: f64| density_modes(&[0.0, 1.0], &[1.0 - weight, weight], 0.1);
        assert_eq!(modes(0.1).len(), 2);
        assert_eq!(modes(0.05).len(), 1);
    }

    #[test]
    fn the_modes_follow_a_change_of_units() {
        // Two clusters, about −0.6 and 0.4, with uneven weights.
        let values: Vec<f64> = (0..40)
            .map(|i| if i < 25 { -0.6 } else { 0.4 } + 0.01 * f64::from(i % 5))
            .collect();
        let weights = vec![1.0 / 40.0; 40];
        let modes = density_modes(&values, &weights, 0.3);
        assert_eq!(modes.len(), 2, "{modes:?}");

        let scaled: Vec<f64> = values.iter().map(|v| 1e-3 * v - 5.0).collect();
        let scaled_modes = density_modes(&scaled, &weights, 0.3e-3);
        assert_eq!(scaled_modes.len(), 2, "{scaled_modes:?}");
        for (mode, scaled) in modes.iter().zip(&scaled_modes) {
            assert!((1e-3 * mode - 5.0 - scaled).abs() < 1e-9, "{mode} {scaled}");
        }
    }
}
