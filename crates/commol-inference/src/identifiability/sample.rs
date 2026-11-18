//! Practical identifiability from a parameter sample: what each parameter
//! learned against its reference (the bounds or the priors), which
//! parameters move together, which marginals have several modes, which
//! bounds set a side of a marginal, and which directions of the parameters
//! the sample leaves as spread as the reference.
//!
//! A parameter is poorly determined when its contraction is below the
//! threshold, it is correlated with another above the threshold, or its
//! marginal has several modes; otherwise it is determined. A bound that sets
//! a side (`at_bound`) is a cause apart, which does not set the verdict.
//!
//! Every statistic reads the distinct rows of the sample, each with the sum
//! of the weights of its copies ([`ParameterSample::distinct`]). The shape
//! measures (the contraction, the correlation, the density and its modes,
//! the density at a bound, the directions) read `ln θ` for the parameters of
//! `log_parameters`, `θ` for the others; the masses at each side and the
//! overlap read the distribution function of the reference, which no scale
//! changes.

mod marginal;

use commol_calibration::first_infinite_bound;
use commol_random::sample::exponential;
use commol_random::{Generator, Stream};
use nalgebra::{DMatrix, SymmetricEigen};

use self::marginal::Density;
use super::PracticalVerdict;
use crate::distribution::{BoundSide, Distribution, Family};
use crate::error::MethodError;
use crate::ranges::{BOUND_SHARE, CONTRACTION_THRESHOLD, CORRELATION_THRESHOLD};
use crate::sample::ParameterSample;
use crate::settings::identifiability::SampleIdentifiabilitySettings;

/// The replicates of the Bayesian bootstrap that corrects the contraction of
/// each direction among the determined parameters.
const BOOTSTRAP_REPLICATES: u64 = 100;

/// The verdict of one parameter and its causes.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleVerdict {
    /// Poorly determined when its contraction is below the threshold, it is
    /// correlated with another parameter, or its marginal has several modes.
    pub verdict: PracticalVerdict,
    /// Whether its contraction is below the threshold.
    pub low_contraction: bool,
    /// The other parameters, ascending, whose absolute correlation with it is
    /// above the threshold, with that correlation: `(k, r)`.
    pub correlated_with: Vec<(usize, f64)>,
    /// The number of modes of its marginal.
    pub modes: usize,
    /// The sides whose finite bound, not the data, sets that side of its
    /// marginal; this cause does not set the verdict.
    pub at_bound: Vec<BoundSide>,
}

/// A direction of the parameters: an eigenvector of the covariance of the
/// sample with each parameter divided by the standard deviation of its
/// reference, `D⁻¹ Σ D⁻¹`.
#[derive(Debug, Clone, PartialEq)]
pub struct Direction {
    /// `1 − λ`, with `λ` the eigenvalue: near 0 when the sample spreads as
    /// the reference along the direction, below 0 when it spreads more.
    pub contraction: f64,
    /// The loading of each parameter, on its scale divided by the standard
    /// deviation of its reference; the largest in absolute value is
    /// positive.
    pub loadings: Vec<f64>,
}

/// The result of [`sample_identifiability`].
#[derive(Debug, Clone, PartialEq)]
pub struct SampleIdentifiability {
    /// Kish's effective size of the distinct rows.
    pub effective_size: f64,
    /// The number of distinct rows of the sample.
    pub distinct_rows: usize,
    /// `1 − Var(sample) / Var(reference)` of each parameter, on its scale:
    /// near 0 when the sample spreads as the reference, near 1 when it sits
    /// on one value.
    pub contraction: Vec<f64>,
    /// The weighted correlation matrix, on the scale of each parameter.
    pub correlation: Vec<Vec<f64>>,
    /// The pairs `(j, k, r)`, `j < k`, whose absolute correlation `r` is
    /// above the threshold.
    pub correlated_pairs: Vec<(usize, usize, f64)>,
    /// The positions of the modes of each marginal, in parameter units.
    pub modes: Vec<Vec<f64>>,
    /// The weight of the draws where the distribution function of the
    /// reference is below `bound_share`.
    pub mass_below: Vec<f64>,
    /// The weight of the draws where the distribution function of the
    /// reference is above `1 − bound_share`.
    pub mass_above: Vec<f64>,
    /// `Σ_b min(w_b, 1/B)` of each parameter over the `B` bins of equal
    /// reference mass.
    pub overlap: Vec<f64>,
    /// The number of bins `B = ⌈2 n_eff^{1/3}⌉` of the overlap (Rice's
    /// rule).
    pub bins: usize,
    /// The directions, from the least contracted; empty when the sample has
    /// at most as many distinct rows as parameters.
    pub directions: Vec<Direction>,
    /// How many directions among the determined parameters have a
    /// contraction, corrected by a Bayesian bootstrap, below the threshold;
    /// None when the sample has no directions.
    pub hidden_ridges: Option<usize>,
    /// The effective size per parameter, `n_eff / p`.
    pub draws_per_parameter: f64,
    /// The verdict of each parameter.
    pub verdicts: Vec<SampleVerdict>,
}

/// The practical identifiability of `sample` with `settings`; the bootstrap
/// of the directions draws from the stream
/// [`Stream::SampleDirections`] of `seed`.
///
/// The density of each marginal is a weighted Gaussian kernel density on the
/// scale of the parameter, with Silverman's bandwidth
/// `h = 0.9 · min(σ, IQR / 1.34) · n_eff^{−1/5}`, reflected at each finite
/// bound, on the points that a kernel reaches of a grid over the draws ± 3
/// bandwidths cut at the bounds with a step of at most `h/4`. Its standard
/// error at `x` is `√(Σ wᵢ² (Σ K_h(x − cᵢ))²)`, the inner sum over the
/// kernels of draw `i` and its reflections. The
/// modes are the peaks left after removing, one at a time, the lower peak
/// of the shallowest dip while its depth
/// `f(peak) − f(valley) − 3·hypot(se_peak, se_valley)` is not positive.
///
/// # Errors
///
/// [`MethodError::Invalid`] when a threshold is out of its range, the
/// reference is the bounds and a bound is not finite, the priors do not fit
/// the sample's parameters and bounds, or `log_parameters` names a parameter
/// that the sample does not hold, one with a lower bound below 0, or one
/// with a draw at 0.
pub fn sample_identifiability(
    sample: &ParameterSample,
    settings: &SampleIdentifiabilitySettings,
    seed: u64,
) -> Result<SampleIdentifiability, MethodError> {
    CORRELATION_THRESHOLD
        .check(settings.correlation_threshold)
        .and_then(|()| CONTRACTION_THRESHOLD.check(settings.contraction_threshold))
        .and_then(|()| BOUND_SHARE.check(settings.bound_share))
        .map_err(MethodError::Invalid)?;
    let references = references(sample, settings)?;
    let logs = settings
        .log_scales(sample.names(), sample.bounds(), sample.values())
        .map_err(MethodError::Invalid)?;
    let distinct = sample.distinct();
    let p = distinct.names().len();
    let weights = distinct.weights();
    let size = distinct.effective_size();
    let rows = distinct.len();

    let natural: Vec<Vec<f64>> = (0..p)
        .map(|j| distinct.values().iter().map(|draw| draw[j]).collect())
        .collect();
    let columns: Vec<Vec<f64>> = natural
        .iter()
        .zip(&logs)
        .map(|(column, &log)| {
            if log {
                column.iter().map(|v| v.ln()).collect()
            } else {
                column.clone()
            }
        })
        .collect();
    let covariance = scaled_covariance(&columns, weights);
    let deviations: Vec<f64> = references
        .iter()
        .zip(&logs)
        .map(|(reference, &log)| {
            if log {
                reference.log_variance()
            } else {
                reference.variance()
            }
            .sqrt()
        })
        .collect();
    let contraction: Vec<f64> = (0..p)
        .map(|j| 1.0 - covariance[j][j] / (deviations[j] * deviations[j]))
        .collect();
    let correlation: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            (0..p)
                .map(|k| covariance[j][k] / (covariance[j][j] * covariance[k][k]).sqrt())
                .collect()
        })
        .collect();
    let mut correlated_pairs = Vec::new();
    for (j, row) in correlation.iter().enumerate() {
        for (k, &r) in row.iter().enumerate().skip(j + 1) {
            if r.abs() > settings.correlation_threshold {
                correlated_pairs.push((j, k, r));
            }
        }
    }

    let bins = (2.0 * size.cbrt()).ceil() as usize;
    let mut modes = Vec::with_capacity(p);
    let mut sides = Vec::with_capacity(p);
    let (mut mass_below, mut mass_above, mut overlap) = (vec![], vec![], vec![]);
    for j in 0..p {
        let reference = &references[j];
        let levels: Vec<f64> = natural[j].iter().map(|&v| reference.cdf(v)).collect();
        let share = settings.bound_share;
        let mass = |inside: &dyn Fn(f64) -> bool| -> f64 {
            levels
                .iter()
                .zip(weights)
                .filter(|(level, _)| inside(**level))
                .map(|(_, w)| w)
                .sum()
        };
        mass_below.push(mass(&|level| level < share));
        mass_above.push(mass(&|level| level > 1.0 - share));
        overlap.push(overlap_of(&levels, weights, bins));

        let (lower, upper) = sample.bounds()[j];
        let scale_bounds = if logs[j] {
            (lower.ln(), upper.ln())
        } else {
            (lower, upper)
        };
        let unscale = |y: f64| if logs[j] { y.exp() } else { y };
        match Density::new(
            &columns[j],
            weights,
            covariance[j][j].sqrt(),
            size,
            scale_bounds,
        ) {
            None => {
                modes.push(vec![natural[j][0]]);
                sides.push(vec![]);
            }
            Some(density) => {
                modes.push(
                    density
                        .modes
                        .iter()
                        .map(|&g| unscale(density.grid[g]))
                        .collect(),
                );
                let at_bound = [BoundSide::Lower, BoundSide::Upper]
                    .into_iter()
                    .filter(|&side| {
                        let bound = match side {
                            BoundSide::Lower => scale_bounds.0,
                            BoundSide::Upper => scale_bounds.1,
                        };
                        bound.is_finite()
                            && density.reaches(side)
                            && density.at_bound(
                                side,
                                reference.smoothed_at_bound(side, density.bandwidth, logs[j]),
                            )
                    })
                    .collect();
                sides.push(at_bound);
            }
        }
    }

    let verdicts: Vec<SampleVerdict> = (0..p)
        .map(|j| {
            let correlated_with: Vec<(usize, f64)> = correlated_pairs
                .iter()
                .filter_map(|&(a, b, r)| {
                    if a == j {
                        Some((b, r))
                    } else if b == j {
                        Some((a, r))
                    } else {
                        None
                    }
                })
                .collect();
            let low_contraction = contraction[j] < settings.contraction_threshold;
            SampleVerdict {
                verdict: PracticalVerdict::of(
                    low_contraction || !correlated_with.is_empty() || modes[j].len() >= 2,
                ),
                low_contraction,
                correlated_with,
                modes: modes[j].len(),
                at_bound: sides[j].clone(),
            }
        })
        .collect();

    let (directions, hidden_ridges) = if rows > p {
        let determined: Vec<usize> = (0..p)
            .filter(|&j| verdicts[j].verdict == PracticalVerdict::Determined)
            .collect();
        let ridges = hidden_ridges(
            &columns,
            weights,
            &deviations,
            &determined,
            settings.contraction_threshold,
            seed,
        );
        (directions(&covariance, &deviations), Some(ridges))
    } else {
        (vec![], None)
    };

    Ok(SampleIdentifiability {
        effective_size: size,
        distinct_rows: rows,
        contraction,
        correlation,
        correlated_pairs,
        modes,
        mass_below,
        mass_above,
        overlap,
        bins,
        directions,
        hidden_ridges,
        draws_per_parameter: size / p as f64,
        verdicts,
    })
}

/// The reference of each parameter: the uniform distribution on its bounds,
/// or its prior truncated to them.
fn references(
    sample: &ParameterSample,
    settings: &SampleIdentifiabilitySettings,
) -> Result<Vec<Distribution>, MethodError> {
    let priors = settings
        .reference_priors(sample.names(), sample.bounds())
        .map_err(MethodError::Invalid)?;
    if let Some(priors) = priors {
        return Ok(priors.distributions().to_vec());
    }
    if let Some((name, (min, max))) = first_infinite_bound(
        sample
            .names()
            .iter()
            .map(String::as_str)
            .zip(sample.bounds().iter().copied()),
    ) {
        return Err(MethodError::Invalid(format!(
            "reference='bounds' measures the sample against the uniform distribution on the \
             bounds, so they must be finite: parameter '{name}' has the bounds [{min}, {max}]. \
             Give reference='prior' with the priors of the sample"
        )));
    }
    Ok(sample
        .bounds()
        .iter()
        .map(|&(min, max)| {
            Distribution::new(Family::Uniform, min, max).expect("finite ordered bounds")
        })
        .collect())
}

/// The weighted covariance of `columns`, the covariance of the distribution
/// the draws put their weights on: `Σ wᵢ (xᵢ − x̄)(yᵢ − ȳ)` for weights that
/// sum to 1. It is the same for a sample and its distinct draws, never above
/// the largest variance on the bounds, and 0 for one row. Each column is
/// centred once, so each entry is one pass over the rows.
fn scaled_covariance(columns: &[Vec<f64>], weights: &[f64]) -> Vec<Vec<f64>> {
    let total: f64 = weights.iter().sum();
    let centred: Vec<Vec<f64>> = columns
        .iter()
        .map(|column| {
            let mean = column.iter().zip(weights).map(|(x, w)| w * x).sum::<f64>() / total;
            column.iter().map(|x| x - mean).collect()
        })
        .collect();
    let p = columns.len();
    let mut matrix = vec![vec![0.0; p]; p];
    for j in 0..p {
        for k in j..p {
            let value = centred[j]
                .iter()
                .zip(&centred[k])
                .zip(weights)
                .map(|((a, b), w)| w * a * b)
                .sum::<f64>()
                / total;
            matrix[j][k] = value;
            matrix[k][j] = value;
        }
    }
    matrix
}

/// `Σ_b min(w_b, 1/B)` over `bins` bins of equal mass of the reference, from
/// the level `F(θ)` of each draw.
fn overlap_of(levels: &[f64], weights: &[f64], bins: usize) -> f64 {
    let mut mass = vec![0.0; bins];
    for (level, weight) in levels.iter().zip(weights) {
        let bin = ((level * bins as f64) as usize).min(bins - 1);
        mass[bin] += weight;
    }
    let share = 1.0 / bins as f64;
    mass.iter().map(|m| m.min(share)).sum()
}

/// The eigenvalues, descending, and the eigenvectors of `D⁻¹ Σ D⁻¹` of the
/// parameters `columns` (indices) of `covariance`.
fn eigen(covariance: &[Vec<f64>], deviations: &[f64], columns: &[usize]) -> Vec<(f64, Vec<f64>)> {
    let q = columns.len();
    let scaled = DMatrix::from_fn(q, q, |a, b| {
        let (j, k) = (columns[a], columns[b]);
        covariance[j][k] / (deviations[j] * deviations[k])
    });
    let decomposition = SymmetricEigen::new(scaled);
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..q)
        .map(|i| {
            let vector: Vec<f64> = decomposition
                .eigenvectors
                .column(i)
                .iter()
                .copied()
                .collect();
            (decomposition.eigenvalues[i], vector)
        })
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    pairs
}

/// The directions of every parameter, from the least contracted.
fn directions(covariance: &[Vec<f64>], deviations: &[f64]) -> Vec<Direction> {
    let all: Vec<usize> = (0..deviations.len()).collect();
    eigen(covariance, deviations, &all)
        .into_iter()
        .map(|(value, mut loadings)| {
            let largest = loadings
                .iter()
                .copied()
                .max_by(|a, b| a.abs().total_cmp(&b.abs()))
                .expect("a direction has a loading");
            if largest < 0.0 {
                loadings.iter_mut().for_each(|l| *l = -*l);
            }
            Direction {
                contraction: 1.0 - value,
                loadings,
            }
        })
        .collect()
}

/// How many directions among the parameters `determined` have a
/// contraction below `threshold` once corrected for the spread of sample
/// eigenvalues: `2ĉ − mean ĉ*` of each direction, in order, over
/// [`BOOTSTRAP_REPLICATES`] Bayesian-bootstrap reweightings of the rows; 0
/// with fewer than two determined parameters.
fn hidden_ridges(
    columns: &[Vec<f64>],
    weights: &[f64],
    deviations: &[f64],
    determined: &[usize],
    threshold: f64,
    seed: u64,
) -> usize {
    if determined.len() < 2 {
        return 0;
    }
    let columns: Vec<Vec<f64>> = determined.iter().map(|&j| columns[j].clone()).collect();
    let deviations: Vec<f64> = determined.iter().map(|&j| deviations[j]).collect();
    let all: Vec<usize> = (0..determined.len()).collect();
    let contractions = |weights: &[f64]| -> Vec<f64> {
        let covariance = scaled_covariance(&columns, weights);
        eigen(&covariance, &deviations, &all)
            .into_iter()
            .map(|(value, _)| 1.0 - value)
            .collect()
    };
    let estimate = contractions(weights);
    let mut mean = vec![0.0; determined.len()];
    for replicate in 0..BOOTSTRAP_REPLICATES {
        let mut generator =
            Generator::from_seed(Stream::SampleDirections.replicate(seed, replicate));
        let mut drawn: Vec<f64> = weights
            .iter()
            .map(|w| w * exponential(&mut generator, 1.0))
            .collect();
        let total: f64 = drawn.iter().sum();
        drawn.iter_mut().for_each(|w| *w /= total);
        for (m, c) in mean.iter_mut().zip(contractions(&drawn)) {
            *m += c / BOOTSTRAP_REPLICATES as f64;
        }
    }
    estimate
        .iter()
        .zip(&mean)
        .filter(|&(c, m)| 2.0 * c - m < threshold)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_overlap_puts_the_upper_bound_in_the_last_bin() {
        assert_eq!(overlap_of(&[1.0, 0.0], &[0.5, 0.5], 2), 1.0);
        assert_eq!(overlap_of(&[1.0, 1.0], &[0.5, 0.5], 4), 0.25);
    }

    #[test]
    fn a_direction_has_its_largest_loading_positive() {
        let covariance = vec![vec![1.0, -0.9], vec![-0.9, 1.0]];
        let found = directions(&covariance, &[1.0, 1.0]);
        assert!((found[0].contraction - (1.0 - 1.9)).abs() < 1e-12);
        assert!((found[1].contraction - (1.0 - 0.1)).abs() < 1e-12);
        for direction in &found {
            let largest = direction
                .loadings
                .iter()
                .copied()
                .max_by(|a, b| a.abs().total_cmp(&b.abs()))
                .unwrap();
            assert!(largest > 0.0, "{direction:?}");
        }
    }
}
