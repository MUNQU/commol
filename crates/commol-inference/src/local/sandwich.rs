//! The sandwich covariance `H⁻¹ B H⁻¹` of a fit, with the meat `B` summed
//! over blocks of observations.
//!
//! `B = G/(G − 1) · Σ_b s_b s_bᵀ` sums the outer products of the block
//! scores `s_b = Σ_{i∈b} gᵢ`, the gradients of the terms of the observations
//! of block `b` ([`Curvature::gradients`]): the clustered estimator CR1 with
//! the blocks of [`BlockScheme`]. It holds under dependence inside blocks,
//! noise whose size changes, and a wrong noise model, and needs no noise
//! model. Intervals and draws read a Student t with `G − 1` degrees of
//! freedom (Cameron and Miller 2015).

use commol_calibration::CalibrationProblem;
use commol_core::SimulationEngine;

use super::approximation::{CurvatureKind, curvature_at};
use super::curvature::Curvature;
use super::gaussian::{Gaussian, inverse};
use crate::error::MethodError;
use crate::resampling::{BlockDrawing, BlockScheme, Estimates};
use crate::sensitivity::Matrix;

/// The sandwich of `curvature` with the blocks of `blocks`: a Student t on
/// the unbounded scale, centred on the fit, of covariance `H⁻¹ B H⁻¹` and
/// `G − 1` degrees of freedom.
///
/// # Errors
///
/// [`MethodError::Invalid`] when there are not more blocks than parameters,
/// since `B` would have rank below the parameters, or when `H` or the
/// sandwich is singular.
pub fn sandwich(curvature: &Curvature, blocks: &BlockScheme) -> Result<Gaussian, MethodError> {
    let count = blocks.count();
    let p = curvature.fit().len();
    if count <= p {
        let plural = |n: usize, word: &str| {
            if n == 1 {
                format!("{n} {word}")
            } else {
                format!("{n} {word}s")
            }
        };
        return Err(MethodError::Invalid(format!(
            "block_length {} makes {} for {}: the sum of the scores over blocks needs more \
             blocks than parameters, or its rank is below the parameters. Use shorter blocks",
            blocks.length(),
            plural(count, "block"),
            plural(p, "parameter")
        )));
    }
    let meat = block_meat(curvature, blocks);
    let bread = inverse(curvature.matrix())?;
    let covariance = &bread * meat * &bread;
    Gaussian::new(
        curvature.transform().clone(),
        curvature.unbounded_fit().to_vec(),
        covariance,
        Some((blocks.count() - 1) as f64),
    )
}

/// The meat `B = G/(G − 1) · Σ_b s_b s_bᵀ` of the gradients of
/// `curvature` over the blocks of `blocks`, on the unbounded scale. With
/// `G ≤ p` its rank is below the parameters `p`, since the block scores of a
/// fit sum to zero.
///
/// `blocks` has at least two blocks.
pub fn block_meat(curvature: &Curvature, blocks: &BlockScheme) -> Matrix {
    let count = blocks.count();
    debug_assert!(count >= 2, "the meat of one block");
    let scores = block_scores(curvature, blocks);
    scores.transpose() * scores * (count as f64 / (count - 1) as f64)
}

/// The block scores `s_b = Σ_{i∈b} gᵢ` of the gradients of `curvature` over
/// the blocks of `blocks`, on the unbounded scale: one row per block.
pub fn block_scores(curvature: &Curvature, blocks: &BlockScheme) -> Matrix {
    let p = curvature.fit().len();
    let gradients = curvature.gradients();
    let mut scores = Matrix::zeros(blocks.count(), p);
    for (b, block) in blocks.blocks().iter().enumerate() {
        for &i in block {
            for j in 0..p {
                scores[(b, j)] += gradients[(i, j)];
            }
        }
    }
    scores
}

/// The blocks of `problem` of `block_length` positions, or of the
/// automatic length of non-overlapping blocks of the residuals at the fit of
/// `curvature` (Politis and White 2004) for `None`, with the estimate of
/// each series ([`BlockScheme::resolve`]).
///
/// # Errors
///
/// [`MethodError::Invalid`] when the block length does not fit the
/// positions.
pub fn blocks_at<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    curvature: &Curvature,
    block_length: Option<usize>,
) -> Result<BlockScheme, MethodError> {
    let observed = problem.observed_data();
    let residuals: Vec<f64> = observed
        .iter()
        .zip(curvature.predictions())
        .map(|(obs, prediction)| obs.value - prediction)
        .collect();
    BlockScheme::resolve(
        observed,
        &residuals,
        block_length,
        BlockDrawing::NonOverlapping,
    )
}

/// A sandwich and the blocks it summed over.
#[derive(Debug, Clone)]
pub struct SandwichRun {
    /// The curvature `H` of the bread.
    pub curvature: Curvature,
    /// The Student t of the sandwich.
    pub gaussian: Gaussian,
    /// The number of blocks `G`.
    pub blocks: usize,
    /// The positions per block.
    pub block_length: usize,
    /// With the automatic length, the estimate of each series by label.
    pub estimates: Option<Estimates>,
}

/// The sandwich of `problem` at `fit` with blocks of `block_length`
/// positions, or of the automatic length of non-overlapping blocks of the
/// residuals at the fit (Politis and White 2004) for `None`.
///
/// # Errors
///
/// As [`curvature_at`] and [`sandwich`], and [`MethodError::Invalid`] when
/// the block length does not fit the positions.
pub fn sandwich_approximation<E>(
    problem: &CalibrationProblem<E>,
    fit: &[f64],
    kind: CurvatureKind,
    block_length: Option<usize>,
) -> Result<SandwichRun, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    let curvature = curvature_at(problem, fit, kind)?;
    let blocks = blocks_at(problem, &curvature, block_length)?;
    let gaussian = sandwich(&curvature, &blocks)?;
    Ok(SandwichRun {
        curvature,
        gaussian,
        blocks: blocks.count(),
        block_length: blocks.length(),
        estimates: blocks.estimates().map(<[_]>::to_vec),
    })
}
