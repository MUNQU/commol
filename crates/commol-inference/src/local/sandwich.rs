//! The sandwich covariance `H⁻¹ B H⁻¹` of a fit, with the meat `B` summed
//! over blocks of observations.
//!
//! `B = G/(G − 1) · Σ_b s_b s_bᵀ` sums the outer products of the block
//! scores `s_b = Σ_{i∈b} gᵢ`, the gradients of the terms of the observations
//! of block `b` ([`Curvature::scaled_gradients`]): the clustered estimator CR1
//! with the blocks of [`BlockScheme`]. It holds under dependence inside blocks,
//! noise whose size changes, and a wrong noise model, and needs no noise
//! model. Intervals and draws read a Student t with `G − 1` degrees of
//! freedom (Cameron and Miller 2015).
//!
//! The covariance is `G/(G − 1) · Σ_b a_b a_bᵀ`, with the influence
//! `a_b = H⁻¹ s_b` of each block on the fit
//! ([`super::InverseCurvature::sandwich`]): the square of a score and the
//! product of two inverses can be too small or too large for a number when the
//! covariance is not.

use commol_calibration::CalibrationProblem;
use commol_core::SimulationEngine;

use super::approximation::{CurvatureKind, curvature_at};
use super::curvature::Curvature;
use super::gaussian::Gaussian;
use crate::error::MethodError;
use crate::resampling::BlockScheme;
use crate::sensitivity::Matrix;

/// The sandwich of `curvature` with the blocks of `blocks`: a Student t on
/// the unbounded scale, centred on the fit, of covariance `H⁻¹ B H⁻¹` and
/// `G − 1` degrees of freedom.
///
/// # Errors
///
/// [`MethodError::Invalid`] when there are not more blocks than parameters,
/// since `B` would have rank below the parameters, when the data do not
/// determine every parameter ([`Curvature::inverse`]), or when the sandwich is
/// singular ([`Gaussian::new`]); [`MethodError::Failed`] when the curvature of
/// a parameter is too large for its covariance to be a number
/// ([`super::InverseCurvature::sandwich`]).
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
    let covariance = curvature.inverse()?.sandwich(blocks)?;
    Gaussian::new(
        curvature.transform().clone(),
        curvature.unbounded_fit().to_vec(),
        covariance,
        Some((count - 1) as f64),
    )
}

/// The block scores `s_b = Σ_{i∈b} gᵢ` of the gradients of `curvature` over
/// the blocks of `blocks`, in the units of [`Curvature::scales`]: one row per
/// block.
pub fn scaled_block_scores(curvature: &Curvature, blocks: &BlockScheme) -> Matrix {
    let p = curvature.fit().len();
    let gradients = curvature.scaled_gradients();
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
}

/// The sandwich of `problem` at `fit` with blocks of `block_length`
/// positions.
///
/// # Errors
///
/// As [`curvature_at`], [`BlockScheme::new`] and [`sandwich`].
pub fn sandwich_approximation<E>(
    problem: &CalibrationProblem<E>,
    fit: &[f64],
    kind: CurvatureKind,
    block_length: usize,
) -> Result<SandwichRun, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    let curvature = curvature_at(problem, fit, kind)?;
    let blocks = BlockScheme::new(problem.observed_data(), block_length)?;
    let gaussian = sandwich(&curvature, &blocks)?;
    Ok(SandwichRun {
        curvature,
        gaussian,
        blocks: blocks.count(),
        block_length: blocks.length(),
    })
}
