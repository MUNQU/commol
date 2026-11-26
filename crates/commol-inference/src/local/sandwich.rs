//! The sandwich covariance of a fit, with the meat summed over blocks of
//! observations.
//!
//! The meat sums the outer products of the block scores
//! `s_b = Σ_{i∈b} gᵢ`, the gradients of the terms of the observations of
//! block `b` ([`Curvature::scaled_gradients`]), over the blocks of
//! [`BlockScheme`]. It holds under dependence inside blocks, noise whose size
//! changes, and a wrong noise model, and needs no noise model. Each block's
//! scores are read after one Gauss–Newton step to the zero of their sum `g`,
//! `s̃_b = s_b − J_b J⁻¹ g` ([`BlockShares::corrected_scores`]), so they sum
//! to 0 by construction, as the scores of an exact minimum do: the gradient
//! that an optimizer leaves at the fit does not fill a rank that the blocks
//! do not have, and the covariance does not follow the error of the fit.
//!
//! The covariance is `√(f_j f_k) · Σ_b a_bj a_bk`, with the influence
//! `a_b = H⁻¹ s̃_b` of each block on the fit
//! ([`super::InverseCurvature::sandwich`]): the square of a score and the
//! product of two inverses can be too small or too large for a number when the
//! covariance is not. The factor `f_j` and the degrees of freedom `ν_j` of
//! each parameter read the share of its curvature that each block holds
//! ([`BlockShares`], Bell and McCaffrey 2002): with equal shares they are
//! `G/(G − 1)` and `G − 1`, the clustered estimator CR1 (Cameron and Miller
//! 2015), and a block with no curvature changes neither. Draws read a grouped
//! t of the `ν_j` ([`Gaussian`]).

use commol_calibration::CalibrationProblem;
use commol_core::SimulationEngine;
use commol_statistics::linear_algebra::singular;

use super::approximation::{CurvatureKind, curvature_at};
use super::curvature::Curvature;
use super::gaussian::Gaussian;
use super::shares::BlockShares;
use crate::error::MethodError;
use crate::resampling::BlockScheme;
use crate::sensitivity::Matrix;

/// The sandwich of `curvature` with the blocks of `blocks`: a grouped t on
/// the unbounded scale, centred on the fit, of covariance
/// `√(f_j f_k) · Σ_b a_bj a_bk` and the degrees of freedom `ν_j` of each
/// parameter ([`BlockShares`]).
///
/// # Errors
///
/// [`MethodError::Invalid`] when the data do not determine every parameter
/// ([`Curvature::inverse`], [`BlockShares::new`]), when there are not more
/// blocks with curvature than parameters, those whose share of the curvature
/// `tr(J⁻¹ J_b) / p` is above [`super::shares::ROUNDED_SHARE`]
/// ([`BlockShares::with_curvature`]), since the sum of the scores over blocks
/// would have rank below the parameters, when the blocks each hold all
/// the curvature of a parameter ([`BlockShares::factor`]), or when the
/// block scores have rank below the parameters, so that the sandwich is
/// singular on a diagonal of 1 ([`singular`]) or not positive definite
/// ([`Gaussian::new`]); [`MethodError::Failed`] when the
/// curvature of a parameter is too large for its covariance to be a number
/// ([`super::InverseCurvature::sandwich`]).
pub fn sandwich(curvature: &Curvature, blocks: &BlockScheme) -> Result<Gaussian, MethodError> {
    let inverse = curvature.inverse()?;
    let shares = BlockShares::new(curvature, blocks)?;
    let read = shares.with_curvature();
    let p = curvature.fit().len();
    if read <= p {
        let plural = |n: usize, word: &str| {
            if n == 1 {
                format!("{n} {word}")
            } else {
                format!("{n} {word}s")
            }
        };
        return Err(MethodError::Invalid(format!(
            "block_length {} makes {} with curvature, of {} in all, for {}: the sum of the \
             scores over blocks needs more blocks with curvature than parameters, or its rank \
             is below the parameters. Use shorter blocks",
            blocks.length(),
            plural(read, "block"),
            blocks.count(),
            plural(p, "parameter")
        )));
    }
    let factors = (0..p)
        .map(|j| {
            shares.factor(j).ok_or_else(|| {
                MethodError::Invalid(format!(
                    "block_length {} makes blocks that each hold all the curvature of '{}', \
                     where its scores sum to zero at the fit. Use shorter blocks",
                    blocks.length(),
                    curvature.names()[j]
                ))
            })
        })
        .collect::<Result<Vec<f64>, _>>()?;
    let scores = shares.corrected_scores(scaled_block_scores(curvature, blocks));
    let covariance = inverse.sandwich(scores, &factors)?;
    if below_rank(&covariance) {
        return Err(MethodError::Invalid(format!(
            "block_length {} makes block scores of rank below the parameters: in every block \
             the scores of some parameters move together, so the sandwich gives them no \
             variance apart. Use another block length, or a method that does not read block \
             scores, such as Laplace or a bootstrap",
            blocks.length()
        )));
    }
    Gaussian::new(
        curvature.transform().clone(),
        curvature.unbounded_fit().to_vec(),
        covariance,
        Some(shares.parameter_degrees_of_freedom()),
    )
}

/// Whether the sandwich `covariance` is singular to working precision on a
/// diagonal of 1 ([`singular`]): the block scores have rank below the
/// parameters, which the rounding of a Cholesky factor can hide. A variance
/// that is not a positive number is left to [`Gaussian::new`].
fn below_rank(covariance: &Matrix) -> bool {
    let roots: Vec<f64> = covariance
        .diagonal()
        .iter()
        .map(|entry| entry.sqrt())
        .collect();
    if !roots.iter().all(|root| root.is_finite() && *root > 0.0) {
        return false;
    }
    let p = roots.len();
    let unit = Matrix::from_fn(p, p, |j, k| covariance[(j, k)] / roots[j] / roots[k]);
    unit.iter().all(|entry| entry.is_finite()) && singular(&unit.symmetric_eigenvalues())
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
    /// The grouped t of the sandwich.
    pub gaussian: Gaussian,
    /// The number of blocks `G`, those with no curvature included.
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
