//! The inverse of a curvature, which the Laplace approximation, the sandwich,
//! the Laplace mixture and information matching share, with its one rule of
//! what the data do not determine, and the covariances that leave its units.

use commol_statistics::linear_algebra::symmetric_inverse;

use super::curvature::Curvature;
use super::sandwich::scaled_block_scores;
use crate::error::MethodError;
use crate::resampling::BlockScheme;
use crate::sensitivity::Matrix;

/// The inverse of a [`Curvature`], kept in its units ([`Curvature::inverse`]).
#[derive(Debug, Clone)]
pub struct InverseCurvature<'a> {
    curvature: &'a Curvature,
    matrix: Matrix,
}

impl<'a> InverseCurvature<'a> {
    /// The inverse of `curvature`, or [`MethodError::Invalid`] when the data
    /// do not determine it ([`Curvature::inverse`]).
    pub(super) fn new(curvature: &'a Curvature) -> Result<Self, MethodError> {
        let matrix = symmetric_inverse(curvature.scaled_matrix()).ok_or_else(|| {
            MethodError::Invalid(
                "The curvature at the fit is not positive definite: the data do not \
                 determine every parameter there, or the fit is not a minimum. \
                 Calibrator.identifiability(...) shows which parameters the data determine"
                    .to_string(),
            )
        })?;
        Ok(Self { curvature, matrix })
    }

    /// The inverse `Ĥ⁻¹`, in the units of [`Curvature::scales`].
    pub fn scaled_matrix(&self) -> &Matrix {
        &self.matrix
    }

    /// The inverse of `curvature_scale` times the curvature on the unbounded
    /// scale, `(c·H_u)⁻¹ = S⁻¹ Ĥ⁻¹ S⁻¹ / c`, exactly symmetric. An entry above
    /// the largest number is infinite.
    ///
    /// # Errors
    ///
    /// [`MethodError::Failed`] when a variance is below the smallest number:
    /// the curvature of its parameter is too large for its covariance to be a
    /// number.
    pub fn covariance(&self, curvature_scale: f64) -> Result<Matrix, MethodError> {
        let scales = self.curvature.scales();
        let p = scales.len();
        let covariance = Matrix::from_fn(p, p, |j, k| {
            let (j, k) = (j.min(k), j.max(k));
            self.matrix[(j, k)] / scales[j] / scales[k] / curvature_scale
        });
        // A diagonal of the inverse is above 0, so a variance of 0 is one below
        // the smallest number.
        self.check_variances(&covariance, |_| true)?;
        Ok(covariance)
    }

    /// The sandwich covariance on the unbounded scale,
    /// `G/(G − 1) · Σ_b a_b a_bᵀ` over the `G` blocks of `blocks`, with the
    /// influence `a_b = H_u⁻¹ s_b` of the block score `s_b` of the curvature
    /// ([`scaled_block_scores`]), exactly symmetric. `blocks` has at least two
    /// blocks.
    ///
    /// # Errors
    ///
    /// [`MethodError::Failed`] when a variance is below the smallest number
    /// while an influence on its parameter is not 0: the curvature of that
    /// parameter is too large for its covariance to be a number.
    pub fn sandwich(&self, blocks: &BlockScheme) -> Result<Matrix, MethodError> {
        let count = blocks.count();
        debug_assert!(count >= 2, "the sandwich of one block");
        let scaled = scaled_block_scores(self.curvature, blocks) * &self.matrix;
        let mut influences = scaled.clone();
        for (mut column, &scale) in influences.column_iter_mut().zip(self.curvature.scales()) {
            column.unscale_mut(scale);
        }
        let correction = count as f64 / (count - 1) as f64;
        let p = influences.ncols();
        let covariance = Matrix::from_fn(p, p, |j, k| {
            correction * influences.column(j).dot(&influences.column(k))
        });
        self.check_variances(&covariance, |j| {
            scaled.column(j).iter().any(|&entry| entry != 0.0)
        })?;
        Ok(covariance)
    }

    /// Refuse a covariance with a variance of 0 for a parameter `j` that
    /// `reads(j)`: a variance that leaves the units below the smallest number.
    fn check_variances(
        &self,
        covariance: &Matrix,
        reads: impl Fn(usize) -> bool,
    ) -> Result<(), MethodError> {
        match (0..covariance.nrows()).find(|&j| covariance[(j, j)] == 0.0 && reads(j)) {
            Some(j) => Err(MethodError::Failed(format!(
                "The curvature of '{}' at the fit is too large for its covariance to be a \
                 number: its variance on the unbounded scale is below the smallest number",
                self.curvature.names()[j]
            ))),
            None => Ok(()),
        }
    }
}
