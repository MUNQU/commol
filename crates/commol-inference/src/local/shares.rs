//! The shares of the blocks of a sandwich in the Gauss–Newton curvature,
//! which give each parameter, and each output of a band, its factor and its
//! degrees of freedom (Bell and McCaffrey 2002).
//!
//! For a direction `c` of the parameters, with `J_b` the Gauss–Newton
//! curvature of block `b` ([`Curvature::scaled_block_matrix`]) and
//! `J = Σ_b J_b`, the variance of `cᵀ θ̂` that the blocks read is
//! `v_bc = cᵀ J⁻¹ V_bc J⁻¹ c`, with `V_bc = J_b δ_bc − J_b J⁻¹ J_c` the
//! covariance of the block scores after the fit when the scores of the terms
//! are independent with the variance of their curvature. The factor of a
//! parameter `j` is `f_j = (J⁻¹)_jj / Σ_b v_bb`, and the degrees of freedom of
//! a direction are `ν = (Σ_b v_bb)² / Σ_bc v_bc²`, Satterthwaite's for the
//! sum of the squares of the influences of the blocks. When every block holds
//! `1/G` of `J`, `f_j = G/(G − 1)` (CR1) and `ν = G − 1`; a block with no
//! curvature adds nothing to either, and one with a curvature that goes to 0
//! adds less and less. The shares are those of Gauss–Newton also for a
//! sandwich whose bread is the Hessian.
//!
//! The sums are read without the `G × G` matrix of the `v_bc` and without a
//! difference, which would lose the share of a block that holds nearly all
//! the curvature of the direction: with `u = J⁻¹ c`, `y_b = J_b u` and
//! `z_b = J⁻¹ y_b`, `v_bb = z_bᵀ Σ_{c≠b} y_c`, since `Σ_c y_c = J u`, and
//! `v_bc = −z_bᵀ y_c` for `b ≠ c`, so
//! `Σ_bc v_bc² = Σ_b v_bb² + 2 Σ_b z_bᵀ (Σ_{c<b} y_c y_cᵀ) z_b`.

use commol_statistics::linear_algebra::symmetric_inverse;
use nalgebra::DVector;

use super::curvature::Curvature;
use crate::error::MethodError;
use crate::resampling::BlockScheme;
use crate::sensitivity::Matrix;

/// The share of its scale at or below which a sum that the blocks read is
/// the rounding of 0. A block whose share of the curvature,
/// `tr(J⁻¹ J_b) / p`, is not above it holds no curvature
/// ([`BlockShares::with_curvature`]); `Σ_b v_bb` of a direction `c` at or
/// below it of `cᵀ J⁻¹ c` is the rounding of blocks that each hold all the
/// curvature of `c` ([`BlockShares::factor`]); and information matching
/// (`smc::information_matching_eta`) reads it for the trace of its block
/// scores and its degrees of freedom.
pub const ROUNDED_SHARE: f64 = 1e-10;

/// The Gauss–Newton curvature of each block of a sandwich that has one, and
/// the inverse of their sum, in the units of [`Curvature::scales`].
#[derive(Debug, Clone)]
pub struct BlockShares {
    inverse: Matrix,
    blocks: Vec<Matrix>,
    /// The index in the scheme of each block of `blocks`.
    rows: Vec<usize>,
}

impl BlockShares {
    /// The shares of the blocks of `blocks` in the Gauss–Newton curvature of
    /// `curvature`. A block whose curvature is 0 holds no share.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when the sum of the curvatures of the blocks
    /// is not positive definite, by the rule of [`Curvature::inverse`]: with
    /// a Hessian bread, the Gauss–Newton curvature can be singular where the
    /// Hessian is not.
    pub fn new(curvature: &Curvature, blocks: &BlockScheme) -> Result<Self, MethodError> {
        let p = curvature.fit().len();
        let (rows, blocks): (Vec<usize>, Vec<Matrix>) = blocks
            .blocks()
            .iter()
            .map(|block| curvature.scaled_block_matrix(block))
            .enumerate()
            .filter(|(_, matrix)| matrix.diagonal().iter().any(|&entry| entry != 0.0))
            .unzip();
        let total = blocks
            .iter()
            .fold(Matrix::zeros(p, p), |total, block| total + block);
        let inverse = symmetric_inverse(&total).ok_or_else(|| {
            MethodError::Invalid(
                "The Gauss–Newton curvature of the blocks at the fit is not positive definite: \
                 the sandwich reads the share of each block in it, and the first derivatives \
                 of the model do not determine every parameter there. \
                 Calibrator.identifiability(...) shows which parameters the data determine"
                    .to_string(),
            )
        })?;
        Ok(Self {
            inverse,
            blocks,
            rows,
        })
    }

    /// The number of blocks with curvature: those whose share of the
    /// curvature, `tr(J⁻¹ J_b) / p`, is above [`ROUNDED_SHARE`]. A block with a
    /// smaller share still enters `J` and the sums of the factors and the
    /// degrees of freedom, which are continuous in it.
    pub fn with_curvature(&self) -> usize {
        let p = self.inverse.nrows() as f64;
        // Both are symmetric, so their entrywise product is the trace.
        self.blocks
            .iter()
            .filter(|block| self.inverse.dot(block) > ROUNDED_SHARE * p)
            .count()
    }

    /// The block scores `scores`, one row per block of the scheme in the
    /// units of [`Curvature::scales`] (`super::scaled_block_scores`), after
    /// one Gauss–Newton step to the zero of their sum `g`:
    /// `s̃_b = s_b − J_b J⁻¹ g`. They sum to 0 whatever the error of the fit,
    /// and equal `s_b` at an exact minimum, where `V_bc` is their covariance.
    /// A block with no curvature keeps its scores.
    pub fn corrected_scores(&self, mut scores: Matrix) -> Matrix {
        let gradient = scores.row_sum().transpose();
        let step = &self.inverse * gradient;
        for (&row, block) in self.rows.iter().zip(&self.blocks) {
            let correction = block * &step;
            for (j, value) in correction.iter().enumerate() {
                scores[(row, j)] -= value;
            }
        }
        scores
    }

    /// The factor `f_j = (J⁻¹)_jj / Σ_b v_bb` of parameter `j`, or `None`
    /// when the blocks each hold all its curvature: `Σ_b v_bb` is then at most
    /// `10⁻¹⁰ (J⁻¹)_jj`, where the scores of `j` sum to zero in each block at
    /// the fit.
    pub fn factor(&self, j: usize) -> Option<f64> {
        let sums = self.sums(&unit(self.inverse.nrows(), j));
        (sums.read > ROUNDED_SHARE * sums.variance).then(|| sums.variance / sums.read)
    }

    /// The degrees of freedom `ν = (Σ_b v_bb)² / Σ_bc v_bc²` of the direction
    /// `direction` of the parameters, in the units of [`Curvature::scales`]:
    /// at least 1, the least value of the ratio, also for a direction that no
    /// block reads. They do not depend on the size of `direction`.
    pub fn degrees_of_freedom(&self, direction: &[f64]) -> f64 {
        let sums = self.sums(&DVector::from_column_slice(direction));
        (sums.read * sums.read / sums.squares).max(1.0)
    }

    /// The degrees of freedom of each parameter ([`Self::degrees_of_freedom`]).
    pub fn parameter_degrees_of_freedom(&self) -> Vec<f64> {
        let p = self.inverse.nrows();
        (0..p)
            .map(|j| self.degrees_of_freedom(unit(p, j).as_slice()))
            .collect()
    }

    /// `cᵀ J⁻¹ c`, `Σ_b v_bb` and `Σ_bc v_bc²` of the direction `c`.
    fn sums(&self, direction: &DVector<f64>) -> Sums {
        let p = direction.len();
        let u = &self.inverse * direction;
        let shares: Vec<DVector<f64>> = self.blocks.iter().map(|block| block * &u).collect();
        // The sum of the shares after each block, so that the sum of the
        // others is never a difference.
        let mut after = vec![DVector::zeros(p); shares.len()];
        for b in (1..shares.len()).rev() {
            after[b - 1] = &after[b] + &shares[b];
        }
        let mut sums = Sums {
            variance: direction.dot(&u),
            read: 0.0,
            squares: 0.0,
        };
        let mut before = DVector::zeros(p);
        let mut outer_before = Matrix::zeros(p, p);
        for (share, after) in shares.iter().zip(&after) {
            let reach = &self.inverse * share;
            let own = reach.dot(&(&before + after));
            sums.read += own;
            sums.squares += own * own + 2.0 * reach.dot(&(&outer_before * &reach));
            before += share;
            outer_before += share * share.transpose();
        }
        sums
    }
}

/// The sums of a direction that [`BlockShares`] reads.
struct Sums {
    /// `cᵀ J⁻¹ c`.
    variance: f64,
    /// `Σ_b v_bb`.
    read: f64,
    /// `Σ_bc v_bc²`.
    squares: f64,
}

/// The `j`-th of the `p` unit vectors.
fn unit(p: usize, j: usize) -> DVector<f64> {
    DVector::from_fn(p, |k, _| if k == j { 1.0 } else { 0.0 })
}
