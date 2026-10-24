//! The moves of the particles: the [`Kernel`] port and random-walk
//! Metropolis on the unbounded scale.

use commol_random::Generator;
use commol_random::sample::normal;
use commol_statistics::statistics::covariance;
use nalgebra::{DMatrix, DVector};

use crate::transform::Transform;

/// The scale of the covariance of a random walk in `d` dimensions:
/// `2.38² / d` (Roberts, Gelman and Gilks 1997).
const OPTIMAL_SCALE: f64 = 2.38 * 2.38;

/// A proposal of a Metropolis–Hastings move: the point, and the log of the
/// ratio of the proposal densities `q(θ | θ') / q(θ' | θ)` on the scale of
/// the parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    /// The proposed values.
    pub values: Vec<f64>,
    /// `log q(θ | θ') − log q(θ' | θ)`.
    pub log_ratio: f64,
}

/// The port of a move that keeps a distribution invariant with the
/// Metropolis–Hastings rule: the sampler accepts a proposal with
/// probability `min(1, π(θ') / π(θ) · exp(log_ratio))`.
pub trait Kernel: Sync {
    /// Fit the proposal to the weighted particles of a step.
    ///
    /// # Errors
    ///
    /// Returns a message when the particles cannot set a proposal, such as
    /// a parameter with no spread; the sampler reports it as
    /// [`MethodError::Failed`](crate::MethodError::Failed).
    fn adapt(
        &mut self,
        bounds: &[(f64, f64)],
        particles: &[Vec<f64>],
        weights: &[f64],
    ) -> Result<(), String>;

    /// A proposal from `current`, drawn from `stream`.
    fn propose(&self, current: &[f64], stream: &mut Generator) -> Proposal;
}

/// Random-walk Metropolis on the unbounded scale of
/// [`crate::transform::Transform`]: the step is normal with the weighted
/// covariance of the particles on that scale, times `2.38² / d`.
#[derive(Debug, Clone, Default)]
pub struct RandomWalk {
    transform: Option<Transform>,
    cholesky: Option<DMatrix<f64>>,
}

impl RandomWalk {
    /// A random walk with no proposal yet; [`Kernel::adapt`] sets it.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Kernel for RandomWalk {
    fn adapt(
        &mut self,
        bounds: &[(f64, f64)],
        particles: &[Vec<f64>],
        weights: &[f64],
    ) -> Result<(), String> {
        let transform = Transform::new(bounds);
        let d = bounds.len();
        let columns: Vec<Vec<f64>> = {
            let unbounded: Vec<Vec<f64>> = particles
                .iter()
                .map(|values| transform.to_unbounded(values))
                .collect();
            (0..d)
                .map(|j| unbounded.iter().map(|u| u[j]).collect())
                .collect()
        };
        let scale = OPTIMAL_SCALE / d as f64;
        let mut matrix = DMatrix::zeros(d, d);
        for j in 0..d {
            for k in 0..=j {
                let c = scale * covariance(&columns[j], &columns[k], weights);
                matrix[(j, k)] = c;
                matrix[(k, j)] = c;
            }
            if matrix[(j, j)].is_nan() || matrix[(j, j)] <= 0.0 {
                return Err(format!(
                    "The particles have no spread in parameter {j}, so the random walk has no \
                     step; use more particles or a smaller target_cess"
                ));
            }
        }
        self.transform = Some(transform);
        self.cholesky = Some(cholesky_or_diagonal(matrix));
        Ok(())
    }

    fn propose(&self, current: &[f64], stream: &mut Generator) -> Proposal {
        let transform = self.transform.as_ref().expect("adapt sets the proposal");
        let cholesky = self.cholesky.as_ref().expect("adapt sets the proposal");
        let step = cholesky * DVector::from_fn(current.len(), |_, _| normal(stream));
        let unbounded: Vec<f64> = transform
            .to_unbounded(current)
            .iter()
            .zip(step.iter())
            .map(|(u, s)| u + s)
            .collect();
        let values = transform.from_unbounded(&unbounded);
        // The walk is symmetric on the unbounded scale; on the scale of the
        // parameters the ratio is that of the Jacobians.
        let log_ratio = transform.log_jacobian(&values) - transform.log_jacobian(current);
        Proposal { values, log_ratio }
    }
}

/// The Cholesky factor of a covariance, or, for perfectly correlated
/// particles, the square root of its diagonal: each parameter moves alone.
fn cholesky_or_diagonal(matrix: DMatrix<f64>) -> DMatrix<f64> {
    match matrix.clone().cholesky() {
        Some(cholesky) => cholesky.l(),
        None => DMatrix::from_diagonal(&matrix.diagonal().map(f64::sqrt)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particles_with_no_spread_have_no_step() {
        let mut walk = RandomWalk::new();
        let error = walk
            .adapt(&[(0.0, 1.0)], &[vec![0.5], vec![0.5]], &[0.5, 0.5])
            .unwrap_err();
        assert!(error.contains("no spread in parameter 0"), "{error}");
    }

    #[test]
    fn perfectly_correlated_particles_move_each_parameter_alone() {
        let singular = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 1.0]);
        let factor = cholesky_or_diagonal(singular);
        assert_eq!(factor, DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn a_walk_on_the_real_line_has_no_jacobian() {
        let mut walk = RandomWalk::new();
        let particles = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 6.0]];
        let bounds = [(f64::NEG_INFINITY, f64::INFINITY); 2];
        walk.adapt(&bounds, &particles, &[1.0 / 3.0; 3]).unwrap();
        let proposal = walk.propose(&[2.0, 4.0], &mut Generator::from_seed(1));
        assert_eq!(proposal.log_ratio, 0.0);
        assert!(proposal.values.iter().all(|v| v.is_finite()));
    }
}
