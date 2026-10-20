//! A normal or Student-t distribution on the unbounded scale, and the
//! Laplace approximation `N(û, H_u⁻¹)` of a curvature.

use commol_random::{Generator, stream_seed};
use nalgebra::DVector;

use super::LocalError;
use super::curvature::Curvature;
use crate::distribution::{Distribution, Family};
use crate::sensitivity::Matrix;
use crate::smc::kernel::standard_normals;
use crate::transform::Transform;

/// The stream of the draws of a local approximation ([`stream_seed`]).
const DRAWS_STREAM: u64 = 1;

/// A normal distribution on the unbounded scale, or a multivariate Student t
/// with `degrees_of_freedom`, whose draws map back to the parameters.
#[derive(Debug, Clone)]
pub struct Gaussian {
    transform: Transform,
    center: Vec<f64>,
    covariance: Matrix,
    factor: Matrix,
    degrees_of_freedom: Option<f64>,
}

impl Gaussian {
    /// The distribution of centre `center` and covariance `covariance` on
    /// the unbounded scale of `transform`; a Student t with
    /// `degrees_of_freedom`, or a normal with `None`.
    ///
    /// # Errors
    ///
    /// [`LocalError::Refused`] when the covariance is not positive definite.
    pub fn new(
        transform: Transform,
        center: Vec<f64>,
        covariance: Matrix,
        degrees_of_freedom: Option<f64>,
    ) -> Result<Self, LocalError> {
        let factor = covariance
            .clone()
            .cholesky()
            .ok_or_else(|| {
                LocalError::Refused(
                    "The covariance is not positive definite: the data do not determine every \
                     parameter at the fit. Calibrator.identifiability(...) shows which"
                        .to_string(),
                )
            })?
            .l();
        Ok(Self {
            transform,
            center,
            covariance,
            factor,
            degrees_of_freedom,
        })
    }

    /// The centre on the unbounded scale.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// The covariance on the unbounded scale.
    pub fn covariance(&self) -> &Matrix {
        &self.covariance
    }

    /// The lower Cholesky factor of the covariance.
    pub fn factor(&self) -> &Matrix {
        &self.factor
    }

    /// The degrees of freedom of a Student t, or `None` for a normal.
    pub fn degrees_of_freedom(&self) -> Option<f64> {
        self.degrees_of_freedom
    }

    /// The map back to the parameters.
    pub fn transform(&self) -> &Transform {
        &self.transform
    }

    /// `L⁻¹ (u − c)`: the point `u` of the unbounded scale in standard units;
    /// its norm is the Mahalanobis distance from the centre.
    pub fn standardized(&self, unbounded: &[f64]) -> Vec<f64> {
        let difference = DVector::from_iterator(
            unbounded.len(),
            unbounded.iter().zip(&self.center).map(|(u, c)| u - c),
        );
        self.factor
            .solve_lower_triangular(&difference)
            .expect("a Cholesky factor has a positive diagonal")
            .iter()
            .copied()
            .collect()
    }

    /// The log density at a point `u` of the unbounded scale.
    pub fn log_density(&self, unbounded: &[f64]) -> f64 {
        let d = self.center.len() as f64;
        let squared: f64 = self.standardized(unbounded).iter().map(|z| z * z).sum();
        let log_determinant: f64 = self.factor.diagonal().iter().map(|l| 2.0 * l.ln()).sum();
        match self.degrees_of_freedom {
            None => -0.5 * (d * (2.0 * std::f64::consts::PI).ln() + log_determinant + squared),
            Some(nu) => {
                libm::lgamma(0.5 * (nu + d))
                    - libm::lgamma(0.5 * nu)
                    - 0.5 * d * (nu * std::f64::consts::PI).ln()
                    - 0.5 * log_determinant
                    - 0.5 * (nu + d) * (squared / nu).ln_1p()
            }
        }
    }

    /// One draw on the unbounded scale: `c + L z`, with `z` standard normal,
    /// divided by `√(w/ν)` with `w ~ χ²_ν` for a Student t.
    pub fn draw_unbounded(&self, generator: &mut Generator) -> Vec<f64> {
        let normals = DVector::from_vec(standard_normals(generator, self.center.len()));
        let mut step = &self.factor * normals;
        if let Some(nu) = self.degrees_of_freedom {
            let chi_square = Distribution::new(
                Family::Gamma {
                    shape: 0.5 * nu,
                    rate: 0.5,
                },
                0.0,
                f64::INFINITY,
            )
            .expect("a chi-square of positive degrees of freedom")
            .sample(generator);
            step *= (nu / chi_square).sqrt();
        }
        self.center
            .iter()
            .zip(step.iter())
            .map(|(c, s)| c + s)
            .collect()
    }

    /// `count` draws mapped back to the parameters, from the stream of
    /// draws of `seed`; every draw lies inside the bounds.
    pub fn draws(&self, count: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut generator = Generator::from_seed(stream_seed(seed, DRAWS_STREAM));
        (0..count)
            .map(|_| {
                self.transform
                    .from_unbounded(&self.draw_unbounded(&mut generator))
            })
            .collect()
    }
}

/// The Laplace approximation of a curvature: the normal `N(û, H_u⁻¹)` on the
/// unbounded scale, centred on the fit.
///
/// # Errors
///
/// [`LocalError::Refused`] when the curvature is not positive definite.
pub fn laplace(curvature: &Curvature) -> Result<Gaussian, LocalError> {
    let covariance = inverse(curvature.matrix())?;
    Gaussian::new(
        curvature.transform().clone(),
        curvature.unbounded_fit().to_vec(),
        covariance,
        None,
    )
}

/// The inverse of a symmetric positive definite matrix, by its Cholesky
/// factor.
///
/// # Errors
///
/// [`LocalError::Refused`] when the matrix is not positive definite.
pub(crate) fn inverse(matrix: &Matrix) -> Result<Matrix, LocalError> {
    matrix
        .clone()
        .cholesky()
        .map(|cholesky| cholesky.inverse())
        .ok_or_else(|| {
            LocalError::Refused(
                "The curvature at the fit is not positive definite: the data do not \
                 determine every parameter there, or the fit is not a minimum. \
                 Calibrator.identifiability(...) shows which parameters the data determine"
                    .to_string(),
            )
        })
}
