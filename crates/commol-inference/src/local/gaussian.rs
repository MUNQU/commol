//! A normal or Student-t distribution on the unbounded scale, and the
//! Laplace approximation `N(û, H_u⁻¹)` of a curvature.

use commol_random::Generator;
use commol_random::sample::{gamma, normal};
use commol_statistics::numeric::{normal_log_density, student_t_log_density};
use nalgebra::DVector;

use super::curvature::Curvature;
use crate::error::MethodError;
use crate::sensitivity::Matrix;
use crate::transform::Transform;
use commol_random::Stream;

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
    /// [`MethodError::Invalid`] when the covariance is not positive definite,
    /// or an entry is not finite.
    pub fn new(
        transform: Transform,
        center: Vec<f64>,
        covariance: Matrix,
        degrees_of_freedom: Option<f64>,
    ) -> Result<Self, MethodError> {
        let factor = Some(&covariance)
            .filter(|covariance| covariance.iter().all(|entry| entry.is_finite()))
            .and_then(|covariance| covariance.clone().cholesky())
            .ok_or_else(|| {
                MethodError::Invalid(
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

    /// The log density at a point `u` of the unbounded scale: its squared
    /// Mahalanobis distance from the centre and the log determinant of the
    /// Cholesky factor, read by the one normal or Student t density.
    pub fn log_density(&self, unbounded: &[f64]) -> f64 {
        let dimensions = self.center.len();
        let squared: f64 = self.standardized(unbounded).iter().map(|z| z * z).sum();
        let log_scale: f64 = self.factor.diagonal().iter().map(|&l| libm::log(l)).sum();
        match self.degrees_of_freedom {
            None => normal_log_density(squared, dimensions, log_scale),
            Some(nu) => student_t_log_density(squared, nu, dimensions, log_scale),
        }
    }

    /// One draw on the unbounded scale: `c + L z`, with `z` standard normal,
    /// divided by `√(w/ν)` with `w ~ χ²_ν` for a Student t.
    pub fn draw_unbounded(&self, generator: &mut Generator) -> Vec<f64> {
        let normals = DVector::from_fn(self.center.len(), |_, _| normal(generator));
        let mut step = &self.factor * normals;
        if let Some(nu) = self.degrees_of_freedom {
            let chi_square = gamma(generator, 0.5 * nu, 0.5);
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
        let mut generator = Generator::from_seed(Stream::LocalDraws.seed(seed));
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
/// As [`Curvature::inverse`], [`super::InverseCurvature::covariance`] and
/// [`Gaussian::new`].
pub fn laplace(curvature: &Curvature) -> Result<Gaussian, MethodError> {
    Gaussian::new(
        curvature.transform().clone(),
        curvature.unbounded_fit().to_vec(),
        curvature.inverse()?.covariance(1.0)?,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian(degrees_of_freedom: Option<f64>) -> Gaussian {
        let transform = Transform::new(&[(f64::NEG_INFINITY, f64::INFINITY)]);
        Gaussian::new(
            transform,
            vec![1.0],
            Matrix::from_element(1, 1, 4.0),
            degrees_of_freedom,
        )
        .unwrap()
    }

    #[test]
    fn densities_have_their_closed_forms() {
        // At u = 3, one standard deviation of 2 from the centre 1.
        let normal = -0.5 * (2.0 * std::f64::consts::PI).ln() - 2f64.ln() - 0.5;
        assert!((gaussian(None).log_density(&[3.0]) - normal).abs() < 1e-12);
        // A t of 3 degrees of freedom and scale 2: Γ(2)/(Γ(3/2)√(3π)·2)·(1 + 1/3)⁻².
        let t = (1.0 / (libm::tgamma(1.5) * (3.0 * std::f64::consts::PI).sqrt() * 2.0)
            * (1.0f64 + 1.0 / 3.0).powi(-2))
        .ln();
        assert!((gaussian(Some(3.0)).log_density(&[3.0]) - t).abs() < 1e-12);
    }

    #[test]
    fn a_covariance_that_is_not_finite_is_refused() {
        // A Cholesky factor of a diagonal of infinities exists, and draws from
        // it are not numbers.
        let transform = Transform::new(&[(f64::NEG_INFINITY, f64::INFINITY); 2]);
        for entry in [f64::INFINITY, f64::NAN] {
            let covariance = Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, entry]);
            let error = Gaussian::new(transform.clone(), vec![0.0; 2], covariance, None);
            assert!(
                matches!(&error, Err(MethodError::Invalid(message))
                    if message.starts_with("The covariance is not positive definite")),
                "{entry}: {error:?}"
            );
        }
    }
}
