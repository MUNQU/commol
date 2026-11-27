//! A normal or Student-t distribution on the unbounded scale, and the
//! Laplace approximation `N(û, H_u⁻¹)` of a curvature.
//!
//! A bound less than [`NEAR_BOUND`] sd from the centre, on the scale of the
//! parameters, is noted ([`Gaussian::near_bounds`]): the distribution mapped
//! back is then far from a normal there.
//!
//! A Student t has degrees of freedom `ν_j` for each parameter: the grouped t
//! of Daul, De Giorgi, Lindskog and McNeil (2003), whose margin of parameter
//! `j` is a Student t of `ν_j`, with one χ² level common to all. With equal
//! `ν_j` it is the multivariate t.

use commol_random::Generator;
use commol_random::sample::{gamma, normal};
use commol_statistics::numeric::{normal_log_density, student_t_log_density};
use commol_statistics::special::{
    chi_square_quantile, gamma_p, normal_quantile_sf, student_t_two_sided_tail,
};
use nalgebra::DVector;

use super::curvature::Curvature;
use crate::error::MethodError;
use crate::sensitivity::Matrix;
use crate::transform::{Scale, Transform};
use commol_random::Stream;

/// A bound less than this many sd from the centre, on the scale of the
/// parameters, is noted ([`Gaussian::near_bounds`]): a normal puts 0.135 % of
/// its mass past a bound 3 sd from its centre.
pub const NEAR_BOUND: f64 = 3.0;

/// A normal distribution on the unbounded scale, or a grouped Student t with
/// `degrees_of_freedom` for each parameter, whose draws map back to the
/// parameters.
#[derive(Debug, Clone)]
pub struct Gaussian {
    transform: Transform,
    center: Vec<f64>,
    covariance: Matrix,
    factor: Matrix,
    degrees_of_freedom: Option<Vec<f64>>,
}

impl Gaussian {
    /// The distribution of centre `center` and covariance `covariance` on
    /// the unbounded scale of `transform`; a grouped Student t with
    /// `degrees_of_freedom`, one per parameter, or a normal with `None`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when the covariance is not positive definite,
    /// or an entry is not finite.
    pub fn new(
        transform: Transform,
        center: Vec<f64>,
        covariance: Matrix,
        degrees_of_freedom: Option<Vec<f64>>,
    ) -> Result<Self, MethodError> {
        debug_assert!(
            degrees_of_freedom
                .as_ref()
                .is_none_or(|nu| nu.len() == center.len()),
            "one degree of freedom per parameter"
        );
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

    /// The degrees of freedom of each parameter of a Student t, or `None` for
    /// a normal.
    pub fn degrees_of_freedom(&self) -> Option<&[f64]> {
        self.degrees_of_freedom.as_deref()
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
    /// Cholesky factor, read by the one normal or Student t density. `None`
    /// for a grouped t of unequal degrees of freedom, whose density has no
    /// closed form.
    pub fn log_density(&self, unbounded: &[f64]) -> Option<f64> {
        let dimensions = self.center.len();
        let squared: f64 = self.standardized(unbounded).iter().map(|z| z * z).sum();
        let log_scale: f64 = self.factor.diagonal().iter().map(|&l| libm::log(l)).sum();
        match self.degrees_of_freedom.as_deref() {
            None => Some(normal_log_density(squared, dimensions, log_scale)),
            Some([nu, rest @ ..]) if rest.iter().all(|other| other == nu) => {
                Some(student_t_log_density(squared, *nu, dimensions, log_scale))
            }
            Some(_) => None,
        }
    }

    /// One draw on the unbounded scale: `c + L z`, with `z` standard normal,
    /// and for a Student t each entry `j` divided by `√(w_j/ν_j)`. The level
    /// `w_1 ~ χ²_{ν_1}` is drawn, and `w_j` is the quantile of `χ²_{ν_j}` at
    /// its probability, so `w_j = w_1` where `ν_j = ν_1`.
    pub fn draw_unbounded(&self, generator: &mut Generator) -> Vec<f64> {
        let normals = DVector::from_fn(self.center.len(), |_, _| normal(generator));
        let mut step = &self.factor * normals;
        if let Some(degrees) = &self.degrees_of_freedom {
            let first = degrees[0];
            let chi_square = gamma(generator, 0.5 * first, 0.5);
            let mut level = None;
            for (entry, &nu) in step.iter_mut().zip(degrees) {
                let own = if nu == first {
                    chi_square
                } else {
                    let level =
                        *level.get_or_insert_with(|| gamma_p(0.5 * first, 0.5 * chi_square));
                    chi_square_quantile(nu, level)
                };
                *entry *= (nu / own).sqrt();
            }
        }
        self.center
            .iter()
            .zip(step.iter())
            .map(|(c, s)| c + s)
            .collect()
    }

    /// Each bound less than [`NEAR_BOUND`] sd from the centre, on the scale
    /// of the parameters, the lower bound of a parameter first.
    ///
    /// The scale of parameter `j` there is `(dθ/du)_j · √Σ_jj`, with the
    /// covariance `Σ` on the unbounded scale. With `σ = √Σ_jj` and the centre
    /// `u`, the distance `d` in that scale is `(1 + eᵘ)/σ` to the lower bound
    /// of a logit and `(1 + e⁻ᵘ)/σ` to its upper bound; to the bound of a log,
    /// `1/σ`. None reads `dθ/du`, which a fit near a bound rounds. For a
    /// normal the distance is `d`. For a Student t of `ν_j` degrees of freedom
    /// it is the distance of a normal with the same mass past the bound,
    /// `z = Φ⁻¹(1 − P(T_ν > d))`, at most `d`, so that the rule reads the
    /// mass past the bound whatever the tails.
    pub fn near_bounds(&self) -> Vec<NearBound> {
        let mut near = Vec::new();
        for (j, scale) in self.transform.scales().iter().enumerate() {
            let sd = self.covariance[(j, j)].sqrt();
            let u = self.center[j];
            let distances = match *scale {
                Scale::Logit { lower, upper } => vec![
                    (lower, (1.0 + libm::exp(u)) / sd),
                    (upper, (1.0 + libm::exp(-u)) / sd),
                ],
                Scale::Log { lower } => vec![(lower, 1.0 / sd)],
                Scale::LogFromAbove { upper } => vec![(upper, 1.0 / sd)],
                Scale::Identity => Vec::new(),
            };
            // An infinite `ν` is a normal.
            let degrees_of_freedom = self
                .degrees_of_freedom
                .as_ref()
                .map(|degrees| degrees[j])
                .filter(|nu| nu.is_finite());
            for (bound, scaled) in distances {
                let distance = match degrees_of_freedom {
                    Some(nu) if (scaled * scaled).is_finite() => {
                        let tail = 0.5 * student_t_two_sided_tail(scaled * scaled, nu);
                        (normal_quantile_sf(tail) + 0.0).max(0.0)
                    }
                    _ => scaled,
                };
                if distance < NEAR_BOUND {
                    near.push(NearBound {
                        parameter: j,
                        bound,
                        distance,
                        degrees_of_freedom,
                    });
                }
            }
        }
        near
    }

    /// [`Self::near_bounds`] in words, with `names` the names of the
    /// parameters: the draws that a bound skews, and the profile likelihood
    /// that describes such a parameter.
    pub fn bound_notes(&self, names: &[String]) -> Vec<String> {
        self.near_bounds()
            .iter()
            .map(|near| {
                format!(
                    "{}: the draws, mapped back from the unbounded scale, are skewed by the \
                     bound and can spread far from the fit. A profile likelihood describes the \
                     uncertainty of a parameter near its bound: \
                     Calibrator.uncertainty(ProfileLikelihood(...))",
                    near.head(names)
                )
            })
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

/// A bound less than [`NEAR_BOUND`] sd from the centre of a [`Gaussian`]
/// ([`Gaussian::near_bounds`]).
#[derive(Debug, Clone, PartialEq)]
pub struct NearBound {
    /// The index of the parameter.
    pub parameter: usize,
    /// The bound.
    pub bound: f64,
    /// The distance from the centre in sd, on the scale of the parameters:
    /// for a Student t, that of a normal with its mass past the bound.
    pub distance: f64,
    /// The degrees of freedom of a Student t, or `None` for a normal.
    pub degrees_of_freedom: Option<f64>,
}

impl NearBound {
    /// The words that every note of a bound near the centre starts with, with
    /// `names` the names of the parameters: the parameter, its distance in sd
    /// and the bound.
    pub fn head(&self, names: &[String]) -> String {
        let read = self.degrees_of_freedom.map_or_else(String::new, |nu| {
            format!(
                " (as a normal with the mass of its Student t of {nu:.1} degrees of freedom \
                 past the bound)"
            )
        });
        format!(
            "'{}' is {} sd from its bound {}{read}, within {NEAR_BOUND} sd",
            names[self.parameter],
            in_sd(self.distance),
            self.bound
        )
    }
}

/// A distance below [`NEAR_BOUND`] in sd: two decimals, or two digits below
/// 0.01, and never one that rounds up to [`NEAR_BOUND`].
fn in_sd(distance: f64) -> String {
    if distance < 0.01 {
        return format!("{distance:.1e}");
    }
    let shown = format!("{distance:.2}");
    if shown.parse::<f64>().is_ok_and(|shown| shown >= NEAR_BOUND) {
        format!("{:.2}", NEAR_BOUND - 0.01)
    } else {
        shown
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
    use commol_statistics::special::student_t_quantile;

    fn gaussian(degrees_of_freedom: Option<f64>) -> Gaussian {
        let transform = Transform::new(&[(f64::NEG_INFINITY, f64::INFINITY)]);
        Gaussian::new(
            transform,
            vec![1.0],
            Matrix::from_element(1, 1, 4.0),
            degrees_of_freedom.map(|nu| vec![nu]),
        )
        .unwrap()
    }

    #[test]
    fn densities_have_their_closed_forms() {
        // At u = 3, one standard deviation of 2 from the centre 1.
        let normal = -0.5 * (2.0 * std::f64::consts::PI).ln() - 2f64.ln() - 0.5;
        assert!((gaussian(None).log_density(&[3.0]).unwrap() - normal).abs() < 1e-12);
        // A t of 3 degrees of freedom and scale 2: Γ(2)/(Γ(3/2)√(3π)·2)·(1 + 1/3)⁻².
        let t = (1.0 / (libm::tgamma(1.5) * (3.0 * std::f64::consts::PI).sqrt() * 2.0)
            * (1.0f64 + 1.0 / 3.0).powi(-2))
        .ln();
        assert!((gaussian(Some(3.0)).log_density(&[3.0]).unwrap() - t).abs() < 1e-12);
    }

    /// A Student t on the real line in two dimensions, of covariance
    /// `[[4, 1], [1, 2]]` and `degrees_of_freedom`.
    fn pair(degrees_of_freedom: Vec<f64>) -> Gaussian {
        let transform = Transform::new(&[(f64::NEG_INFINITY, f64::INFINITY); 2]);
        let covariance = Matrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 2.0]);
        Gaussian::new(
            transform,
            vec![1.0, -1.0],
            covariance,
            Some(degrees_of_freedom),
        )
        .unwrap()
    }

    #[test]
    fn equal_degrees_of_freedom_draw_the_multivariate_t() {
        // Each draw is c + L z √(ν/w), with z two normals and w ~ χ²_ν next in
        // the stream.
        let gaussian = pair(vec![3.0, 3.0]);
        let mut generator = Generator::from_seed(Stream::LocalDraws.seed(5));
        let factor = gaussian.factor().clone();
        for draw in gaussian.draws(20, 5) {
            let normals = DVector::from_fn(2, |_, _| normal(&mut generator));
            let scale = (3.0 / gamma(&mut generator, 1.5, 0.5)).sqrt();
            let expected = &factor * normals * scale;
            assert_eq!(draw, vec![1.0 + expected[0], -1.0 + expected[1]]);
        }
        assert!(gaussian.log_density(&[0.0, 0.0]).is_some());
    }

    #[test]
    fn a_grouped_t_has_the_margin_of_the_degrees_of_freedom_of_each_parameter() {
        // The quartiles of t₁ and t₃₀ are 1 and 0.683; with 4000 draws the error
        // of a quartile is about 0.04 and 0.02 standard deviations.
        let gaussian = pair(vec![1.0, 30.0]);
        let draws = gaussian.draws(4000, 7);
        for (j, (sd, nu)) in [(2.0, 1.0), (2f64.sqrt(), 30.0)].into_iter().enumerate() {
            let mut standardized: Vec<f64> = draws
                .iter()
                .map(|draw| (draw[j] - gaussian.center()[j]) / sd)
                .collect();
            standardized.sort_by(f64::total_cmp);
            let quartile = student_t_quantile(0.75, nu);
            for (index, expected) in [(1000, -quartile), (3000, quartile)] {
                let value = standardized[index];
                assert!(
                    (value - expected).abs() < 0.2,
                    "{nu}: {value} against {expected}"
                );
            }
        }
        assert_eq!(gaussian.log_density(&[0.0, 0.0]), None);
    }

    #[test]
    fn a_grouped_t_is_continuous_where_its_degrees_of_freedom_are_equal() {
        let equal = pair(vec![3.0, 3.0]).draws(50, 11);
        let close = pair(vec![3.0, 3.0 + 1e-9]).draws(50, 11);
        for (equal, close) in equal.iter().zip(&close) {
            assert_eq!(equal[0], close[0]);
            assert!((equal[1] - close[1]).abs() < 1e-7 * (1.0 + equal[1].abs()));
        }
    }

    #[test]
    fn a_distance_in_sd_never_reads_as_the_threshold() {
        assert_eq!(in_sd(0.5), "0.50");
        assert_eq!(in_sd(2.994), "2.99");
        assert_eq!(in_sd(2.996), "2.99");
        assert_eq!(in_sd(1e-3), "1.0e-3");
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
