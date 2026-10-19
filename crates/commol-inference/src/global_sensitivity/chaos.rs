//! Polynomial chaos expansions and the Sobol indices of their coefficients
//! (Sudret 2008).
//!
//! Every input is taken to `u = F(x)` through the distribution function of
//! its distribution; for any continuous distribution `u` is uniform on
//! `[0, 1]`, so one basis serves every input: orthonormal Legendre
//! polynomials of `u` on a total-degree basis, every product
//! `Π P_{α_i}(u_i)` with `Σ α_i ≤ degree`. Polynomials matched to `x`, such
//! as Hermite for a normal, would not be orthogonal under the truncated
//! distributions of commol. The coefficients come from least
//! squares; the leave-one-out error, from the diagonal of the hat matrix,
//! says how well the expansion predicts a point it was not fitted to.
//!
//! Orthonormality makes the variance the sum of the squares of the
//! coefficients but the constant one, so the first-order index of an input
//! sums the squares of the terms in that input alone, and the total index
//! the squares of every term that holds it.

use nalgebra::DMatrix;

/// A polynomial chaos expansion of each quantity, on the scale `u = F(x)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChaosExpansion {
    /// The total degree of the basis.
    pub degree: usize,
    /// The degree of each input in each term, `[term][input]`; term 0 is the
    /// constant.
    pub indices: Vec<Vec<usize>>,
    /// The coefficient of each term for each quantity, `[quantity][term]`.
    pub coefficients: Vec<Vec<f64>>,
    /// The leave-one-out error of each quantity, relative to its variance
    /// over the design; NaN for a quantity of no variance.
    pub loo_error: Vec<f64>,
}

impl ChaosExpansion {
    /// The expansion of `degree` in `inputs` inputs with the coefficients of
    /// each quantity, `[quantity][term]`, on the terms of [`Self::terms`].
    ///
    /// # Errors
    ///
    /// Returns a message when there is no input, or a row of coefficients
    /// has not one value per term.
    pub fn new(inputs: usize, degree: usize, coefficients: Vec<Vec<f64>>) -> Result<Self, String> {
        if inputs == 0 {
            return Err("Polynomial chaos needs at least one input".to_string());
        }
        let indices = total_degree(inputs, degree);
        if let Some(row) = coefficients.iter().find(|row| row.len() != indices.len()) {
            return Err(format!(
                "An expansion of degree {degree} in {inputs} inputs has {} terms; a row of \
                 coefficients has {}",
                indices.len(),
                row.len()
            ));
        }
        Ok(Self {
            degree,
            loo_error: vec![f64::NAN; coefficients.len()],
            indices,
            coefficients,
        })
    }

    /// The number of terms of a total-degree basis of `degree` in `inputs`
    /// inputs: `(inputs + degree)! / (inputs! degree!)`.
    pub fn terms(inputs: usize, degree: usize) -> usize {
        (1..=degree).fold(1usize, |terms, d| terms * (inputs + d) / d)
    }

    /// Fit an expansion of `degree` to the quantities of each point,
    /// `outputs` (`[point][quantity]`), at `unit` (`[point][input]`, each in
    /// `[0, 1]`).
    ///
    /// # Errors
    ///
    /// Returns a message when the points and the outputs differ in number,
    /// there is no input, there are not more points than terms, or the
    /// polynomials of the points are collinear.
    pub fn fit(unit: &[Vec<f64>], outputs: &[Vec<f64>], degree: usize) -> Result<Self, String> {
        let points = unit.len();
        if outputs.len() != points {
            return Err(format!(
                "Polynomial chaos needs one row of quantities per point: {points} points, {} rows",
                outputs.len()
            ));
        }
        let inputs = unit.first().map_or(0, Vec::len);
        if inputs == 0 {
            return Err("Polynomial chaos needs at least one input".to_string());
        }
        let indices = total_degree(inputs, degree);
        let terms = indices.len();
        if points <= terms {
            return Err(format!(
                "Polynomial chaos of degree {degree} in {inputs} inputs has {terms} terms and \
                 needs more points than terms, got {points}"
            ));
        }
        let rows: Vec<Vec<f64>> = unit.iter().map(|u| terms_at(&indices, u, degree)).collect();
        let basis = DMatrix::from_fn(points, terms, |row, term| rows[row][term]);
        let qr = basis.clone().qr();
        let r = qr.r();
        let scale = r.diagonal().amax();
        if (0..terms).any(|i| r[(i, i)].abs() <= 1e-12 * scale) {
            return Err(
                "The polynomials of the points are collinear: polynomial chaos needs points that \
                 spread over every input"
                    .to_string(),
            );
        }
        let q = qr.q();
        let leverage: Vec<f64> = (0..points).map(|row| q.row(row).norm_squared()).collect();
        let quantities = outputs.first().map_or(0, Vec::len);
        let mut coefficients = Vec::with_capacity(quantities);
        let mut loo_error = Vec::with_capacity(quantities);
        for k in 0..quantities {
            let y = nalgebra::DVector::from_iterator(points, outputs.iter().map(|row| row[k]));
            if y.iter().all(|&v| v == y[0]) {
                // A constant is the constant term alone, exactly.
                let mut c = vec![0.0; terms];
                c[0] = y[0];
                coefficients.push(c);
                loo_error.push(f64::NAN);
                continue;
            }
            let c = r
                .solve_upper_triangular(&(q.transpose() * &y))
                .expect("a triangular factor of full rank");
            let fitted = &basis * &c;
            let mean = y.mean();
            let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / points as f64;
            let loo = (0..points)
                .map(|i| ((y[i] - fitted[i]) / (1.0 - leverage[i])).powi(2))
                .sum::<f64>()
                / points as f64;
            loo_error.push(loo / variance);
            coefficients.push(c.iter().copied().collect());
        }
        Ok(Self {
            degree,
            indices,
            coefficients,
            loo_error,
        })
    }

    /// The value of each quantity at a point on the scale `u = F(x)`, with
    /// one value per input.
    pub fn evaluate(&self, unit: &[f64]) -> Vec<f64> {
        let terms = terms_at(&self.indices, unit, self.degree);
        self.coefficients
            .iter()
            .map(|c| c.iter().zip(&terms).map(|(c, t)| c * t).sum())
            .collect()
    }

    /// The mean of each quantity: the constant coefficient.
    pub fn mean(&self) -> Vec<f64> {
        self.coefficients.iter().map(|c| c[0]).collect()
    }

    /// The variance of each quantity: the sum of the squares of every
    /// coefficient but the constant.
    pub fn variance(&self) -> Vec<f64> {
        self.coefficients
            .iter()
            .map(|c| c[1..].iter().map(|v| v * v).sum())
            .collect()
    }

    /// The first-order Sobol index of each input for each quantity,
    /// `[quantity][input]`; NaN for a quantity of no variance.
    pub fn first(&self) -> Vec<Vec<f64>> {
        self.shares(|index, input| {
            index[input] > 0 && index.iter().enumerate().all(|(j, &a)| j == input || a == 0)
        })
    }

    /// The total Sobol index of each input for each quantity,
    /// `[quantity][input]`; NaN for a quantity of no variance.
    pub fn total(&self) -> Vec<Vec<f64>> {
        self.shares(|index, input| index[input] > 0)
    }

    fn shares(&self, counts: impl Fn(&[usize], usize) -> bool) -> Vec<Vec<f64>> {
        let inputs = self.indices[0].len();
        self.coefficients
            .iter()
            .zip(self.variance())
            .map(|(c, variance)| {
                (0..inputs)
                    .map(|input| {
                        if variance <= 0.0 {
                            return f64::NAN;
                        }
                        self.indices
                            .iter()
                            .zip(c)
                            .filter(|(index, _)| counts(index, input))
                            .map(|(_, v)| v * v)
                            .sum::<f64>()
                            / variance
                    })
                    .collect()
            })
            .collect()
    }
}

/// Every multi-index of `inputs` degrees that sum to at most `degree`, by
/// total degree; the constant first.
fn total_degree(inputs: usize, degree: usize) -> Vec<Vec<usize>> {
    let mut indices = Vec::new();
    for total in 0..=degree {
        let mut index = vec![0; inputs];
        fill(&mut index, 0, total, &mut indices);
    }
    indices
}

/// Every way to give `left` degrees to the inputs from `position` on.
fn fill(index: &mut Vec<usize>, position: usize, left: usize, out: &mut Vec<Vec<usize>>) {
    if position == index.len() - 1 {
        index[position] = left;
        out.push(index.clone());
        return;
    }
    for here in (0..=left).rev() {
        index[position] = here;
        fill(index, position + 1, left - here, out);
    }
}

/// The value of each term `Π P_{α_i}(u_i)` at `unit`, with the orthonormal
/// Legendre polynomials of `[0, 1]`.
fn terms_at(indices: &[Vec<usize>], unit: &[f64], degree: usize) -> Vec<f64> {
    let tables: Vec<Vec<f64>> = unit.iter().map(|&u| legendre(u, degree)).collect();
    indices
        .iter()
        .map(|index| {
            index
                .iter()
                .zip(&tables)
                .map(|(&n, table)| table[n])
                .product()
        })
        .collect()
}

/// The orthonormal Legendre polynomials of `[0, 1]` at `u`, degrees 0 to
/// `degree`: `√(2n + 1) · P_n(2u − 1)`.
fn legendre(u: f64, degree: usize) -> Vec<f64> {
    let t = 2.0 * u - 1.0;
    let mut p = vec![1.0, t];
    for n in 1..degree {
        let n_f = n as f64;
        p.push(((2.0 * n_f + 1.0) * t * p[n] - n_f * p[n - 1]) / (n_f + 1.0));
    }
    p.truncate(degree + 1);
    p.iter()
        .enumerate()
        .map(|(n, v)| v * ((2 * n + 1) as f64).sqrt())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_basis_holds_every_index_of_total_degree_at_most_the_degree() {
        let indices = total_degree(2, 2);
        assert_eq!(
            indices,
            vec![
                vec![0, 0],
                vec![1, 0],
                vec![0, 1],
                vec![2, 0],
                vec![1, 1],
                vec![0, 2]
            ]
        );
        assert_eq!(total_degree(3, 10).len(), 286);
        assert_eq!(ChaosExpansion::terms(3, 10), 286);
        assert_eq!(ChaosExpansion::terms(20, 4), 10_626);
    }

    #[test]
    fn the_legendre_polynomials_are_orthonormal_on_the_unit_interval() {
        // The midpoint rule on many points is exact enough for degree 4.
        let n = 20_000;
        for a in 0..=4 {
            for b in 0..=4 {
                let integral: f64 = (0..n)
                    .map(|i| {
                        let u = (i as f64 + 0.5) / n as f64;
                        let p = legendre(u, 4);
                        p[a] * p[b]
                    })
                    .sum::<f64>()
                    / n as f64;
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!((integral - expected).abs() < 1e-6, "{a} {b}: {integral}");
            }
        }
    }

    #[test]
    fn a_polynomial_of_the_basis_is_fitted_exactly() {
        // y = 1 + 2 P1(u1) + 3 P1(u1) P1(u2), with P1 = √3 (2u − 1).
        let unit: Vec<Vec<f64>> = (0..30)
            .map(|i| vec![(i as f64 + 0.5) / 30.0, ((i * 7) % 30) as f64 / 30.0 + 0.01])
            .collect();
        let p1 = |u: f64| 3f64.sqrt() * (2.0 * u - 1.0);
        let outputs: Vec<Vec<f64>> = unit
            .iter()
            .map(|u| vec![1.0 + 2.0 * p1(u[0]) + 3.0 * p1(u[0]) * p1(u[1])])
            .collect();

        let chaos = ChaosExpansion::fit(&unit, &outputs, 2).unwrap();

        assert!((chaos.mean()[0] - 1.0).abs() < 1e-10);
        assert!((chaos.variance()[0] - 13.0).abs() < 1e-9);
        assert!((chaos.first()[0][0] - 4.0 / 13.0).abs() < 1e-10);
        assert!(chaos.first()[0][1].abs() < 1e-10);
        assert!((chaos.total()[0][1] - 9.0 / 13.0).abs() < 1e-10);
        assert!(chaos.loo_error[0] < 1e-20);
        assert!(
            (chaos.evaluate(&[0.3, 0.6])[0] - (1.0 + 2.0 * p1(0.3) + 3.0 * p1(0.3) * p1(0.6)))
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn a_fit_needs_more_points_than_terms_and_points_that_spread() {
        let unit = vec![vec![0.5]; 10];
        let outputs = vec![vec![1.0]; 10];
        assert!(
            ChaosExpansion::fit(&unit[..2], &outputs[..2], 2)
                .unwrap_err()
                .contains("more points")
        );
        assert!(
            ChaosExpansion::fit(&unit, &outputs, 2)
                .unwrap_err()
                .contains("collinear")
        );
        assert!(ChaosExpansion::fit(&unit, &outputs[..3], 2).is_err());
        assert!(ChaosExpansion::fit(&vec![vec![]; 4], &outputs[..4], 1).is_err());
    }

    #[test]
    fn an_expansion_is_rebuilt_from_its_degree_and_coefficients() {
        let chaos = ChaosExpansion::new(2, 1, vec![vec![1.0, 2.0, 0.0]]).unwrap();
        assert_eq!(chaos.indices, vec![vec![0, 0], vec![1, 0], vec![0, 1]]);
        let p1 = 3f64.sqrt() * (2.0 * 0.75 - 1.0);
        assert!((chaos.evaluate(&[0.75, 0.1])[0] - (1.0 + 2.0 * p1)).abs() < 1e-12);
        assert!(
            ChaosExpansion::new(2, 1, vec![vec![1.0, 2.0]])
                .unwrap_err()
                .contains("3 terms")
        );
        assert!(ChaosExpansion::new(0, 1, vec![]).is_err());
    }

    #[test]
    fn a_quantity_of_no_variance_has_no_indices() {
        let unit: Vec<Vec<f64>> = (0..6).map(|i| vec![(i as f64 + 0.5) / 6.0]).collect();
        let chaos = ChaosExpansion::fit(&unit, &vec![vec![2.0]; 6], 1).unwrap();
        assert!(chaos.first()[0][0].is_nan());
        assert!(chaos.loo_error[0].is_nan());
    }
}
