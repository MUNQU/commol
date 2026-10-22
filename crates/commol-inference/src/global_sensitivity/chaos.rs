//! Polynomial chaos expansions and the Sobol indices of their coefficients
//! (Sudret 2008).
//!
//! Every input is taken to `u = F(x)` through the distribution function of
//! its distribution; for any continuous distribution `u` is uniform on
//! `[0, 1]`, so one basis serves every input: orthonormal Legendre
//! polynomials of `u`, every product `Π P_{α_i}(u_i)` that a
//! [`Truncation`] keeps. The total-degree basis keeps `Σ α_i ≤ degree`; a
//! hyperbolic (q-norm) truncation keeps `Σ α_i^q ≤ degree^q`, and a largest
//! interaction order keeps the terms of at most that many inputs, so that a
//! basis of many inputs keeps the terms of low interaction order and stays
//! small. Polynomials matched to `x`, such
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

use super::check_numbers;
use crate::error::MethodError;

/// The terms a basis of `degree` keeps: those whose degrees `α` have
/// `Σ α_i^q ≤ degree^q` and hold at most `max_interaction` inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Truncation {
    q_norm: f64,
    max_interaction: Option<usize>,
}

impl Truncation {
    /// The total-degree basis: `q = 1` and no limit on the inputs of a term.
    pub const TOTAL_DEGREE: Self = Self {
        q_norm: 1.0,
        max_interaction: None,
    };

    /// The truncation of q-norm `q_norm` whose terms hold at most
    /// `max_interaction` inputs; `None` sets no limit.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when `q_norm` is not in `(0, 1]` or
    /// `max_interaction` is 0.
    pub fn new(q_norm: f64, max_interaction: Option<usize>) -> Result<Self, MethodError> {
        if !(q_norm > 0.0 && q_norm <= 1.0) {
            return Err(MethodError::Invalid(format!(
                "The q-norm of a polynomial chaos basis must be in (0, 1], got {q_norm}"
            )));
        }
        if max_interaction == Some(0) {
            return Err(MethodError::Invalid(
                "The largest interaction order of a polynomial chaos basis must be at least 1, \
                 got 0"
                    .to_string(),
            ));
        }
        Ok(Self {
            q_norm,
            max_interaction,
        })
    }

    /// The q of the q-norm, in `(0, 1]`.
    pub fn q_norm(&self) -> f64 {
        self.q_norm
    }

    /// The largest number of inputs of a term; `None` for no limit.
    pub fn max_interaction(&self) -> Option<usize> {
        self.max_interaction
    }

    /// `α^q`.
    fn power(&self, degree: usize) -> f64 {
        (degree as f64).powf(self.q_norm)
    }

    /// The bound of `Σ α_i^q` for a basis of `degree`, with room for the
    /// rounding of the powers.
    fn bound(&self, degree: usize) -> f64 {
        self.power(degree) * (1.0 + 1e-12)
    }

    /// Whether a term of `used` inputs whose degrees give `Σ α_i^q = sum`
    /// is kept under `bound`.
    fn keeps(&self, used: usize, sum: f64, bound: f64) -> bool {
        self.max_interaction.is_none_or(|limit| used <= limit) && sum <= bound
    }
}

/// A polynomial chaos expansion of each quantity, on the scale `u = F(x)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChaosExpansion {
    /// The degree of the basis: the bound of the q-norm of every term, the
    /// total degree when the q-norm is 1.
    pub degree: usize,
    /// The terms the basis keeps.
    pub truncation: Truncation,
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
    /// The expansion of `degree` in `inputs` inputs, truncated by
    /// `truncation`, with the coefficients of each quantity,
    /// `[quantity][term]`, on the terms of [`Self::terms`].
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when there is no input, or a row of
    /// coefficients has not one value per term.
    pub fn new(
        inputs: usize,
        degree: usize,
        truncation: Truncation,
        coefficients: Vec<Vec<f64>>,
    ) -> Result<Self, MethodError> {
        if inputs == 0 {
            return Err(MethodError::Invalid(
                "Polynomial chaos needs at least one input".to_string(),
            ));
        }
        let indices = basis(inputs, degree, truncation);
        if let Some(row) = coefficients.iter().find(|row| row.len() != indices.len()) {
            return Err(MethodError::Invalid(format!(
                "An expansion of degree {degree} in {inputs} inputs has {} terms; a row of \
                 coefficients has {}",
                indices.len(),
                row.len()
            )));
        }
        Ok(Self {
            degree,
            truncation,
            loo_error: vec![f64::NAN; coefficients.len()],
            indices,
            coefficients,
        })
    }

    /// The number of terms of a basis of `degree` in `inputs` inputs,
    /// truncated by `truncation`, with no basis built: the sum over the
    /// interaction orders `r` of `C(inputs, r)` times the number of
    /// ordered tuples of `r` positive degrees the truncation keeps. The
    /// total-degree basis has `(inputs + degree)! / (inputs! degree!)`
    /// terms. A count beyond `usize` saturates.
    pub fn terms(inputs: usize, degree: usize, truncation: Truncation) -> usize {
        let largest = truncation
            .max_interaction
            .map_or(degree, |limit| limit.min(degree))
            .min(inputs);
        let mut tuples = Tuples {
            truncation,
            bound: truncation.bound(degree),
            counts: vec![0; largest + 1],
        };
        tuples.count(degree, 0, 0.0, 1, 0);
        let total = tuples
            .counts
            .iter()
            .enumerate()
            .fold(0u128, |total, (order, &count)| {
                total.saturating_add(binomial(inputs, order).saturating_mul(count))
            });
        usize::try_from(total).unwrap_or(usize::MAX)
    }

    /// Fit an expansion of `degree`, truncated by `truncation`, to the
    /// quantities of each point, `outputs` (`[point][quantity]`), at `unit`
    /// (`[point][input]`, each in `[0, 1]`).
    ///
    /// # Errors
    ///
    /// [`MethodError::Prediction`] of [`PredictiveError::NotANumber`](crate::predictive::PredictiveError::NotANumber) with the first quantity that is NaN at
    /// a point, before any fit; [`MethodError::Invalid`] when the points
    /// and the outputs differ in number, there is no input, there are not
    /// more points than terms, or the polynomials of the points are
    /// collinear.
    pub fn fit(
        unit: &[Vec<f64>],
        outputs: &[Vec<f64>],
        degree: usize,
        truncation: Truncation,
    ) -> Result<Self, MethodError> {
        check_numbers(outputs)?;
        Self::least_squares(unit, outputs, degree, truncation).map_err(MethodError::Invalid)
    }

    fn least_squares(
        unit: &[Vec<f64>],
        outputs: &[Vec<f64>],
        degree: usize,
        truncation: Truncation,
    ) -> Result<Self, String> {
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
        let indices = basis(inputs, degree, truncation);
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
            truncation,
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

/// Every multi-index of `inputs` degrees that `truncation` keeps in a basis
/// of `degree`, by total degree; the constant first.
fn basis(inputs: usize, degree: usize, truncation: Truncation) -> Vec<Vec<usize>> {
    let bound = truncation.bound(degree);
    let mut indices = Vec::new();
    for total in 0..=degree {
        let mut index = vec![0; inputs];
        let mut kept = Kept {
            truncation,
            bound,
            out: &mut indices,
        };
        kept.fill(&mut index, 0, total, 0, 0.0);
    }
    indices
}

/// The multi-indices a truncation keeps, as they are found.
struct Kept<'a> {
    truncation: Truncation,
    bound: f64,
    out: &'a mut Vec<Vec<usize>>,
}

impl Kept<'_> {
    /// Every way to give `left` degrees to the inputs from `position` on,
    /// after `used` inputs whose degrees give `Σ α_i^q = sum`. The degrees
    /// left need one input more at least and add `left^q` at least, as
    /// `x^q` is concave, so a branch that cannot keep that stops.
    fn fill(
        &mut self,
        index: &mut Vec<usize>,
        position: usize,
        left: usize,
        used: usize,
        sum: f64,
    ) {
        if left > 0
            && !self
                .truncation
                .keeps(used + 1, sum + self.truncation.power(left), self.bound)
        {
            return;
        }
        if position == index.len() - 1 {
            index[position] = left;
            self.out.push(index.clone());
            return;
        }
        for here in (0..=left).rev() {
            index[position] = here;
            let (used, sum) = if here == 0 {
                (used, sum)
            } else {
                (used + 1, sum + self.truncation.power(here))
            };
            self.fill(index, position + 1, left - here, used, sum);
        }
    }
}

/// The number of ordered tuples of `r` positive degrees a truncation keeps,
/// `counts[r]`, for every `r` below the length of `counts`.
struct Tuples {
    truncation: Truncation,
    bound: f64,
    counts: Vec<u128>,
}

impl Tuples {
    /// Count every tuple that continues a tuple of `length` degrees whose
    /// powers sum to `sum`, the last `run` of which equal `last`;
    /// `orderings` is the number of distinct orders of those degrees. Each
    /// tuple is found once, in non-increasing order, and counts its
    /// orderings.
    fn count(&mut self, last: usize, length: usize, sum: f64, orderings: u128, run: usize) {
        self.counts[length] = self.counts[length].saturating_add(orderings);
        if length + 1 == self.counts.len() {
            return;
        }
        for part in 1..=last {
            let next = sum + self.truncation.power(part);
            if next > self.bound {
                break;
            }
            let run = if part == last { run + 1 } else { 1 };
            // A multinomial with one degree more: n! / Π m_v! times n / m_v.
            let orderings = orderings.saturating_mul((length + 1) as u128) / run as u128;
            self.count(part, length + 1, next, orderings, run);
        }
    }
}

/// `C(n, r)`, saturated at `u128::MAX`.
fn binomial(n: usize, r: usize) -> u128 {
    (0..r).fold(1u128, |c, i| {
        c.saturating_mul((n - i) as u128) / (i + 1) as u128
    })
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
        let indices = basis(2, 2, Truncation::TOTAL_DEGREE);
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
        assert_eq!(basis(3, 10, Truncation::TOTAL_DEGREE).len(), 286);
        assert_eq!(ChaosExpansion::terms(3, 10, Truncation::TOTAL_DEGREE), 286);
        assert_eq!(
            ChaosExpansion::terms(20, 4, Truncation::TOTAL_DEGREE),
            10_626
        );
    }

    #[test]
    fn a_q_norm_of_one_and_no_interaction_limit_is_the_total_degree_basis() {
        let truncation = Truncation::new(1.0, None).unwrap();
        assert_eq!(truncation, Truncation::TOTAL_DEGREE);
        for (inputs, degree) in [(1, 5), (2, 3), (4, 4), (6, 2)] {
            let total = basis(inputs, degree, truncation);
            assert_eq!(
                total.len(),
                ChaosExpansion::terms(inputs, degree, truncation)
            );
            // Every index of total degree at most the degree, once, by total.
            assert!(
                total
                    .iter()
                    .all(|index| index.iter().sum::<usize>() <= degree)
            );
            assert!(
                total
                    .windows(2)
                    .all(|pair| pair[0].iter().sum::<usize>() <= pair[1].iter().sum::<usize>())
            );
            let mut unique = total.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), total.len());
        }
        // An interaction limit at the degree or above removes no term.
        assert_eq!(
            basis(5, 3, Truncation::new(1.0, Some(3)).unwrap()),
            basis(5, 3, Truncation::TOTAL_DEGREE)
        );
    }

    #[test]
    fn an_interaction_limit_keeps_the_terms_of_at_most_that_many_inputs() {
        let truncation = Truncation::new(1.0, Some(2)).unwrap();
        let indices = basis(20, 4, truncation);
        // 1 + 20 · 4 + C(20, 2) · 6.
        assert_eq!(indices.len(), 1_221);
        assert_eq!(ChaosExpansion::terms(20, 4, truncation), 1_221);
        let used = |index: &Vec<usize>| index.iter().filter(|&&a| a > 0).count();
        assert!(indices.iter().all(|index| used(index) <= 2));
        // Every term of the total-degree basis of at most two inputs is kept.
        let kept = basis(20, 4, Truncation::TOTAL_DEGREE)
            .into_iter()
            .filter(|index| used(index) <= 2)
            .collect::<Vec<_>>();
        assert_eq!(indices, kept);
        assert_eq!(
            basis(3, 3, Truncation::new(1.0, Some(1)).unwrap()),
            vec![
                vec![0, 0, 0],
                vec![1, 0, 0],
                vec![0, 1, 0],
                vec![0, 0, 1],
                vec![2, 0, 0],
                vec![0, 2, 0],
                vec![0, 0, 2],
                vec![3, 0, 0],
                vec![0, 3, 0],
                vec![0, 0, 3]
            ]
        );
    }

    #[test]
    fn a_q_norm_below_one_keeps_the_terms_of_low_interaction() {
        // At degree 4 and q = 0.5, Σ √α ≤ 2: a single input to degree 4, or
        // two inputs at degree 1 each.
        let truncation = Truncation::new(0.5, None).unwrap();
        assert_eq!(
            basis(2, 4, truncation),
            vec![
                vec![0, 0],
                vec![1, 0],
                vec![0, 1],
                vec![2, 0],
                vec![1, 1],
                vec![0, 2],
                vec![3, 0],
                vec![0, 3],
                vec![4, 0],
                vec![0, 4]
            ]
        );
        for (inputs, degree, q) in [(5, 4, 0.5), (8, 6, 0.75), (3, 10, 0.4), (10, 5, 0.6)] {
            let truncation = Truncation::new(q, None).unwrap();
            let indices = basis(inputs, degree, truncation);
            assert_eq!(
                indices.len(),
                ChaosExpansion::terms(inputs, degree, truncation)
            );
            let bound = (degree as f64).powf(q) + 1e-9;
            assert!(
                indices.iter().all(|index| {
                    index.iter().map(|&a| (a as f64).powf(q)).sum::<f64>() <= bound
                })
            );
            // Fewer terms than the total-degree basis.
            assert!(
                indices.len() < ChaosExpansion::terms(inputs, degree, Truncation::TOTAL_DEGREE)
            );
            let both = Truncation::new(q, Some(2)).unwrap();
            assert_eq!(
                basis(inputs, degree, both).len(),
                ChaosExpansion::terms(inputs, degree, both)
            );
        }
    }

    #[test]
    fn a_count_too_large_for_the_machine_saturates() {
        assert_eq!(
            ChaosExpansion::terms(usize::MAX, 3, Truncation::TOTAL_DEGREE),
            usize::MAX
        );
    }

    #[test]
    fn a_truncation_needs_a_q_norm_in_the_unit_interval_and_one_input_at_least() {
        for q in [0.0, -0.5, 1.5, f64::NAN] {
            assert!(
                Truncation::new(q, None)
                    .unwrap_err()
                    .to_string()
                    .contains("must be in (0, 1]")
            );
        }
        assert!(
            Truncation::new(0.5, Some(0))
                .unwrap_err()
                .to_string()
                .contains("at least 1")
        );
        let truncation = Truncation::new(0.5, Some(3)).unwrap();
        assert_eq!(truncation.q_norm(), 0.5);
        assert_eq!(truncation.max_interaction(), Some(3));
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

        let chaos = ChaosExpansion::fit(&unit, &outputs, 2, Truncation::TOTAL_DEGREE).unwrap();

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
            ChaosExpansion::fit(&unit[..2], &outputs[..2], 2, Truncation::TOTAL_DEGREE)
                .unwrap_err()
                .to_string()
                .contains("more points")
        );
        assert!(
            ChaosExpansion::fit(&unit, &outputs, 2, Truncation::TOTAL_DEGREE)
                .unwrap_err()
                .to_string()
                .contains("collinear")
        );
        assert!(ChaosExpansion::fit(&unit, &outputs[..3], 2, Truncation::TOTAL_DEGREE).is_err());
        assert!(
            ChaosExpansion::fit(&vec![vec![]; 4], &outputs[..4], 1, Truncation::TOTAL_DEGREE)
                .is_err()
        );
    }

    #[test]
    fn an_expansion_is_rebuilt_from_its_degree_and_coefficients() {
        let chaos =
            ChaosExpansion::new(2, 1, Truncation::TOTAL_DEGREE, vec![vec![1.0, 2.0, 0.0]]).unwrap();
        assert_eq!(chaos.indices, vec![vec![0, 0], vec![1, 0], vec![0, 1]]);
        let p1 = 3f64.sqrt() * (2.0 * 0.75 - 1.0);
        assert!((chaos.evaluate(&[0.75, 0.1])[0] - (1.0 + 2.0 * p1)).abs() < 1e-12);
        assert!(
            ChaosExpansion::new(2, 1, Truncation::TOTAL_DEGREE, vec![vec![1.0, 2.0]])
                .unwrap_err()
                .to_string()
                .contains("3 terms")
        );
        assert!(ChaosExpansion::new(0, 1, Truncation::TOTAL_DEGREE, vec![]).is_err());
        // The terms of a truncated basis come from its truncation.
        let single = Truncation::new(1.0, Some(1)).unwrap();
        let chaos = ChaosExpansion::new(2, 2, single, vec![vec![1.0, 0.0, 0.0, 0.0, 0.0]]).unwrap();
        assert_eq!(chaos.indices.len(), 5);
        assert_eq!(chaos.truncation, single);
    }

    #[test]
    fn a_quantity_of_no_variance_has_no_indices() {
        let unit: Vec<Vec<f64>> = (0..6).map(|i| vec![(i as f64 + 0.5) / 6.0]).collect();
        let chaos =
            ChaosExpansion::fit(&unit, &vec![vec![2.0]; 6], 1, Truncation::TOTAL_DEGREE).unwrap();
        assert!(chaos.first()[0][0].is_nan());
        assert!(chaos.loo_error[0].is_nan());
    }
}
