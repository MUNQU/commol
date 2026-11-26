//! The linear algebra of commol: the spectral radius of a square matrix,
//! which the model compiler gives each matrix between categories, and the
//! inverse of a symmetric matrix with its rule of what is singular, which the
//! local approximations and the collinearity index share.
//!
//! The eigenvalues of a square matrix come from the real Schur decomposition
//! of `nalgebra`, so they hold for any finite matrix: a complex pair as well
//! as two moduli close to each other. Those of a symmetric matrix come from
//! its symmetric decomposition.

use nalgebra::{DMatrix, DVector, Schur};

/// The sweeps of the QR algorithm per row of a matrix after which its
/// decomposition is given up.
pub const MAX_SWEEPS: usize = 100;

/// The spectral radius of the square matrix of `size` rows whose values,
/// row by row, are `rows`: the largest modulus of its eigenvalues, real or
/// complex. 0 for a matrix of no rows, and NaN when a value is not finite or
/// the decomposition does not converge within [`MAX_SWEEPS`] sweeps per row.
///
/// # Panics
///
/// Panics when `rows` does not hold `size × size` values.
pub fn spectral_radius(rows: &[f64], size: usize) -> f64 {
    assert_eq!(rows.len(), size * size, "a square matrix of {size} rows");
    if rows.iter().any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    if size == 0 {
        return 0.0;
    }
    let matrix = DMatrix::from_row_slice(size, size, rows);
    let Some(schur) = Schur::try_new(matrix, f64::EPSILON, MAX_SWEEPS * size) else {
        // coverage: the sweeps of a finite matrix converge long before the bound
        return f64::NAN;
    };
    schur
        .complex_eigenvalues()
        .iter()
        .map(|eigenvalue| libm::hypot(eigenvalue.re, eigenvalue.im))
        .fold(0.0, f64::max)
}

/// Whether a symmetric matrix with a diagonal of 1 is singular to working
/// precision, by its `eigenvalues`: one is NaN, or the smallest is at most
/// `p·ε` times the largest, with `p` the number of rows and `ε`
/// [`f64::EPSILON`]. The eigenvalues of a matrix of dependent rows are 0 but
/// for a rounding of this size.
pub fn singular(eigenvalues: &DVector<f64>) -> bool {
    let smallest = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let largest = eigenvalues.iter().copied().fold(0.0, f64::max);
    eigenvalues.iter().any(|value| value.is_nan())
        || smallest <= eigenvalues.len() as f64 * f64::EPSILON * largest
}

/// The inverse of a symmetric positive definite matrix `M`, exactly symmetric,
/// whatever the scales of its rows: `M⁻¹ = D E⁻¹ D` with `E = D M D` and
/// `D = diag(M)^(−1/2)`, so that the diagonal of `E` is 1. `None` when the
/// diagonal of `M` is not above 0, `E` is not finite or is [`singular`], or an
/// entry of the inverse is above the largest number.
///
/// `E⁻¹ = L⁻ᵀ L⁻¹` comes from the Cholesky factor `L` of `E`, which keeps the
/// digits of an entry far below the others; the eigenvectors of `E` do not.
pub fn symmetric_inverse(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let size = matrix.nrows();
    let roots: Vec<f64> = matrix.diagonal().iter().map(|entry| entry.sqrt()).collect();
    // A diagonal that is not above 0 makes its own entry of `E` not finite.
    let equilibrated = DMatrix::from_fn(size, size, |j, k| matrix[(j, k)] / roots[j] / roots[k]);
    if equilibrated.iter().any(|entry| !entry.is_finite())
        || singular(&equilibrated.symmetric_eigenvalues())
    {
        return None;
    }
    let factor = equilibrated.cholesky()?;
    let inverse_factor = factor
        .l()
        .solve_lower_triangular(&DMatrix::identity(size, size))?;
    let root = DMatrix::from_fn(size, size, |m, j| inverse_factor[(m, j)] / roots[j]);
    // The product of two columns is the same in either order.
    let inverse = DMatrix::from_fn(size, size, |j, k| root.column(j).dot(&root.column(k)));
    inverse
        .iter()
        .all(|entry| entry.is_finite())
        .then_some(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(value: f64, expected: f64) {
        assert!(
            (value - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{value} != {expected}"
        );
    }

    #[test]
    fn the_radius_of_a_symmetric_matrix_is_its_largest_eigenvalue() {
        close(
            spectral_radius(&[2.0, 1.0, 1.0, 3.0], 2),
            (5.0 + 5.0_f64.sqrt()) / 2.0,
        );
    }

    #[test]
    fn the_radius_reads_the_modulus_of_a_complex_pair() {
        // A rotation by a quarter turn, scaled by 2: eigenvalues ±2i.
        close(spectral_radius(&[0.0, -2.0, 2.0, 0.0], 2), 2.0);
        // Eigenvalues 1 and the pair -1 ± 2i of modulus √5.
        let rows = [1.0, 0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 2.0, -1.0];
        close(spectral_radius(&rows, 3), 5.0_f64.sqrt());
    }

    /// The symmetric matrix of `size` rows `D E D`, with `E` of the diagonal
    /// 1 and the other entries `correlation`, and the scales `D`.
    fn correlated(scales: &[f64], correlation: f64) -> DMatrix<f64> {
        let size = scales.len();
        DMatrix::from_fn(size, size, |j, k| {
            let entry = if j == k { 1.0 } else { correlation };
            scales[j] * entry * scales[k]
        })
    }

    #[test]
    fn the_inverse_of_equal_correlations_is_symmetric_and_has_its_closed_form() {
        // The inverse of a 3 × 3 matrix of equal correlations ρ has the
        // diagonal (1 + ρ)/((1 − ρ)(1 + 2ρ)) and −ρ/((1 − ρ)(1 + 2ρ)) off it.
        let rho = 0.25;
        let inverse = symmetric_inverse(&correlated(&[1.0; 3], rho)).unwrap();
        let determinant = (1.0 - rho) * (1.0 + 2.0 * rho);
        for j in 0..3 {
            for k in 0..3 {
                let expected = if j == k { 1.0 + rho } else { -rho } / determinant;
                close(inverse[(j, k)], expected);
                assert_eq!(inverse[(j, k)], inverse[(k, j)]);
            }
        }
    }

    #[test]
    fn the_inverse_does_not_depend_on_the_scales_of_the_rows() {
        // The entries of the matrix go from 1e-280 to 1e280.
        let scales = [1e-140, 1.0, 1e140, 1e-3];
        let expected = symmetric_inverse(&correlated(&[1.0; 4], 0.9)).unwrap();

        let inverse = symmetric_inverse(&correlated(&scales, 0.9)).unwrap();

        for j in 0..4 {
            for k in 0..4 {
                close(
                    inverse[(j, k)] * scales[j] * scales[k] / expected[(j, k)],
                    1.0,
                );
                assert_eq!(inverse[(j, k)], inverse[(k, j)]);
            }
        }
    }

    #[test]
    fn the_inverse_keeps_the_digits_of_a_weak_coupling() {
        // Two rows of correlation 1/2, and one coupled to them by a and b, far
        // below the rounding of the other entries: its entries of the inverse
        // are −(a − b/2)/(3/4) and −(b − a/2)/(3/4), wherever its row is.
        let (a, b) = (1e-30, 3e-200);
        for weak in 0..3 {
            let others: Vec<usize> = (0..3).filter(|&row| row != weak).collect();
            let mut matrix = DMatrix::identity(3, 3);
            for (j, k, entry) in [
                (others[0], others[1], 0.5),
                (others[0], weak, a),
                (others[1], weak, b),
            ] {
                matrix[(j, k)] = entry;
                matrix[(k, j)] = entry;
            }

            let inverse = symmetric_inverse(&matrix).unwrap();

            close(inverse[(others[0], weak)] / (-(a - 0.5 * b) / 0.75), 1.0);
            close(inverse[(others[1], weak)] / (-(b - 0.5 * a) / 0.75), 1.0);
            close(inverse[(weak, weak)], 1.0);
        }
    }

    #[test]
    fn a_matrix_singular_to_working_precision_has_no_inverse() {
        // Of two rows of correlation 1 − δ the eigenvalues are δ and 2 − δ,
        // so the limit is δ = 2ε · 2 but for its rounding, whatever the scales.
        for scales in [[1.0, 1.0], [1e-140, 1e150], [3.7, 1.0 / 7.0]] {
            for units in [0.0, 1.0, 2.0] {
                let correlation = 1.0 - units * f64::EPSILON;
                assert_eq!(
                    symmetric_inverse(&correlated(&scales, correlation)),
                    None,
                    "{units} units of {scales:?}"
                );
            }
            assert!(symmetric_inverse(&correlated(&scales, 1.0 - 8.0 * f64::EPSILON)).is_some());
        }
        // Three rows of which one is the sum of the others, with no exact
        // number in it.
        for a in [0.1, 0.3, 0.7, 1.9] {
            for b in [0.2, 0.6, 1.1] {
                let rows = DMatrix::from_row_slice(2, 3, &[a, b, a + b, 1.0, -a * b, 1.0 - a * b]);
                let gram = rows.transpose() * &rows;
                assert_eq!(symmetric_inverse(&gram), None, "{a} {b}");
            }
        }
    }

    #[test]
    fn a_matrix_that_is_not_positive_definite_has_no_inverse() {
        for diagonal in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, diagonal]);
            assert_eq!(symmetric_inverse(&matrix), None, "{diagonal}");
            let matrix = DMatrix::from_element(1, 1, diagonal);
            assert_eq!(symmetric_inverse(&matrix), None, "{diagonal}");
        }
        // The eigenvalues −1 and 3.
        assert_eq!(symmetric_inverse(&correlated(&[1.0; 2], 2.0)), None);
        let overflowing = DMatrix::from_row_slice(2, 2, &[1e-300, 1e200, 1e200, 1e-300]);
        assert_eq!(symmetric_inverse(&overflowing), None);
    }

    #[test]
    fn an_inverse_above_the_largest_number_is_none() {
        // The inverse of 1e-310 is 1e310.
        assert_eq!(
            symmetric_inverse(&DMatrix::from_element(1, 1, 1e-310)),
            None
        );
        let inverse = symmetric_inverse(&DMatrix::from_element(1, 1, 1e-300)).unwrap();
        close(inverse[(0, 0)] * 1e-300, 1.0);
    }

    #[test]
    fn eigenvalues_that_are_not_numbers_are_singular() {
        assert!(singular(&DVector::from_vec(vec![f64::NAN, f64::NAN])));
        assert!(singular(&DVector::from_vec(vec![1.0, f64::NAN])));
        assert!(!singular(&DVector::from_vec(vec![1.0, 0.5])));
    }

    #[test]
    fn the_radius_reads_a_negative_eigenvalue_by_its_modulus() {
        close(spectral_radius(&[-4.0, 0.0, 0.0, 1.0], 2), 4.0);
        close(spectral_radius(&[0.5], 1), 0.5);
        assert_eq!(spectral_radius(&[], 0), 0.0);
        assert!(spectral_radius(&[1.0, f64::NAN, 0.0, 1.0], 2).is_nan());
    }
}
