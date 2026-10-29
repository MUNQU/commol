//! The linear algebra that a model and a calibration share: the spectral
//! radius of a square matrix.
//!
//! The eigenvalues come from the real Schur decomposition of `nalgebra`, so
//! they hold for any finite matrix: a complex pair as well as two moduli
//! close to each other.

use nalgebra::{DMatrix, Schur};

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

    #[test]
    fn the_radius_reads_a_negative_eigenvalue_by_its_modulus() {
        close(spectral_radius(&[-4.0, 0.0, 0.0, 1.0], 2), 4.0);
        close(spectral_radius(&[0.5], 1), 0.5);
        assert_eq!(spectral_radius(&[], 0), 0.0);
        assert!(spectral_radius(&[1.0, f64::NAN, 0.0, 1.0], 2).is_nan());
    }
}
