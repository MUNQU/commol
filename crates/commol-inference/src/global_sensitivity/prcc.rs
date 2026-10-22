//! Partial rank correlation coefficients (Marino et al. 2008), weighted.
//!
//! Each input and each quantity is replaced by its rank; the PRCC of an
//! input is the correlation of its rank with the rank of the quantity once
//! the ranks of the other inputs are accounted for. It measures how
//! monotone the quantity is in the input, whatever the scale of either.
//!
//! The draws carry weights, as every sample of commol does. A rank is the
//! weighted mid-rank, the weight of the draws below a value plus half the
//! weight of the draws equal to it; the partial correlations come from the
//! weighted covariance of the ranks, through the Schur complement of the
//! other inputs; and the significance uses Kish's
//! effective size in place of the number of draws. Equal weights give the
//! unweighted PRCC and its usual test.
//!
//! A NaN has no rank, so PRCC refuses a NaN input or quantity before it
//! ranks any value.

use nalgebra::DMatrix;

use super::check_numbers;
use crate::error::MethodError;
use crate::special::beta_i;
use crate::statistics::effective_size;

/// The PRCC of every input for every quantity.
#[derive(Debug, Clone, PartialEq)]
pub struct Prcc {
    /// The coefficient of each input for each quantity, `[quantity][input]`,
    /// in `[−1, 1]`; NaN for a quantity with one value in every draw, or one
    /// the other inputs explain entirely.
    pub coefficients: Vec<Vec<f64>>,
    /// The two-sided p-value of each coefficient against no partial
    /// correlation, `[quantity][input]`.
    pub p_values: Vec<Vec<f64>>,
    /// Kish's effective size of the draws, the number of draws the test
    /// counts.
    pub effective_size: f64,
}

/// The PRCC of `inputs` (`[draw][input]`) for `outputs` (`[draw][quantity]`),
/// with the weight of each draw.
///
/// # Errors
///
/// [`MethodError::Prediction`] of [`PredictiveError::NotANumber`](crate::predictive::PredictiveError::NotANumber) with the first quantity that is NaN at a
/// draw, and every draw where it is, before any value is ranked;
/// [`MethodError::Invalid`] when the draws, the outputs and the weights
/// differ in number, there is no input, an input is NaN, a weight is
/// negative or every weight is zero, the effective size is not above one
/// more than the number of inputs, or the ranks of the inputs are collinear.
pub fn prcc(
    inputs: &[Vec<f64>],
    outputs: &[Vec<f64>],
    weights: &[f64],
) -> Result<Prcc, MethodError> {
    let draws = inputs.len();
    if outputs.len() != draws || weights.len() != draws {
        return Err(MethodError::Invalid(format!(
            "PRCC needs one row of outputs and one weight per draw: {draws} draws, {} rows of \
             outputs and {} weights",
            outputs.len(),
            weights.len()
        )));
    }
    check_numbers(outputs)?;
    let quantities = outputs.first().map_or(0, Vec::len);
    let invalid = |message: String| Err(MethodError::Invalid(message));
    let count = inputs.first().map_or(0, Vec::len);
    if count == 0 {
        return invalid("PRCC needs at least one input".to_string());
    }
    if let Some(draw) = inputs.iter().position(|row| row.iter().any(|v| v.is_nan())) {
        return invalid(format!(
            "PRCC ranks numbers: an input of draw {draw} is NaN"
        ));
    }
    let total: f64 = weights.iter().sum();
    if weights.iter().any(|w| w.is_nan() || *w < 0.0) || total.is_nan() || total <= 0.0 {
        return invalid("The weights of PRCC are not negative, and some are above 0".to_string());
    }
    let weights: Vec<f64> = weights.iter().map(|w| w / total).collect();
    let size = effective_size(&weights);
    let freedom = size - count as f64 - 1.0;
    if freedom.is_nan() || freedom <= 0.0 {
        return invalid(format!(
            "PRCC of {count} inputs needs an effective size above {}, got {size}",
            count + 1
        ));
    }
    let column = |rows: &[Vec<f64>], j: usize| rows.iter().map(|row| row[j]).collect::<Vec<_>>();
    let input_ranks: Vec<Vec<f64>> = (0..count)
        .map(|j| weighted_ranks(&column(inputs, j), &weights))
        .collect();
    // The ranks of the inputs are collinear, to the precision of the
    // arithmetic, when the smallest eigenvalue of their covariance is below
    // 1e-12 of the largest: the relative test of polynomial chaos.
    let eigenvalues = covariance(&input_ranks, &weights).symmetric_eigenvalues();
    if eigenvalues.min() <= 1e-12 * eigenvalues.max() {
        return invalid(
            "The ranks of the inputs are collinear: PRCC needs inputs that vary on their own"
                .to_string(),
        );
    }
    let mut coefficients = Vec::with_capacity(quantities);
    let mut p_values = Vec::with_capacity(quantities);
    for q in 0..quantities {
        let values = column(outputs, q);
        if values.iter().all(|&v| v == values[0]) {
            coefficients.push(vec![f64::NAN; count]);
            p_values.push(vec![f64::NAN; count]);
            continue;
        }
        let mut ranks = input_ranks.clone();
        ranks.push(weighted_ranks(&values, &weights));
        let full = covariance(&ranks, &weights);
        let row: Vec<f64> = (0..count)
            .map(|j| partial_correlation(&full, j, count))
            .collect();
        p_values.push(row.iter().map(|&r| p_value(r, freedom)).collect());
        coefficients.push(row);
    }
    Ok(Prcc {
        coefficients,
        p_values,
        effective_size: size,
    })
}

/// The partial correlation of columns `a` and `b` of the covariance
/// `full`, given every other column: the covariance of the two once the
/// others are regressed out (the Schur complement), over the product of
/// their standard deviations once the others are regressed out. It needs
/// only the covariance of the others to be invertible, so a quantity whose
/// rank is the rank of one input has the coefficient 1 for it; NaN when
/// nothing of `b` is left once the others are regressed out.
fn partial_correlation(full: &DMatrix<f64>, a: usize, b: usize) -> f64 {
    let others: Vec<usize> = (0..full.nrows()).filter(|&i| i != a && i != b).collect();
    let pick = |rows: &[usize], cols: &[usize]| {
        DMatrix::from_fn(rows.len(), cols.len(), |i, j| full[(rows[i], cols[j])])
    };
    let pair = [a, b];
    let mut conditional = pick(&pair, &pair);
    if !others.is_empty() {
        let inverse = pick(&others, &others)
            .try_inverse()
            .expect("the ranks of the inputs are not collinear");
        let cross = pick(&pair, &others);
        conditional -= &cross * inverse * cross.transpose();
    }
    let (va, vb, cab) = (
        conditional[(0, 0)],
        conditional[(1, 1)],
        conditional[(0, 1)],
    );
    if vb <= 1e-12 * full[(b, b)] || va <= 0.0 {
        return f64::NAN;
    }
    (cab / (va * vb).sqrt()).clamp(-1.0, 1.0)
}

/// The weighted mid-rank of each value, in `[0, 1]`: the weight of the
/// values below it plus half the weight of the values equal to it. With
/// equal weights it is `(r − ½) / n` for the usual mid-rank `r`. No value is
/// NaN, as [`prcc()`] checks: a NaN is equal to no value, itself included.
fn weighted_ranks(values: &[f64], weights: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut below = 0.0;
    let mut start = 0;
    while start < order.len() {
        let mut end = start;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let tied: f64 = order[start..end].iter().map(|&i| weights[i]).sum();
        for &i in &order[start..end] {
            ranks[i] = below + tied / 2.0;
        }
        below += tied;
        start = end;
    }
    ranks
}

/// The weighted covariance matrix of the columns, for weights that sum
/// to 1; its scale does not change a correlation.
fn covariance(columns: &[Vec<f64>], weights: &[f64]) -> DMatrix<f64> {
    let means: Vec<f64> = columns
        .iter()
        .map(|column| column.iter().zip(weights).map(|(v, w)| v * w).sum())
        .collect();
    DMatrix::from_fn(columns.len(), columns.len(), |a, b| {
        columns[a]
            .iter()
            .zip(&columns[b])
            .zip(weights)
            .map(|((x, y), w)| w * (x - means[a]) * (y - means[b]))
            .sum()
    })
}

/// The two-sided p-value of a partial correlation `r` with `freedom`
/// degrees of freedom: Student's t with `t² = r² ν / (1 − r²)`.
fn p_value(r: f64, freedom: f64) -> f64 {
    let rest = 1.0 - r * r;
    if rest <= 0.0 {
        return 0.0;
    }
    let t_squared = r * r * freedom / rest;
    beta_i(freedom / 2.0, 0.5, freedom / (freedom + t_squared))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_weights_give_shifted_mid_ranks() {
        let ranks = weighted_ranks(&[3.0, 1.0, 3.0, 2.0], &[0.25; 4]);
        // Mid-ranks 3.5, 1, 3.5, 2, as (r − ½) / 4.
        assert_eq!(ranks, vec![0.75, 0.125, 0.75, 0.375]);
    }

    #[test]
    fn a_weight_is_the_mass_below_and_half_the_mass_at_a_value() {
        let ranks = weighted_ranks(&[1.0, 2.0, 3.0], &[0.5, 0.25, 0.25]);
        assert_eq!(ranks, vec![0.25, 0.625, 0.875]);
    }

    #[test]
    fn the_p_value_of_no_correlation_is_one_and_of_a_perfect_one_is_zero() {
        assert!((p_value(0.0, 10.0) - 1.0).abs() < 1e-12);
        assert_eq!(p_value(1.0, 10.0), 0.0);
        // t = 2.228 is the 97.5 % quantile of Student's t with 10 degrees of
        // freedom: r = t / √(t² + ν).
        let r = 2.228 / (2.228f64 * 2.228 + 10.0).sqrt();
        assert!((p_value(r, 10.0) - 0.05).abs() < 1e-3);
    }

    #[test]
    fn prcc_checks_its_arguments() {
        let inputs = vec![vec![0.0], vec![1.0], vec![2.0]];
        let outputs = vec![vec![0.0], vec![1.0], vec![2.0]];
        assert!(prcc(&inputs, &outputs[..2], &[1.0; 3]).is_err());
        assert!(prcc(&[vec![], vec![]], &outputs[..2], &[1.0; 2]).is_err());
        assert!(prcc(&inputs, &outputs, &[1.0, -1.0, 1.0]).is_err());
        assert!(prcc(&inputs, &outputs, &[0.0; 3]).is_err());
        let error = prcc(&inputs[..2], &outputs[..2], &[1.0; 2]).unwrap_err();
        assert!(
            error.to_string().contains("effective size above 2"),
            "{error}"
        );
        let nan_input = vec![vec![0.0], vec![f64::NAN], vec![2.0]];
        let error = prcc(&nan_input, &outputs, &[1.0; 3]).unwrap_err();
        assert!(error.to_string().contains("draw 1 is NaN"), "{error}");
    }

    /// `prcc` on another thread, with the time it may take before the test
    /// fails as a hang.
    fn within_a_second(inputs: Vec<Vec<f64>>, outputs: Vec<Vec<f64>>) -> Result<Prcc, MethodError> {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let weights = vec![1.0; inputs.len()];
            sender.send(prcc(&inputs, &outputs, &weights)).ok();
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("PRCC returns within a second")
    }

    #[test]
    fn a_nan_value_is_an_error_not_a_hang() {
        let inputs: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        let outputs: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![i as f64, if i % 2 == 1 { f64::NAN } else { 1.0 }])
            .collect();
        let error = within_a_second(inputs, outputs).unwrap_err();
        assert_eq!(
            error,
            MethodError::Prediction(crate::predictive::PredictiveError::NotANumber {
                quantity: 1,
                points: vec![1, 3],
                run: None
            })
        );
        let message = error.to_string();
        assert!(
            message.contains("quantity 1") && message.contains("[1, 3]"),
            "{message}"
        );
    }

    #[test]
    fn a_constant_quantity_has_no_coefficient() {
        let inputs: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        let outputs = vec![vec![1.0]; 5];
        let result = prcc(&inputs, &outputs, &[1.0; 5]).unwrap();
        assert!(result.coefficients[0][0].is_nan());
        assert!(result.p_values[0][0].is_nan());
    }

    #[test]
    fn a_quantity_ranked_as_one_input_has_the_coefficient_one_for_it() {
        let inputs: Vec<Vec<f64>> = (0..8)
            .map(|i| vec![i as f64, ((i * 5) % 8) as f64])
            .collect();
        let outputs: Vec<Vec<f64>> = inputs.iter().map(|x| vec![2.0 * x[0]]).collect();
        let result = prcc(&inputs, &outputs, &[1.0; 8]).unwrap();
        assert!((result.coefficients[0][0] - 1.0).abs() < 1e-12);
        assert!(result.coefficients[0][1].is_nan());
        assert_eq!(result.p_values[0][0], 0.0);
    }

    #[test]
    fn nearly_collinear_inputs_are_named() {
        let inputs: Vec<Vec<f64>> = (0..6)
            .map(|i| vec![i as f64, i as f64 + 1e-9 * (i % 2) as f64])
            .collect();
        let outputs: Vec<Vec<f64>> = (0..6).map(|i| vec![(i * i) as f64]).collect();
        let error = prcc(&inputs, &outputs, &[1.0; 6]).unwrap_err();
        assert!(error.to_string().contains("collinear"), "{error}");
    }

    #[test]
    fn collinear_inputs_are_named() {
        let inputs: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64, 2.0 * i as f64]).collect();
        let outputs: Vec<Vec<f64>> = (0..6).map(|i| vec![(i * i) as f64]).collect();
        let error = prcc(&inputs, &outputs, &[1.0; 6]).unwrap_err();
        assert!(error.to_string().contains("collinear"), "{error}");
    }
}
