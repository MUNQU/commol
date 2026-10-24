//! Practical identifiability from the starts of a multi-start optimization:
//! starts that converge to different points with the same loss show
//! parameters the data do not determine.

use crate::error::MethodError;

/// The result of [`starts_identifiability`].
#[derive(Debug, Clone, PartialEq)]
pub struct StartsIdentifiability {
    /// The starts, by index, that converged within the loss tolerance of the
    /// best.
    pub near_best: Vec<usize>,
    /// The largest difference of each parameter between those starts, as a
    /// share of the width of its bounds.
    pub spread: Vec<f64>,
    /// Whether that spread is above the parameter tolerance.
    pub flagged: Vec<bool>,
}

/// Compare the ends of the starts that converged: those whose loss exceeds
/// the best by at most `loss_tolerance · |best|` are near the best, and a
/// parameter whose ends among them differ by more than
/// `parameter_tolerance` of its bound width is flagged.
///
/// # Errors
///
/// [`MethodError::Invalid`] when the inputs differ in length, a
/// tolerance is negative, or fewer than two starts converged.
pub fn starts_identifiability(
    ends: &[Vec<f64>],
    losses: &[f64],
    converged: &[bool],
    bounds: &[(f64, f64)],
    parameter_tolerance: f64,
    loss_tolerance: f64,
) -> Result<StartsIdentifiability, MethodError> {
    if ends.len() != losses.len() || ends.len() != converged.len() {
        return Err(MethodError::Invalid(
            "One end, loss and convergence flag per start".to_string(),
        ));
    }
    crate::ranges::PARAMETER_TOLERANCE
        .check(parameter_tolerance)
        .and(crate::ranges::LOSS_TOLERANCE.check(loss_tolerance))
        .map_err(MethodError::Invalid)?;
    let finished: Vec<usize> = (0..ends.len()).filter(|&i| converged[i]).collect();
    if finished.len() < 2 {
        return Err(MethodError::Invalid(format!(
            "Comparing starts needs at least two that converged, got {}; run the calibration \
             with more starts (Calibrator.run(starts=n))",
            finished.len()
        )));
    }
    let best = finished
        .iter()
        .map(|&i| losses[i])
        .fold(f64::INFINITY, f64::min);
    let near_best: Vec<usize> = finished
        .into_iter()
        .filter(|&i| losses[i] - best <= loss_tolerance * best.abs())
        .collect();
    let spread: Vec<f64> = bounds
        .iter()
        .enumerate()
        .map(|(j, (min, max))| {
            let values = near_best.iter().map(|&i| ends[i][j]);
            let high = values.clone().fold(f64::NEG_INFINITY, f64::max);
            let low = values.fold(f64::INFINITY, f64::min);
            (high - low) / (max - min)
        })
        .collect();
    let flagged = spread.iter().map(|&s| s > parameter_tolerance).collect();
    Ok(StartsIdentifiability {
        near_best,
        spread,
        flagged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare(losses: &[f64], parameter_tolerance: f64) -> StartsIdentifiability {
        let ends = [vec![1.5], vec![2.5], vec![2.9]];
        starts_identifiability(
            &ends,
            losses,
            &[true; 3],
            &[(1.0, 3.0)],
            parameter_tolerance,
            0.1,
        )
        .unwrap()
    }

    #[test]
    fn the_starts_near_the_best_lose_at_most_its_share_of_the_best_loss() {
        // 10.5 is within 0.1 × 10 of the best; 12 is not.
        let result = compare(&[10.0, 10.5, 12.0], 0.1);
        assert_eq!(result.near_best, vec![0, 1]);
    }

    #[test]
    fn the_spread_is_a_share_of_the_bound_width_and_flags_above_the_tolerance() {
        let result = compare(&[10.0, 10.5, 12.0], 0.5);
        // The ends 1.5 and 2.5 differ by half the width of (1, 3).
        assert_eq!(result.spread, vec![0.5]);
        assert_eq!(result.flagged, vec![false]);
        assert_eq!(compare(&[10.0, 10.5, 12.0], 0.4).flagged, vec![true]);
    }

    #[test]
    fn the_inputs_have_one_value_per_start_and_tolerances_of_zero_or_more() {
        let check = |losses: &[f64], converged: &[bool], tolerances: (f64, f64)| {
            starts_identifiability(
                &[vec![1.0], vec![2.0]],
                losses,
                converged,
                &[(0.0, 3.0)],
                tolerances.0,
                tolerances.1,
            )
        };
        assert!(check(&[1.0, 1.0], &[true, true], (0.1, 0.1)).is_ok());
        assert!(check(&[1.0], &[true, true], (0.1, 0.1)).is_err());
        assert!(check(&[1.0, 1.0], &[true], (0.1, 0.1)).is_err());
        assert!(check(&[1.0, 1.0], &[true, true], (-0.1, 0.1)).is_err());
        assert!(check(&[1.0, 1.0], &[true, true], (0.1, -0.1)).is_err());
    }
}
