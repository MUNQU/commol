//! Practical identifiability from the starts of a multi-start optimization:
//! starts that converge to different points with the same loss show
//! parameters the data do not determine.

use super::IdentifiabilityError;

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
/// [`IdentifiabilityError::Invalid`] when the inputs differ in length, a
/// tolerance is negative, or fewer than two starts converged.
pub fn starts_identifiability(
    ends: &[Vec<f64>],
    losses: &[f64],
    converged: &[bool],
    bounds: &[(f64, f64)],
    parameter_tolerance: f64,
    loss_tolerance: f64,
) -> Result<StartsIdentifiability, IdentifiabilityError> {
    if ends.len() != losses.len() || ends.len() != converged.len() {
        return Err(IdentifiabilityError::Invalid(
            "One end, loss and convergence flag per start".to_string(),
        ));
    }
    if !(parameter_tolerance >= 0.0 && loss_tolerance >= 0.0) {
        return Err(IdentifiabilityError::Invalid(format!(
            "The tolerances must not be negative, got {parameter_tolerance} and {loss_tolerance}"
        )));
    }
    let finished: Vec<usize> = (0..ends.len()).filter(|&i| converged[i]).collect();
    if finished.len() < 2 {
        return Err(IdentifiabilityError::Invalid(format!(
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
