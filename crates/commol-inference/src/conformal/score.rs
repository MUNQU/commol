//! The nonconformity scores, how quantiles pool them, and the settings every
//! variant shares.

use std::collections::HashMap;

use commol_calibration::ObservedDataPoint;
use commol_calibration::discrepancy::series_rms;

use crate::error::MethodError;

/// How a predicted and an observed value become a score: `|y − ŷ| / s`,
/// where the scale `s` depends on the score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonconformityScore {
    /// `s = 1`.
    Absolute,
    /// `s` is the root mean square of the observed values of the series in
    /// the problem, the inverse of its normalization factor
    /// ([`commol_calibration::discrepancy::observation_normalization`]).
    Normalized,
    /// `s = σ̂`, the spread of the predictions at that point.
    Spread,
}

/// Why `score="spread"` needs a sample.
const NO_SPREAD: &str = "score=\"spread\" divides by the spread of the predictions, and a fit has \
                         none: give a sample, whose predictive gives the prediction and its spread";

impl NonconformityScore {
    /// The score of `observed` against `predicted` on the scale `scale`.
    pub fn of(self, observed: f64, predicted: f64, scale: f64) -> f64 {
        (observed - predicted).abs() / scale
    }

    /// Check that a predictor with a spread or without one (a fit) gives the
    /// scale of this score: `"spread"` needs the spread of a sample.
    ///
    /// # Errors
    ///
    /// Returns a message that names the fix.
    pub fn check_spread(self, has_spread: bool) -> Result<(), String> {
        if self == Self::Spread && !has_spread {
            return Err(NO_SPREAD.to_string());
        }
        Ok(())
    }

    /// Check that a method that refits once per origin, so has no spread,
    /// takes this score.
    ///
    /// # Errors
    ///
    /// Returns a message that names the scores it takes, for `"spread"`.
    pub fn check_rolling(self) -> Result<(), String> {
        if self == Self::Spread {
            return Err(
                "RollingConformal fits once per origin, so its predictions have no spread for \
                 score=\"spread\"; use \"absolute\" or \"normalized\""
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// Which scores a quantile is taken of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    /// The scores of each series (observation label) apart.
    Series,
    /// The scores of every series together.
    All,
}

impl Pooling {
    /// The group of a series: its label, or one group for every series.
    pub fn group(self, label: &str) -> &str {
        match self {
            Self::Series => label,
            Self::All => "",
        }
    }
}

/// The score, the pooling and the confidence level of a conformal method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalSettings {
    /// The score.
    pub score: NonconformityScore,
    /// Which scores each quantile reads.
    pub by: Pooling,
    /// The level of the intervals, `1 − α`.
    pub confidence_level: f64,
}

impl ConformalSettings {
    /// Check the pair of a score and a pooling.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when `"normalized"` is taken per series:
    /// its factor is one constant per series, so its intervals would equal
    /// those of `"absolute"`.
    pub fn check_pair(score: NonconformityScore, by: Pooling) -> Result<(), MethodError> {
        if score == NonconformityScore::Normalized && by == Pooling::Series {
            return Err(MethodError::Invalid(
                "score=\"normalized\" with by=\"series\" divides every score of a series by one \
                 constant, so its intervals equal those of \"absolute\"; use by=\"all\" to pool \
                 the series on one scale, or score=\"absolute\""
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// The settings of a score, a pooling and a confidence level.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when the pair is rejected
    /// ([`Self::check_pair`]) or `confidence_level` is not strictly between 0
    /// and 1 ([`commol_core::CONFIDENCE_LEVEL`]).
    pub fn new(
        score: NonconformityScore,
        by: Pooling,
        confidence_level: f64,
    ) -> Result<Self, MethodError> {
        Self::check_pair(score, by)?;
        commol_core::CONFIDENCE_LEVEL
            .check(confidence_level)
            .map_err(MethodError::Invalid)?;
        Ok(Self {
            score,
            by,
            confidence_level,
        })
    }
}

/// The scale of each of `points` under `score`.
///
/// `spread` holds the spread of the predictions at each point, when the
/// method has one.
///
/// # Errors
///
/// [`MethodError::Invalid`] when `"normalized"` meets a series the problem
/// does not observe, `"spread"` has no spread, or a spread is not positive.
pub fn scales(
    score: NonconformityScore,
    observed: &[ObservedDataPoint],
    points: &[ObservedDataPoint],
    spread: Option<&[f64]>,
) -> Result<Vec<f64>, MethodError> {
    scale_values(score, observed, points, spread).map_err(MethodError::Invalid)
}

fn scale_values(
    score: NonconformityScore,
    observed: &[ObservedDataPoint],
    points: &[ObservedDataPoint],
    spread: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    match score {
        NonconformityScore::Absolute => Ok(vec![1.0; points.len()]),
        NonconformityScore::Normalized => {
            let labels: Vec<String> = observed.iter().map(|obs| obs.label.clone()).collect();
            let values: Vec<f64> = observed.iter().map(|obs| obs.value).collect();
            let rms: HashMap<&str, f64> = labels
                .iter()
                .map(String::as_str)
                .zip(series_rms(&labels, &values))
                .collect();
            points
                .iter()
                .map(|point| {
                    rms.get(point.label.as_str()).copied().ok_or_else(|| {
                        format!(
                            "score=\"normalized\" scales a series by its observations in the \
                             problem, and the problem has none of '{}'",
                            point.label
                        )
                    })
                })
                .collect()
        }
        NonconformityScore::Spread => {
            let Some(spread) = spread else {
                return Err(NO_SPREAD.to_string());
            };
            points
                .iter()
                .zip(spread)
                .map(|(point, &s)| {
                    if s > 0.0 && s.is_finite() {
                        Ok(s)
                    } else {
                        Err(format!(
                            "The spread of the predictions of '{}' at {} is {s}; \
                             score=\"spread\" needs a positive spread at every point",
                            point.label, point.at
                        ))
                    }
                })
                .collect()
        }
    }
}
