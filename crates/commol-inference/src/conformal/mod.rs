//! Conformal prediction intervals: intervals for observations, with a
//! coverage guarantee that needs no noise model, no prior and no correct
//! model.
//!
//! A [`NonconformityScore`] turns a predicted and an observed value into a
//! score; each variant only chooses which predictions and which held-out
//! values feed it, and the quantile of the scores ([`conformal_quantile`])
//! sets the width of every interval.
//!
//! | Module | Holds |
//! |---|---|
//! | [`score`] | [`NonconformityScore`], [`Pooling`], [`ConformalSettings`] and the scale of a score |
//! | [`quantile`] | The conformal quantile and the order statistics of jackknife+ |
//! | [`intervals`] | [`PredictionIntervals`] and [`ConformalRun`], the result of every variant |
//! | [`predict`] | Point predictions and their spread at any observations, and the target points |
//! | [`split`] | [`split_conformal`]: held-out observations scored against one predictor |

pub mod intervals;
pub mod predict;
pub mod quantile;
pub mod score;
pub mod split;

pub use intervals::{ConformalRun, PredictionIntervals};
pub use predict::{Predictions, Predictor, predict, set_predictions, targets};
pub use quantile::{conformal_quantile, group_quantiles, lower_quantile, upper_quantile};
pub use score::{ConformalSettings, NonconformityScore, Pooling};
pub use split::split_conformal;

use crate::predictive::PredictiveError;

/// Why a conformal method gave no intervals.
#[derive(Debug, Clone, PartialEq)]
pub enum ConformalError {
    /// The settings or the data do not fit the method; the message says
    /// why.
    Invalid(String),
    /// A point could not be predicted.
    Prediction(PredictiveError),
    /// The method's refits failed; the message gives the first reasons.
    Failed(String),
}

impl From<PredictiveError> for ConformalError {
    fn from(error: PredictiveError) -> Self {
        Self::Prediction(error)
    }
}

impl std::fmt::Display for ConformalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) | Self::Failed(message) => f.write_str(message),
            Self::Prediction(error) => error.fmt(f),
        }
    }
}
