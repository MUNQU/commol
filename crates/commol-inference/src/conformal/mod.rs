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
//! | [`adaptive`] | [`AdaptiveLevel`] and [`adaptive_conformal`]: a working level that moves with each miss |
//! | [`enbpi`](mod@enbpi) | [`enbpi()`]: replicates of blocks drawn with replacement, and their leave-one-out aggregate |
//! | [`intervals`] | [`PredictionIntervals`] and [`ConformalRun`], the result of every variant |
//! | [`predict`](mod@predict) | Point predictions and their spread at any observations, and the target points |
//! | [`split`] | [`split_conformal`]: held-out observations scored against one predictor |
//! | [`jackknife_plus`](mod@jackknife_plus) | [`jackknife_plus()`]: one refit per block left out, and the CV+ bound |
//! | [`rolling`] | [`rolling_conformal`]: one fit per origin, one quantile per horizon |

pub mod adaptive;
pub mod enbpi;
pub mod intervals;
pub mod jackknife_plus;
pub mod predict;
pub mod quantile;
pub mod rolling;
pub mod score;
pub mod split;

pub use adaptive::{AdaptiveLevel, adaptive_conformal};
pub use enbpi::{BlockMultinomial, enbpi};
pub use intervals::{ConformalRun, PredictionIntervals};
pub use jackknife_plus::{JackknifePlusRun, cv_plus_bound, jackknife_plus};
pub use predict::{Predictions, Predictor, predict, set_predictions, targets};
pub use quantile::{conformal_quantile, group_quantiles, lower_quantile, upper_quantile};
pub use rolling::rolling_conformal;
pub use score::{ConformalSettings, NonconformityScore, Pooling, scales};
pub use split::split_conformal;
