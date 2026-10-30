//! The settings of the conformal methods. The score and the pooling of
//! each are checked together when it is read
//! ([`ConformalSettings::check_pair`]).

use commol_calibration::Observation;
use commol_calibration::ranges::STARTS;
use commol_core::{CONFIDENCE_LEVEL, Rule, checked};
use serde::{Deserialize, Serialize};

use super::BlockLength;
use crate::conformal::{ConformalSettings, NonconformityScore, Pooling};
use crate::error::MethodError;
use crate::ranges::{BLOCK_LENGTH, STEP, TWO_REPLICATES};

/// The score, the pooling and the level of a conformal method.
///
/// # Errors
///
/// As [`ConformalSettings::new`].
fn conformal(
    score: NonconformityScore,
    by: Pooling,
    confidence_level: f64,
) -> Result<ConformalSettings, MethodError> {
    ConformalSettings::new(score, by, confidence_level)
}

/// The check of a pair of a score and a pooling, as a reader takes it.
fn pair(score: NonconformityScore, by: Pooling) -> Result<(), String> {
    ConformalSettings::check_pair(score, by).map_err(|error| error.to_string())
}

/// Check that the list `field` holds at least one `what`: a method with an
/// empty one would return no interval.
fn given<T>(list: &[T], field: &str, what: &str) -> Result<(), String> {
    if list.is_empty() {
        return Err(format!("{field} is empty: give at least one {what}"));
    }
    Ok(())
}

/// The check of the score of a method whose predictor is a sample or the
/// fit, before the fit runs.
fn predictor(score: NonconformityScore, sample: bool) -> Result<(), MethodError> {
    score.check_spread(sample).map_err(MethodError::Invalid)
}

/// The score and the pooling of a conformal method, which the Python class
/// checks before its other fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreAndPooling {
    /// The nonconformity score.
    pub score: NonconformityScore,
    /// Which scores each quantile reads.
    pub by: Pooling,
}

impl ScoreAndPooling {
    /// Check the pair.
    ///
    /// # Errors
    ///
    /// As [`ConformalSettings::check_pair`].
    pub fn check(self) -> Result<(), MethodError> {
        ConformalSettings::check_pair(self.score, self.by)
    }
}

/// Split conformal prediction, as the Python class `SplitConformal` writes
/// it; its sample crosses as a handle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct SplitConformalSettings {
    /// The nonconformity score.
    pub score: NonconformityScore,
    /// Which scores each quantile reads.
    pub by: Pooling,
    /// The level of the intervals.
    pub confidence_level: f64,
    /// The held-out observations.
    pub calibration_data: Vec<Observation>,
    /// Where the intervals are.
    pub positions: Vec<f64>,
    /// Starts of the fit, when the predictor is the fit.
    pub starts: usize,
}

checked!(
    SplitConformalSettings,
    "SplitConformal",
    [Rule::Real(CONFIDENCE_LEVEL), Rule::Count(STARTS)],
    check = |settings: &SplitConformalSettings| {
        pair(settings.score, settings.by)?;
        given(
            &settings.calibration_data,
            "calibration_data",
            "held-out observation",
        )?;
        given(&settings.positions, "positions", "position")
    }
);

impl SplitConformalSettings {
    /// Check that the score fits the predictor: a sample when `sample`,
    /// otherwise the fit.
    ///
    /// # Errors
    ///
    /// As [`NonconformityScore::check_spread`].
    pub fn check_predictor(&self, sample: bool) -> Result<(), MethodError> {
        predictor(self.score, sample)
    }
}

/// Rolling-origin conformal prediction, as the Python class
/// `RollingConformal` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct RollingConformalSettings {
    /// The nonconformity score.
    pub score: NonconformityScore,
    /// Which scores each quantile reads.
    pub by: Pooling,
    /// The level of the intervals.
    pub confidence_level: f64,
    /// The positions of the origins.
    pub origins: Vec<f64>,
    /// Where the intervals are.
    pub positions: Vec<f64>,
    /// Starts of the fit and of each refit.
    pub starts: usize,
}

checked!(
    RollingConformalSettings,
    "RollingConformal",
    [Rule::Real(CONFIDENCE_LEVEL), Rule::Count(STARTS)],
    check = |settings: &RollingConformalSettings| {
        pair(settings.score, settings.by)?;
        settings.score.check_rolling()?;
        given(&settings.origins, "origins", "origin")?;
        given(&settings.positions, "positions", "position")
    }
);

/// Jackknife+ with blocks, as the Python class `JackknifePlus` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct JackknifePlusSettings {
    /// The nonconformity score.
    pub score: NonconformityScore,
    /// Which scores each quantile reads.
    pub by: Pooling,
    /// The level of the intervals.
    pub confidence_level: f64,
    /// Positions per block.
    pub block_length: BlockLength,
    /// Where the intervals are.
    pub positions: Vec<f64>,
    /// Starts of the fit and of each refit.
    pub starts: usize,
}

checked!(
    JackknifePlusSettings,
    "JackknifePlus",
    [
        Rule::Real(CONFIDENCE_LEVEL),
        Rule::Count(BLOCK_LENGTH),
        Rule::Count(STARTS),
    ],
    check = |settings: &JackknifePlusSettings| {
        pair(settings.score, settings.by)?;
        given(&settings.positions, "positions", "position")
    }
);

/// Ensemble batch prediction intervals, as the Python class `EnbPI` writes
/// them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct EnbPiSettings {
    /// The nonconformity score.
    pub score: NonconformityScore,
    /// Which scores each quantile reads.
    pub by: Pooling,
    /// The level of the intervals.
    pub confidence_level: f64,
    /// Number of replicates.
    pub replicates: usize,
    /// Positions per block.
    pub block_length: BlockLength,
    /// Where the intervals are.
    pub positions: Vec<f64>,
    /// Starts of the fit and of each replicate.
    pub starts: usize,
}

checked!(
    EnbPiSettings,
    "EnbPI",
    [
        Rule::Real(CONFIDENCE_LEVEL),
        Rule::Count(TWO_REPLICATES),
        Rule::Count(BLOCK_LENGTH),
        Rule::Count(STARTS),
    ],
    check = |settings: &EnbPiSettings| {
        pair(settings.score, settings.by)?;
        given(&settings.positions, "positions", "position")
    }
);

/// Adaptive conformal inference, as the Python class `AdaptiveConformal`
/// writes it; its sample crosses as a handle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct AdaptiveConformalSettings {
    /// The nonconformity score.
    pub score: NonconformityScore,
    /// Which scores each quantile reads.
    pub by: Pooling,
    /// The target level of the intervals.
    pub confidence_level: f64,
    /// The step of the working level.
    pub step: f64,
    /// The observations that arrive.
    pub stream: Vec<Observation>,
    /// Starts of the fit, when the predictor is the fit.
    pub starts: usize,
}

checked!(
    AdaptiveConformalSettings,
    "AdaptiveConformal",
    [
        Rule::Real(CONFIDENCE_LEVEL),
        Rule::Real(STEP),
        Rule::Count(STARTS),
    ],
    check = |settings: &AdaptiveConformalSettings| {
        pair(settings.score, settings.by)?;
        given(&settings.stream, "stream", "observation that arrives")
    }
);

impl AdaptiveConformalSettings {
    /// Check that the score fits the predictor: a sample when `sample`,
    /// otherwise the fit.
    ///
    /// # Errors
    ///
    /// As [`NonconformityScore::check_spread`].
    pub fn check_predictor(&self, sample: bool) -> Result<(), MethodError> {
        predictor(self.score, sample)
    }
}

macro_rules! conformal_settings {
    ($($type:ty),*) => {
        $(
            impl $type {
                /// The score, the pooling and the level of the method.
                ///
                /// # Errors
                ///
                /// As [`ConformalSettings::new`].
                pub fn conformal(&self) -> Result<ConformalSettings, MethodError> {
                    conformal(self.score, self.by, self.confidence_level)
                }
            }
        )*
    };
}

conformal_settings!(
    SplitConformalSettings,
    RollingConformalSettings,
    JackknifePlusSettings,
    EnbPiSettings,
    AdaptiveConformalSettings
);
