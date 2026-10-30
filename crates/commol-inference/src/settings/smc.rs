//! The settings of the sequential Monte Carlo methods, of the calibrations
//! of `η` and of the distances of ABC.

use commol_calibration::ranges::STARTS;
use commol_core::json::{self, Value};
use commol_core::{CONFIDENCE_LEVEL, Rule, checked, tagged};
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};

use super::{BlockLength, PriorsSettings};
use crate::ranges::{
    BLOCK_LENGTH, ETA, MIN_ACCEPTANCE, PARTICLES, QUANTILE, REPLICATES, RESAMPLE_THRESHOLD,
    STAY_PROBABILITY, TARGET_CESS, TWO_DRAWS,
};
use crate::smc::{Distance, SeriesMetric};

/// The calibration of `η` by information matching, as the Python class
/// `InformationMatchingEta` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct InformationMatchingEta {
    /// Positions per block of the scores.
    pub block_length: BlockLength,
    /// Starts of the fit.
    pub starts: usize,
}

checked!(
    InformationMatchingEta,
    "InformationMatchingEta",
    [Rule::Count(BLOCK_LENGTH), Rule::Count(STARTS)]
);

/// The calibration of `η` by bootstrap coverage, as the Python class
/// `BootstrapCoverageEta` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct BootstrapCoverageEta {
    /// Number of Bayesian-bootstrap data sets.
    pub replicates: usize,
    /// Positions per block.
    pub block_length: BlockLength,
    /// The level of the intervals.
    pub confidence_level: f64,
    /// Starts of the fit.
    pub starts: usize,
}

checked!(
    BootstrapCoverageEta,
    "BootstrapCoverageEta",
    [
        Rule::Count(REPLICATES),
        Rule::Count(BLOCK_LENGTH),
        Rule::Real(CONFIDENCE_LEVEL),
        Rule::Count(STARTS),
    ]
);

/// A calibration of `η`: the union of its classes, by `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", rename_all = "snake_case")]
pub enum EtaCalibration {
    /// `InformationMatchingEta`, kind `"information_matching"`.
    InformationMatching(InformationMatchingEta),
    /// `BootstrapCoverageEta`, kind `"bootstrap_coverage"`.
    BootstrapCoverage(BootstrapCoverageEta),
}

tagged!(EtaCalibration);

/// The `η` of a Gibbs posterior: a number, or a calibration of it. The
/// range rule of the number is the rule `eta` of the method that holds it.
#[derive(Debug, Clone, PartialEq)]
pub enum Eta {
    /// The scale itself.
    Value(f64),
    /// The calibration that gives it.
    Calibrated(EtaCalibration),
}

impl Serialize for Eta {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Value(eta) => serializer.serialize_f64(*eta),
            Self::Calibrated(calibration) => calibration.serialize(serializer),
        }
    }
}

/// What a method takes when its eta is missing.
const ETA_FIX: &str = "give a number above 0, InformationMatchingEta(block_length=...) or \
                       BootstrapCoverageEta(replicates=..., block_length=..., \
                       confidence_level=...); the loss has no scale of its own, so eta has no \
                       default";

impl<'de> Deserialize<'de> for Eta {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // A missing field fails to read as a value; a null reads as one.
        let value = Value::deserialize(deserializer)
            .map_err(|_: D::Error| de::Error::custom(format!("eta is required: {ETA_FIX}")))?;
        match value {
            Value::Null => Err(de::Error::custom(format!("null is not an eta: {ETA_FIX}"))),
            value @ (Value::Int(_) | Value::Float(_)) => json::from_value(value).map(Self::Value),
            // A number that is not finite crosses as its text.
            Value::String(text) if json::non_finite(&text).is_some() => {
                json::from_value(Value::String(text)).map(Self::Value)
            }
            value @ Value::Map(_) => json::from_value(value).map(Self::Calibrated),
            _ => Err(de::Error::custom(
                "eta is a number or a calibration of eta, InformationMatchingEta or \
                 BootstrapCoverageEta",
            )),
        }
    }
}

/// The distance of ABC: the union `LossDistance | SeriesDistance`, by
/// `kind`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", rename_all = "snake_case", deny_unknown_fields)]
pub enum DistanceSettings {
    /// `LossDistance`, kind `"loss"`.
    Loss,
    /// `SeriesDistance`, kind `"series"`.
    Series {
        /// The metric of each series.
        metric: SeriesMetric,
    },
}

tagged!(DistanceSettings);

impl DistanceSettings {
    /// The distance of a run from the observations.
    pub fn distance(&self) -> Distance {
        match *self {
            Self::Loss => Distance::Loss,
            Self::Series { metric } => Distance::Series(metric),
        }
    }
}

/// A Gibbs posterior, as the Python class `GibbsPosterior` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct GibbsPosteriorSettings {
    /// The prior of each parameter.
    pub priors: PriorsSettings,
    /// The scale of the loss, or its calibration.
    pub eta: Eta,
    /// Number of particles.
    pub particles: usize,
    /// The conditional effective sample size of each step.
    pub target_cess: f64,
    /// Resample below this share of effective size.
    pub resample_threshold: f64,
    /// The probability that a particle never moves in a step.
    pub stay_probability: f64,
}

checked!(
    GibbsPosteriorSettings,
    "GibbsPosterior",
    [
        Rule::Real(ETA),
        Rule::Count(PARTICLES),
        Rule::Real(TARGET_CESS),
        Rule::Real(RESAMPLE_THRESHOLD),
        Rule::Real(STAY_PROBABILITY),
    ]
);

/// The posterior of the observation models, as the Python class
/// `BayesianPosterior` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct BayesianPosteriorSettings {
    /// The prior of each parameter.
    pub priors: PriorsSettings,
    /// Number of particles.
    pub particles: usize,
    /// The conditional effective sample size of each step.
    pub target_cess: f64,
    /// Resample below this share of effective size.
    pub resample_threshold: f64,
    /// The probability that a particle never moves in a step.
    pub stay_probability: f64,
}

checked!(
    BayesianPosteriorSettings,
    "BayesianPosterior",
    [
        Rule::Count(PARTICLES),
        Rule::Real(TARGET_CESS),
        Rule::Real(RESAMPLE_THRESHOLD),
        Rule::Real(STAY_PROBABILITY),
    ]
);

/// ABC by sequential Monte Carlo, as the Python class `ABCSMC` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct AbcSmcSettings {
    /// The prior of each parameter.
    pub priors: PriorsSettings,
    /// How far a run is from the observations.
    pub distance: DistanceSettings,
    /// Number of particles.
    pub particles: usize,
    /// The weighted share of particles each tolerance keeps.
    pub target_cess: f64,
    /// Stop when the first move of a step accepts less.
    pub min_acceptance: f64,
    /// The probability that a particle never moves in a step.
    pub stay_probability: f64,
}

checked!(
    AbcSmcSettings,
    "ABCSMC",
    [
        Rule::Count(PARTICLES),
        Rule::Real(TARGET_CESS),
        Rule::Real(MIN_ACCEPTANCE),
        Rule::Real(STAY_PROBABILITY),
    ]
);

/// ABC rejection, as the Python class `ABCRejection` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct AbcRejectionSettings {
    /// The prior of each parameter.
    pub priors: PriorsSettings,
    /// How far a run is from the observations.
    pub distance: DistanceSettings,
    /// Number of draws.
    pub draws: usize,
    /// The share of draws kept.
    pub quantile: f64,
}

checked!(
    AbcRejectionSettings,
    "ABCRejection",
    [Rule::Count(TWO_DRAWS), Rule::Real(QUANTILE)]
);
