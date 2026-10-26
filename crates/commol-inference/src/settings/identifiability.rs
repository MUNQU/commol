//! The settings of the identifiability methods. The point of `Collinearity`,
//! the sample of `SampleIdentifiability` and the fit of
//! `StartsIdentifiability` cross as handles, so their types have no field
//! for them.

use commol_core::{Rule, checked};
use serde::{Deserialize, Serialize};

use super::PriorsSettings;
use crate::ranges::{
    CORRELATION_THRESHOLD, LOSS_TOLERANCE, MAX_SUBSET_SIZE, PARAMETER_TOLERANCE, POINTS, THRESHOLD,
};

/// Where the structural rank reads the model, as the Python API names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankDesign {
    /// At the positions and outputs of the observations.
    Observations,
    /// At every output on a dense grid of positions.
    Dense,
}

/// The structural rank, as the Python class `StructuralRank` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct StructuralRankSettings {
    /// Number of random points.
    pub points: usize,
    /// Where the rank reads the model.
    pub design: RankDesign,
}

checked!(
    StructuralRankSettings,
    "StructuralRank",
    [Rule::Count(POINTS)]
);

/// The collinearity analysis, as the Python class `Collinearity` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct CollinearitySettings {
    /// The largest subset of parameters read.
    pub max_subset_size: usize,
    /// The collinearity index at or above which a subset is not
    /// identifiable.
    pub threshold: f64,
}

checked!(
    CollinearitySettings,
    "Collinearity",
    [Rule::Count(MAX_SUBSET_SIZE), Rule::Real(THRESHOLD)]
);

/// What the contraction of a sample is measured against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reference {
    /// The width of the bounds.
    Bounds,
    /// The spread of the priors.
    Prior,
}

/// The identifiability of a sample, as the Python class
/// `SampleIdentifiability` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct SampleIdentifiabilitySettings {
    /// What the contraction is measured against.
    pub reference: Reference,
    /// The correlation at or above which a pair is flagged.
    pub correlation_threshold: f64,
    /// The priors of `reference="prior"`, or None.
    pub priors: Option<PriorsSettings>,
}

checked!(
    SampleIdentifiabilitySettings,
    "SampleIdentifiability",
    [Rule::Real(CORRELATION_THRESHOLD)]
);

/// The identifiability of the starts of a fit, as the Python class
/// `StartsIdentifiability` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct StartsIdentifiabilitySettings {
    /// The spread of an end, as a share of its bounds, that flags it.
    pub parameter_tolerance: f64,
    /// The loss above the best within which a start counts as at the best.
    pub loss_tolerance: f64,
}

checked!(
    StartsIdentifiabilitySettings,
    "StartsIdentifiability",
    [Rule::Real(PARAMETER_TOLERANCE), Rule::Real(LOSS_TOLERANCE)]
);
