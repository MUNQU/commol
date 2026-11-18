//! The settings of the identifiability methods. The point of `Collinearity`,
//! the sample of `SampleIdentifiability` and the fit of
//! `StartsIdentifiability` cross as handles, so their types have no field
//! for them.

use commol_core::{Rule, checked};
use serde::{Deserialize, Serialize};

use super::PriorsSettings;
use crate::priors::Priors;
use crate::ranges::{
    BOUND_SHARE, CONTRACTION_THRESHOLD, CORRELATION_THRESHOLD, LOSS_TOLERANCE, MAX_SUBSET_SIZE,
    PARAMETER_TOLERANCE, POINTS, THRESHOLD,
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
    /// The absolute correlation above which a pair is flagged.
    pub correlation_threshold: f64,
    /// The contraction below which a parameter is poorly determined.
    pub contraction_threshold: f64,
    /// The share of the reference at each side in which the mass of the
    /// sample is measured.
    pub bound_share: f64,
    /// The parameters whose shape measures read `ln θ`.
    pub log_parameters: Vec<String>,
    /// The priors of `reference="prior"`, or None.
    pub priors: Option<PriorsSettings>,
}

checked!(
    SampleIdentifiabilitySettings,
    "SampleIdentifiability",
    [
        Rule::Real(CORRELATION_THRESHOLD),
        Rule::Real(CONTRACTION_THRESHOLD),
        Rule::Real(BOUND_SHARE)
    ],
    check = SampleIdentifiabilitySettings::check
);

impl SampleIdentifiabilitySettings {
    /// Check that the priors go with `reference="prior"`, and only with it.
    ///
    /// # Errors
    ///
    /// Returns a message that names the fix.
    pub fn check(&self) -> Result<(), String> {
        match (self.reference, &self.priors) {
            (Reference::Prior, None) => Err(
                "reference='prior' needs priors: the prior of each parameter, which the \
                 contraction is measured against"
                    .to_string(),
            ),
            (Reference::Bounds, Some(_)) => Err(
                "priors are read only with reference='prior'; with reference='bounds' leave \
                 them out"
                    .to_string(),
            ),
            _ => Ok(()),
        }
    }

    /// The priors the contraction is measured against, for the parameters
    /// `names` with `bounds`: None for `reference="bounds"`.
    ///
    /// # Errors
    ///
    /// The message of [`Self::check`], or of [`PriorsSettings::priors`].
    pub fn reference_priors(
        &self,
        names: &[String],
        bounds: &[(f64, f64)],
    ) -> Result<Option<Priors>, String> {
        self.check()?;
        self.priors
            .as_ref()
            .map(|priors| priors.priors(names, bounds))
            .transpose()
    }

    /// Whether each parameter of a sample with `names`, `bounds` and the
    /// draws `values` reads `ln θ`.
    ///
    /// # Errors
    ///
    /// When `log_parameters` names a parameter the sample does not hold,
    /// one with a lower bound below 0, or one with a draw at 0.
    pub fn log_scales(
        &self,
        names: &[String],
        bounds: &[(f64, f64)],
        values: &[Vec<f64>],
    ) -> Result<Vec<bool>, String> {
        for name in &self.log_parameters {
            let Some(j) = names.iter().position(|n| n == name) else {
                return Err(format!(
                    "log_parameters names '{name}', which is not a parameter of the sample: \
                     {names:?}"
                ));
            };
            let (lower, upper) = bounds[j];
            if lower < 0.0 {
                return Err(format!(
                    "log_parameters reads ln θ, so each of its parameters has a lower bound of \
                     at least 0: '{name}' has the bounds [{lower}, {upper}]"
                ));
            }
            if values.iter().any(|draw| draw[j] == 0.0) {
                return Err(format!(
                    "log_parameters reads ln θ, so each of its parameters has every draw above \
                     0: '{name}' has a draw at 0"
                ));
            }
        }
        Ok(names
            .iter()
            .map(|n| self.log_parameters.contains(n))
            .collect())
    }
}

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
