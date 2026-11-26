//! The settings of a dependence analysis: its reference, as the Python
//! classes `AtParameters` and `Refitted` write it.

use commol_calibration::ranges::STARTS;
use commol_core::{Rule, checked, tagged};
use serde::{Deserialize, Serialize};

use crate::ranges::REPLICATES;

/// A reference at the parameters, as the Python class `AtParameters` writes
/// it: data sets drawn from the observation models at the parameters of each
/// draw, with no refit and no run of the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct AtParametersSettings {
    /// The data sets of the reference.
    pub replicates: usize,
}

checked!(
    AtParametersSettings,
    "AtParameters",
    [Rule::Count(REPLICATES)]
);

/// A reference of refits, as the Python class `Refitted` writes it: data
/// sets drawn at the fit, each fitted again by the optimizer of the problem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct RefittedSettings {
    /// The data sets of the reference.
    pub replicates: usize,
    /// Starts of each refit.
    pub starts: usize,
}

checked!(
    RefittedSettings,
    "Refitted",
    [Rule::Count(REPLICATES), Rule::Count(STARTS)]
);

/// The reference of a dependence analysis of residuals: the union of the two
/// reference classes, by `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", rename_all = "snake_case")]
pub enum Reference {
    /// `AtParameters`, kind `"at_parameters"`.
    AtParameters(AtParametersSettings),
    /// `Refitted`, kind `"refitted"`.
    Refitted(RefittedSettings),
}

tagged!(Reference);

impl Reference {
    /// The data sets of the reference.
    pub fn replicates(&self) -> usize {
        match self {
            Self::AtParameters(settings) => settings.replicates,
            // coverage: invariant, a sample of many draws refuses Refitted
            Self::Refitted(settings) => settings.replicates,
        }
    }
}
