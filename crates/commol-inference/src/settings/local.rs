//! The settings of the local approximations.

use commol_calibration::ranges::STARTS;
use commol_core::{Rule, checked, tagged};
use serde::{Deserialize, Serialize};

use super::{BlockLength, PriorsSettings};
use crate::local::CurvatureKind;
use crate::ranges::{BLOCK_LENGTH, DRAWS, ETA, MIXTURE_DRAWS};

/// The Laplace approximation, as the Python class `Laplace` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct LaplaceSettings {
    /// Number of draws.
    pub draws: usize,
    /// The curvature at the fit.
    pub curvature: CurvatureKind,
    /// Starts of the fit.
    pub starts: usize,
}

checked!(
    LaplaceSettings,
    "Laplace",
    [Rule::Count(DRAWS), Rule::Count(STARTS)]
);

/// The sandwich, as the Python class `Sandwich` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct SandwichSettings {
    /// Positions per block of the meat.
    pub block_length: BlockLength,
    /// Number of draws.
    pub draws: usize,
    /// The curvature at the fit.
    pub curvature: CurvatureKind,
    /// Starts of the fit.
    pub starts: usize,
}

checked!(
    SandwichSettings,
    "Sandwich",
    [
        Rule::Count(BLOCK_LENGTH),
        Rule::Count(DRAWS),
        Rule::Count(STARTS),
    ]
);

/// The `η` of the target of a Gibbs posterior, as the Python class
/// `GibbsTarget` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct GibbsTargetSettings {
    /// The prior of each parameter.
    pub priors: PriorsSettings,
    /// The scale of the loss.
    pub eta: f64,
}

checked!(GibbsTargetSettings, "GibbsTarget", [Rule::Real(ETA)]);

/// The target of a Laplace mixture: the union `LikelihoodTarget |
/// GibbsTarget`, by `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", rename_all = "snake_case", deny_unknown_fields)]
pub enum MixtureTarget {
    /// `LikelihoodTarget`, kind `"likelihood"`: the posterior of the
    /// observation models.
    Likelihood {
        /// The prior of each parameter.
        priors: PriorsSettings,
    },
    /// `GibbsTarget`, kind `"gibbs"`: a Gibbs posterior.
    Gibbs(GibbsTargetSettings),
}

tagged!(MixtureTarget);

impl MixtureTarget {
    /// The prior of each parameter.
    pub fn priors(&self) -> &PriorsSettings {
        match self {
            Self::Likelihood { priors } => priors,
            Self::Gibbs(target) => &target.priors,
        }
    }
}

/// A Laplace mixture, as the Python class `LaplaceMixture` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct LaplaceMixtureSettings {
    /// Starts of the fit; the end of each gives a component.
    pub starts: usize,
    /// Number of draws.
    pub draws: usize,
    /// The posterior sampled.
    pub target: MixtureTarget,
    /// The curvature at each end.
    pub curvature: CurvatureKind,
}

checked!(
    LaplaceMixtureSettings,
    "LaplaceMixture",
    [Rule::Count(STARTS), Rule::Count(MIXTURE_DRAWS)]
);
