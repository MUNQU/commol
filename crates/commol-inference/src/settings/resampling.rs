//! The settings of the bootstraps, the block jackknife and the observation
//! noises.

use commol_calibration::FitRecord;
use commol_calibration::ranges::STARTS;
use commol_core::{Rule, checked, tagged};
use serde::{Deserialize, Serialize};

use super::BlockLength;
use crate::ranges::{BLOCK_LENGTH, REPLICATES};
use crate::resampling::{Multipliers, ResidualBlocks, ResidualScale};

/// A Bayesian bootstrap, as the Python class `BayesianBootstrap` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct BayesianBootstrapSettings {
    /// Number of replicates.
    pub replicates: usize,
    /// Positions per block.
    pub block_length: BlockLength,
    /// Starts of the fit and of each replicate, the fit first.
    pub starts: usize,
}

checked!(
    BayesianBootstrapSettings,
    "BayesianBootstrap",
    [
        Rule::Count(REPLICATES),
        Rule::Count(BLOCK_LENGTH),
        Rule::Count(STARTS),
    ]
);

/// A parametric bootstrap, as the Python class `ParametricBootstrap` writes
/// it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct ParametricBootstrapSettings {
    /// Number of replicates.
    pub replicates: usize,
    /// Starts of the fit and of each replicate, the fit first.
    pub starts: usize,
}

checked!(
    ParametricBootstrapSettings,
    "ParametricBootstrap",
    [Rule::Count(REPLICATES), Rule::Count(STARTS)]
);

/// A residual block bootstrap, as the Python class `ResidualBootstrap`
/// writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct ResidualBootstrapSettings {
    /// Number of replicates.
    pub replicates: usize,
    /// Positions per block.
    pub block_length: BlockLength,
    /// The scale of the residuals.
    pub residual_scale: ResidualScale,
    /// How the blocks are drawn.
    pub scheme: ResidualBlocks,
    /// Starts of the fit and of each replicate, the fit first.
    pub starts: usize,
}

checked!(
    ResidualBootstrapSettings,
    "ResidualBootstrap",
    [
        Rule::Count(REPLICATES),
        Rule::Count(BLOCK_LENGTH),
        Rule::Count(STARTS),
    ]
);

/// A wild block bootstrap, as the Python class `WildBootstrap` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct WildBootstrapSettings {
    /// Number of replicates.
    pub replicates: usize,
    /// Positions per block.
    pub block_length: BlockLength,
    /// The distribution of the multipliers.
    pub multipliers: Multipliers,
    /// Starts of the fit and of each replicate, the fit first.
    pub starts: usize,
}

checked!(
    WildBootstrapSettings,
    "WildBootstrap",
    [
        Rule::Count(REPLICATES),
        Rule::Count(BLOCK_LENGTH),
        Rule::Count(STARTS),
    ]
);

/// A bootstrap: the union of the four bootstrap classes, by `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", rename_all = "snake_case")]
pub enum Bootstrap {
    /// `BayesianBootstrap`, kind `"bayesian"`.
    Bayesian(BayesianBootstrapSettings),
    /// `ParametricBootstrap`, kind `"parametric"`.
    Parametric(ParametricBootstrapSettings),
    /// `ResidualBootstrap`, kind `"residual"`.
    Residual(ResidualBootstrapSettings),
    /// `WildBootstrap`, kind `"wild"`.
    Wild(WildBootstrapSettings),
}

tagged!(Bootstrap);

impl Bootstrap {
    /// The number of replicates.
    pub fn replicates(&self) -> usize {
        match self {
            Self::Bayesian(settings) => settings.replicates,
            Self::Parametric(settings) => settings.replicates,
            Self::Residual(settings) => settings.replicates,
            Self::Wild(settings) => settings.replicates,
        }
    }

    /// The starts of the fit and of each replicate.
    pub fn starts(&self) -> usize {
        match self {
            Self::Bayesian(settings) => settings.starts,
            Self::Parametric(settings) => settings.starts,
            Self::Residual(settings) => settings.starts,
            Self::Wild(settings) => settings.starts,
        }
    }

    /// The name of the method that a sample of the bootstrap records.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bayesian(_) => "bayesian_bootstrap",
            Self::Parametric(_) => "parametric_bootstrap",
            Self::Residual(_) => "residual_bootstrap",
            Self::Wild(_) => "wild_bootstrap",
        }
    }
}

/// A block jackknife, as the Python class `BlockJackknife` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct BlockJackknifeSettings {
    /// Positions per block.
    pub block_length: BlockLength,
    /// Starts of the fit and of each refit, the fit first.
    pub starts: usize,
}

checked!(
    BlockJackknifeSettings,
    "BlockJackknife",
    [Rule::Count(BLOCK_LENGTH), Rule::Count(STARTS)]
);

/// Residual noise, as the Python class `ResidualNoise` writes it: blocks of
/// the residuals of a fit added to each draw.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct ResidualNoiseSettings {
    /// The fit whose residuals are added.
    pub fit: FitRecord,
    /// Positions per block.
    pub block_length: usize,
    /// The scale of the residuals.
    pub residual_scale: ResidualScale,
}

checked!(
    ResidualNoiseSettings,
    "ResidualNoise",
    [Rule::Count(BLOCK_LENGTH)]
);

/// The noise of a draw of observations: the union `NoNoise |
/// ResidualNoise | DistributionNoise`, by `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", rename_all = "snake_case")]
pub enum NoiseSettings {
    /// `NoNoise`, kind `"none"`: the prediction itself.
    None,
    /// `ResidualNoise`, kind `"residual"`.
    Residual(ResidualNoiseSettings),
    /// `DistributionNoise`, kind `"distribution"`: a draw of the
    /// observation models of the problem.
    Distribution,
}

tagged!(NoiseSettings);
