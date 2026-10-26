//! The settings of a sensitivity analysis: its inputs and its methods. The
//! sample of inputs from a sample crosses as a handle, so
//! [`InputSpaceSettings`] has no field for it.

use commol_calibration::CalibrationParameter;
use commol_core::json::Pairs;
use commol_core::{CONFIDENCE_LEVEL, Rule, checked};
use serde::{Deserialize, Serialize};

use crate::distribution::Family;
use crate::quantity::Quantity;
use crate::ranges::{
    DEGREE, DRAWS, LEVELS, MAX_INTERACTION, Q_NORM, TRAJECTORIES, TWO_DRAWS, TWO_REPLICATES,
};

/// What a sensitivity analysis varies, as the Python class `InputSpace`
/// writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSpaceSettings {
    /// The inputs, in order.
    pub parameters: Vec<CalibrationParameter>,
    /// The distribution of each input by name, for independent inputs.
    pub distributions: Option<Pairs<Family>>,
    /// The bin or compartment that takes what the initial conditions leave.
    pub initial_condition_remainder: Option<String>,
}

impl InputSpaceSettings {
    /// The name of each input, in order.
    pub fn names(&self) -> Vec<String> {
        self.parameters.iter().map(|p| p.id.clone()).collect()
    }

    /// The bounds of each input, in order.
    pub fn bounds(&self) -> Vec<(f64, f64)> {
        self.parameters
            .iter()
            .map(|p| (p.min_bound, p.max_bound))
            .collect()
    }
}

/// Monte Carlo propagation, as the Python class `Propagation` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct PropagationSettings {
    /// Number of draws of independent inputs; None for a sample.
    pub draws: Option<usize>,
    /// The quantities of each run.
    pub quantities: Vec<Quantity>,
}

checked!(PropagationSettings, "Propagation", [Rule::Count(DRAWS)]);

/// Partial rank correlation coefficients, as the Python class `PRCC` writes
/// them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct PrccSettings {
    /// Number of draws of independent inputs; None for a sample.
    pub draws: Option<usize>,
    /// The quantities of each run.
    pub quantities: Vec<Quantity>,
}

checked!(PrccSettings, "PRCC", [Rule::Count(DRAWS)]);

/// Morris screening, as the Python class `Morris` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct MorrisSettings {
    /// Number of trajectories.
    pub trajectories: usize,
    /// Number of levels of the grid of each input.
    pub levels: usize,
    /// The quantities of each run.
    pub quantities: Vec<Quantity>,
}

checked!(
    MorrisSettings,
    "Morris",
    [Rule::Count(TRAJECTORIES), Rule::Count(LEVELS)]
);

/// Sobol indices, as the Python class `Sobol` writes them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct SobolSettings {
    /// Number of draws of each matrix of the design.
    pub draws: usize,
    /// Number of bootstrap replicates of the intervals.
    pub replicates: usize,
    /// The level of the intervals.
    pub confidence_level: f64,
    /// The quantities of each run.
    pub quantities: Vec<Quantity>,
}

checked!(
    SobolSettings,
    "Sobol",
    [
        Rule::Count(TWO_DRAWS),
        Rule::Count(TWO_REPLICATES),
        Rule::Real(CONFIDENCE_LEVEL),
    ]
);

/// A polynomial chaos expansion, as the Python class `PolynomialChaos`
/// writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct PolynomialChaosSettings {
    /// The largest degree of a term.
    pub degree: usize,
    /// The q of the q-norm of the basis.
    pub q_norm: f64,
    /// The largest number of inputs of a term, or None for no limit.
    pub max_interaction: Option<usize>,
    /// Number of points of the Latin hypercube.
    pub draws: usize,
    /// The quantities of each run.
    pub quantities: Vec<Quantity>,
}

checked!(
    PolynomialChaosSettings,
    "PolynomialChaos",
    [
        Rule::Count(DEGREE),
        Rule::Real(Q_NORM),
        Rule::Count(MAX_INTERACTION),
        Rule::Count(TWO_DRAWS),
    ]
);
