use serde::{Deserialize, Serialize};

use super::dynamics::{StratificationCondition, Transition};

/// A disease state (compartment) in the model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bin {
    pub id: String,
    pub name: String,
}

/// A cumulative event counter tracked by the engine but excluded from population totals.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Accumulator {
    pub id: String,
    pub name: String,
}

/// A stratification dimension with its categories
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stratification {
    pub id: String,
    pub categories: Vec<String>,
    /// Conditions that must be satisfied (by already-applied stratifications) for
    /// this stratification to expand a compartment. When `None`, it always applies
    /// (standard full Cartesian product). When set, only compartments whose
    /// already-applied categories satisfy ALL conditions are further expanded;
    /// others are kept as-is without this stratification's categories appended.
    ///
    /// Conditions may only reference stratifications declared before this one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditions: Option<Vec<StratificationCondition>>,
}

/// Specifies the fraction of population in a particular bin
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinFraction {
    pub bin: String,
    /// Fraction value - None indicates the initial condition needs calibration
    pub fraction: Option<f64>,
}

/// Specifies the fraction of population in a stratification category
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratificationFraction {
    pub category: String,
    pub fraction: f64,
}

/// Groups stratification fractions for a specific stratification dimension
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratificationFractions {
    pub stratification: String,
    pub fractions: Vec<StratificationFraction>,
}

/// Initial conditions for the population model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitialConditions {
    pub population_size: u64,
    pub bin_fractions: Vec<BinFraction>,
    pub stratification_fractions: Vec<StratificationFractions>,
}

/// Complete population structure including disease states, stratifications, and initial conditions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Population {
    pub bins: Vec<Bin>,
    #[serde(default)]
    pub accumulators: Vec<Accumulator>,
    pub stratifications: Vec<Stratification>,
    pub transitions: Vec<Transition>,
    pub initial_conditions: InitialConditions,
}
