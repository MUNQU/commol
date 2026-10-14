//! The structure of a compartmental model: bins expanded over
//! stratifications, joined by transitions, and the initial conditions.

use serde::{Deserialize, Serialize};

use super::transitions::{StratificationCondition, Transition};

/// A state of the model. Stratifications expand it into compartments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bin {
    /// Identifier used in rates, initial conditions and outputs.
    pub id: String,
    /// Display name.
    pub name: String,
}

/// A cumulative event counter tracked by the engine but excluded from population totals.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Accumulator {
    /// Identifier, and the output name before expansion.
    pub id: String,
    /// Display name.
    pub name: String,
}

/// A stratification dimension with its categories
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stratification {
    /// Identifier used in conditions and initial conditions.
    pub id: String,
    /// Categories, in the order their compartments are expanded.
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
    /// Id of the bin.
    pub bin: String,
    /// Fraction value - None indicates the initial condition needs calibration
    pub fraction: Option<f64>,
}

/// Specifies the fraction of population in a stratification category
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratificationFraction {
    /// The category.
    pub category: String,
    /// Fraction of the group the stratification splits, in `0.0..=1.0`.
    pub fraction: f64,
}

/// Groups stratification fractions for a specific stratification dimension
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratificationFractions {
    /// Id of the stratification.
    pub stratification: String,
    /// One fraction per category; they sum to 1.0.
    pub fractions: Vec<StratificationFraction>,
}

/// Specifies an override fraction for a single expanded compartment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompartmentFraction {
    /// Name of the expanded compartment, such as `A_g1`.
    pub compartment: String,
    /// Fraction of the total population, applied after the uniform expansion
    pub fraction: f64,
}

/// How the compartments of a model start.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitialConditions {
    /// Total population at step 0.
    pub population_size: u64,
    /// Fraction of the population in each bin.
    pub bin_fractions: Vec<BinFraction>,
    /// Category fractions of each stratification.
    pub stratification_fractions: Vec<StratificationFractions>,
    /// Overrides of single expanded compartments, applied after the expansion.
    #[serde(default)]
    pub compartment_fractions: Vec<CompartmentFraction>,
}

/// The structure of a compartmental model: its bins, accumulators,
/// stratifications, transitions and initial conditions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompartmentalStructure {
    /// Bins, in declaration order.
    pub bins: Vec<Bin>,
    /// Accumulators, in declaration order.
    #[serde(default)]
    pub accumulators: Vec<Accumulator>,
    /// Stratifications, in declaration order. The order defines the expansion.
    pub stratifications: Vec<Stratification>,
    /// Transitions, in declaration order.
    pub transitions: Vec<Transition>,
    /// How the compartments start.
    pub initial_conditions: InitialConditions,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_conditions_without_compartment_fractions_deserialize() {
        let json = r#"{
            "population_size": 1000,
            "bin_fractions": [{"bin": "A", "fraction": 1.0}],
            "stratification_fractions": []
        }"#;
        let conditions: InitialConditions = serde_json::from_str(json).unwrap();

        assert!(conditions.compartment_fractions.is_empty());
    }

    #[test]
    fn compartment_fractions_round_trip() {
        let json = r#"{
            "population_size": 1000,
            "bin_fractions": [{"bin": "A", "fraction": 1.0}],
            "stratification_fractions": [],
            "compartment_fractions": [{"compartment": "A_g1", "fraction": 0.25}]
        }"#;
        let conditions: InitialConditions = serde_json::from_str(json).unwrap();

        assert_eq!(conditions.compartment_fractions.len(), 1);
        assert_eq!(conditions.compartment_fractions[0].compartment, "A_g1");
        assert_eq!(conditions.compartment_fractions[0].fraction, 0.25);
    }
}
