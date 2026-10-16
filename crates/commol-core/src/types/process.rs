//! The process of a model: what changes and how, written once for every
//! structure.
//!
//! A [`Process`] holds the bins, the accumulators, the stratifications and
//! the transitions. The structure of the model
//! ([`ModelStructure`](super::ModelStructure)) says how the process is laid
//! out, such as amounts per compartment ([`Aggregate`](super::Aggregate)).

use serde::{Deserialize, Serialize};

use super::transitions::{StratificationCondition, Transition};

/// Prefix of the rate variables that read the neighbors of a unit:
/// `neighbors_<bin>` is the amount of a bin among them.
pub const NEIGHBORS_PREFIX: &str = "neighbors_";

/// The rate variable that reads the total of the neighbors of a unit.
pub const NEIGHBORS: &str = "neighbors";

/// A state of the model. Stratifications expand it into compartments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bin {
    /// Identifier used in rates, initial conditions and outputs.
    pub id: String,
    /// Display name.
    pub name: String,
}

/// A cumulative event counter tracked by the engine but excluded from the total.
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

/// What changes in a model and how: bins, accumulators, stratifications and
/// transitions, written once for every structure.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Process {
    /// Bins, in declaration order.
    #[serde(default)]
    pub bins: Vec<Bin>,
    /// Accumulators, in declaration order.
    #[serde(default)]
    pub accumulators: Vec<Accumulator>,
    /// Stratifications, in declaration order. The order defines the expansion.
    #[serde(default)]
    pub stratifications: Vec<Stratification>,
    /// Transitions, in declaration order.
    #[serde(default)]
    pub transitions: Vec<Transition>,
}

impl Process {
    /// Whether the process holds nothing.
    pub fn is_empty(&self) -> bool {
        self.bins.is_empty()
            && self.accumulators.is_empty()
            && self.stratifications.is_empty()
            && self.transitions.is_empty()
    }
}
