//! The aggregate structure: amounts per compartment.
//!
//! An [`Aggregate`] lays a [`Process`] out as amounts: the bins expand over
//! the stratifications into compartments, each compartment holds an amount,
//! and the structure says how much starts where (an amount per bin, in the
//! unit of the bin, and the split of each bin among the categories) and who
//! the neighbors of a unit are ([`Neighbors`]).

use std::collections::BTreeMap;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::process::Process;

/// How the compartments of a model start: the amount of each bin and the
/// split of it among the categories, each keyed by the id it applies to, in
/// the order the model declares them.
///
/// Every amount is in the unit of its bin, so bins of different units, such
/// as people and a concentration, start without a total over them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitialConditions {
    /// Amount of each bin, by bin id, finite and not negative; `None` needs
    /// calibration.
    pub bin_amounts: IndexMap<String, Option<f64>>,
    /// Fraction of each category of the group a stratification splits, by
    /// stratification id and category: each from 0 to 1, and a sum of 1
    /// (`commol_model::rules::check_fractions`).
    #[serde(default)]
    pub stratification_fractions: IndexMap<String, IndexMap<String, f64>>,
    /// Amount of single expanded compartments, by compartment name, applied
    /// after the expansion.
    #[serde(default)]
    pub compartment_amounts: IndexMap<String, f64>,
}

/// Who the neighbors of a unit are, for the rate variables `neighbors_<bin>`
/// and `neighbors`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Neighbors {
    /// Every unit is a neighbor of every other: `neighbors_<bin>` is the bin
    /// total and `neighbors` is `total`.
    Complete,
    /// Units are neighbors by the categories of one stratification. In a
    /// compartment of category `g`, `neighbors_<bin>` is
    /// `Σ_g' w[g][g'] × (amount of the bin in g')` and `neighbors` is
    /// `Σ_g' w[g][g'] × (amount of every bin in g')`.
    Groups {
        /// Id of the stratification whose categories group the units.
        stratification: String,
        /// `weights[g][g']`, the weight of category `g'` for a unit in `g`;
        /// `None` is the identity: a unit sees only its own category.
        #[serde(default)]
        weights: Option<BTreeMap<String, BTreeMap<String, f64>>>,
    },
}

impl Neighbors {
    /// The weight of category `other` for a unit in category `own`.
    pub fn weight(&self, own: &str, other: &str) -> f64 {
        match self {
            Self::Complete => 1.0,
            Self::Groups { weights: None, .. } => f64::from(u8::from(own == other)),
            Self::Groups {
                weights: Some(weights),
                ..
            } => weights
                .get(own)
                .and_then(|row| row.get(other))
                .copied()
                .unwrap_or(0.0),
        }
    }

    /// Check the neighbors against the process.
    ///
    /// # Errors
    ///
    /// Returns a message when `Groups` names no stratification of the
    /// process, or its weights are not one finite, non-negative number for
    /// every pair of categories of that stratification.
    pub fn check(&self, process: &Process) -> Result<(), String> {
        let Self::Groups {
            stratification,
            weights,
        } = self
        else {
            return Ok(());
        };
        let Some(found) = process
            .stratifications
            .iter()
            .find(|candidate| &candidate.id == stratification)
        else {
            let ids: Vec<&str> = process
                .stratifications
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect();
            return Err(format!(
                "The neighbors group by the stratification '{stratification}', which the \
                 process does not have; its stratifications are {ids:?}"
            ));
        };
        let Some(weights) = weights else {
            return Ok(());
        };
        let categories: Vec<&String> = found.categories.iter().collect();
        let same_keys = |keys: Vec<&String>| {
            let mut keys = keys;
            let mut expected = categories.clone();
            keys.sort();
            expected.sort();
            keys == expected
        };
        if !same_keys(weights.keys().collect()) {
            return Err(format!(
                "The weights of the neighbors need one row per category of '{stratification}' \
                 ({categories:?}); they have {:?}",
                weights.keys().collect::<Vec<_>>()
            ));
        }
        for (own, row) in weights {
            if !same_keys(row.keys().collect()) {
                return Err(format!(
                    "The weights of the neighbors in category '{own}' need one weight per \
                     category of '{stratification}' ({categories:?}); they have {:?}",
                    row.keys().collect::<Vec<_>>()
                ));
            }
            if let Some((other, weight)) = row
                .iter()
                .find(|(_, weight)| !(weight.is_finite() && **weight >= 0.0))
            {
                return Err(format!(
                    "The weight of category '{other}' for a unit in '{own}' must be a \
                     finite number of at least 0, got {weight}"
                ));
            }
        }
        Ok(())
    }
}

/// A narrower form of the aggregate structure: the same fields and engines,
/// with checks of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregateVariant {
    /// The variant of the tests: a process with no stratifications.
    #[cfg(feature = "test-kind")]
    Test,
}

impl AggregateVariant {
    /// Name of the variant, as the `variant` of a model structure.
    pub fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "test-kind")]
            Self::Test => "test",
        }
    }

    /// Check a process against the variant.
    ///
    /// # Errors
    ///
    /// Returns a message that names the variant and what it does not allow.
    #[cfg_attr(not(feature = "test-kind"), allow(unused_variables))]
    pub fn check(self, process: &Process) -> Result<(), String> {
        match self {
            #[cfg(feature = "test-kind")]
            Self::Test if !process.stratifications.is_empty() => Err(format!(
                "A {} aggregate structure takes no stratifications",
                self.name()
            )),
            #[cfg(feature = "test-kind")]
            Self::Test => Ok(()),
        }
    }
}

/// Amounts per compartment: how much starts where, and who the neighbors of
/// a unit are.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Aggregate {
    /// The variant of the structure, or `None` for the structure itself.
    #[serde(default)]
    pub variant: Option<AggregateVariant>,
    /// How the compartments start.
    #[serde(flatten)]
    pub initial_conditions: InitialConditions,
    /// Who the neighbors of a unit are; required when a rate reads
    /// `neighbors_<bin>` or `neighbors`.
    #[serde(default)]
    pub neighbors: Option<Neighbors>,
}

impl Aggregate {
    /// Check the structure against a process: its variant and its
    /// neighbors.
    ///
    /// # Errors
    ///
    /// Returns the message of the first check that fails.
    pub fn check(&self, process: &Process) -> Result<(), String> {
        if let Some(variant) = self.variant {
            variant.check(process)?;
        }
        if let Some(neighbors) = &self.neighbors {
            neighbors.check(process)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::process::Stratification;

    fn grouped_process() -> Process {
        Process {
            stratifications: vec![Stratification {
                id: "group".to_string(),
                categories: vec!["g1".to_string(), "g2".to_string()],
                conditions: None,
            }],
            ..Process::default()
        }
    }

    #[test]
    fn every_unit_weighs_one_for_every_other_under_complete_neighbors() {
        assert_eq!(Neighbors::Complete.weight("g1", "g2"), 1.0);
        assert_eq!(Neighbors::Complete.weight("g1", "g1"), 1.0);
    }

    #[test]
    fn the_weights_of_the_neighbors_have_one_row_per_category() {
        let neighbors = Neighbors::Groups {
            stratification: "group".to_string(),
            weights: Some(BTreeMap::from([(
                "g1".to_string(),
                BTreeMap::from([("g1".to_string(), 1.0), ("g2".to_string(), 0.5)]),
            )])),
        };

        let error = neighbors.check(&grouped_process()).unwrap_err();

        assert_eq!(
            error,
            "The weights of the neighbors need one row per category of 'group' \
             ([\"g1\", \"g2\"]); they have [\"g1\"]"
        );
    }

    #[test]
    fn initial_conditions_without_compartment_amounts_deserialize() {
        let json = r#"{"bin_amounts": {"A": 1000.0}}"#;
        let conditions: InitialConditions = serde_json::from_str(json).unwrap();

        assert!(conditions.compartment_amounts.is_empty());
        assert!(conditions.stratification_fractions.is_empty());
    }

    #[test]
    fn amounts_keep_their_order_and_uncalibrated_values() {
        let json = r#"{
            "bin_amounts": {"B": 750.0, "A": null},
            "stratification_fractions": {"group": {"g2": 0.5, "g1": 0.5}},
            "compartment_amounts": {"B_g1": 250.0}
        }"#;
        let conditions: InitialConditions = serde_json::from_str(json).unwrap();

        let bins: Vec<(&String, &Option<f64>)> = conditions.bin_amounts.iter().collect();
        assert_eq!(
            bins,
            [(&"B".to_string(), &Some(750.0)), (&"A".to_string(), &None)]
        );
        assert_eq!(
            conditions.stratification_fractions["group"]
                .keys()
                .collect::<Vec<_>>(),
            ["g2", "g1"]
        );
        assert_eq!(conditions.compartment_amounts["B_g1"], 250.0);
    }
}
