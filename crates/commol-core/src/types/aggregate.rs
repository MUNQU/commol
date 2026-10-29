//! The aggregate structure: amounts per compartment.
//!
//! An [`Aggregate`] lays a [`Process`] out as amounts: the bins expand over
//! the stratifications into compartments, each compartment holds an amount,
//! and the structure says how much starts where (an amount per bin, in the
//! unit of the bin, and the split of each bin among the categories) and who
//! the neighbors of a unit are ([`Neighbors`]).

use std::collections::BTreeMap;

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize, de};

use super::process::Process;
use crate::json::{self, Value};

/// The key of a split by category that names the earlier stratification it
/// differs by, as in `{"by": "group", "g1": {...}, "g2": {...}}`.
pub const FRACTIONS_BY: &str = "by";

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
    /// The split of each stratification, by stratification id: one split of
    /// the group it splits, or one split per category of an earlier
    /// stratification ([`StratificationFractions`]). Each fraction is from 0
    /// to 1, and each split sums to 1
    /// (`commol_model::rules::check_fractions`).
    #[serde(default)]
    pub stratification_fractions: IndexMap<String, StratificationFractions>,
    /// Amount of single expanded compartments, by compartment name, applied
    /// after the expansion.
    #[serde(default)]
    pub compartment_amounts: IndexMap<String, f64>,
}

/// How a stratification splits the amount of a compartment among its
/// categories.
///
/// The JSON of [`Self::Split`] is the fraction of each category, such as
/// `{"t1": 0.4, "t2": 0.6}`; the JSON of [`Self::ByCategory`] names the
/// earlier stratification under [`FRACTIONS_BY`] and gives one split per
/// category of it, such as `{"by": "group", "g1": {"t1": 0.25, "t2": 0.75},
/// "g2": {"t1": 0.5, "t2": 0.5}}`.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum StratificationFractions {
    /// One split for every group the stratification splits: the fraction of
    /// each category, by category.
    Split(IndexMap<String, f64>),
    /// One split per category of the stratification `by`, declared before
    /// the one it splits: the fraction of each category, by category of
    /// `by` and category.
    ByCategory {
        /// The id of the earlier stratification the split differs by.
        by: String,
        /// The split of each category of `by`, by category of `by`.
        #[serde(flatten)]
        splits: IndexMap<String, IndexMap<String, f64>>,
    },
}

impl StratificationFractions {
    /// The fraction of `category` in a compartment of the `(stratification,
    /// category)` pairs `categories`: the split of the category of `by` the
    /// compartment has, for a split by category. `None` when the split names
    /// no such category, or the compartment has no category of `by`.
    pub fn fraction(&self, category: &str, categories: &[(String, String)]) -> Option<f64> {
        let split = match self {
            Self::Split(split) => split,
            Self::ByCategory { by, splits } => categories
                .iter()
                .find(|(stratification, _)| stratification == by)
                .and_then(|(_, of)| splits.get(of))?,
        };
        split.get(category).copied()
    }
}

impl<'de> Deserialize<'de> for StratificationFractions {
    /// A split by category when the field [`FRACTIONS_BY`] of the object is
    /// a text, one split otherwise, so a category may be named `by`.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let Value::Map(mut fields) = Value::deserialize(deserializer)? else {
            return Err(de::Error::custom(
                "expected the fraction of each category, such as {\"t1\": 0.4, \"t2\": \
                 0.6}, or a split per category of an earlier stratification, such as \
                 {\"by\": \"group\", \"g1\": {\"t1\": 0.4, \"t2\": 0.6}, ...}",
            ));
        };
        let by = fields
            .iter()
            .enumerate()
            .find_map(|(position, (key, value))| match value {
                Value::String(by) if key == FRACTIONS_BY => Some((position, by.clone())),
                _ => None,
            });
        let Some((position, by)) = by else {
            return json::from_value(Value::Map(fields)).map(Self::Split);
        };
        fields.remove(position);
        let splits = json::from_value(Value::Map(fields))?;
        Ok(Self::ByCategory { by, splits })
    }
}

/// Who the neighbors of a unit are, for the rate variables `neighbors_<bin>`
/// and `neighbors`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
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
            json::write(&conditions.stratification_fractions["group"]),
            r#"{"g2":0.5,"g1":0.5}"#
        );
        assert_eq!(conditions.compartment_amounts["B_g1"], 250.0);
    }

    #[test]
    fn a_split_by_category_reads_and_writes_back_with_its_stratification() {
        let text = r#"{"by":"group","g1":{"t1":0.25,"t2":0.75},"g2":{"t1":0.5,"t2":0.5}}"#;

        let fractions: StratificationFractions = json::read(text).unwrap();

        let pairs = |group: &str| vec![("group".to_string(), group.to_string())];
        assert_eq!(fractions.fraction("t2", &pairs("g1")), Some(0.75));
        assert_eq!(fractions.fraction("t1", &pairs("g2")), Some(0.5));
        assert_eq!(fractions.fraction("t1", &[]), None);
        assert_eq!(json::write(&fractions), text);
    }

    #[test]
    fn a_category_may_be_named_by() {
        let fractions: StratificationFractions = json::read(r#"{"by":0.25,"other":0.75}"#).unwrap();

        assert_eq!(fractions.fraction("by", &[]), Some(0.25));
        let error = json::read::<StratificationFractions>("[1]").unwrap_err();
        assert!(
            error.starts_with("expected the fraction of each category"),
            "{error}"
        );
    }

    #[test]
    fn a_split_reads_a_fraction_that_is_not_finite() {
        let fractions: StratificationFractions = json::read(r#"{"t1":"NaN","t2":0.5}"#).unwrap();

        assert!(fractions.fraction("t1", &[]).unwrap().is_nan());
        assert!(matches!(fractions, StratificationFractions::Split(_)));
    }
}
