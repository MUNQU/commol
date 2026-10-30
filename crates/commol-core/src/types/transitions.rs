//! Transitions of a process and their stratified rates.

use serde::{Deserialize, Serialize};

use super::process::Stratification;

/// Condition that specifies a stratification category for rate matching.
///
/// The condition holds for a compartment whose category in the
/// stratification is `category`, or one of `categories`.
///
/// When `to` is set, the matched category in the source compartment is replaced
/// with the `to` category in the target compartment name. This enables
/// cross-category transitions within the same bin (for example `g1` → `g2`).
///
/// When `category` and `categories` are `None`, the condition acts as a
/// target-only override: it does not filter source compartments but still
/// contributes a `to` category to the computed target compartment name. This is
/// required when the source compartment lacks a stratification that the target
/// compartment must have (e.g., an `A_cat0` compartment routing into an
/// `A_cat1_sub0` compartment).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratificationCondition {
    /// Id of the stratification.
    pub stratification: String,
    /// Category the source compartment must have; `None` filters nothing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Categories the source compartment must have one of; `None` filters
    /// nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub categories: Option<Vec<String>>,
    /// Category the target compartment gets instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}

impl StratificationCondition {
    /// Whether the condition filters compartments by their category.
    pub fn filters(&self) -> bool {
        self.category.is_some() || self.categories.is_some()
    }

    /// Whether a compartment whose category in the stratification of the
    /// condition is `category` (`None` when it has none) holds the condition.
    /// A condition that does not filter holds for every compartment.
    pub fn holds(&self, category: Option<&str>) -> bool {
        match (&self.category, &self.categories) {
            (Some(one), _) => category == Some(one.as_str()),
            (None, Some(set)) => category.is_some_and(|category| set.iter().any(|c| c == category)),
            (None, None) => true,
        }
    }

    /// Check the condition against the stratifications of its process.
    /// `owner` names what holds the condition, as a message starts.
    ///
    /// # Errors
    ///
    /// Returns a message that names `owner` and the stratification when the
    /// stratification is not one of `stratifications`, the condition gives
    /// both `category` and `categories`, an empty set, or `to` with a set,
    /// and that also names the category when the set names it twice or the
    /// stratification does not have it.
    pub fn check(&self, owner: &str, stratifications: &[Stratification]) -> Result<(), String> {
        let id = &self.stratification;
        let Some(stratification) = stratifications.iter().find(|s| &s.id == id) else {
            return Err(format!(
                "{owner}: a condition names '{id}', which is not a stratification of the process"
            ));
        };
        let set = self.categories.as_deref().unwrap_or_default();
        if let (Some(category), Some(_)) = (&self.category, &self.categories) {
            return Err(format!(
                "{owner}: the condition on '{id}' gives both the category '{category}' and \
                 the categories {set:?}; give one of them"
            ));
        }
        if self.categories.is_some() && set.is_empty() {
            return Err(format!(
                "{owner}: the condition on '{id}' has an empty set of categories"
            ));
        }
        if let (Some(to), Some(_)) = (&self.to, &self.categories) {
            return Err(format!(
                "{owner}: the condition on '{id}' moves to '{to}' from a set of categories; \
                 'to' takes the condition of one 'category'"
            ));
        }
        if let Some(twice) = set
            .iter()
            .enumerate()
            .find_map(|(index, category)| set[..index].contains(category).then_some(category))
        {
            return Err(format!(
                "{owner}: the condition on '{id}' names the category '{twice}' twice"
            ));
        }
        if let Some(unknown) = self
            .category
            .iter()
            .chain(set)
            .find(|category| !stratification.categories.contains(category))
        {
            return Err(format!(
                "{owner}: the condition on '{id}' names '{unknown}', which is not a category \
                 of '{id}' ({:?})",
                stratification.categories
            ));
        }
        Ok(())
    }
}

/// Rate that applies to a specific stratification condition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratifiedRate {
    /// Conditions that must all match the source compartment.
    pub conditions: Vec<StratificationCondition>,
    /// The rate expression.
    pub rate: String,
    /// Forces the rate to be absolute (`true`) or proportional (`false`); `None` infers it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub absolute: Option<bool>,
}

/// A flow between bins, or between the categories of one stratification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transition {
    /// Identifier.
    pub id: String,
    /// Bins (or one category) that lose the flow; empty for an inflow.
    pub source: Vec<String>,
    /// Bins (or one category) that gain the flow; empty for an outflow.
    pub target: Vec<String>,
    /// Ids of the accumulators that add up the flow.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accumulators: Vec<String>,
    /// The rate, or the fallback rate when `stratified_rates` is set.
    pub rate: Option<String>,
    /// Rates for compartments with some categories.
    pub stratified_rates: Option<Vec<StratifiedRate>>,
    /// When true, the id of each source and target bin in a rate is replaced
    /// with the compartment of each flow; any other bin keeps its total. A
    /// transition between categories ignores it.
    pub per_compartment: Option<bool>,
    /// Positions on the axis where the transition is an impulse: at each one,
    /// the rate is a fraction (proportional) or an amount (absolute) moved at
    /// once, and the transition moves nothing between them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impulse_at: Option<Vec<f64>>,
    /// `(period, offset)`: the transition is an impulse at `offset`,
    /// `offset + period`, `offset + 2 * period`, …
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impulse_every: Option<(f64, f64)>,
}
