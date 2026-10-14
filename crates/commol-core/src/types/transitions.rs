//! Transitions of a compartmental model and their stratified rates.

use serde::{Deserialize, Serialize};

use crate::math_expression::RateMathExpression;

/// Condition that specifies a stratification category for rate matching.
///
/// When `to` is set, the matched category in the source compartment is replaced
/// with the `to` category in the target compartment name. This enables
/// cross-category transitions within the same bin (for example `g1` → `g2`).
///
/// When `category` is `None`, the condition acts as a target-only override:
/// it does not filter source compartments but still contributes a `to` category
/// to the computed target compartment name. This is required when the source
/// compartment lacks a stratification that the target compartment must have
/// (e.g., an `A_cat0` compartment routing into an `A_cat1_sub0` compartment).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratificationCondition {
    /// Id of the stratification.
    pub stratification: String,
    /// Category the source compartment must have; `None` filters nothing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Category the target compartment gets instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}

/// Rate that applies to a specific stratification condition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratifiedRate {
    /// Conditions that must all match the source compartment.
    pub conditions: Vec<StratificationCondition>,
    /// The rate expression.
    pub rate: String,
    /// Forces the rate to be absolute (`true`) or per-capita (`false`); `None` infers it.
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
    pub rate: Option<RateMathExpression>,
    /// Rates for compartments with some categories.
    pub stratified_rates: Option<Vec<StratifiedRate>>,
    /// When true, base compartment names in the rate expression are replaced
    /// with the specific stratified compartment name for each expanded flow.
    pub per_compartment: Option<bool>,
    /// Positions on the axis where the transition is an impulse: at each one,
    /// the rate is a fraction (per-capita) or an amount (absolute) moved at
    /// once, and the transition moves nothing between them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impulse_at: Option<Vec<f64>>,
    /// `(period, offset)`: the transition is an impulse at `offset`,
    /// `offset + period`, `offset + 2 * period`, …
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impulse_every: Option<(f64, f64)>,
}

impl Transition {
    /// Whether the transition is an impulse rather than a continuous flow.
    pub fn is_impulse(&self) -> bool {
        self.impulse_at.is_some() || self.impulse_every.is_some()
    }
}
