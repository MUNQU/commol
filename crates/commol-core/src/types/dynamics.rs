use serde::{Deserialize, Serialize};

use super::conditions::Condition;
use crate::math_expression::RateMathExpression;

/// Model typologies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelTypes {
    /// Discrete time: each step applies every flow once.
    #[serde(rename = "DifferenceEquations")]
    DifferenceEquations,
    /// Defined for model definitions; no engine solves it yet.
    #[serde(rename = "DifferentialEquations")]
    DifferentialEquations,
}

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
    /// A condition stored with the transition. The engine does not evaluate it.
    #[allow(dead_code)]
    pub condition: Option<Condition>,
    /// When true, base compartment names in the rate expression are replaced
    /// with the specific stratified compartment name for each expanded flow.
    pub per_compartment: Option<bool>,
}

/// Model dynamics specification including type and transitions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dynamics {
    /// The typology of the model.
    pub typology: ModelTypes,
    /// The transitions the engine plans and simulates.
    pub transitions: Vec<Transition>,
}
