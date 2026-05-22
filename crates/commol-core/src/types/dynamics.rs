use serde::{Deserialize, Serialize};

use super::conditions::Condition;
use crate::math_expression::RateMathExpression;

/// Supported model types for disease dynamics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelTypes {
    #[serde(rename = "DifferenceEquations")]
    DifferenceEquations,
}

/// Condition that specifies a stratification category for rate matching.
///
/// When `to` is set, the matched category in the source compartment is replaced
/// with the `to` category in the target compartment name. This enables
/// cross-category transitions within the same bin (e.g., aging: y60 → oe60).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratificationCondition {
    pub stratification: String,
    pub category: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}

/// Rate that applies to a specific stratification condition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratifiedRate {
    pub conditions: Vec<StratificationCondition>,
    pub rate: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub absolute: Option<bool>,
}

/// Transition between disease states with optional stratified rates and conditions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transition {
    pub id: String,
    pub source: Vec<String>,
    pub target: Vec<String>,
    pub rate: Option<RateMathExpression>,
    pub stratified_rates: Option<Vec<StratifiedRate>>,
    /// Conditional logic for when this transition should be active.
    /// Note: Currently not evaluated by the simulation engine but preserved for
    /// future functionality and backward compatibility with the Python API.
    #[allow(dead_code)]
    pub condition: Option<Condition>,
    /// When true, base compartment names in the rate expression are replaced
    /// with the specific stratified compartment name for each expanded flow.
    pub per_compartment: Option<bool>,
}

/// Model dynamics specification including type and transitions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dynamics {
    pub typology: ModelTypes,
    pub transitions: Vec<Transition>,
}
