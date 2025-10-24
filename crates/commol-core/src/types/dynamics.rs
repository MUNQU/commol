use serde::{Deserialize, Serialize};

use super::conditions::Condition;
use crate::math_expression::RateMathExpression;

/// Supported model types for disease dynamics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelTypes {
    #[serde(rename = "DifferenceEquations")]
    DifferenceEquations,
}

/// Condition that specifies a stratification category
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratificationCondition {
    pub stratification: String,
    pub category: String,
}

/// Rate that applies to a specific stratification condition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratifiedRate {
    pub conditions: Vec<StratificationCondition>,
    pub rate: String,
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
}

/// Model dynamics specification including type and transitions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dynamics {
    pub typology: ModelTypes,
    pub transitions: Vec<Transition>,
}
