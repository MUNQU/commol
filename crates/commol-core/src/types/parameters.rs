use serde::{Deserialize, Serialize};

/// Prefixes used to identify variable types in expressions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VariablePrefixes {
    #[serde(rename = "state")]
    State,
    #[serde(rename = "strat")]
    Strat,
}

/// Represents different types of parameter values
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParameterValue {
    /// Constant numeric value
    Constant(f64),
    /// Formula expression that can reference other parameters or special variables
    Formula(String),
}

impl ParameterValue {
    /// Check if this is a constant value
    pub fn is_constant(&self) -> bool {
        matches!(self, ParameterValue::Constant(_))
    }

    /// Check if this is a formula
    pub fn is_formula(&self) -> bool {
        matches!(self, ParameterValue::Formula(_))
    }

    /// Get the constant value if this is a constant, otherwise None
    pub fn as_constant(&self) -> Option<f64> {
        match self {
            ParameterValue::Constant(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the formula string if this is a formula, otherwise None
    pub fn as_formula(&self) -> Option<&str> {
        match self {
            ParameterValue::Formula(s) => Some(s),
            _ => None,
        }
    }
}

impl From<f64> for ParameterValue {
    fn from(value: f64) -> Self {
        ParameterValue::Constant(value)
    }
}

impl From<String> for ParameterValue {
    fn from(value: String) -> Self {
        ParameterValue::Formula(value)
    }
}

impl From<&str> for ParameterValue {
    fn from(value: &str) -> Self {
        ParameterValue::Formula(value.to_string())
    }
}

/// A parameter definition with its value and optional description
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parameter {
    pub id: String,
    pub value: ParameterValue,
    pub description: Option<String>,
}

impl Parameter {
    /// Create a new parameter with a constant value
    pub fn new_constant(id: String, value: f64, description: Option<String>) -> Self {
        Self {
            id,
            value: ParameterValue::Constant(value),
            description,
        }
    }

    /// Create a new parameter with a formula
    pub fn new_formula(id: String, formula: String, description: Option<String>) -> Self {
        Self {
            id,
            value: ParameterValue::Formula(formula),
            description,
        }
    }

    /// Check if this parameter has a constant value
    pub fn is_constant(&self) -> bool {
        self.value.is_constant()
    }

    /// Check if this parameter has a formula
    pub fn is_formula(&self) -> bool {
        self.value.is_formula()
    }
}
