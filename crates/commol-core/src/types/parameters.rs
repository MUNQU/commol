use serde::{Deserialize, Serialize};

/// Prefixes used to identify variable types in expressions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VariablePrefixes {
    #[serde(rename = "state")]
    State,
    #[serde(rename = "strat")]
    Strat,
}

/// A parameter definition with its value and optional description
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parameter {
    pub id: String,
    pub value: f64,
    pub description: Option<String>,
}
