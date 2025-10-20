use serde::{Deserialize, Serialize};

/// Logic operators for conditional rules and expressions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LogicOperator {
    #[serde(rename = "and")]
    And,
    #[serde(rename = "or")]
    Or,
    #[serde(rename = "eq")]
    Eq,
    #[serde(rename = "neq")]
    Neq,
    #[serde(rename = "gt")]
    Gt,
    #[serde(rename = "get")]
    Get,
    #[serde(rename = "lt")]
    Lt,
    #[serde(rename = "let")]
    Let,
}

/// Value type that can be used in conditional rules
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RuleValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// A rule that compares a variable to a value using a logic operator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    pub variable: String,
    pub operator: LogicOperator,
    pub value: RuleValue,
}

/// A condition composed of multiple rules combined with a logic operator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Condition {
    pub logic: LogicOperator,
    pub rules: Vec<Rule>,
}
