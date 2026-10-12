use serde::{Deserialize, Serialize};

/// Operators of a transition `Condition`: how rules combine, and how a rule compares.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LogicOperator {
    /// Every rule holds.
    #[serde(rename = "and")]
    And,
    /// At least one rule holds.
    #[serde(rename = "or")]
    Or,
    /// Equal to.
    #[serde(rename = "eq")]
    Eq,
    /// Not equal to.
    #[serde(rename = "neq")]
    Neq,
    /// Greater than.
    #[serde(rename = "gt")]
    Gt,
    /// Greater than or equal to.
    #[serde(rename = "get")]
    Get,
    /// Less than.
    #[serde(rename = "lt")]
    Lt,
    /// Less than or equal to.
    #[serde(rename = "let")]
    Let,
}

/// Value type that can be used in conditional rules
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RuleValue {
    /// A text value.
    String(String),
    /// An integer value.
    Int(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
}

/// A rule that compares a variable to a value using a logic operator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    /// The variable, as `state:<bin>` or `strat:<stratification>`.
    pub variable: String,
    /// The comparison.
    pub operator: LogicOperator,
    /// The value compared with.
    pub value: RuleValue,
}

/// Rules combined with `and` or `or`. Stored with a transition; the engine does not evaluate it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Condition {
    /// How the rules combine: `And` or `Or`.
    pub logic: LogicOperator,
    /// The rules.
    pub rules: Vec<Rule>,
}
