//! Error types for mathematical expression evaluation

use evalexpr::EvalexprError;

/// Errors that can occur during expression evaluation
#[derive(Debug, thiserror::Error)]
pub enum MathExpressionError {
    /// The parser could not read the formula.
    #[error("it does not parse: {0}")]
    ParseError(#[from] EvalexprError),
    /// The expression names a variable with no value.
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    /// The expression cannot be parsed or compiled.
    #[error("Invalid expression: {0}")]
    InvalidExpression(String),
    /// The formula is not in the language of formulas
    /// ([`MathExpression::check`](crate::MathExpression::check)).
    #[error("Formula '{formula}': {reason}")]
    NotInLanguage {
        /// The formula, as it was written.
        formula: String,
        /// The first part that is not in the language, and why.
        reason: String,
    },
}
