//! Error types for mathematical expression evaluation

use evalexpr::EvalexprError;

/// Errors that can occur during expression evaluation
#[derive(Debug, thiserror::Error)]
pub enum MathExpressionError {
    /// evalexpr could not evaluate the expression.
    #[error("Evaluation error: {0}")]
    EvalError(#[from] EvalexprError),
    /// The expression names a variable with no value.
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    /// The expression cannot be parsed or compiled.
    #[error("Invalid expression: {0}")]
    InvalidExpression(String),
}
