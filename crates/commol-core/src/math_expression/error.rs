//! Error types for mathematical expression evaluation

use evalexpr::EvalexprError;

/// Errors that can occur during expression evaluation
#[derive(Debug, thiserror::Error)]
pub enum MathExpressionError {
    #[error("Evaluation error: {0}")]
    EvalError(#[from] EvalexprError),
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    #[error("Invalid expression: {0}")]
    InvalidExpression(String),
}
