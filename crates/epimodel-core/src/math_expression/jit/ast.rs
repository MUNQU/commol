//! Abstract Syntax Tree for mathematical expressions
//!
//! This module defines the AST representation used for JIT compilation.

use std::fmt;

/// Mathematical expression AST node
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Constant floating-point value
    Constant(f64),

    /// Variable reference (parameter or compartment name)
    Variable(String),

    /// Binary operation (e.g., a + b, x * y)
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Unary operation (e.g., -x, !condition)
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expr>,
    },

    /// Function call (e.g., sin(x), pow(x, 2))
    FunctionCall { name: String, args: Vec<Expr> },

    /// Conditional expression: if(condition, true_expr, false_expr)
    Conditional {
        condition: Box<Expr>,
        true_expr: Box<Expr>,
        false_expr: Box<Expr>,
    },
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,

    // Comparison
    Lt, // <
    Gt, // >
    Le, // <=
    Ge, // >=
    Eq, // ==
    Ne, // !=

    // Logical
    And, // &&
    Or,  // ||
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    /// Negation: -x
    Neg,

    /// Logical NOT: !x
    Not,
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Constant(val) => write!(f, "{}", val),
            Expr::Variable(name) => write!(f, "{}", name),
            Expr::BinaryOp { op, left, right } => {
                write!(f, "({} {} {})", left, op, right)
            }
            Expr::UnaryOp { op, operand } => {
                write!(f, "({}{})", op, operand)
            }
            Expr::FunctionCall { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Expr::Conditional {
                condition,
                true_expr,
                false_expr,
            } => {
                write!(f, "if({}, {}, {})", condition, true_expr, false_expr)
            }
        }
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BinaryOperator::Add => "+",
            BinaryOperator::Sub => "-",
            BinaryOperator::Mul => "*",
            BinaryOperator::Div => "/",
            BinaryOperator::Mod => "%",
            BinaryOperator::Pow => "^",
            BinaryOperator::Lt => "<",
            BinaryOperator::Gt => ">",
            BinaryOperator::Le => "<=",
            BinaryOperator::Ge => ">=",
            BinaryOperator::Eq => "==",
            BinaryOperator::Ne => "!=",
            BinaryOperator::And => "&&",
            BinaryOperator::Or => "||",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            UnaryOperator::Neg => "-",
            UnaryOperator::Not => "!",
        };
        write!(f, "{}", s)
    }
}

impl Expr {
    /// Helper to create a binary operation
    pub fn binary(op: BinaryOperator, left: Expr, right: Expr) -> Self {
        Expr::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Helper to create a unary operation
    pub fn unary(op: UnaryOperator, operand: Expr) -> Self {
        Expr::UnaryOp {
            op,
            operand: Box::new(operand),
        }
    }

    /// Helper to create a function call
    pub fn call(name: impl Into<String>, args: Vec<Expr>) -> Self {
        Expr::FunctionCall {
            name: name.into(),
            args,
        }
    }

    /// Helper to create a conditional
    pub fn if_then_else(condition: Expr, true_expr: Expr, false_expr: Expr) -> Self {
        Expr::Conditional {
            condition: Box::new(condition),
            true_expr: Box::new(true_expr),
            false_expr: Box::new(false_expr),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_display() {
        let expr = Expr::binary(
            BinaryOperator::Mul,
            Expr::Variable("beta".to_string()),
            Expr::Constant(2.0),
        );
        assert_eq!(expr.to_string(), "(beta * 2)");
    }

    #[test]
    fn test_function_call() {
        let expr = Expr::call("sin", vec![Expr::Variable("x".to_string())]);
        assert_eq!(expr.to_string(), "sin(x)");
    }
}
