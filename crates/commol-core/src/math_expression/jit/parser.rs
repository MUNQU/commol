//! Parser for converting evalexpr AST to our JIT AST
//!
//! This module bridges evalexpr's parser with our JIT compiler by converting
//! evalexpr's AST into our simplified AST representation.

use super::ast::{BinaryOperator, Expr, UnaryOperator};
use crate::math_expression::error::MathExpressionError;
use evalexpr::{Node, Operator};

/// Macro to reduce boilerplate for binary operators
macro_rules! binary_op {
    ($node:expr, $op_name:expr, $binary_op:expr) => {{
        let children = $node.children();
        if children.len() != 2 {
            return Err(MathExpressionError::InvalidExpression(format!(
                "{} requires exactly 2 operands",
                $op_name
            )));
        }
        Ok(Expr::binary(
            $binary_op,
            convert_node(&children[0])?,
            convert_node(&children[1])?,
        ))
    }};
}

/// Macro to reduce boilerplate for unary operators
macro_rules! unary_op {
    ($node:expr, $op_name:expr, $unary_op:expr) => {{
        let children = $node.children();
        if children.len() != 1 {
            return Err(MathExpressionError::InvalidExpression(format!(
                "{} requires exactly 1 operand",
                $op_name
            )));
        }
        Ok(Expr::unary($unary_op, convert_node(&children[0])?))
    }};
}

/// Parse a preprocessed formula string into our AST
pub fn parse_expression(preprocessed: &str) -> Result<Expr, MathExpressionError> {
    // Use evalexpr to parse the expression into its AST
    let evalexpr_tree =
        evalexpr::build_operator_tree(preprocessed).map_err(MathExpressionError::EvalError)?;

    // Convert evalexpr's AST to our AST
    convert_node(&evalexpr_tree)
}

/// Convert an evalexpr Node to our Expr
fn convert_node(node: &Node) -> Result<Expr, MathExpressionError> {
    match node.operator() {
        // RootNode - just unwrap and convert the child
        Operator::RootNode => {
            let children = node.children();
            if children.len() == 1 {
                convert_node(&children[0])
            } else {
                Err(MathExpressionError::InvalidExpression(
                    "RootNode should have exactly 1 child".to_string(),
                ))
            }
        }

        // Constants
        Operator::Const { value } => match value {
            evalexpr::Value::Float(f) => Ok(Expr::Constant(*f)),
            evalexpr::Value::Int(i) => Ok(Expr::Constant(*i as f64)),
            evalexpr::Value::Boolean(b) => Ok(Expr::Constant(if *b { 1.0 } else { 0.0 })),
            _ => Err(MathExpressionError::InvalidExpression(
                "Unsupported constant type".to_string(),
            )),
        },

        // Variables
        Operator::VariableIdentifierRead { identifier } => Ok(Expr::Variable(identifier.clone())),

        // Binary operations - Arithmetic
        Operator::Add => binary_op!(node, "Add", BinaryOperator::Add),
        Operator::Sub => binary_op!(node, "Sub", BinaryOperator::Sub),
        Operator::Mul => binary_op!(node, "Mul", BinaryOperator::Mul),
        Operator::Div => binary_op!(node, "Div", BinaryOperator::Div),
        Operator::Mod => binary_op!(node, "Mod", BinaryOperator::Mod),
        Operator::Exp => binary_op!(node, "Exp", BinaryOperator::Pow),

        // Binary operations - Comparison
        Operator::Lt => binary_op!(node, "Lt", BinaryOperator::Lt),
        Operator::Gt => binary_op!(node, "Gt", BinaryOperator::Gt),
        Operator::Leq => binary_op!(node, "Leq", BinaryOperator::Le),
        Operator::Geq => binary_op!(node, "Geq", BinaryOperator::Ge),
        Operator::Eq => binary_op!(node, "Eq", BinaryOperator::Eq),
        Operator::Neq => binary_op!(node, "Neq", BinaryOperator::Ne),

        // Binary operations - Logical
        Operator::And => binary_op!(node, "And", BinaryOperator::And),
        Operator::Or => binary_op!(node, "Or", BinaryOperator::Or),

        // Unary operations
        Operator::Neg => unary_op!(node, "Neg", UnaryOperator::Neg),
        Operator::Not => unary_op!(node, "Not", UnaryOperator::Not),

        // Function calls
        Operator::FunctionIdentifier { identifier } => {
            let children = node.children();

            // Extract arguments - they might be wrapped in a Tuple
            // evalexpr represents multi-argument functions as: FunctionIdentifier -> RootNode -> Tuple -> [args]
            let arg_exprs: Vec<Expr> = if children.len() == 1 {
                // Check if the child is a RootNode wrapping a Tuple
                let child = &children[0];
                if matches!(child.operator(), Operator::RootNode) && child.children().len() == 1 {
                    let grandchild = &child.children()[0];
                    if matches!(grandchild.operator(), Operator::Tuple) {
                        // Extract tuple elements and convert each
                        grandchild
                            .children()
                            .iter()
                            .map(convert_node)
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        // Single argument, not in a tuple
                        vec![convert_node(child)?]
                    }
                } else if matches!(child.operator(), Operator::Tuple) {
                    // Direct tuple (less common, but handle it)
                    child
                        .children()
                        .iter()
                        .map(convert_node)
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    // Single argument, not in a tuple
                    vec![convert_node(child)?]
                }
            } else if children.is_empty() {
                // No arguments
                Vec::new()
            } else {
                // Multiple direct children - convert each one
                children
                    .iter()
                    .map(convert_node)
                    .collect::<Result<Vec<_>, _>>()?
            };

            // Special case for 'if' function (conditional)
            if identifier == "if" {
                if arg_exprs.len() != 3 {
                    return Err(MathExpressionError::InvalidExpression(format!(
                        "if requires exactly 3 arguments, got {}",
                        arg_exprs.len()
                    )));
                }
                return Ok(Expr::if_then_else(
                    arg_exprs[0].clone(),
                    arg_exprs[1].clone(),
                    arg_exprs[2].clone(),
                ));
            }

            // Regular function call - args are already converted Exprs
            Ok(Expr::call(identifier.clone(), arg_exprs))
        }

        // Tuple (used for function arguments)
        // Tuples should only appear within function calls, but if we encounter one
        // at the top level, try to handle it gracefully
        Operator::Tuple => {
            let children = node.children();
            if children.len() == 1 {
                // Single-element tuple, unwrap it
                convert_node(&children[0])
            } else {
                Err(MathExpressionError::InvalidExpression(format!(
                    "Unexpected tuple with {} elements in expression",
                    children.len()
                )))
            }
        }

        // Chain (sequence of expressions - take the last one)
        Operator::Chain => {
            let children = node.children();
            if children.is_empty() {
                return Err(MathExpressionError::InvalidExpression(
                    "Empty chain".to_string(),
                ));
            }
            convert_node(&children[children.len() - 1])
        }

        // Unsupported operators
        _ => Err(MathExpressionError::InvalidExpression(format!(
            "Unsupported operator: {:?}",
            node.operator()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_constant() {
        let expr = parse_expression("3.14").unwrap();
        assert!(matches!(expr, Expr::Constant(_)));
    }

    #[test]
    fn test_parse_variable() {
        let expr = parse_expression("beta").unwrap();
        assert!(matches!(expr, Expr::Variable(_)));
    }

    #[test]
    fn test_parse_binary_op() {
        let expr = parse_expression("beta * gamma").unwrap();
        if let Expr::BinaryOp { op, .. } = expr {
            assert_eq!(op, BinaryOperator::Mul);
        } else {
            panic!("Expected BinaryOp");
        }
    }

    #[test]
    fn test_parse_complex_expression() {
        let expr = parse_expression("beta * S * I / N").unwrap();
        // Should parse as ((beta * S) * I) / N
        assert!(matches!(expr, Expr::BinaryOp { .. }));
    }

    #[test]
    fn test_parse_function_call() {
        let expr = parse_expression("math::sin(x)").unwrap();
        if let Expr::FunctionCall { name, args } = expr {
            assert_eq!(name, "math::sin");
            assert_eq!(args.len(), 1);
        } else {
            panic!("Expected FunctionCall");
        }
    }
}
