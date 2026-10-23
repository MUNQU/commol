//! Parser for converting evalexpr AST to our JIT AST
//!
//! This module bridges evalexpr's parser with our JIT compiler by converting
//! evalexpr's AST into our simplified AST representation.

use super::ast::{BinaryOperator, Expr, UnaryOperator};
use crate::math_expression::error::MathExpressionError;
use evalexpr::{Node, Operator};

/// Macro to reduce boilerplate for binary operators
macro_rules! binary_op {
    ($node:expr, $binary_op:expr) => {{
        let children = $node.children();
        if children.len() != 2 {
            return Err(MathExpressionError::InvalidExpression(format!(
                "the operator '{}' needs two operands",
                $binary_op
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
    ($node:expr, $unary_op:expr) => {{
        let children = $node.children();
        if children.len() != 1 {
            return Err(MathExpressionError::InvalidExpression(format!(
                "the operator '{}' needs one operand",
                $unary_op
            )));
        }
        Ok(Expr::unary($unary_op, convert_node(&children[0])?))
    }};
}

/// Parse a preprocessed formula string into our AST
pub fn parse_expression(preprocessed: &str) -> Result<Expr, MathExpressionError> {
    // Use evalexpr to parse the expression into its AST
    let evalexpr_tree =
        evalexpr::build_operator_tree(preprocessed).map_err(MathExpressionError::ParseError)?;

    // Convert evalexpr's AST to our AST
    convert_node(&evalexpr_tree)
}

/// Convert an evalexpr Node to our Expr
fn convert_node(node: &Node) -> Result<Expr, MathExpressionError> {
    match node.operator() {
        // A root holds the formula or the inside of a pair of parentheses.
        Operator::RootNode => match node.children() {
            [child] => convert_node(child),
            _ => Err(MathExpressionError::InvalidExpression(
                "the formula, or a pair of parentheses in it, holds nothing".to_string(),
            )),
        },

        // Constants
        Operator::Const { value } => match value {
            evalexpr::Value::Float(f) => Ok(Expr::Constant(*f)),
            evalexpr::Value::Int(i) => Ok(Expr::Constant(*i as f64)),
            evalexpr::Value::Boolean(b) => Err(MathExpressionError::InvalidExpression(format!(
                "the boolean '{b}' is not a number"
            ))),
            _ => Err(MathExpressionError::InvalidExpression(
                "Unsupported constant type".to_string(),
            )),
        },

        // Variables
        Operator::VariableIdentifierRead { identifier } => Ok(Expr::Variable(identifier.clone())),

        // Binary operations - Arithmetic
        Operator::Add => binary_op!(node, BinaryOperator::Add),
        Operator::Sub => binary_op!(node, BinaryOperator::Sub),
        Operator::Mul => binary_op!(node, BinaryOperator::Mul),
        Operator::Div => binary_op!(node, BinaryOperator::Div),
        Operator::Mod => binary_op!(node, BinaryOperator::Mod),
        Operator::Exp => binary_op!(node, BinaryOperator::Pow),

        // Binary operations - Comparison
        Operator::Lt => binary_op!(node, BinaryOperator::Lt),
        Operator::Gt => binary_op!(node, BinaryOperator::Gt),
        Operator::Leq => binary_op!(node, BinaryOperator::Le),
        Operator::Geq => binary_op!(node, BinaryOperator::Ge),
        Operator::Eq => binary_op!(node, BinaryOperator::Eq),
        Operator::Neq => binary_op!(node, BinaryOperator::Ne),

        // Binary operations - Logical
        Operator::And => binary_op!(node, BinaryOperator::And),
        Operator::Or => binary_op!(node, BinaryOperator::Or),

        // Unary operations
        Operator::Neg => unary_op!(node, UnaryOperator::Neg),
        Operator::Not => unary_op!(node, UnaryOperator::Not),

        // Function calls: evalexpr gives a call one child, a root that holds
        // a tuple of the arguments, the one argument or nothing.
        Operator::FunctionIdentifier { identifier } => {
            let [arguments] = node.children() else {
                // coverage: invariant, evalexpr gives a call one child
                unreachable!("evalexpr gives a call one child")
            };
            let arg_exprs: Vec<Expr> = match arguments.children() {
                [] if matches!(arguments.operator(), Operator::RootNode) => Vec::new(),
                [tuple] if matches!(tuple.operator(), Operator::Tuple) => tuple
                    .children()
                    .iter()
                    .map(convert_node)
                    .collect::<Result<Vec<_>, _>>()?,
                _ => vec![convert_node(arguments)?],
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

        // A tuple outside the arguments of a call: evalexpr keeps a single
        // value in parentheses as a root, so a tuple has two values or more.
        Operator::Tuple => Err(MathExpressionError::InvalidExpression(format!(
            "Unexpected tuple with {} elements in expression",
            node.children().len()
        ))),

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
        let expr = parse_expression("k1").unwrap();
        assert!(matches!(expr, Expr::Variable(_)));
    }

    #[test]
    fn test_parse_binary_op() {
        let expr = parse_expression("k1 * k2").unwrap();
        assert!(matches!(
            expr,
            Expr::BinaryOp {
                op: BinaryOperator::Mul,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_complex_expression() {
        let expr = parse_expression("k1 * A * B / total").unwrap();
        // Should parse as ((k1 * A) * B) / total
        assert!(matches!(expr, Expr::BinaryOp { .. }));
    }

    #[test]
    fn test_parse_function_call() {
        let expr = parse_expression("sin(x)").unwrap();
        assert!(
            matches!(&expr, Expr::FunctionCall { name, args } if name == "sin" && args.len() == 1)
        );
    }

    #[test]
    fn a_parsed_expression_prints_as_a_formula_that_reads_back_as_itself() {
        for formula in [
            "x + 1",
            "x - 1",
            "x * 2",
            "x / 2",
            "x % 2",
            "x ^ 2",
            "x < 1",
            "x > 1",
            "x <= 1",
            "x >= 1",
            "x == 1",
            "x != 1",
            "x && y",
            "x || y",
            "-x",
            "!x",
            "if(x, 1, 2)",
            "max(x, y, 2)",
            "sin(x)",
        ] {
            let printed = parse_expression(formula).unwrap().to_string();
            let again = parse_expression(&printed).unwrap().to_string();
            assert_eq!(again, printed, "{formula}");
        }
        assert_eq!(
            parse_expression("max(x, -y) <= 1").unwrap().to_string(),
            "(max(x, (-y)) <= 1)"
        );
    }

    #[test]
    fn a_call_without_arguments_has_none() {
        assert_eq!(parse_expression("f()").unwrap(), Expr::call("f", vec![]));
    }

    #[test]
    fn what_the_jit_does_not_compile_is_an_error_that_says_why() {
        for (formula, reason) in [
            ("\"a\"", "Unsupported constant type"),
            ("a = 1", "Unsupported operator: Assign"),
            ("1; 2", "Unsupported operator: Chain"),
            ("1, 2", "Unexpected tuple with 2 elements in expression"),
            ("if(1, 2)", "if requires exactly 3 arguments, got 2"),
            ("if(true, 1, 2)", "the boolean 'true' is not a number"),
        ] {
            assert_eq!(
                parse_expression(formula).unwrap_err().to_string(),
                format!("Invalid expression: {reason}"),
                "{formula}"
            );
        }
    }
}
