//! The tree of a formula as JSON: the port through which an edge reads the
//! tree that the parser reads ([`Expr`](super::jit::ast::Expr)), so that no
//! edge reads the text of a formula.
//!
//! The tree is the one the JIT compiles, with the grouping of the parser: a
//! chain of one level applies from the left, and a chain of powers groups
//! from the right. Each node is an object with one key, the kind of the
//! node:
//!
//! | Node | JSON |
//! |---|---|
//! | A number | `{"Constant": 0.5}`, the value the parser read; a value that is not finite, such as `1e400`, is its text: `{"Constant": "Infinity"}`, as the JSON of the port writes it |
//! | A variable | `{"Variable": "k"}` |
//! | A chain | `{"Chain": {"first": node, "rest": [["Mul", node], ...]}}`, with the operators `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Lt`, `Gt`, `Le`, `Ge`, `Eq` and `Ne` |
//! | A power | `{"Power": {"base": node, "exponent": node}}` |
//! | A sign or a negation | `{"UnaryOp": {"op": "Neg", "operand": node}}`; `"Not"` for `!` |
//! | A call | `{"FunctionCall": {"name": "sin", "args": [node, ...]}}` |
//! | `if` | `{"Conditional": {"condition": node, "true_expr": node, "false_expr": node}}` |
//! | `matrix_sum` | `{"MatrixSum": {"matrix": "M", "expression": node}}`: the id of the matrix is not a variable |

use serde::Serializer;

use super::{MathExpression, MathExpressionError};

impl MathExpression {
    /// The tree of the formula as JSON, in the shape of this module.
    ///
    /// # Errors
    ///
    /// Returns the error of [`MathExpression::check`] when the formula is
    /// not in the language of formulas.
    pub fn tree_json(&self) -> Result<String, MathExpressionError> {
        self.check()?;
        let tree = self.tree()?;
        Ok(serde_json::to_string(&tree).expect("the tree of a formula is JSON"))
    }
}

/// The value of a number of the tree: a JSON number when it is finite, and
/// its text (`"Infinity"`) when it is not, which JSON has no number for, as
/// the JSON of the port writes it ([`crate::json::non_finite_text`]).
pub(crate) fn number<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if value.is_finite() {
        serializer.serialize_f64(*value)
    } else {
        serializer.serialize_str(crate::json::non_finite_text(*value))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn tree(formula: &str) -> Value {
        let json = MathExpression::new(formula.to_string())
            .tree_json()
            .unwrap();
        serde_json::from_str(&json).unwrap()
    }

    fn variable(name: &str) -> Value {
        json!({ "Variable": name })
    }

    #[test]
    fn the_tree_keeps_the_grouping_of_the_parser() {
        let (a, b, c) = (variable("a"), variable("b"), variable("c"));

        assert_eq!(
            tree("a - b - c"),
            json!({ "Chain": { "first": a, "rest": [["Sub", b], ["Sub", c]] } })
        );
        assert_eq!(
            tree("a ^ b ^ c"),
            json!({ "Power": { "base": a, "exponent":
                { "Power": { "base": b, "exponent": c } } } })
        );
        assert_eq!(
            tree("(a - b) * -2.5"),
            json!({ "Chain": {
                "first": { "Chain": { "first": a, "rest": [["Sub", b]] } },
                "rest": [["Mul", { "UnaryOp": { "op": "Neg", "operand": { "Constant": 2.5 } } }]]
            } })
        );
        assert_eq!(
            tree("if(!(a > 1e400), max(a, 0x10), c)"),
            json!({ "Conditional": {
                "condition": { "UnaryOp": { "op": "Not", "operand":
                    { "Chain": { "first": a, "rest": [["Gt", { "Constant": "Infinity" }]] } } } },
                "true_expr": { "FunctionCall": { "name": "max",
                    "args": [a, { "Constant": 16.0 }] } },
                "false_expr": c
            } })
        );
    }

    #[test]
    fn a_matrix_sum_names_its_matrix_apart_from_its_variables() {
        assert_eq!(
            tree("matrix_sum(M, b)"),
            json!({ "MatrixSum": { "matrix": "M", "expression": variable("b") } })
        );
    }

    #[test]
    fn a_formula_outside_the_language_has_no_tree() {
        let error = MathExpression::new("a && b".to_string())
            .tree_json()
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            MathExpression::new("a && b".to_string())
                .check()
                .unwrap_err()
                .to_string()
        );
    }
}
