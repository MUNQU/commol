//! The tree of a formula, the one the check of the language reads and the
//! JIT compiles.
//!
//! A chain of operators of one level, such as `a + b - c`, is one
//! [`Expr::Chain`] node, so the depth of a tree does not grow with the terms
//! of a sum or the factors of a product, and every walk of a tree recurses
//! only as deep as the formula nests
//! ([`MAX_NESTING`](super::parser::MAX_NESTING)).
//!
//! The tree serializes to the JSON that
//! [`MathExpression::tree_json`](crate::MathExpression::tree_json) gives an
//! edge ([`tree`](crate::math_expression::tree) documents its shape).

use std::fmt;

use serde::Serialize;

use crate::math_expression::language::MATRIX_SUM;

/// A node of the tree of a formula.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Expr {
    /// A number.
    Constant(#[serde(serialize_with = "crate::math_expression::tree::number")] f64),

    /// A variable: a parameter, a bin, a compartment or an engine variable.
    Variable(String),

    /// Operators of one level applied from the left: `a + b - c` is
    /// `(a + b) - c`. Every operator of `rest` has the level of the first
    /// ([`BinaryOperator::level`]), and `rest` holds one pair or more.
    Chain {
        /// The first operand.
        first: Box<Expr>,
        /// Each operator, with the operand on its right.
        rest: Vec<(BinaryOperator, Expr)>,
    },

    /// A power, `base ^ exponent`. A chain of powers groups from the right:
    /// `a ^ b ^ c` is `a ^ (b ^ c)`.
    Power {
        /// The base.
        base: Box<Expr>,
        /// The exponent.
        exponent: Box<Expr>,
    },

    /// A sign or a negation: `-x`, `!condition`.
    UnaryOp {
        /// The operator.
        op: UnaryOperator,
        /// The operand.
        operand: Box<Expr>,
    },

    /// A call of a function, such as `sin(x)` or `max(x, y, 2)`.
    FunctionCall {
        /// The name of the function.
        name: String,
        /// The arguments.
        args: Vec<Expr>,
    },

    /// `if(condition, true_expr, false_expr)`.
    Conditional {
        /// The condition.
        condition: Box<Expr>,
        /// The value when the condition is true.
        true_expr: Box<Expr>,
        /// The value when the condition is false.
        false_expr: Box<Expr>,
    },

    /// `matrix_sum(matrix, expression)`: in a compartment of category `i`
    /// of the stratification of the matrix, `Σ_j matrix[i, j] · e_j`, where
    /// `e_j` is the expression with each bin read as its amount in category
    /// `j` and `total` as the total of category `j`. A model writes it as
    /// that sum before the JIT compiles the rate
    /// (`commol_model`, module `matrices`).
    MatrixSum {
        /// The id of the matrix: not a variable.
        matrix: String,
        /// The expression read for each category `j`.
        expression: Box<Expr>,
    },
}

/// An operator of a [`Expr::Chain`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BinaryOperator {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Mod,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `<=`
    Le,
    /// `>=`
    Ge,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `&&`
    And,
    /// `||`
    Or,
}

/// The level of an operator of a chain, from the one that binds least.
/// The operators of one level make one chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// `||`
    Or,
    /// `&&`
    And,
    /// `< > <= >= == !=`
    Comparison,
    /// `+ -`
    Sum,
    /// `* / %`
    Product,
}

impl BinaryOperator {
    /// The level of the operator.
    pub fn level(self) -> Level {
        match self {
            Self::Or => Level::Or,
            Self::And => Level::And,
            Self::Lt | Self::Gt | Self::Le | Self::Ge | Self::Eq | Self::Ne => Level::Comparison,
            Self::Add | Self::Sub => Level::Sum,
            Self::Mul | Self::Div | Self::Mod => Level::Product,
        }
    }

    /// Whether the operator compares two numbers.
    pub fn is_comparison(self) -> bool {
        self.level() == Level::Comparison
    }
}

/// A unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum UnaryOperator {
    /// The sign: `-x`.
    Neg,

    /// The negation of a condition: `!x`.
    Not,
}

/// How tightly a node binds when a formula writes it without parentheses,
/// from the loosest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Binding {
    /// A chain of a level.
    Chain(Level),
    /// A sign or a negation.
    Sign,
    /// A power.
    Power,
    /// A number, a variable, a call or an `if`.
    Primary,
}

impl Expr {
    /// How tightly the node binds.
    fn binding(&self) -> Binding {
        match self {
            Expr::Chain { rest, .. } => Binding::Chain(rest[0].0.level()),
            Expr::UnaryOp { .. } => Binding::Sign,
            Expr::Power { .. } => Binding::Power,
            Expr::Constant(_)
            | Expr::Variable(_)
            | Expr::FunctionCall { .. }
            | Expr::Conditional { .. }
            | Expr::MatrixSum { .. } => Binding::Primary,
        }
    }
}

/// A node in parentheses when `.1` holds.
struct Grouped<'a>(&'a Expr, bool);

impl<'a> Grouped<'a> {
    /// `expr` where a formula reads only nodes that bind more tightly than
    /// `than`.
    fn tighter(expr: &'a Expr, than: Binding) -> Self {
        Grouped(expr, expr.binding() <= than)
    }

    /// `expr` where a formula reads only nodes that bind at least as tightly
    /// as `least`.
    fn at_least(expr: &'a Expr, least: Binding) -> Self {
        Grouped(expr, expr.binding() < least)
    }
}

impl fmt::Display for Grouped<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Grouped(expr, true) => write!(f, "({expr})"),
            Grouped(expr, false) => write!(f, "{expr}"),
        }
    }
}

/// A formula that reads back as the same tree, with the parentheses the tree
/// needs and no other.
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Constant(val) => write!(f, "{val}"),
            Expr::Variable(name) => write!(f, "{name}"),
            Expr::Chain { first, rest } => {
                // An operand binds more tightly than its chain: a chain of
                // the same level, or of a looser one, is a group.
                let binding = self.binding();
                write!(f, "{}", Grouped::tighter(first, binding))?;
                for (op, right) in rest {
                    write!(f, " {op} {}", Grouped::tighter(right, binding))?;
                }
                Ok(())
            }
            Expr::Power { base, exponent } => write!(
                f,
                "{} ^ {}",
                Grouped::at_least(base, Binding::Primary),
                Grouped::at_least(exponent, Binding::Sign)
            ),
            Expr::UnaryOp { op, operand } => {
                write!(f, "{op}{}", Grouped::at_least(operand, Binding::Sign))
            }
            Expr::FunctionCall { name, args } => {
                write!(f, "{name}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Expr::Conditional {
                condition,
                true_expr,
                false_expr,
            } => write!(f, "if({condition}, {true_expr}, {false_expr})"),
            Expr::MatrixSum { matrix, expression } => {
                write!(f, "{MATRIX_SUM}({matrix}, {expression})")
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
            BinaryOperator::Lt => "<",
            BinaryOperator::Gt => ">",
            BinaryOperator::Le => "<=",
            BinaryOperator::Ge => ">=",
            BinaryOperator::Eq => "==",
            BinaryOperator::Ne => "!=",
            BinaryOperator::And => "&&",
            BinaryOperator::Or => "||",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            UnaryOperator::Neg => "-",
            UnaryOperator::Not => "!",
        };
        write!(f, "{s}")
    }
}

impl Expr {
    /// A variable.
    pub fn variable(name: impl Into<String>) -> Self {
        Expr::Variable(name.into())
    }

    /// A chain: `first`, then each operator with its operand.
    pub fn chain(first: Expr, rest: Vec<(BinaryOperator, Expr)>) -> Self {
        Expr::Chain {
            first: Box::new(first),
            rest,
        }
    }

    /// A chain of one operator: `left op right`.
    pub fn binary(op: BinaryOperator, left: Expr, right: Expr) -> Self {
        Self::chain(left, vec![(op, right)])
    }

    /// `base ^ exponent`.
    pub fn power(base: Expr, exponent: Expr) -> Self {
        Expr::Power {
            base: Box::new(base),
            exponent: Box::new(exponent),
        }
    }

    /// A unary operation.
    pub fn unary(op: UnaryOperator, operand: Expr) -> Self {
        Expr::UnaryOp {
            op,
            operand: Box::new(operand),
        }
    }

    /// A call of a function.
    pub fn call(name: impl Into<String>, args: Vec<Expr>) -> Self {
        Expr::FunctionCall {
            name: name.into(),
            args,
        }
    }

    /// `if(condition, true_expr, false_expr)`.
    pub fn if_then_else(condition: Expr, true_expr: Expr, false_expr: Expr) -> Self {
        Expr::Conditional {
            condition: Box::new(condition),
            true_expr: Box::new(true_expr),
            false_expr: Box::new(false_expr),
        }
    }

    /// `matrix_sum(matrix, expression)`.
    pub fn matrix_sum(matrix: impl Into<String>, expression: Expr) -> Self {
        Expr::MatrixSum {
            matrix: matrix.into(),
            expression: Box::new(expression),
        }
    }

    /// The tree with each node for which `replace` gives a node replaced by
    /// it, and every other node rebuilt from its operands, mapped the same
    /// way. `replace` sees a node before its operands.
    ///
    /// # Errors
    ///
    /// Returns the first error that `replace` gives.
    pub fn try_map<E, F>(&self, replace: &mut F) -> Result<Expr, E>
    where
        F: FnMut(&Expr) -> Option<Result<Expr, E>>,
    {
        if let Some(node) = replace(self) {
            return node;
        }
        // One loop over the operands keeps each level of the recursion one
        // small frame, so a tree at the nesting limit maps on a worker thread.
        let operands = self.operands();
        let mut mapped = Vec::with_capacity(operands.len());
        for operand in operands {
            mapped.push(operand.try_map(replace)?);
        }
        Ok(self.with_operands(mapped))
    }

    /// The node with its operands, in the order of [`Self::operands`],
    /// replaced by `operands`.
    fn with_operands(&self, operands: Vec<Expr>) -> Expr {
        let mut operands = operands.into_iter();
        let mut next = || {
            operands
                .next()
                .expect("one operand for each operand of the node")
        };
        match self {
            Expr::Constant(_) | Expr::Variable(_) => self.clone(),
            Expr::Chain { rest, .. } => {
                let first = next();
                let rest = rest.iter().map(|(op, _)| (*op, next())).collect();
                Expr::chain(first, rest)
            }
            Expr::Power { .. } => Expr::power(next(), next()),
            Expr::UnaryOp { op, .. } => Expr::unary(*op, next()),
            Expr::FunctionCall { name, args } => {
                Expr::call(name.clone(), args.iter().map(|_| next()).collect())
            }
            Expr::Conditional { .. } => Expr::if_then_else(next(), next(), next()),
            Expr::MatrixSum { matrix, .. } => Expr::matrix_sum(matrix.clone(), next()),
        }
    }

    /// Whether the tree holds a `matrix_sum`, which only a model writes as
    /// a formula the JIT compiles.
    pub fn reads_matrix_sum(&self) -> bool {
        matches!(self, Expr::MatrixSum { .. })
            || self
                .operands()
                .into_iter()
                .any(|operand| operand.reads_matrix_sum())
    }

    /// The operands of the tree one level down, in the order the formula
    /// writes them.
    pub fn operands(&self) -> Vec<&Expr> {
        match self {
            Expr::Constant(_) | Expr::Variable(_) => Vec::new(),
            Expr::Chain { first, rest } => std::iter::once(first.as_ref())
                .chain(rest.iter().map(|(_, operand)| operand))
                .collect(),
            Expr::Power { base, exponent } => vec![base, exponent],
            Expr::UnaryOp { operand, .. } => vec![operand],
            Expr::FunctionCall { args, .. } => args.iter().collect(),
            Expr::Conditional {
                condition,
                true_expr,
                false_expr,
            } => vec![condition, true_expr, false_expr],
            Expr::MatrixSum { expression, .. } => vec![expression],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_map_replaces_the_nodes_it_gives_and_rebuilds_the_others() {
        let tree = Expr::matrix_sum(
            "M",
            Expr::binary(
                BinaryOperator::Div,
                Expr::variable("B"),
                Expr::variable("x"),
            ),
        );
        let renamed = tree.try_map(&mut |node| match node {
            Expr::Variable(name) if name == "x" => Some(Ok::<_, ()>(Expr::variable("y"))),
            _ => None,
        });
        let failed = tree.try_map(&mut |node| match node {
            Expr::Variable(name) if name == "B" => Some(Err(name.clone())),
            _ => None,
        });

        assert_eq!(renamed.unwrap().to_string(), "matrix_sum(M, B / y)");
        assert_eq!(failed, Err("B".to_string()));
    }

    #[test]
    fn a_tree_prints_as_a_formula_with_its_groups() {
        let x = || Expr::variable("x");
        let tree = Expr::chain(
            x(),
            vec![
                (BinaryOperator::Add, Expr::power(x(), Expr::Constant(2.0))),
                (
                    BinaryOperator::Sub,
                    Expr::unary(UnaryOperator::Neg, Expr::call("sin", vec![x()])),
                ),
            ],
        );

        assert_eq!(tree.to_string(), "x + x ^ 2 - -sin(x)");
        assert_eq!(
            Expr::if_then_else(
                Expr::binary(BinaryOperator::Gt, x(), Expr::Constant(1.0)),
                x(),
                Expr::Constant(0.5)
            )
            .to_string(),
            "if(x > 1, x, 0.5)"
        );
        let ops = [
            BinaryOperator::Eq,
            BinaryOperator::Ne,
            BinaryOperator::Le,
            BinaryOperator::Ge,
            BinaryOperator::Lt,
        ];
        assert_eq!(
            ops.map(|op| Expr::binary(op, x(), x()).to_string()),
            ["x == x", "x != x", "x <= x", "x >= x", "x < x"]
        );
    }

    #[test]
    fn the_operands_of_a_node_are_in_the_order_of_the_formula() {
        let tree = Expr::chain(
            Expr::variable("a"),
            vec![
                (BinaryOperator::Mul, Expr::variable("b")),
                (BinaryOperator::Div, Expr::variable("c")),
            ],
        );

        assert_eq!(
            tree.operands(),
            [
                &Expr::variable("a"),
                &Expr::variable("b"),
                &Expr::variable("c")
            ]
        );
        assert!(Expr::Constant(1.0).operands().is_empty());
    }
}
