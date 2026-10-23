//! The language of formulas: its functions and the check of a formula.
//!
//! A formula is a number: numbers, variables, the arithmetic operators
//! `+ - * / % ^` (`**` is `^`), the sign `-`, the calls of [`FUNCTIONS`] and
//! `if(condition, a, b)`. A condition is a comparison of two numbers
//! (`< > <= >= == !=`) or `!` of a condition, and it is the only place a
//! comparison or `!` may be. The check of a formula
//! ([`MathExpression::check`](crate::MathExpression::check)) reads the tree the JIT compiles
//! ([`Expr`]), so a formula it accepts compiles, and the JIT compiles the
//! functions of [`FUNCTIONS`] and no other.

use std::collections::HashSet;

use super::jit::ast::{BinaryOperator, Expr, Level, UnaryOperator};

/// How many arguments a function takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arguments {
    /// Exactly this many.
    Exactly(usize),
    /// This many or more.
    AtLeast(usize),
}

impl Arguments {
    /// Whether a call with `count` arguments has the arguments it needs.
    pub fn accepts(self, count: usize) -> bool {
        match self {
            Self::Exactly(expected) => count == expected,
            Self::AtLeast(least) => count >= least,
        }
    }
}

impl std::fmt::Display for Arguments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exactly(1) => write!(f, "1 argument"),
            Self::Exactly(count) => write!(f, "{count} arguments"),
            Self::AtLeast(count) => write!(f, "{count} arguments or more"),
        }
    }
}

/// A function of the language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Function {
    /// The name a formula calls it by.
    pub name: &'static str,
    /// The arguments it takes.
    pub arguments: Arguments,
    /// The C function that JIT code calls (`address` in `jit/libm.rs`), or
    /// `None` for `min` and `max`, which compile to comparisons.
    pub symbol: Option<&'static str>,
}

/// A [`Function`]: `one(name, symbol)` takes one argument and JIT code calls
/// it by `symbol`, `two(name)` takes two and JIT code calls it by its name,
/// and `variadic(name)` takes two or more and compiles to comparisons.
macro_rules! function {
    (one($name:literal, $symbol:literal)) => {
        Function {
            name: $name,
            arguments: Arguments::Exactly(1),
            symbol: Some($symbol),
        }
    };
    (two($name:literal)) => {
        Function {
            name: $name,
            arguments: Arguments::Exactly(2),
            symbol: Some($name),
        }
    };
    (variadic($name:literal)) => {
        Function {
            name: $name,
            arguments: Arguments::AtLeast(2),
            symbol: None,
        }
    };
}

/// The functions of the language, the ones
/// `docs/guide/mathematical-expressions.md` lists. `if` is not among them:
/// it is the conditional of the tree ([`Expr::Conditional`]).
pub const FUNCTIONS: &[Function] = &[
    function!(one("sin", "sin")),
    function!(one("cos", "cos")),
    function!(one("tan", "tan")),
    function!(one("asin", "asin")),
    function!(one("acos", "acos")),
    function!(one("atan", "atan")),
    function!(two("atan2")),
    function!(one("sinh", "sinh")),
    function!(one("cosh", "cosh")),
    function!(one("tanh", "tanh")),
    function!(one("asinh", "asinh")),
    function!(one("acosh", "acosh")),
    function!(one("atanh", "atanh")),
    function!(one("exp", "exp")),
    function!(one("log", "log")),
    function!(one("ln", "log")),
    function!(one("log10", "log10")),
    function!(one("log2", "log2")),
    function!(one("sqrt", "sqrt")),
    function!(one("cbrt", "cbrt")),
    function!(two("pow")),
    function!(two("hypot")),
    function!(variadic("min")),
    function!(variadic("max")),
    function!(one("abs", "fabs")),
    function!(one("floor", "floor")),
    function!(one("ceil", "ceil")),
    function!(one("round", "round")),
];

/// The function of the language called `name`.
pub fn function(name: &str) -> Option<&'static Function> {
    FUNCTIONS.iter().find(|function| function.name == name)
}

/// The names of the functions, as a message lists them.
fn function_names() -> String {
    FUNCTIONS
        .iter()
        .map(|function| function.name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Check that `tree` is a formula of the language, and count its nodes:
/// numbers, variables, operators and calls, `if` among them.
///
/// # Errors
///
/// Returns the first part of the tree that is not in the language, and why.
pub(crate) fn check(tree: &Expr) -> Result<usize, String> {
    let mut nodes = 0;
    number(tree, &mut nodes)?;
    Ok(nodes)
}

/// The reason `&&` and `||` are rejected.
fn logical(op: BinaryOperator) -> String {
    format!(
        "the operator '{op}' is not in the language; nest if(condition, a, b) calls \
         instead"
    )
}

/// The reason a comparison outside a condition is rejected.
fn comparison(op: BinaryOperator) -> String {
    format!("the comparison '{op}' is only allowed in the condition of if(condition, a, b)")
}

/// The operator of a chain that the tree applies last.
fn last(rest: &[(BinaryOperator, Expr)]) -> BinaryOperator {
    rest.last().expect("a chain has an operator").0
}

/// Check a part of the tree that must be a number.
fn number(expr: &Expr, nodes: &mut usize) -> Result<(), String> {
    *nodes += 1;
    match expr {
        Expr::Constant(_) | Expr::Variable(_) => Ok(()),
        Expr::Chain { first, rest } => match last(rest).level() {
            Level::Or | Level::And => Err(logical(last(rest))),
            Level::Comparison => Err(comparison(last(rest))),
            Level::Sum | Level::Product => {
                // Each operator of the chain is one node.
                *nodes += rest.len() - 1;
                number(first, nodes)?;
                rest.iter()
                    .try_for_each(|(_, operand)| number(operand, nodes))
            }
        },
        Expr::Power { base, exponent } => {
            number(base, nodes)?;
            number(exponent, nodes)
        }
        Expr::UnaryOp {
            op: UnaryOperator::Neg,
            operand,
        } => number(operand, nodes),
        Expr::UnaryOp {
            op: UnaryOperator::Not,
            ..
        } => Err(
            "the operator '!' is only allowed in the condition of if(condition, a, b)".to_string(),
        ),
        Expr::FunctionCall { name, args } => {
            let Some(function) = function(name) else {
                return Err(format!(
                    "'{name}' is not a function of the language, whose functions are \
                     {}, and if(condition, a, b)",
                    function_names()
                ));
            };
            if !function.arguments.accepts(args.len()) {
                return Err(format!(
                    "'{name}' takes {}, got {}",
                    function.arguments,
                    args.len()
                ));
            }
            args.iter().try_for_each(|arg| number(arg, nodes))
        }
        Expr::Conditional {
            condition: test,
            true_expr,
            false_expr,
        } => {
            condition(test, nodes)?;
            number(true_expr, nodes)?;
            number(false_expr, nodes)
        }
    }
}

/// Check a part of the tree that must be a condition.
fn condition(expr: &Expr, nodes: &mut usize) -> Result<(), String> {
    match expr {
        Expr::Chain { first, rest } if last(rest).is_comparison() => {
            // `a < b < c` compares `a < b`, a comparison, with `c`.
            let (_, earlier) = rest.split_last().expect("a chain has an operator");
            if let Some(&(op, _)) = earlier.last() {
                return Err(comparison(op));
            }
            *nodes += 1;
            number(first, nodes)?;
            number(&rest[0].1, nodes)
        }
        Expr::Chain { rest, .. } if matches!(last(rest).level(), Level::Or | Level::And) => {
            Err(logical(last(rest)))
        }
        Expr::UnaryOp {
            op: UnaryOperator::Not,
            operand,
        } => {
            *nodes += 1;
            condition(operand, nodes)
        }
        _ => Err(format!(
            "the condition of if(condition, a, b) is a comparison, such as 'step > 30', \
             or '!' of one; got '{expr}'"
        )),
    }
}

/// The variables `tree` reads, in the order it first reads them.
pub(crate) fn variables(tree: &Expr) -> Vec<String> {
    fn collect<'a>(expr: &'a Expr, seen: &mut HashSet<&'a str>, out: &mut Vec<String>) {
        if let Expr::Variable(name) = expr
            && seen.insert(name)
        {
            out.push(name.clone());
        }
        for operand in expr.operands() {
            collect(operand, seen, out);
        }
    }
    let mut out = Vec::new();
    collect(tree, &mut HashSet::new(), &mut out);
    out
}
