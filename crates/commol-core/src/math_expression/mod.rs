//! Mathematical expressions: the language of formulas, and their compiled
//! functions.
//!
//! A rate, a formula parameter and a calibration constraint are formulas.
//! [`MathExpression::check`] decides whether a formula is in the language
//! that `docs/guide/mathematical-expressions.md` documents, and the JIT
//! (Cranelift) compiles it to machine code once.
//!
//! ## Module Organization
//!
//! - `language` - The functions of the language and the check of a formula
//! - `jit` - The tree of a formula (`jit::ast`), its parser and its compiler
//! - `preprocessing` - The engine variables, and the formula the parser reads
//! - `context` - Evaluation context with parameters and compartments
//! - `error` - Error types for expression evaluation
//!
//! ## The language
//!
//! - Numbers, such as `2`, `0.5` and `1e5`, and variables: parameters, bins,
//!   compartments and the engine variables ([`ENGINE_VARIABLES`]): `total`,
//!   `step`, `t`, and the constants `pi` and `e`.
//! - Arithmetic: `+`, `-` (also unary), `*`, `/`, `%` (remainder), `^` or
//!   `**` (power).
//! - The functions of [`FUNCTIONS`]: `sin`, `cos`, `tan`, `asin`, `acos`,
//!   `atan`, `atan2(y, x)`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`,
//!   `atanh`, `exp`, `log` and `ln` (natural), `log10`, `log2`, `sqrt`,
//!   `cbrt`, `pow(x, y)`, `hypot(x, y)`, `min` and `max` (two arguments or
//!   more), `abs`, `floor`, `ceil`, `round`.
//! - `if(condition, a, b)`: `a` when the condition holds, else `b`. A
//!   condition is a comparison (`<`, `>`, `<=`, `>=`, `==`, `!=`) of two
//!   numbers, or `!` of a condition; a comparison or `!` is nowhere else.
//!   `&&` and `||` are not in the language: nest `if` calls.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

pub mod context;
pub mod error;
pub mod jit;
pub mod language;
pub mod preprocessing;

// Re-export main types
pub use context::MathExpressionContext;
pub use error::MathExpressionError;
pub use jit::{JITCompiler, JITFunction};
pub use language::{Arguments, FUNCTIONS, Function};
pub use preprocessing::{ENGINE_VARIABLES, SPECIAL_VAR_TOTAL, SUBGROUP_TOTAL_PREFIX};

use jit::ast::Expr;
use jit::parser::parse_expression;
use preprocessing::preprocess_formula;

/// Global cache for JIT-compiled functions.
///
/// This cache ensures that the same preprocessed formula always uses the same
/// compiled machine code, which is critical for deterministic floating-point results.
/// Without caching, each JIT compilation can produce slightly different machine code
/// (due to memory layout, instruction scheduling, etc.) that leads to ULP-level
/// differences in floating-point results.
static JIT_CACHE: std::sync::LazyLock<Mutex<HashMap<String, Arc<JITFunction>>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// The JIT function of a preprocessed formula: the cached one, or a new one
/// compiled and cached. Expressions are compiled when a model is built, not
/// while it runs, so one lock serves every caller.
///
/// # Errors
///
/// Returns a message when the formula does not compile.
fn get_or_compile_jit(preprocessed: &str) -> Result<Arc<JITFunction>, String> {
    // A thread that panicked while it held the lock left a complete map: an
    // entry is inserted whole or not at all.
    let mut cache = JIT_CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(jit_fn) = cache.get(preprocessed) {
        return Ok(Arc::clone(jit_fn));
    }
    let jit_fn = Arc::new(
        JITCompiler::new()
            .compile(preprocessed)
            .map_err(|error| error.to_string())?,
    );
    cache.insert(preprocessed.to_string(), Arc::clone(&jit_fn));
    Ok(jit_fn)
}

/// A mathematical expression that can be evaluated
///
/// Expressions are JIT-compiled to native machine code. An expression that
/// does not compile keeps the reason, and evaluating it returns that reason
/// as an error.
///
/// ## Example
/// ```rust
/// use commol_core::MathExpression;
/// use commol_core::MathExpressionContext;
///
/// let expr = MathExpression::new("k1 * A * B".to_string());
/// let mut context = MathExpressionContext::new();
/// context.set_parameter("k1".to_string(), 0.5);
/// context.init_compartments(vec!["A".to_string(), "B".to_string()]);
/// context.set_compartments_by_index(&[990.0, 10.0]);
/// let result = expr.evaluate(&mut context).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MathExpression {
    /// The original mathematical formula as a string
    pub formula: String,
    /// Preprocessed formula (cached for performance)
    preprocessed: String,
    /// The compiled function, or why the formula does not compile.
    compiled: Result<Arc<JITFunction>, String>,
}

impl MathExpression {
    /// Create a new expression (preprocesses and JIT-compiles the formula)
    ///
    /// JIT-compiled functions are cached globally to ensure deterministic
    /// floating-point results. Without caching, each compilation could produce
    /// slightly different machine code leading to ULP-level differences.
    pub fn new(formula: String) -> Self {
        let preprocessed = preprocess_formula(&formula);
        let compiled = get_or_compile_jit(&preprocessed);
        Self {
            formula,
            preprocessed,
            compiled,
        }
    }

    /// The compiled function.
    ///
    /// # Errors
    ///
    /// Returns [`MathExpressionError::InvalidExpression`] with the reason
    /// when the formula does not compile.
    fn jit(&self) -> Result<&JITFunction, MathExpressionError> {
        self.compiled.as_deref().map_err(|reason| {
            MathExpressionError::InvalidExpression(format!(
                "'{}' does not compile: {reason}",
                self.formula
            ))
        })
    }

    /// The tree of the formula, the one the JIT compiles.
    ///
    /// # Errors
    ///
    /// Returns why the formula does not parse.
    pub fn tree(&self) -> Result<Expr, MathExpressionError> {
        parse_expression(&self.preprocessed)
    }

    /// Check that the formula is in the language of formulas, and count the
    /// nodes of its tree: numbers, variables, operators and calls, `if`
    /// among them. A formula that passes the check compiles.
    ///
    /// # Errors
    ///
    /// Returns [`MathExpressionError::NotInLanguage`], which names the
    /// formula and the first part of it that is not in the language: a
    /// formula that does not parse, a function that is not one of
    /// [`FUNCTIONS`] or that has the wrong number of arguments, `&&` or `||`,
    /// or a comparison or `!` outside the condition of an `if`.
    pub fn check(&self) -> Result<usize, MathExpressionError> {
        let outside = |reason: String| MathExpressionError::NotInLanguage {
            formula: self.formula.clone(),
            reason,
        };
        let tree = self.tree().map_err(|error| {
            outside(match error {
                MathExpressionError::InvalidExpression(reason) => reason,
                other => other.to_string(),
            })
        })?;
        let nodes = language::check(&tree).map_err(outside)?;
        self.jit()?;
        Ok(nodes)
    }

    /// Evaluate the expression with the given context
    ///
    /// # Errors
    ///
    /// Returns an error when the formula does not compile or reads a
    /// variable the context does not hold.
    pub fn evaluate(
        &self,
        context: &mut MathExpressionContext,
    ) -> Result<f64, MathExpressionError> {
        self.jit()?.call(context)
    }

    /// The variables the formula reads, in the order it first reads them,
    /// without the engine variables ([`ENGINE_VARIABLES`]): the names a
    /// model must define. Empty when the formula does not parse.
    pub fn get_variables(&self) -> Vec<String> {
        self.tree()
            .map(|tree| language::variables(&tree))
            .unwrap_or_default()
            .into_iter()
            .filter(|name| !ENGINE_VARIABLES.contains(&name.as_str()))
            .collect()
    }

    /// Whether the formula reads the variable `name`, an engine variable
    /// such as `step` included. False when the formula does not parse.
    pub fn reads_variable(&self, name: &str) -> bool {
        self.tree()
            .is_ok_and(|tree| language::variables(&tree).iter().any(|read| read == name))
    }

    /// The formula the parser reads: `**` written as `^`.
    pub fn preprocessed(&self) -> &str {
        &self.preprocessed
    }

    /// The variable names of the compiled function, in the order
    /// `call_jit_with_buffer` reads them, or `None` when the formula does not
    /// compile.
    pub fn jit_variable_names(&self) -> Option<&[String]> {
        self.jit().ok().map(JITFunction::variables)
    }

    /// Invoke the compiled function with a pre-filled value buffer, laid out
    /// in the order of `jit_variable_names`.
    ///
    /// # Errors
    ///
    /// Returns an error when the formula does not compile or the buffer has
    /// another length.
    pub fn call_jit_with_buffer(&self, buffer: &[f64]) -> Result<f64, MathExpressionError> {
        self.jit()?.call_with_buffer(buffer)
    }
}

/// A rate or a formula parameter, by the shape of its tree.
#[derive(Debug, Clone)]
pub enum RateMathExpression {
    /// A single variable that is not an engine variable: a parameter, a bin
    /// or a compartment, such as `k1`.
    Parameter(String),
    /// Any other formula, such as `k1 * 2` or `step`.
    Formula(MathExpression),
    /// A single number, such as `0.3`.
    Constant(f64),
}

impl RateMathExpression {
    /// Classify a formula from its tree: a number is a `Constant`, a single
    /// variable that is not an engine variable ([`ENGINE_VARIABLES`]) is a
    /// `Parameter`, and anything else is a `Formula`, also one that is not in
    /// the language ([`Self::check`]).
    pub fn from_string(s: String) -> Self {
        match parse_expression(&preprocess_formula(&s)) {
            Ok(Expr::Constant(value)) => Self::Constant(value),
            Ok(Expr::Variable(name)) if !ENGINE_VARIABLES.contains(&name.as_str()) => {
                Self::Parameter(name)
            }
            _ => Self::Formula(MathExpression::new(s)),
        }
    }

    /// Check that a formula is in the language
    /// ([`MathExpression::check`]), and count the nodes of its tree; a
    /// parameter or a constant is one node.
    ///
    /// # Errors
    ///
    /// Returns the error of [`MathExpression::check`].
    pub fn check(&self) -> Result<usize, MathExpressionError> {
        match self {
            Self::Formula(expr) => expr.check(),
            Self::Parameter(_) | Self::Constant(_) => Ok(1),
        }
    }

    /// Evaluate the rate expression
    pub fn evaluate(
        &self,
        context: &mut MathExpressionContext,
    ) -> Result<f64, MathExpressionError> {
        match self {
            Self::Parameter(param_name) => context
                .get_parameter(param_name)
                .ok_or_else(|| MathExpressionError::VariableNotFound(param_name.clone())),
            Self::Formula(expr) => expr.evaluate(context),
            Self::Constant(value) => Ok(*value),
        }
    }

    /// The variables this rate expression reads, without the engine
    /// variables ([`MathExpression::get_variables`]).
    pub fn get_variables(&self) -> Vec<String> {
        match self {
            Self::Parameter(param_name) => vec![param_name.clone()],
            Self::Formula(expr) => expr.get_variables(),
            Self::Constant(_) => vec![],
        }
    }

    /// Whether this rate expression reads the variable `name`, an engine
    /// variable such as `step` included.
    pub fn reads_variable(&self, name: &str) -> bool {
        match self {
            Self::Parameter(param_name) => param_name == name,
            Self::Formula(expr) => expr.reads_variable(name),
            Self::Constant(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_formula_compiles_to_one_function() {
        let first = MathExpression::new("k1 * A + k2".to_string()).compiled;
        let second = MathExpression::new("k1 * A + k2".to_string()).compiled;

        assert!(Arc::ptr_eq(&first.unwrap(), &second.unwrap()));
    }

    #[test]
    fn test_basic_expressions() {
        let mut ctx = MathExpressionContext::new();
        ctx.set_parameter("k1".to_string(), 0.5);
        ctx.set_parameter("k2".to_string(), 0.2);

        let expr = MathExpression::new("k1 * k2".to_string());
        let result = expr.evaluate(&mut ctx).unwrap();
        assert!((result - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_complex_expressions() {
        let mut ctx = MathExpressionContext::new();
        ctx.set_parameter("k1".to_string(), 0.5);
        ctx.set_compartment("A".to_string(), 990.0);
        ctx.set_compartment("B".to_string(), 10.0);
        ctx.set_compartment("C".to_string(), 0.0);

        // Test a triple product
        let expr = MathExpression::new("k1 * A * B".to_string());
        let result = expr.evaluate(&mut ctx).unwrap();
        assert!((result - 4950.0).abs() < 1e-10);

        // Test a product divided by the total
        let expr = MathExpression::new("k1 * A * B / total".to_string());
        let result = expr.evaluate(&mut ctx).unwrap();
        assert!((result - 4.95).abs() < 1e-10);
    }

    #[test]
    fn test_math_functions() {
        let mut ctx = MathExpressionContext::new();
        ctx.set_parameter("x".to_string(), 0.0);

        let expr = MathExpression::new("sin(x)".to_string());
        let result = expr.evaluate(&mut ctx).unwrap();
        assert!(result.abs() < 1e-10);

        let expr = MathExpression::new("exp(x)".to_string());
        let result = expr.evaluate(&mut ctx).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn an_expression_that_does_not_compile_fails_with_the_reason() {
        let expr = MathExpression::new("typeof(k1)".to_string());
        let mut ctx = MathExpressionContext::new();
        ctx.set_parameter("k1".to_string(), 0.5);
        let reason = "Invalid expression: 'typeof(k1)' does not compile: Invalid expression: \
                      Unknown function: typeof";

        assert_eq!(expr.evaluate(&mut ctx).unwrap_err().to_string(), reason);
        assert_eq!(
            expr.call_jit_with_buffer(&[0.5]).unwrap_err().to_string(),
            reason
        );
        assert_eq!(expr.jit_variable_names(), None);
        let rate = RateMathExpression::from_string("typeof(k1)".to_string());
        assert_eq!(
            rate.check().unwrap_err().to_string(),
            expr.check().unwrap_err().to_string()
        );
        assert_eq!(
            RateMathExpression::from_string("k1".to_string())
                .check()
                .unwrap(),
            1
        );
    }

    #[test]
    fn a_formula_that_does_not_parse_names_itself() {
        let error = MathExpression::new("(k1 + 2".to_string())
            .check()
            .unwrap_err()
            .to_string();

        assert!(
            error.starts_with("Formula '(k1 + 2': it does not parse: "),
            "{error}"
        );
    }

    #[test]
    fn a_parameter_slot_is_set_by_its_index() {
        let mut ctx = MathExpressionContext::new();
        ctx.reserve_parameters(["k1".to_string()]);
        let index = ctx.parameter_index("k1").unwrap();

        ctx.set_parameter_by_index(index, 0.3).unwrap();

        assert_eq!(ctx.get_parameter("k1"), Some(0.3));
        assert_eq!(ctx.get_parameters()["k1"], 0.3);
        assert_eq!(
            ctx.set_parameter_by_index(index + 1, 0.3),
            Err(format!("Parameter index '{}' not found", index + 1))
        );
    }

    #[test]
    fn an_expression_prints_its_formula_and_the_variables_it_reads() {
        let printed = format!("{:?}", MathExpression::new("k1 * A".to_string()));

        assert!(printed.contains("formula: \"k1 * A\""), "{printed}");
        assert!(
            printed.contains("variable_names: [\"k1\", \"A\"]"),
            "{printed}"
        );
    }

    #[test]
    fn test_rate_math_expression() {
        let mut ctx = MathExpressionContext::new();
        ctx.set_parameter("k1".to_string(), 0.5);

        // Test parameter
        let rate = RateMathExpression::from_string("k1".to_string());
        assert!(matches!(rate, RateMathExpression::Parameter(_)));
        assert!((rate.evaluate(&mut ctx).unwrap() - 0.5).abs() < 1e-10);

        // Test constant
        let rate = RateMathExpression::from_string("0.3".to_string());
        assert!(matches!(rate, RateMathExpression::Constant(_)));
        assert!((rate.evaluate(&mut ctx).unwrap() - 0.3).abs() < 1e-10);

        // Test formula
        let rate = RateMathExpression::from_string("k1 * 2".to_string());
        assert!(matches!(rate, RateMathExpression::Formula(_)));
        assert!((rate.evaluate(&mut ctx).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn a_rate_is_classified_by_its_tree() {
        let mut ctx = MathExpressionContext::new();
        ctx.set_step(3.0);
        let classify = |formula: &str| RateMathExpression::from_string(formula.to_string());

        assert!(matches!(classify("1e5"), RateMathExpression::Constant(value) if value == 1e5));
        assert!(matches!(classify(" k1 "), RateMathExpression::Parameter(name) if name == "k1"));
        assert!(matches!(
            classify("step > 1"),
            RateMathExpression::Formula(_)
        ));
        assert!(matches!(classify("-0.5"), RateMathExpression::Formula(_)));
        let step = classify("step");
        assert!(matches!(step, RateMathExpression::Formula(_)));
        assert_eq!(step.evaluate(&mut ctx).unwrap(), 3.0);
    }
}
