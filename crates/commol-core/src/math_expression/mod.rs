//! Mathematical expression evaluation module
//!
//! This module provides mathematical expression evaluation for compartment models.
//!
//! ## Module Organization
//!
//! - `error` - Error types for expression evaluation
//! - `preprocessing` - Formula preprocessing and syntax conversion
//! - `context` - Evaluation context with parameters and compartments
//! - `compiled` - Compiled expression patterns for fast evaluation
//!
//! ## Special Variables
//!
//! The following variables are automatically available in all expressions:
//! - `total` - The sum of every compartment, accumulators excluded
//! - `total_<cat>…` - A subgroup total: the sum of the compartments with the
//!   categories `<cat>…`
//! - `step` - The step that holds the current position, `floor(t)`
//! - `t` - The current position on the axis; equal to `step` in a difference
//!   engine, continuous in a differential engine
//! - `pi` - Mathematical constant π (≈ 3.14159)
//! - `e` - Mathematical constant e (≈ 2.71828)
//!
//! ## Supported Operators
//!
//! - Arithmetic: `+`, `-`, `*`, `/`, `%` (modulo), `^` or `**` (power)
//! - Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=`
//! - Logical: `&&` (and), `||` (or), `!` (not)
//!
//! Note: Both `^` and `**` are accepted for exponentiation. Python users can use
//! the familiar `**` syntax, which is automatically converted to `^`.
//!
//! ## Supported Functions
//!
//! ### Trigonometric
//! - `sin(x)`, `cos(x)`, `tan(x)` - Basic trigonometric functions (x in radians)
//! - `asin(x)`, `acos(x)`, `atan(x)` - Inverse trigonometric functions
//! - `atan2(y, x)` - Two-argument arctangent
//! - `sinh(x)`, `cosh(x)`, `tanh(x)` - Hyperbolic functions
//! - `asinh(x)`, `acosh(x)`, `atanh(x)` - Inverse hyperbolic functions
//!
//! ### Exponential and Logarithmic
//! - `exp(x)` - e raised to the power of x
//! - `ln(x)` - Natural logarithm (base e)
//! - `log(x)`, `log2(x)`, `log10(x)` - Logarithms with different bases
//! - `pow(x, y)` - x raised to the power of y
//!
//! ### Roots and Absolute Value
//! - `sqrt(x)` - Square root
//! - `cbrt(x)` - Cube root
//! - `hypot(x, y)` - Euclidean distance: sqrt(x² + y²)
//! - `abs(x)` - Absolute value
//!
//! ### Rounding
//! - `floor(x)` - Round down to nearest integer
//! - `ceil(x)` - Round up to nearest integer
//! - `round(x)` - Round to nearest integer
//! - `trunc(x)` - Truncate to integer (towards zero)
//!
//! ### Advanced Math Functions
//! - `fma(x, y, z)` - Fused multiply-add: x*y+z (single rounding, more accurate)
//! - `copysign(x, y)` - Copy sign of y to magnitude of x
//! - `fdim(x, y)` - Positive difference: max(x-y, 0)
//! - `fmax(x, y)` - Maximum (NaN-aware, different from min/max)
//! - `fmin(x, y)` - Minimum (NaN-aware, different from min/max)
//! - `remainder(x, y)` - IEEE remainder
//!
//! ### Special Functions
//! - `erf(x)` - Error function
//! - `erfc(x)` - Complementary error function
//! - `tgamma(x)` - Gamma function
//! - `lgamma(x)` - Natural logarithm of gamma function
//!
//! ### Other
//! - `min(a, b, ...)` - Minimum value (variadic)
//! - `max(a, b, ...)` - Maximum value (variadic)
//! - `if(condition, value_if_true, value_if_false)` - Conditional expression

use evalexpr::{ContextWithMutableVariables, HashMapContext, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

pub mod context;
pub mod error;
pub mod jit;
pub mod preprocessing;

// Re-export main types
pub use context::MathExpressionContext;
pub use error::MathExpressionError;
pub use jit::{JITCompiler, JITFunction};
pub use preprocessing::{ENGINE_VARIABLES, SPECIAL_VAR_TOTAL, SUBGROUP_TOTAL_PREFIX};

use preprocessing::{get_variables, preprocess_formula, reads_variable};

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
            .and_then(|compiler| compiler.compile(preprocessed))
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
/// context.set_compartment("A".to_string(), 990.0);
/// context.set_compartment("B".to_string(), 10.0);
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

    /// Check that the formula compiles.
    ///
    /// # Errors
    ///
    /// Returns the reason when it does not.
    pub fn compiles(&self) -> Result<(), MathExpressionError> {
        self.jit().map(|_| ())
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

    /// Validate that the expression is syntactically correct
    pub fn validate(&self) -> Result<(), MathExpressionError> {
        // First check if we can build the operator tree
        let tree = match evalexpr::build_operator_tree(&self.preprocessed) {
            Ok(tree) => tree,
            Err(e) => return Err(MathExpressionError::EvalError(e)),
        };

        // Create a dummy context with some common variables to validate the expression
        // This will catch issues like incomplete expressions, invalid operator sequences, etc.
        let mut context = HashMapContext::new();

        // Add dummy values for common variables that might be in the expression
        let variables = self.get_variables();
        for var in variables {
            context.set_value(var, Value::Float(1.0)).ok();
        }

        // Add special variables
        context
            .set_value(SPECIAL_VAR_TOTAL.to_string(), Value::Float(1.0))
            .ok();
        context
            .set_value("step".to_string(), Value::Float(1.0))
            .ok();
        context.set_value("t".to_string(), Value::Float(1.0)).ok();
        context
            .set_value("pi".to_string(), Value::Float(std::f64::consts::PI))
            .ok();
        context
            .set_value("e".to_string(), Value::Float(std::f64::consts::E))
            .ok();

        // Try to evaluate with dummy context - this will catch syntax errors
        match tree.eval_with_context(&context) {
            Ok(_) => Ok(()),
            Err(e) => Err(MathExpressionError::EvalError(e)),
        }
    }

    /// Get all variable identifiers used in the expression.
    ///
    /// Returns an empty vector if the expression is syntactically invalid.
    pub fn get_variables(&self) -> Vec<String> {
        get_variables(&self.preprocessed)
    }

    /// Whether the expression reads the variable `name`, special variables
    /// such as `step` included.
    pub fn reads_variable(&self, name: &str) -> bool {
        reads_variable(&self.preprocessed, name)
    }

    /// Get the preprocessed formula string (Python operators normalised,
    /// transcendental functions prefixed).
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

/// Represents different types of rate expressions
#[derive(Debug, Clone)]
pub enum RateMathExpression {
    /// Simple parameter reference (backward compatibility)
    Parameter(String),
    /// Mathematical expression
    Formula(MathExpression),
    /// Constant value
    Constant(f64),
}

impl RateMathExpression {
    /// Parse a string into the appropriate rate expression type.
    ///
    /// - Numeric values become `Constant`
    /// - Single variable names become `Parameter` (e.g., "k1")
    /// - Expressions with an operator or a call become `Formula` (e.g.,
    ///   "k1 * 2"); one that does not compile keeps the reason
    ///   ([`Self::compiles`])
    pub fn from_string(s: String) -> Self {
        // First check if it's a numeric constant
        if let Ok(value) = s.parse::<f64>() {
            return Self::Constant(value);
        }

        // Simple heuristic: if the string contains operators or functions, it's a formula
        // Otherwise, treat it as a simple parameter name
        let has_operators = s.contains('+')
            || s.contains('-')
            || s.contains('*')
            || s.contains('/')
            || s.contains('^')
            || s.contains('%')
            || s.contains('(')
            || s.contains(')');

        if has_operators {
            Self::Formula(MathExpression::new(s))
        } else {
            Self::Parameter(s)
        }
    }

    /// Check that a formula compiles; a parameter or a constant always does.
    ///
    /// # Errors
    ///
    /// Returns the reason when a formula does not compile.
    pub fn compiles(&self) -> Result<(), MathExpressionError> {
        match self {
            Self::Formula(expr) => expr.compiles(),
            Self::Parameter(_) | Self::Constant(_) => Ok(()),
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

    /// Get all variables referenced in this rate expression
    pub fn get_variables(&self) -> Vec<String> {
        match self {
            Self::Parameter(param_name) => vec![param_name.clone()],
            Self::Formula(expr) => expr.get_variables(),
            Self::Constant(_) => vec![],
        }
    }

    /// Whether this rate expression reads the variable `name`, special
    /// variables such as `step` included.
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

        let reason = expr.compiles().unwrap_err().to_string();

        assert!(
            reason.contains("'typeof(k1)' does not compile: "),
            "{reason}"
        );
        assert_eq!(expr.evaluate(&mut ctx).unwrap_err().to_string(), reason);
        assert_eq!(
            expr.call_jit_with_buffer(&[0.5]).unwrap_err().to_string(),
            reason
        );
        assert_eq!(expr.jit_variable_names(), None);
        let rate = RateMathExpression::from_string("typeof(k1)".to_string());
        assert_eq!(rate.compiles().unwrap_err().to_string(), reason);
        assert!(
            RateMathExpression::from_string("k1".to_string())
                .compiles()
                .is_ok()
        );
    }

    #[test]
    fn a_formula_that_does_not_parse_is_not_valid() {
        let error = MathExpression::new("(k1 + 2".to_string())
            .validate()
            .unwrap_err();

        assert!(
            matches!(error, MathExpressionError::EvalError(_)),
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
}
