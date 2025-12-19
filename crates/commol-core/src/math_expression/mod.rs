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
//! - `N` - Total population (sum of all compartments)
//! - `step` - Current simulation step number
//! - `t` - Alias for `step` (for convenience in time-dependent formulas)
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

use evalexpr::{ContextWithMutableVariables, HashMapContext, Node, Value, build_operator_tree};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub mod context;
pub mod error;
pub mod jit;
pub mod preprocessing;

// Re-export main types
pub use context::MathExpressionContext;
pub use error::MathExpressionError;
pub use jit::{JITCompiler, JITFunction};

use preprocessing::{get_variables, preprocess_formula};

/// Global cache for JIT-compiled functions.
///
/// This cache ensures that the same preprocessed formula always uses the same
/// compiled machine code, which is critical for deterministic floating-point results.
/// Without caching, each JIT compilation can produce slightly different machine code
/// (due to memory layout, instruction scheduling, etc.) that leads to ULP-level
/// differences in floating-point results.
static JIT_CACHE: std::sync::LazyLock<RwLock<HashMap<String, Arc<JITFunction>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

/// Get or compile a JIT function for the given preprocessed formula.
///
/// Returns a cached JIT function if available, otherwise compiles and caches a new one.
fn get_or_compile_jit(preprocessed: &str) -> Option<Arc<JITFunction>> {
    // First, try to get from cache with a read lock
    {
        let cache = JIT_CACHE.read().ok()?;
        if let Some(jit_fn) = cache.get(preprocessed) {
            return Some(Arc::clone(jit_fn));
        }
    }

    // Not in cache, need to compile
    // Acquire write lock and check again (another thread might have compiled it)
    let mut cache = JIT_CACHE.write().ok()?;
    if let Some(jit_fn) = cache.get(preprocessed) {
        return Some(Arc::clone(jit_fn));
    }

    // Compile the expression
    let compiler = JITCompiler::new().ok()?;
    let jit_fn = compiler.compile(preprocessed).ok()?;
    let arc_fn = Arc::new(jit_fn);

    // Cache it
    cache.insert(preprocessed.to_string(), Arc::clone(&arc_fn));

    Some(arc_fn)
}

/// Internal representation of compiled expression
#[derive(Debug, Clone)]
enum CompiledExpr {
    /// JIT-compiled function (fast path) - wrapped in Arc for sharing from cache
    Jit(Arc<JITFunction>),
    /// Generic evalexpr fallback
    Generic(Node),
}

/// A mathematical expression that can be evaluated
///
/// Expressions are JIT-compiled to native machine code for optimal performance.
/// If JIT compilation fails, the expression falls back to evalexpr interpretation.
///
/// ## Example
/// ```rust
/// use commol_core::MathExpression;
/// use commol_core::MathExpressionContext;
///
/// let expr = MathExpression::new("beta * S * I".to_string());
/// let mut context = MathExpressionContext::new();
/// context.set_parameter("beta".to_string(), 0.5);
/// context.set_compartment("S".to_string(), 990.0);
/// context.set_compartment("I".to_string(), 10.0);
/// let result = expr.evaluate(&mut context).unwrap();
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct MathExpression {
    /// The original mathematical formula as a string
    pub formula: String,
    /// Preprocessed formula (cached for performance)
    #[serde(skip)]
    preprocessed: String,
    /// Compiled expression (JIT or evalexpr fallback)
    #[serde(skip)]
    compiled: CompiledExpr,
}

// Custom deserialize to ensure preprocessed field is populated
impl<'de> Deserialize<'de> for MathExpression {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct MathExpressionData {
            formula: String,
        }

        let data = MathExpressionData::deserialize(deserializer)?;
        Ok(MathExpression::new(data.formula))
    }
}

impl MathExpression {
    /// Create a new expression (preprocesses and JIT-compiles the formula)
    ///
    /// Attempts to JIT-compile the expression for optimal performance.
    /// Falls back to evalexpr interpretation if JIT compilation fails.
    ///
    /// JIT-compiled functions are cached globally to ensure deterministic
    /// floating-point results. Without caching, each compilation could produce
    /// slightly different machine code leading to ULP-level differences.
    pub fn new(formula: String) -> Self {
        let preprocessed = preprocess_formula(&formula);

        // Try to get JIT function from cache, or compile and cache it
        let compiled = match get_or_compile_jit(&preprocessed) {
            Some(jit_fn) => {
                // JIT compilation succeeded (from cache or fresh) - use fast path
                CompiledExpr::Jit(jit_fn)
            }
            None => {
                // JIT compilation failed - fall back to evalexpr
                // Build the operator tree once and cache it
                match build_operator_tree(&preprocessed) {
                    Ok(tree) => CompiledExpr::Generic(tree),
                    Err(_) => {
                        // If even tree building fails, create a dummy tree
                        // This will error during evaluation, but won't panic during construction
                        // We'll just use an empty/error tree
                        match build_operator_tree("0") {
                            Ok(tree) => CompiledExpr::Generic(tree),
                            Err(_) => unreachable!("Failed to build even a constant tree"),
                        }
                    }
                }
            }
        };

        Self {
            formula,
            preprocessed,
            compiled,
        }
    }

    /// Evaluate the expression with the given context
    pub fn evaluate(
        &self,
        context: &mut MathExpressionContext,
    ) -> Result<f64, MathExpressionError> {
        match &self.compiled {
            CompiledExpr::Jit(jit_fn) => {
                // Use JIT-compiled function (fast path)
                jit_fn.call(context)
            }
            CompiledExpr::Generic(tree) => {
                // Fallback to evalexpr interpretation
                let evalexpr_context = context.get_evalexpr_context();

                match tree.eval_with_context(evalexpr_context) {
                    Ok(Value::Float(result)) => Ok(result),
                    Ok(Value::Int(result)) => Ok(result as f64),
                    Ok(_) => Err(MathExpressionError::InvalidExpression(
                        "Expression must evaluate to a number".to_string(),
                    )),
                    Err(e) => Err(MathExpressionError::EvalError(e)),
                }
            }
        }
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
        context.set_value("N".to_string(), Value::Float(1.0)).ok();
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
}

/// Represents different types of rate expressions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
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
    /// - Single variable names become `Parameter` (e.g., "beta")
    /// - Complex expressions become `Formula` (e.g., "beta * 2")
    /// - Invalid expressions are treated as `Parameter` for backward compatibility
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
            || s.contains('(')
            || s.contains(')');

        if !has_operators {
            // Simple identifier - treat as parameter
            return Self::Parameter(s);
        }

        // Try to parse as expression - just validate it's parseable
        // Use turbofish to specify default numeric types
        if evalexpr::build_operator_tree::<evalexpr::DefaultNumericTypes>(&s).is_ok() {
            return Self::Formula(MathExpression::new(s));
        }

        // Fallback: treat as parameter for backward compatibility
        Self::Parameter(s)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_expressions() {
        let mut ctx = MathExpressionContext::new();
        ctx.set_parameter("beta".to_string(), 0.5);
        ctx.set_parameter("gamma".to_string(), 0.2);

        let expr = MathExpression::new("beta * gamma".to_string());
        let result = expr.evaluate(&mut ctx).unwrap();
        assert!((result - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_complex_expressions() {
        let mut ctx = MathExpressionContext::new();
        ctx.set_parameter("beta".to_string(), 0.5);
        ctx.set_compartment("S".to_string(), 990.0);
        ctx.set_compartment("I".to_string(), 10.0);
        ctx.set_compartment("R".to_string(), 0.0);

        // Test triple product (common in SIR models)
        let expr = MathExpression::new("beta * S * I".to_string());
        let result = expr.evaluate(&mut ctx).unwrap();
        assert!((result - 4950.0).abs() < 1e-10);

        // Test normalized infection rate
        let expr = MathExpression::new("beta * S * I / N".to_string());
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
    fn test_rate_math_expression() {
        let mut ctx = MathExpressionContext::new();
        ctx.set_parameter("beta".to_string(), 0.5);

        // Test parameter
        let rate = RateMathExpression::from_string("beta".to_string());
        assert!(matches!(rate, RateMathExpression::Parameter(_)));
        assert!((rate.evaluate(&mut ctx).unwrap() - 0.5).abs() < 1e-10);

        // Test constant
        let rate = RateMathExpression::from_string("0.3".to_string());
        assert!(matches!(rate, RateMathExpression::Constant(_)));
        assert!((rate.evaluate(&mut ctx).unwrap() - 0.3).abs() < 1e-10);

        // Test formula
        let rate = RateMathExpression::from_string("beta * 2".to_string());
        assert!(matches!(rate, RateMathExpression::Formula(_)));
        assert!((rate.evaluate(&mut ctx).unwrap() - 1.0).abs() < 1e-10);
    }
}
