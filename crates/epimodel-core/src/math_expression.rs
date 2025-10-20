//! # Mathematical Expression Evaluation
//!
//! This module provides mathematical expression evaluation for epidemiological models.
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
//!
//! ### Other
//! - `min(a, b, ...)` - Minimum value
//! - `max(a, b, ...)` - Maximum value
//! - `if(condition, value_if_true, value_if_false)` - Conditional expression
//!
//! ## Example
//! ```rust
//! use epimodel_core::{MathExpression, MathExpressionContext};
//!
//! let expr = MathExpression::new("beta * sin(2 * pi * t / 365)".to_string());
//! let mut context = MathExpressionContext::new();
//! context.set_parameter("beta".to_string(), 0.3);
//! context.set_parameter("t".to_string(), 91.25); // Day 91.25 of year
//! let result = expr.evaluate(&context).unwrap();
//! ```

use evalexpr::{
    ContextWithMutableVariables, EvalexprError, HashMapContext, Node, Value, build_operator_tree,
    eval_with_context,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// Special variable names that are automatically available in all expressions
const SPECIAL_VAR_N: &str = "N";
const SPECIAL_VAR_STEP: &str = "step";
const SPECIAL_VAR_T: &str = "t";
const SPECIAL_VAR_PI: &str = "pi";
const SPECIAL_VAR_E: &str = "e";

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

/// Context for evaluating mathematical expressions
#[derive(Debug, Clone)]
pub struct MathExpressionContext {
    parameters: HashMap<String, f64>,
    compartments: HashMap<String, f64>,
    step: f64,
}

impl MathExpressionContext {
    /// Create a new expression context
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            compartments: HashMap::new(),
            step: 0.0,
        }
    }

    /// Set a parameter value
    pub fn set_parameter(&mut self, name: String, value: f64) {
        self.parameters.insert(name, value);
    }

    /// Set multiple parameters
    pub fn set_parameters(&mut self, parameters: HashMap<String, f64>) {
        self.parameters.extend(parameters);
    }

    /// Set a compartment population value
    pub fn set_compartment(&mut self, name: String, value: f64) {
        self.compartments.insert(name, value);
    }

    /// Set the current step
    pub fn set_step(&mut self, step: f64) {
        self.step = step;
    }

    /// Get a single parameter value by name
    pub fn get_parameter(&self, name: &str) -> Option<f64> {
        self.parameters.get(name).copied()
    }

    /// Get a reference to all parameters
    pub fn get_parameters(&self) -> &HashMap<String, f64> {
        &self.parameters
    }

    /// Convert to evalexpr context
    fn to_evalexpr_context(&self) -> HashMapContext {
        let mut context = HashMapContext::new();

        // Add user-defined parameters
        for (name, value) in &self.parameters {
            context.set_value(name.clone(), Value::Float(*value)).ok();
        }

        // Add compartment values
        for (name, value) in &self.compartments {
            context.set_value(name.clone(), Value::Float(*value)).ok();
        }

        // Add special variables
        let total_pop: f64 = self.compartments.values().sum();
        context
            .set_value(SPECIAL_VAR_N.to_string(), Value::Float(total_pop))
            .ok();
        context
            .set_value(SPECIAL_VAR_STEP.to_string(), Value::Float(self.step))
            .ok();
        context
            .set_value(SPECIAL_VAR_T.to_string(), Value::Float(self.step))
            .ok();
        context
            .set_value(
                SPECIAL_VAR_PI.to_string(),
                Value::Float(std::f64::consts::PI),
            )
            .ok();
        context
            .set_value(SPECIAL_VAR_E.to_string(), Value::Float(std::f64::consts::E))
            .ok();

        context
    }
}

impl Default for MathExpressionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Preprocesses a formula by adding `math::` prefix to mathematical functions that require it
/// and converting Python-style operators to evalexpr syntax.
///
/// Transformations:
/// - `sin(x)` → `math::sin(x)` (adds prefix for math functions)
/// - `**` → `^` (converts Python power operator to evalexpr syntax)
///
/// Note: Functions like `min`, `max`, `floor`, `ceil`, `round`, `if` are available
/// without the `math::` prefix in evalexpr and are not modified.
fn preprocess_formula(formula: &str) -> String {
    // Functions that need math:: prefix (not available as built-ins)
    // Order matters. Longer function names must come first to avoid
    // replacing substrings (e.g., process "asin" before "sin")
    // Note: "log" is not in this list - it's handled separately below
    const MATH_FUNCTIONS: &[&str] = &[
        "asinh", "acosh", "atanh", "asin", "acos", "atan2", "atan", "sinh", "cosh", "tanh", "sin",
        "cos", "tan", "log10", "log2", "ln", "cbrt", "sqrt", "hypot", "exp", "abs", "pow",
    ];

    // First, convert Python-style ** to evalexpr-style ^
    let mut result = formula.replace("**", "^");

    // Replace log( with ln( since evalexpr's math::log requires 2 arguments
    // Process log10 and log2 first (already in MATH_FUNCTIONS), then replace remaining log(
    result = result.replace("log(", "ln(");

    for func in MATH_FUNCTIONS {
        // Replace "func(" with "math::func(" but avoid replacing if already prefixed
        let pattern = format!("{}(", func);
        let replacement = format!("math::{}(", func);

        // We need to check if the match is a valid function call, not part of another identifier
        // Valid function call: preceded by whitespace, operator, or start of string
        let mut new_result = String::new();
        let mut remaining = result.as_str();

        while let Some(pos) = remaining.find(&pattern) {
            // Check if it's already prefixed with "math::"
            let prefix_start = pos.saturating_sub(6);
            let prefix = &remaining[prefix_start..pos];

            // Check if this is a valid function boundary (not part of another identifier)
            let is_valid_boundary = if pos == 0 {
                true // Start of string
            } else {
                let prev_char = remaining.chars().nth(pos - 1);
                match prev_char {
                    Some(c) => !c.is_alphanumeric() && c != '_',
                    None => true,
                }
            };

            if is_valid_boundary && !prefix.ends_with("math::") {
                new_result.push_str(&remaining[..pos]);
                new_result.push_str(&replacement);
                remaining = &remaining[pos + pattern.len()..];
            } else {
                new_result.push_str(&remaining[..pos + pattern.len()]);
                remaining = &remaining[pos + pattern.len()..];
            }
        }
        new_result.push_str(remaining);
        result = new_result;
    }

    result
}

/// Collects all variable identifiers from an AST node, excluding special variables.
fn get_variables_from_node(node: &Node, variables: &mut HashSet<String>) {
    for ident in node.iter_variable_identifiers() {
        // Exclude special variables that are automatically available
        if ident != SPECIAL_VAR_PI
            && ident != SPECIAL_VAR_E
            && ident != SPECIAL_VAR_N
            && ident != SPECIAL_VAR_STEP
            && ident != SPECIAL_VAR_T
        {
            variables.insert(ident.to_string());
        }
    }
}

/// Checks if an AST node is a single variable identifier.
fn is_single_identifier(node: &Node) -> bool {
    let mut var_count = 0;
    for _ in node.iter_variable_identifiers() {
        var_count += 1;
        if var_count > 1 {
            return false;
        }
    }

    var_count == 1 && node.children().is_empty()
}

/// A mathematical expression that can be evaluated
///
/// The formula is preprocessed once during construction for optimal performance.
#[derive(Debug, Clone, Serialize)]
pub struct MathExpression {
    /// The original mathematical formula as a string
    pub formula: String,
    /// Preprocessed formula (cached for performance)
    #[serde(skip)]
    preprocessed: String,
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
    /// Create a new expression (preprocesses the formula once)
    pub fn new(formula: String) -> Self {
        let preprocessed = preprocess_formula(&formula);
        Self {
            formula,
            preprocessed,
        }
    }

    /// Evaluate the expression with the given context
    pub fn evaluate(&self, context: &MathExpressionContext) -> Result<f64, MathExpressionError> {
        let evalexpr_context = context.to_evalexpr_context();

        match eval_with_context(&self.preprocessed, &evalexpr_context) {
            Ok(Value::Float(result)) => Ok(result),
            Ok(Value::Int(result)) => Ok(result as f64),
            Ok(_) => Err(MathExpressionError::InvalidExpression(
                "Expression must evaluate to a number".to_string(),
            )),
            Err(e) => Err(MathExpressionError::EvalError(e)),
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
        let mut variables = HashSet::new();
        if let Ok(node) = build_operator_tree(&self.preprocessed) {
            get_variables_from_node(&node, &mut variables);
        }
        variables.into_iter().collect()
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
        if let Ok(value) = s.parse::<f64>() {
            return Self::Constant(value);
        }

        if let Ok(node) = build_operator_tree(&s) {
            if is_single_identifier(&node) {
                return Self::Parameter(s);
            }
            return Self::Formula(MathExpression::new(s));
        }

        Self::Parameter(s)
    }

    /// Evaluate the rate expression
    pub fn evaluate(&self, context: &MathExpressionContext) -> Result<f64, MathExpressionError> {
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
