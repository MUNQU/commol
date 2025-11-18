//! libm function imports for JIT compilation
//!
//! This module provides infrastructure for importing and calling
//! standard math functions from libm in JIT-compiled code.

use crate::math_expression::error::MathExpressionError;
use cranelift::prelude::*;
use cranelift_module::{FuncId, Linkage, Module};
use std::collections::HashMap;

/// Registry of imported libm functions
#[derive(Default)]
pub struct LibmRegistry {
    /// Map of function name to Cranelift function ID
    functions: HashMap<String, FuncId>,
}

impl LibmRegistry {
    /// Create a new libm registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Generic helper to import a libm function with a given number of f64 parameters
    fn import_function<M: Module>(
        &mut self,
        module: &mut M,
        name: &str,
        param_count: usize,
    ) -> Result<FuncId, MathExpressionError> {
        // Check if already imported
        if let Some(&func_id) = self.functions.get(name) {
            return Ok(func_id);
        }

        // Create signature with `param_count` f64 parameters and f64 return
        let mut sig = module.make_signature();
        for _ in 0..param_count {
            sig.params.push(AbiParam::new(types::F64));
        }
        sig.returns.push(AbiParam::new(types::F64));

        // Declare external function
        let func_id = module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| {
                MathExpressionError::InvalidExpression(format!(
                    "Failed to declare libm function '{}': {}",
                    name, e
                ))
            })?;

        self.functions.insert(name.to_string(), func_id);
        Ok(func_id)
    }

    /// Import a single-argument f64 function from libm
    pub fn import_f64_to_f64<M: Module>(
        &mut self,
        module: &mut M,
        name: &str,
    ) -> Result<FuncId, MathExpressionError> {
        self.import_function(module, name, 1)
    }

    /// Import a two-argument f64 function from libm
    pub fn import_f64_f64_to_f64<M: Module>(
        &mut self,
        module: &mut M,
        name: &str,
    ) -> Result<FuncId, MathExpressionError> {
        self.import_function(module, name, 2)
    }

    /// Import a three-argument f64 function from libm
    pub fn import_f64_f64_f64_to_f64<M: Module>(
        &mut self,
        module: &mut M,
        name: &str,
    ) -> Result<FuncId, MathExpressionError> {
        self.import_function(module, name, 3)
    }

    /// Get a function ID by name (if already imported)
    pub fn get(&self, name: &str) -> Option<FuncId> {
        self.functions.get(name).copied()
    }
}

/// List of all single-argument math functions we support
pub const SINGLE_ARG_FUNCTIONS: &[&str] = &[
    // Trigonometric
    "sin", "cos", "tan", "asin", "acos", "atan", // Hyperbolic
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", // Exponential and logarithmic
    "exp", "ln", "log", "log2", "log10", // Roots and absolute value
    "sqrt", "cbrt", "abs", // Rounding
    "floor", "ceil", "round", "trunc",  // Special functions
    "erf",    // Error function
    "erfc",   // Complementary error function
    "tgamma", // Gamma function
    "lgamma", // Log gamma function
    // Bessel functions (may not be available on all platforms)
    "j0", "j1", // Bessel functions of first kind
    "y0", "y1", // Bessel functions of second kind
];

/// List of all two-argument math functions we support
pub const TWO_ARG_FUNCTIONS: &[&str] = &[
    "pow",       // x^y
    "atan2",     // atan(y/x) with correct quadrant
    "hypot",     // sqrt(x^2 + y^2)
    "fmod",      // floating-point modulo
    "copysign",  // copy sign of y to x
    "fdim",      // positive difference: max(x-y, 0)
    "fmax",      // maximum (NaN-aware)
    "fmin",      // minimum (NaN-aware)
    "remainder", // IEEE remainder
];

/// List of all three-argument math functions we support
pub const THREE_ARG_FUNCTIONS: &[&str] = &[
    "fma", // fused multiply-add: x*y+z
];

/// Map user-friendly function names to libm names
pub fn map_function_name(name: &str) -> &str {
    // Handle preprocessing transformations
    match name {
        "math::sin" => "sin",
        "math::cos" => "cos",
        "math::tan" => "tan",
        "math::asin" => "asin",
        "math::acos" => "acos",
        "math::atan" => "atan",
        "math::sinh" => "sinh",
        "math::cosh" => "cosh",
        "math::tanh" => "tanh",
        "math::asinh" => "asinh",
        "math::acosh" => "acosh",
        "math::atanh" => "atanh",
        "math::exp" => "exp",
        "math::ln" => "log", // ln -> log (natural log in C)
        "math::log" => "log",
        "math::log2" => "log2",
        "math::log10" => "log10",
        "math::sqrt" => "sqrt",
        "math::cbrt" => "cbrt",
        "math::abs" => "fabs", // abs -> fabs for floating-point
        "math::floor" => "floor",
        "math::ceil" => "ceil",
        "math::round" => "round",
        "math::pow" => "pow",
        "math::atan2" => "atan2",
        "math::hypot" => "hypot",

        // Direct names (without math:: prefix)
        "ln" => "log",
        "abs" => "fabs",

        // Default: return as-is
        _ => name,
    }
}

/// Check if a function is a single-argument function
pub fn is_single_arg_function(name: &str) -> bool {
    let mapped = map_function_name(name);
    SINGLE_ARG_FUNCTIONS.contains(&mapped)
        || mapped == "log"  // ln maps to log
        || mapped == "fabs" // abs maps to fabs
}

/// Check if a function is a two-argument function
pub fn is_two_arg_function(name: &str) -> bool {
    let mapped = map_function_name(name);
    TWO_ARG_FUNCTIONS.contains(&mapped)
}

/// Check if a function is a three-argument function
pub fn is_three_arg_function(name: &str) -> bool {
    let mapped = map_function_name(name);
    THREE_ARG_FUNCTIONS.contains(&mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_name_mapping() {
        assert_eq!(map_function_name("math::sin"), "sin");
        assert_eq!(map_function_name("math::ln"), "log");
        assert_eq!(map_function_name("math::abs"), "fabs");
        assert_eq!(map_function_name("ln"), "log");
        assert_eq!(map_function_name("abs"), "fabs");
    }

    #[test]
    fn test_function_classification() {
        assert!(is_single_arg_function("math::sin"));
        assert!(is_single_arg_function("math::ln"));
        assert!(is_two_arg_function("math::pow"));
        assert!(is_two_arg_function("math::atan2"));
    }
}
