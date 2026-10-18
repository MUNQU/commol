//! libm function imports for JIT compilation
//!
//! JIT-compiled code calls each math function through [`address`], so the JIT
//! never looks a symbol up in the process: a Linux binary links the system
//! libm only for the functions Rust calls. A function of the standard library
//! comes from it, most of them from the system libm; the others (`erf`,
//! `erfc`, `tgamma`, `lgamma`, the Bessel functions, `fdim`, `remainder`)
//! come from the `libm` crate, a port of musl in Rust.

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
    // Bessel functions
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

/// The address of the C-ABI function that JIT code calls for the libm name
/// `name`, or `None` when no supported function has that name.
pub fn address(name: &str) -> Option<*const u8> {
    macro_rules! addresses {
        ($($function:ident($($argument:ident),+) => $value:expr),+ $(,)?) => {
            match name {
                $(stringify!($function) => {
                    extern "C" fn function($($argument: f64),+) -> f64 {
                        $value
                    }
                    Some(function as *const u8)
                })+
                _ => None,
            }
        };
    }
    addresses!(
        sin(x) => x.sin(),
        cos(x) => x.cos(),
        tan(x) => x.tan(),
        asin(x) => x.asin(),
        acos(x) => x.acos(),
        atan(x) => x.atan(),
        sinh(x) => x.sinh(),
        cosh(x) => x.cosh(),
        tanh(x) => x.tanh(),
        asinh(x) => x.asinh(),
        acosh(x) => x.acosh(),
        atanh(x) => x.atanh(),
        exp(x) => x.exp(),
        log(x) => x.ln(),
        log2(x) => x.log2(),
        log10(x) => x.log10(),
        sqrt(x) => x.sqrt(),
        cbrt(x) => x.cbrt(),
        fabs(x) => x.abs(),
        floor(x) => x.floor(),
        ceil(x) => x.ceil(),
        round(x) => x.round(),
        trunc(x) => x.trunc(),
        pow(x, y) => x.powf(y),
        atan2(y, x) => y.atan2(x),
        hypot(x, y) => x.hypot(y),
        fmod(x, y) => x % y,
        copysign(x, y) => x.copysign(y),
        fmax(x, y) => x.max(y),
        fmin(x, y) => x.min(y),
        fma(x, y, z) => x.mul_add(y, z),
        erf(x) => libm::erf(x),
        erfc(x) => libm::erfc(x),
        tgamma(x) => libm::tgamma(x),
        lgamma(x) => libm::lgamma(x),
        j0(x) => libm::j0(x),
        j1(x) => libm::j1(x),
        y0(x) => libm::y0(x),
        y1(x) => libm::y1(x),
        fdim(x, y) => libm::fdim(x, y),
        remainder(x, y) => libm::remainder(x, y),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_supported_function_has_an_address() {
        let names = SINGLE_ARG_FUNCTIONS
            .iter()
            .chain(TWO_ARG_FUNCTIONS)
            .chain(THREE_ARG_FUNCTIONS);
        for name in names {
            assert!(address(map_function_name(name)).is_some(), "{name}");
        }
        assert!(address("system").is_none());
    }

    fn call2(name: &str, x: f64, y: f64) -> f64 {
        let address = address(name).unwrap();
        // SAFETY: `address` gives a two-argument function for these names.
        let function: extern "C" fn(f64, f64) -> f64 = unsafe { std::mem::transmute(address) };
        function(x, y)
    }

    #[test]
    fn the_functions_keep_the_meaning_of_c() {
        assert_eq!(call2("atan2", 1.0, 0.0), std::f64::consts::FRAC_PI_2);
        assert_eq!(call2("fmod", -7.0, 3.0), -1.0);
        assert_eq!(call2("remainder", 7.0, 3.0), 1.0);
        assert_eq!(call2("fdim", 2.0, 5.0), 0.0);
        assert_eq!(call2("fmax", f64::NAN, 1.0), 1.0);
        assert_eq!(call2("fmin", 1.0, f64::NAN), 1.0);
        let fma = address("fma").unwrap();
        // SAFETY: `address` gives a three-argument function for `fma`.
        let fma: extern "C" fn(f64, f64, f64) -> f64 = unsafe { std::mem::transmute(fma) };
        assert_eq!(fma(2.0, 3.0, 4.0), 10.0);
    }

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
