//! The math functions JIT code calls.
//!
//! JIT-compiled code calls each function through [`address`], so the JIT
//! never looks a symbol up in the process: a Linux binary links the system
//! libm only for the functions Rust calls. Each function is the method of
//! the Rust `f64` of the same meaning, most of them from the system libm.
//! The functions are those of the language
//! ([`FUNCTIONS`](crate::math_expression::language::FUNCTIONS)), with
//! `fmod` for `%` and `pow` for `^`.

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

    /// Import the C function `name` of `arguments` f64 arguments and an f64
    /// result, once per module.
    ///
    /// # Errors
    ///
    /// Returns a message when Cranelift cannot declare the function.
    pub fn import<M: Module>(
        &mut self,
        module: &mut M,
        name: &str,
        arguments: usize,
    ) -> Result<FuncId, MathExpressionError> {
        if let Some(&func_id) = self.functions.get(name) {
            return Ok(func_id);
        }

        let mut sig = module.make_signature();
        for _ in 0..arguments {
            sig.params.push(AbiParam::new(types::F64));
        }
        sig.returns.push(AbiParam::new(types::F64));

        let func_id = module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| {
                // coverage: an error that only Cranelift returns, an import declared twice
                MathExpressionError::InvalidExpression(format!(
                    "Failed to declare libm function '{}': {}",
                    name, e
                ))
            })?;

        self.functions.insert(name.to_string(), func_id);
        Ok(func_id)
    }
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
        pow(x, y) => x.powf(y),
        atan2(y, x) => y.atan2(x),
        hypot(x, y) => x.hypot(y),
        fmod(x, y) => x % y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_expression::language::FUNCTIONS;

    #[test]
    fn every_function_of_the_language_has_an_address() {
        for function in FUNCTIONS {
            if let Some(symbol) = function.symbol {
                assert!(address(symbol).is_some(), "{}", function.name);
            }
        }
        assert!(address("fmod").is_some());
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
    }
}
