//! JIT compiler for mathematical expressions
//!
//! This module provides the main JIT compilation interface.

use super::ast::Expr;
use super::codegen::CodeGenerator;
use super::libm::LibmRegistry;
use super::parser::parse_expression;
use crate::math_expression::MathExpressionContext;
use crate::math_expression::error::MathExpressionError;
use cranelift::prelude::*;
use cranelift_codegen::ir::AbiParam;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A compiled JIT function that can be called to evaluate an expression
pub struct JITFunction {
    /// The compiled function pointer
    func_ptr: extern "C" fn(*const f64) -> f64,

    /// Keep the JIT module alive (it owns the executable memory)
    /// Wrapped in Arc<Mutex<>> for safe sharing across threads
    /// The Mutex is never actually locked after creation - it only exists for Send+Sync
    _module: Arc<Mutex<JITModule>>,

    /// Ordered list of variable names (for context preparation)
    variable_names: Vec<String>,
}

impl std::fmt::Debug for JITFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JITFunction")
            .field(
                "func_ptr",
                &format_args!("{:p}", self.func_ptr as *const ()),
            )
            .field("variable_names", &self.variable_names)
            .finish()
    }
}

impl Clone for JITFunction {
    fn clone(&self) -> Self {
        Self {
            func_ptr: self.func_ptr,
            _module: Arc::clone(&self._module),
            variable_names: self.variable_names.clone(),
        }
    }
}

impl JITFunction {
    /// Call the JIT-compiled function with a context
    pub fn call(&self, context: &MathExpressionContext) -> Result<f64, MathExpressionError> {
        // Prepare a flat array of variable values
        let mut values = vec![0.0; self.variable_names.len()];

        for (i, name) in self.variable_names.iter().enumerate() {
            let value = if name == "step" || name == "t" {
                context.step
            } else if name == "N" {
                // Sum all compartments
                context.compartments.values().sum()
            } else if name.starts_with("N_") {
                // Stratified population sum
                let suffix = &name[1..]; // Keep "_age_young" etc.
                context
                    .compartments
                    .iter()
                    .filter(|(comp_name, _)| comp_name.ends_with(suffix))
                    .map(|(_, val)| val)
                    .sum()
            } else {
                // Try parameter first, then compartment
                context
                    .get_parameter(name)
                    .or_else(|| context.compartments.get(name).copied())
                    .ok_or_else(|| MathExpressionError::VariableNotFound(name.clone()))?
            };

            values[i] = value;
        }

        // Call the JIT function with pointer to values array
        let result = (self.func_ptr)(values.as_ptr());

        Ok(result)
    }

    /// Get the list of variables used by this function
    pub fn variables(&self) -> &[String] {
        &self.variable_names
    }
}

/// JIT compiler for mathematical expressions
pub struct JITCompiler;

impl JITCompiler {
    /// Create a new JIT compiler
    pub fn new() -> Result<Self, MathExpressionError> {
        Ok(Self)
    }

    /// Compile a preprocessed expression to a JIT function
    pub fn compile(&self, preprocessed: &str) -> Result<JITFunction, MathExpressionError> {
        // Create a new JIT module for this function
        let builder = JITBuilder::new(cranelift_module::default_libcall_names()).map_err(|e| {
            MathExpressionError::InvalidExpression(format!("Failed to create JIT builder: {}", e))
        })?;
        let mut module = JITModule::new(builder);
        // Step 1: Parse to AST
        let ast = parse_expression(preprocessed)?;

        // Step 2: Collect all variables in the expression
        let variable_names = collect_variables(&ast);
        let mut variable_indices = HashMap::new();
        for (i, name) in variable_names.iter().enumerate() {
            variable_indices.insert(name.clone(), i);
        }

        // Step 3: Define function signature
        // extern "C" fn(context_ptr: *const f64) -> f64
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64)); // context pointer
        sig.returns.push(AbiParam::new(types::F64)); // return f64

        // Step 4: Declare function
        let func_id = module
            .declare_function("eval_expr", Linkage::Export, &sig)
            .map_err(|e| {
                MathExpressionError::InvalidExpression(format!("Failed to declare function: {}", e))
            })?;

        // Step 5: Build function body
        let mut ctx = module.make_context();
        ctx.func.signature = sig;

        let mut func_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_builder_ctx);

        // Create entry block
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        // Get the context pointer parameter
        let context_ptr = builder.block_params(entry_block)[0];

        // Step 6: Generate code
        let mut libm_registry = LibmRegistry::new();
        let mut codegen = CodeGenerator::new(
            builder,
            context_ptr,
            variable_indices.clone(),
            &mut module,
            &mut libm_registry,
        );
        let result_val = codegen.generate(&ast)?;

        // Step 7: Return the result
        codegen.builder_mut().ins().return_(&[result_val]);
        codegen.finalize();

        // Step 8: Compile to machine code
        module.define_function(func_id, &mut ctx).map_err(|e| {
            MathExpressionError::InvalidExpression(format!("Failed to define function: {}", e))
        })?;

        module.clear_context(&mut ctx);

        // Step 9: Finalize and get function pointer
        module.finalize_definitions().map_err(|e| {
            MathExpressionError::InvalidExpression(format!("Failed to finalize definitions: {}", e))
        })?;

        let code_ptr = module.get_finalized_function(func_id);

        // Step 10: Convert code pointer to function pointer
        // This is the only unavoidable unsafe in JIT compilation - we must convert
        // a raw memory pointer to executable code into a callable function pointer.
        // We wrap this in a helper function with explicit safety documentation.
        let func_ptr = Self::code_ptr_to_fn(code_ptr);

        Ok(JITFunction {
            func_ptr,
            _module: Arc::new(Mutex::new(module)),
            variable_names,
        })
    }

    /// Convert a code pointer from Cranelift to a callable function pointer
    ///
    /// # Safety
    ///
    /// This function performs an unsafe transmute from *const u8 to a function pointer.
    /// This is safe only when:
    /// 1. The pointer comes from Cranelift's `get_finalized_function()`
    /// 2. The function signature matches the declared Cranelift signature
    /// 3. The JIT module remains alive for the lifetime of the function pointer
    ///
    /// All these conditions are guaranteed by the JITCompiler's design.
    #[inline]
    fn code_ptr_to_fn(code_ptr: *const u8) -> extern "C" fn(*const f64) -> f64 {
        // This is the standard way to call JIT-compiled code
        // It's used by all JIT compilers (LLVM, Cranelift, etc.)
        unsafe { std::mem::transmute(code_ptr) }
    }
}

/// Collect all variable names used in an expression
fn collect_variables(expr: &Expr) -> Vec<String> {
    let mut vars = Vec::new();
    collect_variables_recursive(expr, &mut vars);

    // Remove duplicates while preserving order
    let mut seen = HashMap::new();
    let mut result = Vec::new();
    for var in vars {
        if !seen.contains_key(&var) {
            seen.insert(var.clone(), true);
            result.push(var);
        }
    }

    result
}

/// Recursively collect variables from an expression
fn collect_variables_recursive(expr: &Expr, vars: &mut Vec<String>) {
    match expr {
        Expr::Variable(name) => {
            // Skip special constants that are compiled directly
            if name != "pi" && name != "e" {
                vars.push(name.clone());
            }
        }
        Expr::BinaryOp { left, right, .. } => {
            collect_variables_recursive(left, vars);
            collect_variables_recursive(right, vars);
        }
        Expr::UnaryOp { operand, .. } => {
            collect_variables_recursive(operand, vars);
        }
        Expr::FunctionCall { args, .. } => {
            for arg in args {
                collect_variables_recursive(arg, vars);
            }
        }
        Expr::Conditional {
            condition,
            true_expr,
            false_expr,
        } => {
            collect_variables_recursive(condition, vars);
            collect_variables_recursive(true_expr, vars);
            collect_variables_recursive(false_expr, vars);
        }
        Expr::Constant(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_variables() {
        let expr = Expr::binary(
            super::super::ast::BinaryOperator::Mul,
            Expr::Variable("beta".to_string()),
            Expr::Variable("gamma".to_string()),
        );

        let vars = collect_variables(&expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&"beta".to_string()));
        assert!(vars.contains(&"gamma".to_string()));
    }

    #[test]
    fn test_collect_variables_deduplicate() {
        let expr = Expr::binary(
            super::super::ast::BinaryOperator::Add,
            Expr::Variable("beta".to_string()),
            Expr::Variable("beta".to_string()),
        );

        let vars = collect_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0], "beta");
    }
}
