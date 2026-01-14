//! Cranelift IR code generation
//!
//! This module generates Cranelift IR from our AST.

use super::ast::{BinaryOperator, Expr, UnaryOperator};
use super::libm::{
    LibmRegistry, is_single_arg_function, is_three_arg_function, is_two_arg_function,
    map_function_name,
};
use crate::math_expression::error::MathExpressionError;
use cranelift::prelude::*;
use cranelift_codegen::ir::InstBuilder;
use cranelift_module::Module;
use std::collections::HashMap;

/// Code generator that converts AST to Cranelift IR
pub struct CodeGenerator<'a, M: Module> {
    /// Cranelift function builder
    builder: FunctionBuilder<'a>,

    /// Map of variable names to their indices in the context array
    variable_indices: HashMap<String, usize>,

    /// Context pointer parameter (passed to the function)
    context_ptr: Value,

    /// Reference to the JIT module (for importing functions)
    module: &'a mut M,

    /// Registry of imported libm functions
    libm_registry: &'a mut LibmRegistry,
}

impl<'a, M: Module> CodeGenerator<'a, M> {
    /// Create a new code generator
    pub fn new(
        builder: FunctionBuilder<'a>,
        context_ptr: Value,
        variable_indices: HashMap<String, usize>,
        module: &'a mut M,
        libm_registry: &'a mut LibmRegistry,
    ) -> Self {
        Self {
            builder,
            variable_indices,
            context_ptr,
            module,
            libm_registry,
        }
    }

    /// Generate code for an expression, returning the Cranelift value
    pub fn generate(&mut self, expr: &Expr) -> Result<Value, MathExpressionError> {
        match expr {
            Expr::Constant(val) => {
                // Generate constant load
                Ok(self.builder.ins().f64const(*val))
            }

            Expr::Variable(name) => {
                // Load variable from context
                self.load_variable(name)
            }

            Expr::BinaryOp { op, left, right } => {
                // Generate code for operands
                let left_val = self.generate(left)?;
                let right_val = self.generate(right)?;

                // Generate operation
                self.generate_binary_op(*op, left_val, right_val)
            }

            Expr::UnaryOp { op, operand } => {
                // Generate code for operand
                let operand_val = self.generate(operand)?;

                // Generate operation
                self.generate_unary_op(*op, operand_val)
            }

            Expr::FunctionCall { name, args } => {
                // Generate function call
                self.generate_function_call(name, args)
            }

            Expr::Conditional {
                condition,
                true_expr,
                false_expr,
            } => {
                // Generate conditional using Cranelift's select instruction
                self.generate_conditional(condition, true_expr, false_expr)
            }
        }
    }

    /// Load a variable from the context
    fn load_variable(&mut self, name: &str) -> Result<Value, MathExpressionError> {
        // Special variables
        if name == "pi" {
            return Ok(self.builder.ins().f64const(std::f64::consts::PI));
        }
        if name == "e" {
            return Ok(self.builder.ins().f64const(std::f64::consts::E));
        }

        // Get variable index
        let index = self
            .variable_indices
            .get(name)
            .ok_or_else(|| MathExpressionError::VariableNotFound(name.to_string()))?;

        // Calculate offset: base_offset + index * sizeof(f64)
        // Context layout: [variables array at offset 0]
        let offset = (*index as i64) * 8; // 8 bytes per f64

        // Load from memory: context_ptr + offset
        let load_flags = MemFlags::trusted();
        Ok(self
            .builder
            .ins()
            .load(types::F64, load_flags, self.context_ptr, offset as i32))
    }

    /// Generate a binary operation
    fn generate_binary_op(
        &mut self,
        op: BinaryOperator,
        left: Value,
        right: Value,
    ) -> Result<Value, MathExpressionError> {
        let result = match op {
            // Arithmetic operations
            BinaryOperator::Add => self.builder.ins().fadd(left, right),
            BinaryOperator::Sub => self.builder.ins().fsub(left, right),
            BinaryOperator::Mul => self.builder.ins().fmul(left, right),
            BinaryOperator::Div => self.builder.ins().fdiv(left, right),

            BinaryOperator::Mod => {
                // Call libm fmod function
                return self.call_libm_f64_f64_to_f64("fmod", left, right);
            }

            BinaryOperator::Pow => {
                // Call libm pow function
                return self.call_libm_f64_f64_to_f64("pow", left, right);
            }

            // Comparison operations - return 1.0 or 0.0
            BinaryOperator::Lt => {
                let cmp = self.builder.ins().fcmp(FloatCC::LessThan, left, right);
                self.bool_to_float(cmp)
            }
            BinaryOperator::Gt => {
                let cmp = self.builder.ins().fcmp(FloatCC::GreaterThan, left, right);
                self.bool_to_float(cmp)
            }
            BinaryOperator::Le => {
                let cmp = self
                    .builder
                    .ins()
                    .fcmp(FloatCC::LessThanOrEqual, left, right);
                self.bool_to_float(cmp)
            }
            BinaryOperator::Ge => {
                let cmp = self
                    .builder
                    .ins()
                    .fcmp(FloatCC::GreaterThanOrEqual, left, right);
                self.bool_to_float(cmp)
            }
            BinaryOperator::Eq => {
                let cmp = self.builder.ins().fcmp(FloatCC::Equal, left, right);
                self.bool_to_float(cmp)
            }
            BinaryOperator::Ne => {
                let cmp = self.builder.ins().fcmp(FloatCC::NotEqual, left, right);
                self.bool_to_float(cmp)
            }

            // Logical operations
            BinaryOperator::And => {
                // Convert floats to booleans (non-zero = true)
                let zero = self.builder.ins().f64const(0.0);
                let left_bool = self.builder.ins().fcmp(FloatCC::NotEqual, left, zero);
                let right_bool = self.builder.ins().fcmp(FloatCC::NotEqual, right, zero);
                let result_bool = self.builder.ins().band(left_bool, right_bool);
                self.bool_to_float(result_bool)
            }
            BinaryOperator::Or => {
                // Convert floats to booleans (non-zero = true)
                let zero = self.builder.ins().f64const(0.0);
                let left_bool = self.builder.ins().fcmp(FloatCC::NotEqual, left, zero);
                let right_bool = self.builder.ins().fcmp(FloatCC::NotEqual, right, zero);
                let result_bool = self.builder.ins().bor(left_bool, right_bool);
                self.bool_to_float(result_bool)
            }
        };

        Ok(result)
    }

    /// Generate a unary operation
    fn generate_unary_op(
        &mut self,
        op: UnaryOperator,
        operand: Value,
    ) -> Result<Value, MathExpressionError> {
        let result = match op {
            UnaryOperator::Neg => {
                // Negate: 0.0 - operand
                let zero = self.builder.ins().f64const(0.0);
                self.builder.ins().fsub(zero, operand)
            }

            UnaryOperator::Not => {
                // Logical NOT: convert to boolean, negate, convert back to float
                let zero = self.builder.ins().f64const(0.0);
                let bool_val = self.builder.ins().fcmp(FloatCC::Equal, operand, zero);
                self.bool_to_float(bool_val)
            }
        };

        Ok(result)
    }

    /// Convert a boolean (i8) to float (1.0 or 0.0)
    fn bool_to_float(&mut self, bool_val: Value) -> Value {
        // Convert i8 (boolean) to i32, then to f64
        let int_val = self.builder.ins().uextend(types::I32, bool_val);
        self.builder.ins().fcvt_from_uint(types::F64, int_val)
    }

    /// Generate a function call
    fn generate_function_call(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<Value, MathExpressionError> {
        // Map function name (e.g., "math::sin" -> "sin")
        let libm_name = map_function_name(name);

        // Handle special multi-argument functions
        if name == "min" || name == "max" {
            return self.generate_min_max(name, args);
        }

        // Single-argument function
        if is_single_arg_function(name) {
            if args.len() != 1 {
                return Err(MathExpressionError::InvalidExpression(format!(
                    "Function {} requires exactly 1 argument, got {}",
                    name,
                    args.len()
                )));
            }

            let arg_val = self.generate(&args[0])?;
            return self.call_libm_f64_to_f64(libm_name, arg_val);
        }

        // Two-argument function
        if is_two_arg_function(name) {
            if args.len() != 2 {
                return Err(MathExpressionError::InvalidExpression(format!(
                    "Function {} requires exactly 2 arguments, got {}",
                    name,
                    args.len()
                )));
            }

            let arg1_val = self.generate(&args[0])?;
            let arg2_val = self.generate(&args[1])?;
            return self.call_libm_f64_f64_to_f64(libm_name, arg1_val, arg2_val);
        }

        // Three-argument function
        if is_three_arg_function(name) {
            if args.len() != 3 {
                return Err(MathExpressionError::InvalidExpression(format!(
                    "Function {} requires exactly 3 arguments, got {}",
                    name,
                    args.len()
                )));
            }

            let arg1_val = self.generate(&args[0])?;
            let arg2_val = self.generate(&args[1])?;
            let arg3_val = self.generate(&args[2])?;
            return self.call_libm_f64_f64_f64_to_f64(libm_name, arg1_val, arg2_val, arg3_val);
        }

        Err(MathExpressionError::InvalidExpression(format!(
            "Unknown function: {}",
            name
        )))
    }

    /// Generic helper to call a libm function with any number of arguments
    fn call_libm_function(
        &mut self,
        name: &str,
        args: &[Value],
    ) -> Result<Value, MathExpressionError> {
        // Import the function based on argument count
        let func_id = match args.len() {
            1 => self.libm_registry.import_f64_to_f64(self.module, name)?,
            2 => self
                .libm_registry
                .import_f64_f64_to_f64(self.module, name)?,
            3 => self
                .libm_registry
                .import_f64_f64_f64_to_f64(self.module, name)?,
            n => {
                return Err(MathExpressionError::InvalidExpression(format!(
                    "Unsupported argument count {} for libm function {}",
                    n, name
                )));
            }
        };

        // Get function reference in current function
        let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);

        // Generate call
        let call = self.builder.ins().call(func_ref, args);

        // Get return value
        let results = self.builder.inst_results(call);
        Ok(results[0])
    }

    /// Call a single-argument libm function
    #[inline]
    fn call_libm_f64_to_f64(
        &mut self,
        name: &str,
        arg: Value,
    ) -> Result<Value, MathExpressionError> {
        self.call_libm_function(name, &[arg])
    }

    /// Call a two-argument libm function
    #[inline]
    fn call_libm_f64_f64_to_f64(
        &mut self,
        name: &str,
        arg1: Value,
        arg2: Value,
    ) -> Result<Value, MathExpressionError> {
        self.call_libm_function(name, &[arg1, arg2])
    }

    /// Call a three-argument libm function
    #[inline]
    fn call_libm_f64_f64_f64_to_f64(
        &mut self,
        name: &str,
        arg1: Value,
        arg2: Value,
        arg3: Value,
    ) -> Result<Value, MathExpressionError> {
        self.call_libm_function(name, &[arg1, arg2, arg3])
    }

    /// Generate min/max function (variadic)
    fn generate_min_max(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<Value, MathExpressionError> {
        if args.is_empty() {
            return Err(MathExpressionError::InvalidExpression(format!(
                "Function {} requires at least 1 argument",
                name
            )));
        }

        // Generate first argument
        let mut result = self.generate(&args[0])?;

        // Compare with remaining arguments
        for arg in &args[1..] {
            let arg_val = self.generate(arg)?;

            // Compare and select
            // For min: if arg_val < result, select arg_val, else select result
            // For max: if arg_val > result, select arg_val, else select result
            let cmp = if name == "min" {
                self.builder.ins().fcmp(FloatCC::LessThan, arg_val, result)
            } else {
                self.builder
                    .ins()
                    .fcmp(FloatCC::GreaterThan, arg_val, result)
            };

            // Select based on comparison: if cmp is true, use arg_val, else use result
            result = self.builder.ins().select(cmp, arg_val, result);
        }

        Ok(result)
    }

    /// Generate a conditional expression (if/then/else)
    fn generate_conditional(
        &mut self,
        condition: &Expr,
        true_expr: &Expr,
        false_expr: &Expr,
    ) -> Result<Value, MathExpressionError> {
        // Generate condition (result is a float)
        let condition_val = self.generate(condition)?;

        // Convert float to boolean (non-zero = true)
        let zero = self.builder.ins().f64const(0.0);
        let condition_bool = self
            .builder
            .ins()
            .fcmp(FloatCC::NotEqual, condition_val, zero);

        // Generate true and false branches
        let true_val = self.generate(true_expr)?;
        let false_val = self.generate(false_expr)?;

        // Use select instruction to choose based on condition
        Ok(self
            .builder
            .ins()
            .select(condition_bool, true_val, false_val))
    }

    /// Finalize code generation and return the function
    pub fn finalize(mut self) {
        // Seal all blocks (required by Cranelift)
        self.builder.seal_all_blocks();
        self.builder.finalize();
    }

    /// Get a mutable reference to the builder (for external use)
    pub fn builder_mut(&mut self) -> &mut FunctionBuilder<'a> {
        &mut self.builder
    }
}

#[cfg(test)]
mod tests {
    // Note: Full integration tests will be in the main JIT module
    // These are just structural tests
}
