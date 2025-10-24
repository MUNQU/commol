//! JIT compilation module for mathematical expressions
//!
//! This module provides Just-In-Time compilation of mathematical expressions
//! to native machine code using Cranelift, providing significant performance
//! improvements over interpreted evaluation.
//!
//! ## Architecture
//!
//! 1. **Parser** (`parser.rs`): Converts preprocessed formula strings to AST
//! 2. **AST** (`ast.rs`): Abstract syntax tree representation
//! 3. **Code Generator** (`codegen.rs`): Converts AST to Cranelift IR
//! 4. **Compiler** (`compiler.rs`): Main JIT compilation interface
//!
//! ## Usage
//!
//! ```rust,ignore
//! use commol_core::math_expression::jit::{JITCompiler, JITFunction};
//!
//! let mut compiler = JITCompiler::new()?;
//! let jit_fn = compiler.compile("beta * gamma")?;
//! let result = jit_fn.call(&context)?;
//! ```

pub mod ast;
pub mod codegen;
pub mod compiler;
pub mod libm;
pub mod parser;

#[cfg(test)]
mod tests;

// Re-export main types
pub use compiler::{JITCompiler, JITFunction};
