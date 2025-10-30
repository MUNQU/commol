//! # Commol Core
//!
//! Core data structures and abstractions for compartment modeling.
//!
//! This crate provides:
//! - Mathematical expression parsing and evaluation
//! - Common simulation engine interface
//! - Type definitions for models, populations, and dynamics
//!
//! ## Main Components
//!
//! - [`MathExpression`]: Parse and evaluate mathematical expressions
//! - [`SimulationEngine`]: Trait for implementing simulation engines
//! - [`Model`]: Complete model specification including population and dynamics
//! - [`types`]: All data structures for defining compartment models

pub mod math_expression;
pub mod simulation_engine;
pub mod types;

// Re-export commonly used types from math_expression
pub use math_expression::{
    MathExpression, MathExpressionContext, MathExpressionError, RateMathExpression,
};

// Re-export simulation engine trait
pub use simulation_engine::SimulationEngine;

// Re-export all public types from the types module for convenient access
pub use types::{
    Bin, BinFraction, Condition, Dynamics, InitialConditions, LogicOperator, Model, ModelTypes,
    Parameter, ParameterValue, Population, Rule, RuleValue, Stratification,
    StratificationCondition, StratificationFraction, StratificationFractions, StratifiedRate,
    Transition, VariablePrefixes,
};
