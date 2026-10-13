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

#![warn(missing_docs)]

pub mod math_expression;
pub mod simulation_engine;
/// Data types of a model definition.
pub mod types;

// Re-export commonly used types from math_expression
pub use math_expression::{
    MathExpression, MathExpressionContext, MathExpressionError, RateMathExpression,
};

// Re-export simulation engine trait
pub use simulation_engine::{OutputLabels, SimulationEngine};

// Re-export all public types from the types module for convenient access
pub use types::{
    Bin, BinFraction, CompartmentFraction, Dynamics, InitialConditions, Model, ModelTypes,
    Parameter, ParameterValue, Population, SeriesMode, Stratification, StratificationCondition,
    StratificationFraction, StratificationFractions, StratifiedRate, Transition,
};
