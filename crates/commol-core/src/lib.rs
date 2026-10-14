//! # Commol Core
//!
//! Core data structures and abstractions for compartment modeling.
//!
//! This crate provides:
//! - Mathematical expression parsing and evaluation
//! - Common simulation engine interface
//! - Type definitions for models: the header, the engine and the structure of
//!   each model kind
//!
//! ## Main Components
//!
//! - [`MathExpression`]: Parse and evaluate mathematical expressions
//! - [`SimulationEngine`]: Trait for implementing simulation engines
//! - [`Model`]: Complete model definition: its header, engine and structure
//! - [`EngineType`]: The list of engines a model can name
//! - [`types`]: All data structures for defining compartment models

#![warn(missing_docs)]

pub mod math_expression;
pub mod simulation_engine;
/// Data types of a model definition.
pub mod types;

// Re-export commonly used types from math_expression
pub use math_expression::{
    ENGINE_VARIABLES, MathExpression, MathExpressionContext, MathExpressionError,
    RateMathExpression, SPECIAL_VAR_TOTAL, SUBGROUP_TOTAL_PREFIX,
};

// Re-export simulation engine trait
pub use simulation_engine::{ModelValue, SimulationEngine, Trajectory, ValuePlan, check_positions};

// Re-export all public types from the types module for convenient access
pub use types::{
    Accumulator, Bin, BinFraction, CompartmentFraction, CompartmentalStructure, EngineSettings,
    EngineType, InitialConditions, Model, ModelEngine, ModelKind, ModelStructure, ModelValueType,
    OdeMethod, OdeSolver, Parameter, ParameterValue, RateMeaning, SeriesMode, Stratification,
    StratificationCondition, StratificationFraction, StratificationFractions, StratifiedRate,
    Transition,
};
#[cfg(feature = "test-kind")]
pub use types::{DecayOutput, TestStructure};
