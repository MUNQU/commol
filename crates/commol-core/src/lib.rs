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
//! - [`EngineType`]: The list of engines a model can name, and the positions
//!   each one reports at
//! - [`same_position`]: When two positions on the axis are one
//! - [`check_open_unit`]: The one check of a share, strictly between 0 and 1
//! - [`types`]: All data structures for defining a model: its process and
//!   its structure

#![warn(missing_docs)]

pub mod math_expression;
pub mod parallelism;
pub mod position;
pub mod share;
pub mod simulation_engine;
/// Data types of a model definition.
pub mod types;

// Re-export commonly used types from math_expression
pub use math_expression::variables::{SPECIAL_VAR_STEP, SPECIAL_VAR_T};
pub use math_expression::{
    ENGINE_VARIABLES, FUNCTIONS, MathExpression, MathExpressionContext, MathExpressionError,
    RateMathExpression, SPECIAL_VAR_TOTAL, SUBGROUP_TOTAL_PREFIX,
};

pub use parallelism::{Levels, Parallelism, inner_levels_in_parallel};

pub use position::{POSITION_TOLERANCE, same_position};

pub use share::check_open_unit;

// Re-export simulation engine trait
pub use simulation_engine::{
    ModelValue, SimulationEngine, SimulationError, Trajectory, ValuePlan, check_positions,
    run_replicates,
};

// Re-export all public types from the types module for convenient access
pub use types::{
    Accumulator, Aggregate, AggregateVariant, Bin, DynamicsForm, EngineSettings, EngineType,
    EventMethod, InitialConditions, Model, ModelEngine, ModelKind, ModelStructure, ModelValueType,
    NEIGHBORS, NEIGHBORS_PREFIX, Neighbors, OdeMethod, OdeSolver, Parameter, ParameterValue,
    Process, RateMeaning, Rounding, SeriesMode, StochasticEventsSettings, StochasticStepsSettings,
    Stratification, StratificationCondition, StratifiedRate, Transition,
};
#[cfg(feature = "test-kind")]
pub use types::{DecayOutput, TestStructure};
