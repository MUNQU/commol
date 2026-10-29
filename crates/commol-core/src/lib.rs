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
//! - [`range`]: The range rule of each setting ([`Count`], [`Real`]), declared
//!   once as data, and the check of an object against the rules of its
//!   settings ([`Ranges`])
//! - [`types`]: All data structures for defining a model: its process and
//!   its structure
//! - [`ScaledMatrix`]: A matrix between categories at the values of the
//!   parameters that scale it, with its spectral radius

#![warn(missing_docs)]

pub mod json;
pub mod math_expression;
pub mod parallelism;
pub mod position;
pub mod range;
pub mod scaled_matrix;
pub mod simulation_engine;
/// Data types of a model definition.
pub mod types;

// Re-export commonly used types from math_expression
pub use math_expression::variables::{SPECIAL_VAR_STEP, SPECIAL_VAR_T};
pub use math_expression::{
    ENGINE_VARIABLES, FUNCTIONS, MATRIX_SUM, MathExpression, MathExpressionContext,
    MathExpressionError, RateMathExpression, SPECIAL_VAR_TOTAL, SUBGROUP_TOTAL_PREFIX,
};

pub use parallelism::{Levels, Parallelism, inner_levels_in_parallel};

pub use position::{POSITION_TOLERANCE, same_position};

pub use range::{Bound, CONFIDENCE_LEVEL, Count, Ranges, Real, Rule, Setting, check_object};

// Re-export simulation engine trait
pub use simulation_engine::{
    Accuracy, AccuracySettings, ModelValue, NUM_STEPS, REPLICATE_COUNT, SEED, SimulationEngine,
    SimulationError, Trajectory, ValuePlan, check_positions, run_replicates,
};

// The matrix an engine reads, and the public types of the types module.
pub use scaled_matrix::{MatrixScale, ScaledMatrix};
pub use types::{
    Accumulator, Aggregate, AggregateVariant, Bin, CategoryMatrix, DynamicsForm, EngineSettings,
    EngineType, EventMethod, FRACTIONS_BY, InitialConditions, Model, ModelEngine, ModelKind,
    ModelStructure, ModelValueType, NEIGHBORS, NEIGHBORS_PREFIX, Neighbors, OdeMethod, OdeSolver,
    Parameter, ParameterValue, Process, RateKeeps, RateMeaning, Rounding, SeriesMode,
    StochasticEventsSettings, StochasticStepsSettings, Stratification, StratificationCondition,
    StratificationFractions, StratifiedRate, Transition,
};
#[cfg(feature = "test-kind")]
pub use types::{DecayOutput, TestStructure};
