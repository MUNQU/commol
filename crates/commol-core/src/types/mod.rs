//! Core data types for compartment models
//!
//! This module contains all the data structures used to define and configure
//! compartment models, organized into logical submodules:
//!
//! - `conditions`: Logic operators, rules, and conditional expressions
//! - `parameters`: Model parameters and variable prefixes
//! - `population`: Bins, accumulators, stratifications and initial conditions
//! - `dynamics`: Model dynamics, transitions, and rates
//! - `model`: Top-level model structure

/// Transition conditions: operators, rules and conditions.
pub mod conditions;
/// Dynamics: typologies, transitions and stratified rates.
pub mod dynamics;
/// The top-level model definition.
pub mod model;
/// Parameters and their values.
pub mod parameters;
/// Bins, accumulators, stratifications and initial conditions.
pub mod population;

// Re-export all public types for convenient access
pub use conditions::{Condition, LogicOperator, Rule, RuleValue};
pub use dynamics::{Dynamics, ModelTypes, StratificationCondition, StratifiedRate, Transition};
pub use model::Model;
pub use parameters::{Parameter, ParameterValue, SeriesMode, VariablePrefixes};
pub use population::{
    Accumulator, Bin, BinFraction, CompartmentFraction, InitialConditions, Population,
    Stratification, StratificationFraction, StratificationFractions,
};
