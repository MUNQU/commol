//! Core data types for compartment models
//!
//! This module contains all the data structures used to define and configure
//! compartment models, organized into logical submodules:
//!
//! - `conditions`: Logic operators, rules, and conditional expressions
//! - `parameters`: Model parameters and variable prefixes
//! - `population`: Disease states, stratifications, and population structure
//! - `dynamics`: Model dynamics, transitions, and rates
//! - `model`: Top-level model structure

pub mod conditions;
pub mod dynamics;
pub mod model;
pub mod parameters;
pub mod population;

// Re-export all public types for convenient access
pub use conditions::{Condition, LogicOperator, Rule, RuleValue};
pub use dynamics::{Dynamics, ModelTypes, StratificationCondition, StratifiedRate, Transition};
pub use model::Model;
pub use parameters::{Parameter, ParameterValue, VariablePrefixes};
pub use population::{
    Bin, BinFraction, InitialConditions, Population, Stratification, StratificationFraction,
    StratificationFractions,
};
