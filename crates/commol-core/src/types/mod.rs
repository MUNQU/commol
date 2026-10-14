//! Core data types for compartment models
//!
//! This module contains all the data structures used to define and configure
//! compartment models, organized into logical submodules:
//!
//! - `parameters`: Model parameters
//! - `population`: Bins, accumulators, stratifications and initial conditions
//! - `dynamics`: Transitions and their rates
//! - `engine`: The list of engines, their settings and what they mean
//! - `model`: Top-level model structure

/// Transitions and stratified rates.
pub mod dynamics;
/// The list of engines a model can name.
pub mod engine;
/// The top-level model definition.
pub mod model;
/// Parameters and their values.
pub mod parameters;
/// Bins, accumulators, stratifications and initial conditions.
pub mod population;

// Re-export all public types for convenient access
pub use dynamics::{Dynamics, StratificationCondition, StratifiedRate, Transition};
pub use engine::{
    EngineSettings, EngineType, ModelEngine, ModelKind, OdeMethod, OdeSolver, RateMeaning,
};
pub use model::Model;
pub use parameters::{Parameter, ParameterValue, SeriesMode};
pub use population::{
    Accumulator, Bin, BinFraction, CompartmentFraction, InitialConditions, Population,
    Stratification, StratificationFraction, StratificationFractions,
};
