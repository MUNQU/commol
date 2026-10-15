//! Core data types for compartment models
//!
//! This module contains all the data structures used to define and configure
//! compartment models, organized into logical submodules:
//!
//! - `parameters`: Model parameters
//! - `engine`: The list of engines, their settings and what they mean
//! - `structure`: The part of a model that depends on its kind
//! - `compartmental`: The structure of a compartmental model
//! - `transitions`: Transitions of a compartmental model and their rates
//! - `model`: Top-level model definition

pub mod compartmental;
/// The list of engines a model can name.
pub mod engine;
/// The top-level model definition.
pub mod model;
/// Parameters and their values.
pub mod parameters;
pub mod structure;
#[cfg(feature = "test-kind")]
pub mod test_kind;
pub mod transitions;

// Re-export all public types for convenient access
pub use compartmental::{
    Accumulator, Bin, BinFraction, CompartmentFraction, CompartmentalStructure, InitialConditions,
    Stratification, StratificationFraction, StratificationFractions,
};
pub use engine::{
    EngineSettings, EngineType, ModelEngine, ModelKind, ModelValueType, OdeMethod, OdeSolver,
    RateMeaning,
};
pub use model::Model;
pub use parameters::{Parameter, ParameterValue, SeriesMode};
pub use structure::ModelStructure;
#[cfg(feature = "test-kind")]
pub use test_kind::{DecayOutput, TestStructure};
pub use transitions::{StratificationCondition, StratifiedRate, Transition};
