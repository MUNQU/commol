//! Core data types for compartment models
//!
//! This module contains all the data structures used to define and configure
//! compartment models, organized into logical submodules:
//!
//! - `parameters`: Model parameters
//! - `engine`: The list of engines, their settings and what they mean
//! - `process`: What changes and how, written once for every structure
//! - `structure`: How the process is laid out, which changes between kinds
//! - `aggregate`: The aggregate structure: amounts per compartment
//! - `transitions`: Transitions of a process and their rates
//! - `model`: Top-level model definition

pub mod aggregate;
/// The list of engines a model can name.
pub mod engine;
/// The top-level model definition.
pub mod model;
/// Parameters and their values.
pub mod parameters;
pub mod process;
pub mod structure;
#[cfg(feature = "test-kind")]
pub mod test_kind;
pub mod transitions;

// Re-export all public types for convenient access
pub use aggregate::{
    Aggregate, AggregateVariant, FRACTIONS_BY, InitialConditions, Neighbors,
    StratificationFractions,
};
pub use engine::{
    DynamicsForm, EngineSettings, EngineType, EventMethod, ModelEngine, ModelKind, ModelValueType,
    OdeMethod, OdeSolver, RateKeeps, RateMeaning, Rounding, StochasticEventsSettings,
    StochasticStepsSettings,
};
pub use model::Model;
pub use parameters::{Parameter, ParameterValue, SeriesMode};
pub use process::{
    Accumulator, Bin, CategoryMatrix, NEIGHBORS, NEIGHBORS_PREFIX, Process, Stratification,
};
pub use structure::ModelStructure;
#[cfg(feature = "test-kind")]
pub use test_kind::{DecayOutput, TestStructure};
pub use transitions::{StratificationCondition, StratifiedRate, Transition};
