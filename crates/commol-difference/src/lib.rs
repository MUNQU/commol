//! Difference equations simulation engine for compartment models.
//!
//! This crate provides a discrete-time simulation engine for compartment models
//! using difference equations. It is designed for efficient simulation of compartmental
//! models with support for stratifications and complex rate expressions.
//!
//! ## Module Organization
//!
//! - `types` - Core data structures and types
//! - `builder` - Model compilation and construction
//! - `helpers` - Utility functions for stratification and rate resolution
//! - `simulation` - Simulation execution methods
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use commol_core::Model;
//! use commol_difference::DifferenceEquations;
//!
//! // Load a model from JSON
//! let model = Model::from_json_file("sir_model.json")?;
//!
//! // Create the simulation engine
//! let mut engine = DifferenceEquations::from_model(&model);
//!
//! // Run for 100 time steps
//! let results = engine.run(100)?;
//!
//! // Access compartment names and final populations
//! let compartments = engine.compartments();
//! let final_state = engine.population();
//! ```

mod builder;
mod helpers;
mod simulation;
mod types;

pub use types::DifferenceEquations;
