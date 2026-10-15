//! Difference equations simulation engine for compartment models.
//!
//! This crate provides a discrete-step simulation engine for compartment models
//! using difference equations. The model is compiled by `commol-model`; this
//! crate only steps it: `X(n + 1) = X(n) + derivatives(n, X(n))`.
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use commol_core::Model;
//! use commol_difference::DifferenceEquations;
//!
//! // Load a model from JSON
//! let model = Model::from_json_file("model.json")?;
//!
//! // Create the simulation engine
//! let mut engine = DifferenceEquations::from_model(&model);
//!
//! // Run for 100 steps
//! let results = engine.run(100)?;
//!
//! // Access compartment names and the final state
//! let compartments = engine.compartments();
//! let final_state = engine.state();
//! ```

#![warn(missing_docs)]

mod simulation;

pub use simulation::DifferenceEquations;
