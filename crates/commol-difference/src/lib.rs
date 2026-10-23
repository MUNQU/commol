//! Difference equations simulation engine for compartment models.
//!
//! This crate provides a discrete-step simulation engine for compartment models
//! using difference equations. The model is compiled by `commol-model`; this
//! crate only steps it: `X(n + 1) = X(n) + derivatives(n, X(n))`.
//!
//! ## Example Usage
//!
//! ```rust
//! use commol_core::{Model, SimulationEngine};
//! use commol_difference::DifferenceEquations;
//! use commol_model::CompartmentalEngine;
//!
//! // A → B at the rate k1, from A = 900 and B = 100.
//! let json = r#"{"name":"m","parameters":[{"id":"k1","value":0.1}],
//!     "engine":{"type":"DifferenceEquations"},
//!     "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
//!     "accumulators":[],"stratifications":[],
//!     "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k1"}]},
//!     "structure":{"kind":"aggregate","bin_amounts":{"A":900.0,"B":100.0}}}"#;
//! let model: Model = serde_json::from_str(json).unwrap();
//! let mut engine = DifferenceEquations::try_from_model(&model).unwrap();
//!
//! // Run for 100 steps: the rows at the positions 0, 1, …, 100.
//! let trajectory = engine.run(100).unwrap();
//! assert_eq!(trajectory.rows.len(), 101);
//!
//! // The compartment names and the final state.
//! assert_eq!(engine.compartments(), ["A", "B"]);
//! assert_eq!(engine.state().iter().sum::<f64>(), 1000.0);
//! ```

#![warn(missing_docs)]

mod simulation;

pub use simulation::DifferenceEquations;
