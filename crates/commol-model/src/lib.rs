//! Compilation of aggregate models into compartments, shared by every
//! engine of the aggregate kind.
//!
//! This crate expands the process of a [`commol_core::Model`] into its
//! compartments and accumulators, plans its transition flows, compiles their
//! rates (fast path or JIT) and exposes the result as a
//! [`CompiledModel`]. Its
//! [`CompiledModel::derivatives`] is the right-hand side `dX/dt = Σ flows`:
//! `DifferenceEquations` (`commol-difference`) adds it once per step,
//! `DifferentialEquations` (`commol-differential`) integrates it, and
//! `StochasticEvents` and `StochasticSteps` (`commol-stochastic`) move whole
//! units by the change of one event of each flow
//! ([`CompiledModel::flow_changes`]). The four engines reach the compiled
//! model through [`CompartmentalEngine`] and [`delegate_to_compiled_model`].
//!
//! ## Module Organization
//!
//! - `builder` - Expansion, initial state, name checks and compilation
//! - `engine` - What a compartmental engine gives besides the generic trait,
//!   and what the engines share: the delegation to the compiled model, the
//!   run at integer steps, the numbers of their messages
//! - `flows` - Planning of the transition flows
//! - `helpers` - Utility functions for stratification and rate resolution
//! - `initial_conditions` - Initial conditions that calibration writes
//! - `labels` - Labels of the outputs and the sums they name
//! - `layout` - Structured index of the compartments
//! - `matrices` - The variables of a matrix between categories, and the
//!   matrix sums of a rate written as the sums they are
//! - `rates` - Evaluation of a compiled rate
//! - [`rules`] - The rules of the values of a model: an initial amount, an
//!   impulse and the points of a series
//! - `system` - The right-hand side and the accessors of a compiled model
//!
//! ## Example Usage
//!
//! ```rust
//! use commol_core::Model;
//! use commol_model::CompiledModel;
//!
//! let json = r#"{"name":"m","parameters":[{"id":"k1","value":0.1}],
//!     "engine":{"type":"DifferenceEquations"},
//!     "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
//!     "accumulators":[],"stratifications":[],
//!     "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k1"}]},
//!     "structure":{"kind":"aggregate","bin_amounts":{"A":900.0,"B":100.0}}}"#;
//! let model: Model = serde_json::from_str(json).unwrap();
//! let mut compiled = CompiledModel::try_from_model(&model).unwrap();
//! let state = compiled.initial_state().to_vec();
//! let mut rates = vec![0.0; state.len()];
//! compiled.derivatives(0.0, 0.0, &state, &mut rates).unwrap();
//! assert_eq!(rates, [-90.0, 90.0]);
//! ```

#![warn(missing_docs)]

mod builder;
mod engine;
mod flows;
mod helpers;
mod initial_conditions;
mod labels;
mod layout;
mod matrices;
mod rates;
pub mod rules;
mod system;
mod types;

pub use builder::{
    ExpandedOutput, InitialCompartment, as_whole, check_model, expand_accumulators,
    expand_compartments, initial_state, subgroup_total_names,
};
pub use engine::{CompartmentalEngine, IntegerStepper, record_steps, rounded};
pub use flows::{FlowDescription, describe_flows};
pub use initial_conditions::InitialConditionPlan;
pub use labels::{OutputLabels, output_sums};
pub use matrices::{MatrixVariable, matrix_variables, scaled_matrices, variable_values};
pub use types::{CompiledModel, FlowChange};
