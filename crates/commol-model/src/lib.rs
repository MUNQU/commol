//! Compilation of compartment models, shared by every engine.
//!
//! This crate expands a [`commol_core::Model`] into its compartments and
//! accumulators, plans its transition flows, compiles their rates (fast path,
//! JIT, evalexpr fallback) and exposes the result as a [`CompiledModel`]. Its
//! [`CompiledModel::derivatives`] is the right-hand side `dX/dt = Σ flows`:
//! the difference engine (`commol-difference`) adds it once per step, the
//! differential engine (`commol-differential`) integrates it.
//!
//! ## Module Organization
//!
//! - `builder` - Expansion, initial state, name checks and compilation
//! - `flows` - Planning of the transition flows
//! - `helpers` - Utility functions for stratification and rate resolution
//! - `layout` - Structured index of the compartments
//! - `rates` - Evaluation of a compiled rate
//! - `system` - The right-hand side and the accessors of a compiled model
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use commol_core::Model;
//! use commol_model::CompiledModel;
//!
//! let model: Model = serde_json::from_str(json)?;
//! let mut compiled = CompiledModel::try_from_model(&model)?;
//! let state = compiled.initial_state().to_vec();
//! let mut rates = vec![0.0; state.len()];
//! compiled.derivatives(0.0, 0.0, &state, &mut rates)?;
//! ```

#![warn(missing_docs)]

mod builder;
mod flows;
mod helpers;
mod layout;
mod rates;
mod system;
mod types;

pub use builder::{
    ExpandedOutput, InitialCompartment, check_names, expand_accumulators, expand_compartments,
    initial_state,
};
pub use flows::{FlowDescription, describe_flows};
pub use types::CompiledModel;
