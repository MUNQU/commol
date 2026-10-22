//! # Commol Python Bindings
//!
//! High-performance compartment modelling library with Python bindings.
//!
//! This library provides Python access to the Rust-based Commol framework,
//! following the Polars architecture pattern where Python bindings wrap
//! pure Rust implementations.
//!
//! ## Modules
//!
//! - `core`: Core data structures (Model, Process, Aggregate, etc.)
//! - `model`: What a model expands into (compartments, flows, initial state)
//! - `engine`: The list of engines, and the engine a model names
//! - `calibration`: The types of a calibration problem and its optimizers
//! - `inference`: A calibration problem built once, and the methods that read it
//!
//! ## Example
//!
//! ```python
//! from commol_rs import core, engine
//!
//! # Load a model
//! model = core.Model.from_json_file("model.json")
//!
//! # Build the engine the model names
//! simulation = engine.Engine(model)
//!
//! # Run simulation
//! positions, rows = simulation.run(100)
//! ```

#![warn(missing_docs)]

use pyo3::prelude::*;

// Module declarations
mod calibration;
mod conformal;
mod core;
mod engine;
mod identifiability;
mod inference;
mod local;
mod model;
mod profiles;
pub(crate) mod python_observer;
mod quantity;
mod sensitivity;
mod smc;
#[cfg(feature = "test-kind")]
mod test_kind;

/// High-performance compartment modelling library.
///
/// This module provides Python bindings to the Commol Rust library,
/// enabling high-performance compartment modeling with a Pythonic API.
#[pymodule]
fn _commol_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core module: fundamental data structures
    let core_module = PyModule::new(py, "core")?;
    core_module.add(
        "__doc__",
        "Core data structures for compartment models.\n\n\
         Includes Model, Process, Aggregate, Bin, Transition, and related types.",
    )?; // coverage: an error that only PyO3 returns, when the module cannot take it
    core::register(&core_module)?;
    m.add_submodule(&core_module)?;

    // Model module: expansion shared by every engine
    let model_module = PyModule::new(py, "model")?;
    model_module.add(
        "__doc__",
        "What a model expands into, whatever engine runs it.\n\n\
         Provides the initial state, the expanded outputs, the transition\n\
         flows and the name checks of a model.",
    )?; // coverage: an error that only PyO3 returns, when the module cannot take it
    model::register(&model_module)?;
    m.add_submodule(&model_module)?;

    // Engine module: the list of engines and the engine a model names
    let engine_module = PyModule::new(py, "engine")?;
    engine_module.add(
        "__doc__",
        "The engines a model can name.\n\n\
         Provides EngineType, the list of engines and what each one means,\n\
         and Engine, the engine a model names, built from the model.",
    )?; // coverage: an error that only PyO3 returns, when the module cannot take it
    engine::register(&engine_module)?;
    m.add_submodule(&engine_module)?;

    // Calibration module: parameter optimization
    let calibration_module = PyModule::new(py, "calibration")?;
    calibration_module.add(
        "__doc__",
        "Model calibration and parameter optimization.\n\n\
         Provides tools for calibrating model parameters against observed data\n\
         using various optimization algorithms (Nelder-Mead, Particle Swarm, etc.).",
    )?; // coverage: an error that only PyO3 returns, when the module cannot take it
    calibration::register(&calibration_module)?;
    m.add_submodule(&calibration_module)?;

    // Inference module: a built calibration problem and the methods that
    // read it
    let inference_module = PyModule::new(py, "inference")?;
    inference_module.add(
        "__doc__",
        "A calibration problem built once, its optimizers and predictions,\n\
         and the inference methods that read it.",
    )?; // coverage: an error that only PyO3 returns, when the module cannot take it
    inference::register(&inference_module)?;
    m.add_submodule(&inference_module)?;

    // Module-level metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "Commol: High-performance compartment modelling in Rust with Python bindings.",
    )?; // coverage: an error that only PyO3 returns, when the module cannot take it

    Ok(())
}
