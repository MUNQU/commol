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
//! - `json`: The port: every data object crosses as the JSON of its
//!   `model_dump()`, which the engine reads into its own types
//! - `core`: The checks and formulas of a model, read from its JSON
//! - `model`: What a model expands into (compartments, flows, initial state)
//! - `engine`: The list of engines, and the engine a model names
//! - `calibration`: What a calibration reads of a model, and its results
//! - `inference`: A calibration problem built once, and the methods that read it
//! - `parallelism`: The threads of the parallel methods and the levels that run in parallel
//!
//! ## Example
//!
//! ```python
//! from pathlib import Path
//!
//! from commol.commol_rs._commol_rs import engine
//!
//! # Build the engine a model names, from the JSON that `Model.to_json` writes
//! simulation = engine.Engine(Path("model.json").read_text())
//!
//! # Run simulation
//! positions, rows = simulation.run(100)
//! ```

#![warn(missing_docs)]

use pyo3::prelude::*;

// Module declarations
#[cfg(test)]
mod build_info;
mod calibration;
mod conformal;
mod core;
mod counts;
mod engine;
mod identifiability;
mod inference;
mod json;
mod local;
mod model;
mod parallelism;
mod profiles;
pub(crate) mod python_observer;
mod quantity;
mod ranges;
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
    // JSON module: the port of every data object
    let json_module = PyModule::new(py, "json")?;
    json_module.add(
        "__doc__",
        "The port of the bridge: the classes whose JSON the engine reads,\n\
         and what it reads of each.",
    )?; // coverage: an error that only PyO3 returns, when the module cannot take it
    json::register(&json_module)?;
    m.add_submodule(&json_module)?;

    // Core module: the checks and formulas of a model
    let core_module = PyModule::new(py, "core")?;
    core_module.add(
        "__doc__",
        "The checks and formulas of a model, read from its JSON.",
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
        "What a calibration reads of a model, and the results of a fit.",
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

    // Parallelism module: the threads of the parallel methods
    let parallelism_module = PyModule::new(py, "parallelism")?;
    parallelism_module.add(
        "__doc__",
        "The threads of the parallel methods and the levels that run in\n\
         parallel, held per context.",
    )?; // coverage: an error that only PyO3 returns, when the module cannot take it
    parallelism::register(&parallelism_module)?;
    m.add_submodule(&parallelism_module)?;

    // Test kind module: the queries of the test model kind, only in a build
    // with the `test-kind` feature
    #[cfg(feature = "test-kind")]
    {
        let test_kind_module = PyModule::new(py, "test_kind")?;
        test_kind::register(&test_kind_module)?;
        m.add_submodule(&test_kind_module)?;
    }

    // Module-level metadata: the version and the commit of the build, which
    // the build script reads (`build_info.rs`); no commit is an empty string.
    let commit = Some(env!("COMMOL_COMMIT")).filter(|commit| !commit.is_empty());
    m.add("__version__", env!("COMMOL_VERSION"))?;
    m.add("__commit__", commit)?;
    m.add(
        "__doc__",
        "Commol: High-performance compartment modelling in Rust with Python bindings.",
    )?; // coverage: an error that only PyO3 returns, when the module cannot take it

    Ok(())
}
