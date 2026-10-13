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
//! - `core`: Core data structures (Model, Population, etc.)
//! - `model`: What a model expands into (compartments, flows, initial state)
//! - `difference`: Discrete-time difference equation solver
//! - `differential`: Continuous-time ordinary differential equation solver
//! - `calibration`: Parameter calibration and optimization
//!
//! ## Example
//!
//! ```python
//! from commol_rs import core, difference
//!
//! # Load a model
//! model = core.Model.from_json_file("sir_model.json")
//!
//! # Create solver
//! solver = difference.DifferenceEquations(model)
//!
//! # Run simulation
//! results = solver.run(100)
//! ```

#![warn(missing_docs)]

use pyo3::prelude::*;

// Module declarations
mod calibration;
mod core;
mod difference;
mod differential;
mod engine;
mod model;
pub(crate) mod python_observer;

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
         Includes Model, Population, Bin, Transition, and related types.",
    )?;
    core::register(&core_module)?;
    m.add_submodule(&core_module)?;

    // Model module: expansion shared by every engine
    let model_module = PyModule::new(py, "model")?;
    model_module.add(
        "__doc__",
        "What a model expands into, whatever engine runs it.\n\n\
         Provides the initial state, the expanded outputs, the transition\n\
         flows and the name checks of a model.",
    )?;
    model::register(&model_module)?;
    m.add_submodule(&model_module)?;

    // Difference module: discrete-time solver
    let difference_module = PyModule::new(py, "difference")?;
    difference_module.add(
        "__doc__",
        "Discrete-time difference equation solver.\n\n\
         Provides the DifferenceEquations class for simulating compartment\n\
         models using difference equations (discrete time steps).",
    )?;
    difference::register(&difference_module)?;
    m.add_submodule(&difference_module)?;

    // Differential module: continuous-time solver
    let differential_module = PyModule::new(py, "differential")?;
    differential_module.add(
        "__doc__",
        "Continuous-time ordinary differential equation solver.\n\n\
         Provides the DifferentialEquations class for simulating compartment\n\
         models in continuous time with diffsol.",
    )?;
    differential::register(&differential_module)?;
    m.add_submodule(&differential_module)?;

    // Calibration module: parameter optimization
    let calibration_module = PyModule::new(py, "calibration")?;
    calibration_module.add(
        "__doc__",
        "Model calibration and parameter optimization.\n\n\
         Provides tools for calibrating model parameters against observed data\n\
         using various optimization algorithms (Nelder-Mead, Particle Swarm, etc.).",
    )?;
    calibration::register(&calibration_module)?;
    m.add_submodule(&calibration_module)?;

    // Module-level metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "Commol: High-performance compartment modelling in Rust with Python bindings.",
    )?;

    Ok(())
}
