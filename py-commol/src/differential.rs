//! Python bindings for commol-differential (continuous-time solver).

use pyo3::prelude::*;

crate::engine::engine_class!(
    PyDifferentialEquations,
    "DifferentialEquations",
    commol_differential::DifferentialEquations,
    "Continuous-time compartment model solver using ordinary differential equations."
);

/// Register the differential equation solver with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDifferentialEquations>()?;
    Ok(())
}
