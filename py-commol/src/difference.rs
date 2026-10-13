//! Python bindings for commol-difference (discrete-time solver).

use pyo3::prelude::*;

crate::engine::engine_class!(
    PyDifferenceEquations,
    "DifferenceEquations",
    commol_difference::DifferenceEquations,
    "Discrete-time compartment model solver using difference equations."
);

/// Register the difference equation solver with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDifferenceEquations>()?;
    Ok(())
}
