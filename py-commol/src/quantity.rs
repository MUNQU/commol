//! The quantities of interest as they cross from Python.
//!
//! A quantity crosses as the JSON that `model_dump_json()` of its pydantic
//! class writes, read into [`Quantity`] with `kind` as the tag of each enum:
//! a series and a reduction, never a tuple whose fields depend on a kind
//! string. A list of quantities crosses once per analysis.

use commol_inference::quantity::Quantity;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The quantities of `json`, a JSON list of quantities.
///
/// # Errors
///
/// A `ValueError` when `json` is not a list of quantities.
pub(crate) fn read_quantities(json: &str) -> PyResult<Vec<Quantity>> {
    serde_json::from_str(json)
        .map_err(|error| PyValueError::new_err(format!("Not a list of quantities: {error}")))
}

/// The quantities of `json` as Rust reads them, written back as JSON
///
/// Args:
///     json: A JSON list of quantities, as the quantity classes write them
///
/// Returns:
///     The quantities Rust read, as JSON in the same form
///
/// Raises:
///     ValueError: If `json` is not a list of quantities
#[pyfunction]
fn quantities_as_read(json: &str) -> PyResult<String> {
    let read = read_quantities(json)?;
    Ok(serde_json::to_string(&read).expect("a quantity is plain data that JSON writes"))
}

/// Register the quantity bindings in the inference module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantities_as_read, m)?)?;
    Ok(())
}
