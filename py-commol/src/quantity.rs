//! The quantities of interest as they cross from Python.
//!
//! A quantity crosses as the JSON of the `model_dump()` of its pydantic
//! class, read into [`Quantity`] with `kind` as the tag of each enum
//! ([`crate::json`]). A list of quantities crosses inside the JSON of a
//! method, or once per call that reduces runs that are done.

use commol_inference::quantity::{MissingPositions, Quantities, Quantity};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The quantities of `json`, a JSON list of quantities.
///
/// # Errors
///
/// A `ValueError` when `json` is not a list of quantities; the message names
/// the path of the field.
fn read_quantities(json: &str) -> PyResult<Vec<Quantity>> {
    crate::json::read(json)
}

/// The positions a run must report at for the quantities of `json`: sorted,
/// each once
///
/// Args:
///     json: A JSON list of quantities, as the quantity classes write them
///
/// Returns:
///     The positions every quantity reads, and the start of every window
///
/// Raises:
///     ValueError: If `json` is not a list of quantities, or a quantity does
///         not fit its rules
#[pyfunction]
fn quantity_schedule(json: &str) -> PyResult<Vec<f64>> {
    Quantities::schedule_of(&read_quantities(json)?).map_err(PyValueError::new_err)
}

/// The quantities of runs that are done, each run reduced as a propagation
/// reduces its runs
///
/// Each position of the schedule of the quantities is found in `positions`
/// within the tolerance of the axis.
///
/// Args:
///     json: A JSON list of quantities, as the quantity classes write them
///     names: The name of each quantity, for the message of a missing
///         position
///     output_names: The outputs of a row, in order
///     positions: The positions of the runs, ascending
///     runs: The rows of each run, one row per position
///
/// Returns:
///     The value of each scalar of each run, `[run][scalar]`
///
/// Raises:
///     ValueError: If `json` is not a list of quantities, a quantity names
///         an output not in `output_names`, or the runs lack a position a
///         quantity reads; the message lists every such position with its
///         quantity
#[pyfunction]
fn quantities_of_runs(
    json: &str,
    names: Vec<String>,
    output_names: Vec<String>,
    positions: Vec<f64>,
    runs: Vec<Vec<Vec<f64>>>,
) -> PyResult<Vec<Vec<f64>>> {
    let compiled =
        Quantities::new(&read_quantities(json)?, &output_names).map_err(PyValueError::new_err)?;
    runs.iter()
        .map(|rows| {
            compiled
                .evaluate_run(&positions, rows)
                .map_err(|missing| PyValueError::new_err(missing_message(&missing, &names)))
        })
        .collect()
}

/// The message of the positions a run lacks: each quantity by its name, with
/// its positions.
fn missing_message(missing: &[MissingPositions], names: &[String]) -> String {
    let listed: Vec<String> = missing
        .iter()
        .map(|entry| {
            let positions: Vec<String> = entry.positions.iter().map(f64::to_string).collect();
            format!("'{}' reads {}", names[entry.quantity], positions.join(", "))
        })
        .collect();
    format!(
        "The run has no value at positions the quantities read: {}. Run the \
         simulation at quantity_schedule(quantities) with Simulation.run_at, or \
         give those positions to the Predictive or the Scenarios.",
        listed.join("; ")
    )
}

/// Register the quantity bindings in the inference module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantity_schedule, m)?)?;
    m.add_function(wrap_pyfunction!(quantities_of_runs, m)?)?;
    Ok(())
}
