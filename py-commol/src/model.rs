//! Python bindings for commol-model: what a compartmental structure expands
//! into, whatever engine runs it.

use crate::core::{PyCompartmentalStructure, PyModel};
use pyo3::prelude::*;

/// Expand a structure into its compartments and their initial populations
///
/// This is the authority for the initial state of a compartmental model,
/// including per-compartment overrides.
///
/// Args:
///     structure: The compartmental structure
///
/// Returns:
///     One `(name, bin, categories, population)` tuple per expanded compartment,
///     in engine order. `categories` holds the `(stratification, category)`
///     pairs applied to the compartment, in declaration order.
///
/// Raises:
///     ValueError: If an override names an unknown or repeated compartment,
///         holds a fraction outside [0, 1], or makes the initial population
///         exceed population_size
#[pyfunction]
#[allow(clippy::type_complexity)]
fn initial_state(
    structure: &PyCompartmentalStructure,
) -> PyResult<Vec<(String, String, Vec<(String, String)>, f64)>> {
    let state = commol_model::initial_state(&structure.inner)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(state
        .into_iter()
        .map(|compartment| {
            (
                compartment.name,
                compartment.bin,
                compartment.categories,
                compartment.population,
            )
        })
        .collect())
}

/// Expand the bins and accumulators of a structure over its stratifications
///
/// This is the authority for the names a compartmental model expands into.
///
/// Args:
///     structure: The compartmental structure
///
/// Returns:
///     A `(compartments, accumulators)` tuple. Each holds one
///     `(name, source, categories)` tuple per expanded output, in engine order,
///     where `source` is the bin or accumulator id and `categories` the
///     `(stratification, category)` pairs applied to it.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn expanded_outputs(
    structure: &PyCompartmentalStructure,
) -> (
    Vec<(String, String, Vec<(String, String)>)>,
    Vec<(String, String, Vec<(String, String)>)>,
) {
    let as_tuples = |outputs: Vec<commol_model::ExpandedOutput>| {
        outputs
            .into_iter()
            .map(|output| (output.name, output.source, output.categories))
            .collect()
    };
    (
        as_tuples(commol_model::expand_compartments(&structure.inner)),
        as_tuples(commol_model::expand_accumulators(&structure.inner)),
    )
}

/// Every transition flow of a structure, exactly as the engine simulates it
///
/// Args:
///     structure: The compartmental structure, whatever the engine of its
///         model
///
/// Returns:
///     One `(transition, source, target, rate, per_capita_of)` tuple per flow
///     piece, in transition order. `source` or `target` is None for a
///     source-less or target-less transition. `per_capita_of` is the
///     compartment a per-capita rate is multiplied by, or None for an absolute
///     flow; the pieces of a transition with several sources share it.
///
/// Raises:
///     ValueError: If a transition mixes bins and categories, or a category
///         transition has no source or no target
#[pyfunction]
#[allow(clippy::type_complexity)]
fn transition_flows(
    structure: &PyCompartmentalStructure,
) -> PyResult<
    Vec<(
        String,
        Option<String>,
        Option<String>,
        String,
        Option<String>,
    )>,
> {
    let flows = commol_model::describe_flows(&structure.inner)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(flows
        .into_iter()
        .map(|flow| {
            (
                flow.transition,
                flow.source,
                flow.target,
                flow.rate,
                flow.per_capita_of,
            )
        })
        .collect())
}

/// Check the names a model uses against the names the engine provides
///
/// Args:
///     model: A compartmental model
///
/// Raises:
///     ValueError: If the model is not compartmental, a parameter id is an expanded compartment or a
///         subpopulation total, or a rate uses an `N_…` name that is not a
///         subpopulation of the model
#[pyfunction]
fn check_names(model: &PyModel) -> PyResult<()> {
    commol_model::check_names(&model.inner).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

/// Register the model helpers with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initial_state, m)?)?;
    m.add_function(wrap_pyfunction!(expanded_outputs, m)?)?;
    m.add_function(wrap_pyfunction!(transition_flows, m)?)?;
    m.add_function(wrap_pyfunction!(check_names, m)?)?;
    Ok(())
}
