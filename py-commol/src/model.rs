//! Python bindings for commol-model: what the process of an aggregate model
//! expands into, whatever engine runs it.

use crate::core::{PyAggregate, PyModel, PyProcess};
use pyo3::prelude::*;

/// Expand a process into its compartments and their initial amounts
///
/// This is the authority for the initial state of an aggregate model,
/// including per-compartment overrides.
///
/// Args:
///     process: The process of the model
///     structure: The aggregate structure
///
/// Returns:
///     One `(name, bin, categories, amount)` tuple per expanded compartment,
///     in engine order. `categories` holds the `(stratification, category)`
///     pairs applied to the compartment, in declaration order.
///
/// Raises:
///     ValueError: If a bin amount or an override is not finite or is
///         negative, or an override names an unknown compartment
#[pyfunction]
#[allow(clippy::type_complexity)]
fn initial_state(
    process: &PyProcess,
    structure: &PyAggregate,
) -> PyResult<Vec<(String, String, Vec<(String, String)>, f64)>> {
    let state = commol_model::initial_state(&process.inner, &structure.inner)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(state
        .into_iter()
        .map(|compartment| {
            (
                compartment.name,
                compartment.bin,
                compartment.categories,
                compartment.amount,
            )
        })
        .collect())
}

/// Expand the bins and accumulators of a process over its stratifications
///
/// This is the authority for the names an aggregate model expands into.
///
/// Args:
///     process: The process of the model
///
/// Returns:
///     A `(compartments, accumulators)` tuple. Each holds one
///     `(name, source, categories)` tuple per expanded output, in engine order,
///     where `source` is the bin or accumulator id and `categories` the
///     `(stratification, category)` pairs applied to it.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn expanded_outputs(
    process: &PyProcess,
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
        as_tuples(commol_model::expand_compartments(&process.inner)),
        as_tuples(commol_model::expand_accumulators(&process.inner)),
    )
}

/// Every transition flow of a process, exactly as the engine simulates it
///
/// Args:
///     process: The process of the model
///     structure: The aggregate structure, whose neighbors resolve
///         `neighbors_<bin>` and `neighbors`, whatever the engine of its
///         model
///
/// Returns:
///     One `(transition, source, target, rate, proportional_to, flow)` tuple
///     per flow piece, in transition order. `source` or `target` is None for a
///     source-less or target-less transition. `proportional_to` is the
///     compartment a proportional rate is multiplied by, or None for an absolute
///     flow; the pieces of a transition with several sources share it. `flow`
///     is the index of the flow the piece belongs to: its pieces move
///     together.
///
/// Raises:
///     ValueError: If a transition mixes bins and categories, a category
///         transition has no source or no target, or a rate reads the
///         neighbors and the structure has none
#[pyfunction]
#[allow(clippy::type_complexity)]
fn transition_flows(
    process: &PyProcess,
    structure: &PyAggregate,
) -> PyResult<
    Vec<(
        String,
        Option<String>,
        Option<String>,
        String,
        Option<String>,
        usize,
    )>,
> {
    let flows = commol_model::describe_flows(&process.inner, structure.inner.neighbors.as_ref())
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok(flows
        .into_iter()
        .map(|flow| {
            (
                flow.transition,
                flow.source,
                flow.target,
                flow.rate,
                flow.proportional_to,
                flow.flow,
            )
        })
        .collect())
}

/// Check an aggregate model: its variant and neighbors, the positions of its
/// series and impulses, the variables its rates read, and its names against
/// the names the engine provides
///
/// Args:
///     model: An aggregate model
///
/// Raises:
///     ValueError: If the model is not aggregate, its variant or neighbors
///         do not fit the process, a series point or an impulse position is
///         one the engine cannot run, a parameter id is an expanded compartment,
///         a subgroup total or a name of the neighbors, a rate uses a
///         `total_…` name that is not a subgroup total of the model, or a
///         rate reads neighbors the model cannot resolve
#[pyfunction]
fn check_model(model: &PyModel) -> PyResult<()> {
    commol_model::check_model(&model.inner).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

/// Names of the subgroup totals a rate of a process may read, sorted
///
/// This is the authority for those names: the `total_<categories>` names,
/// the bin totals and the partial bin totals. Category suffixes follow the
/// declaration order of the stratifications.
///
/// Args:
///     process: The process of the model
///
/// Returns:
///     The names, sorted. A process without stratifications has none.
#[pyfunction]
fn subgroup_total_names(process: &PyProcess) -> Vec<String> {
    commol_model::subgroup_total_names(&process.inner)
}

/// The error of a value outside its rule, with the message of the rule.
fn value_error(message: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message)
}

/// Check the initial amount of each bin, by bin id; None leaves an amount
/// for calibration and has no rule
///
/// Raises:
///     ValueError: With the message of the rule of the first amount outside
///         it, which names its bin
#[pyfunction]
fn check_bin_amounts(amounts: Vec<(String, Option<f64>)>) -> PyResult<()> {
    amounts
        .iter()
        .filter_map(|(bin, amount)| amount.map(|amount| (bin, amount)))
        .try_for_each(|(bin, amount)| commol_model::rules::check_bin_amount(bin, amount))
        .map_err(value_error)
}

/// Check each initial amount that overrides a compartment, by compartment
/// name
///
/// Raises:
///     ValueError: With the message of the rule of the first amount outside
///         it, which names its compartment
#[pyfunction]
fn check_compartment_amounts(amounts: Vec<(String, f64)>) -> PyResult<()> {
    amounts
        .iter()
        .try_for_each(|(compartment, amount)| {
            commol_model::rules::check_compartment_amount(compartment, *amount)
        })
        .map_err(value_error)
}

/// Check the fractions of each stratification, as `(stratification,
/// [(category, fraction)])`: each from 0 to 1, and a sum of 1 within 1e-6
///
/// Raises:
///     ValueError: With the message of the first rule the fractions break,
///         which names the stratification
#[pyfunction]
fn check_fractions(stratifications: Vec<(String, Vec<(String, f64)>)>) -> PyResult<()> {
    stratifications
        .iter()
        .try_for_each(|(stratification, fractions)| {
            commol_model::rules::check_fractions(
                stratification,
                fractions
                    .iter()
                    .map(|(category, fraction)| (category.as_str(), *fraction)),
            )
        })
        .map_err(value_error)
}

/// Check the impulse of a transition: its positions `at` or its schedule
/// `every`, `(period, offset)`
///
/// Raises:
///     ValueError: With the message of the first rule the impulse breaks,
///         which names the transition
#[pyfunction]
#[pyo3(signature = (transition, at, every))]
fn check_impulse(
    transition: &str,
    at: Option<Vec<f64>>,
    every: Option<(f64, f64)>,
) -> PyResult<()> {
    commol_model::rules::check_impulse(transition, at.as_deref(), every).map_err(value_error)
}

/// Check the `(position, value)` points of a series
///
/// Raises:
///     ValueError: With the message of the first rule the points break
#[pyfunction]
fn check_series(data: Vec<(f64, f64)>) -> PyResult<()> {
    commol_model::rules::check_series(&data).map_err(value_error)
}

/// Register the model helpers with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initial_state, m)?)?;
    m.add_function(wrap_pyfunction!(expanded_outputs, m)?)?;
    m.add_function(wrap_pyfunction!(transition_flows, m)?)?;
    m.add_function(wrap_pyfunction!(check_model, m)?)?;
    m.add_function(wrap_pyfunction!(subgroup_total_names, m)?)?;
    m.add_function(wrap_pyfunction!(check_bin_amounts, m)?)?;
    m.add_function(wrap_pyfunction!(check_compartment_amounts, m)?)?;
    m.add_function(wrap_pyfunction!(check_fractions, m)?)?;
    m.add_function(wrap_pyfunction!(check_impulse, m)?)?;
    m.add_function(wrap_pyfunction!(check_series, m)?)?;
    m.add("NEIGHBORS", commol_core::NEIGHBORS)?;
    m.add("NEIGHBORS_PREFIX", commol_core::NEIGHBORS_PREFIX)?;
    m.add("ENGINE_VARIABLES", commol_core::ENGINE_VARIABLES.to_vec())?;
    m.add("TOTAL", commol_core::SPECIAL_VAR_TOTAL)?;
    m.add("SUBGROUP_TOTAL_PREFIX", commol_core::SUBGROUP_TOTAL_PREFIX)?;
    Ok(())
}
