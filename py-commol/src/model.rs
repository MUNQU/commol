//! Python bindings for commol-model: what the process of an aggregate model
//! expands into, whatever engine runs it.

use crate::core::{aggregate, model, process};
use commol_core::StratificationFractions;
use commol_core::json::Pairs;
use pyo3::prelude::*;

/// Expand a process into its compartments and their initial amounts
///
/// This is the authority for the initial state of an aggregate model,
/// including per-compartment overrides.
///
/// Args:
///     process: The JSON of the process of the model
///     structure: The JSON of the aggregate structure
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
    process: &str,
    structure: &str,
) -> PyResult<Vec<(String, String, Vec<(String, String)>, f64)>> {
    let state = commol_model::initial_state(&self::process(process)?, &aggregate(structure)?)
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
///     process: The JSON of the process of the model
///
/// Returns:
///     A `(compartments, accumulators)` tuple. Each holds one
///     `(name, source, categories)` tuple per expanded output, in engine order,
///     where `source` is the bin or accumulator id and `categories` the
///     `(stratification, category)` pairs applied to it.
///
/// Raises:
///     ValueError: If the JSON is not a process
#[pyfunction]
#[allow(clippy::type_complexity)]
fn expanded_outputs(
    process: &str,
) -> PyResult<(
    Vec<(String, String, Vec<(String, String)>)>,
    Vec<(String, String, Vec<(String, String)>)>,
)> {
    let process = self::process(process)?;
    let as_tuples = |outputs: Vec<commol_model::ExpandedOutput>| {
        outputs
            .into_iter()
            .map(|output| (output.name, output.source, output.categories))
            .collect()
    };
    Ok((
        as_tuples(commol_model::expand_compartments(&process)),
        as_tuples(commol_model::expand_accumulators(&process)),
    ))
}

/// Every transition flow of a process, exactly as the engine simulates it
///
/// Args:
///     process: The JSON of the process of the model
///     structure: The JSON of the aggregate structure, whose neighbors resolve
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
    process: &str,
    structure: &str,
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
    let structure = aggregate(structure)?;
    let flows =
        commol_model::describe_flows(&self::process(process)?, structure.neighbors.as_ref())
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

/// Check an aggregate model: its variant and neighbors, its stratification
/// fractions against its stratifications, the positions of its series and
/// impulses, the variables its rates read, and its names against the names
/// the engine provides
///
/// Args:
///     model: The JSON of an aggregate model
///
/// Raises:
///     ValueError: If the model is not aggregate, its variant, neighbors or
///         stratification fractions do not fit the process, a series point
///         or an impulse position is one the engine cannot run, a parameter
///         id is an expanded compartment, a subgroup total or a name of the
///         neighbors, a rate uses a
///         `total_…` name that is not a subgroup total of the model, or a
///         rate reads neighbors the model cannot resolve
#[pyfunction]
fn check_model(model: &str) -> PyResult<()> {
    commol_model::check_model(&self::model(model)?)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

/// Names of the subgroup totals a rate of a process may read, sorted
///
/// This is the authority for those names: the `total_<categories>` names,
/// the bin totals and the partial bin totals. Category suffixes follow the
/// declaration order of the stratifications.
///
/// Args:
///     process: The JSON of the process of the model
///
/// Returns:
///     The names, sorted. A process without stratifications has none.
///
/// Raises:
///     ValueError: If the JSON is not a process
#[pyfunction]
fn subgroup_total_names(process: &str) -> PyResult<Vec<String>> {
    Ok(commol_model::subgroup_total_names(&self::process(process)?))
}

/// A variable of a matrix as Python reads it: its name, the id of the
/// matrix, its value (None while a scale has no value) and whether it is the
/// spectral radius.
type MatrixVariableRow = (String, String, Option<f64>, bool);

/// The variables of the matrices of a model: for each matrix, its entries
/// `<M>_<i>_<j>` row by row, then its spectral radius `<M>_radius`, of the
/// matrix scaled by the values of the parameters of its rows and columns
///
/// Args:
///     model: The JSON of the model
///
/// Returns:
///     (name, matrix id, value, whether it is the radius) of each variable,
///     in that order; the value is None while a scale is a parameter
///     without a value
///
/// Raises:
///     ValueError: If the JSON is not a model
#[pyfunction]
fn matrix_variables(model: &str) -> PyResult<Vec<MatrixVariableRow>> {
    Ok(commol_model::matrix_variables(&self::model(model)?)
        .into_iter()
        .map(|variable| {
            (
                variable.name,
                variable.matrix,
                variable.value,
                variable.radius,
            )
        })
        .collect())
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

/// Check the fractions of each stratification, the JSON of
/// `stratification_fractions`: in each split, each fraction from 0 to 1, and
/// a sum of 1 within 1e-6
///
/// Raises:
///     ValueError: With the message of the first rule the fractions break,
///         which names the stratification, and the category of the
///         stratification a split by category differs by
#[pyfunction]
fn check_fractions(fractions: &str) -> PyResult<()> {
    crate::json::read::<Pairs<StratificationFractions>>(fractions)?
        .0
        .iter()
        .try_for_each(|(stratification, fractions)| {
            commol_model::rules::check_fractions(stratification, fractions)
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

/// The stratification and the bare name of a category id of the initial
/// conditions, bare (`g1`) or qualified (`group:g1`), among the
/// stratifications given with their categories
///
/// Raises:
///     ValueError: If no stratification has the category, with the
///         categories there are, or a bare category is in several
///         stratifications, with each of them and the qualified ids
#[pyfunction]
fn resolve_category(
    category: &str,
    stratifications: Vec<(String, Vec<String>)>,
) -> PyResult<(String, String)> {
    match commol_model::resolve_category(category, &stratifications).map_err(value_error)? {
        Some((stratification, name)) => Ok((stratification.to_string(), name.to_string())),
        None => {
            let mut available: Vec<&str> = stratifications
                .iter()
                .flat_map(|(_, categories)| categories.iter().map(String::as_str))
                .collect();
            available.sort_unstable();
            available.dedup();
            Err(value_error(format!(
                "Category '{category}' not found in initial conditions. Available \
                 categories: {available:?}"
            )))
        }
    }
}

/// Register the model helpers with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initial_state, m)?)?;
    m.add_function(wrap_pyfunction!(expanded_outputs, m)?)?;
    m.add_function(wrap_pyfunction!(transition_flows, m)?)?;
    m.add_function(wrap_pyfunction!(check_model, m)?)?;
    m.add_function(wrap_pyfunction!(subgroup_total_names, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_variables, m)?)?;
    m.add_function(wrap_pyfunction!(check_bin_amounts, m)?)?;
    m.add_function(wrap_pyfunction!(check_compartment_amounts, m)?)?;
    m.add_function(wrap_pyfunction!(check_fractions, m)?)?;
    m.add_function(wrap_pyfunction!(check_impulse, m)?)?;
    m.add_function(wrap_pyfunction!(check_series, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_category, m)?)?;
    let separator = commol_model::CATEGORY_SEPARATOR.to_string();
    m.add("CATEGORY_SEPARATOR", separator)?;
    m.add("FRACTIONS_BY", commol_core::FRACTIONS_BY)?;
    m.add("NEIGHBORS", commol_core::NEIGHBORS)?;
    m.add("NEIGHBORS_PREFIX", commol_core::NEIGHBORS_PREFIX)?;
    m.add("ENGINE_VARIABLES", commol_core::ENGINE_VARIABLES.to_vec())?;
    m.add("TOTAL", commol_core::SPECIAL_VAR_TOTAL)?;
    m.add("SUBGROUP_TOTAL_PREFIX", commol_core::SUBGROUP_TOTAL_PREFIX)?;
    Ok(())
}
