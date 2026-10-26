//! Python bindings for commol-core types.
//!
//! The Python package builds a model in Python and sends the JSON of its
//! `model_dump_json()`; [`model`], [`process`] and [`aggregate`] read it, so
//! a type of the model is made only from JSON.

use pyo3::prelude::*;

/// Read a model from the JSON of `Model.model_dump_json()`.
///
/// # Errors
///
/// A `ValueError` that names the path of the field, when the JSON is not
/// a model.
pub(crate) fn model(text: &str) -> PyResult<commol_core::Model> {
    crate::json::read(text)
}

/// Read a process from the JSON of `Process.model_dump_json()`.
///
/// # Errors
///
/// A `ValueError` that names the path of the field, when the JSON is not
/// a process.
pub(crate) fn process(text: &str) -> PyResult<commol_core::Process> {
    crate::json::read(text)
}

/// Read an aggregate structure from the JSON of its `model_dump_json()`,
/// with its `kind`.
///
/// # Errors
///
/// A `ValueError` when the JSON is not a structure, or the structure is not
/// aggregate.
pub(crate) fn aggregate(text: &str) -> PyResult<commol_core::Aggregate> {
    let structure: commol_core::ModelStructure = crate::json::read(text)?;
    match structure {
        commol_core::ModelStructure::Aggregate(inner) => Ok(inner),
        // coverage: an arm for the test model kind: Python sends only the JSON of an Aggregate
        // (engine_state.py)
        #[cfg(feature = "test-kind")]
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "The structure is a {} structure, not an aggregate structure",
            other.kind().name()
        ))),
    }
}

/// A formula of the language of formulas (`commol_core::MathExpression`)
#[pyclass(name = "MathExpression", from_py_object)]
#[derive(Clone)]
pub struct PyMathExpression {
    inner: commol_core::MathExpression,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyMathExpression {
    #[new]
    fn new(expression: String) -> Self {
        Self {
            inner: commol_core::MathExpression::new(expression),
        }
    }

    /// Check that the formula is in the language of formulas, and count the
    /// nodes of its tree: numbers, variables, operators and calls
    ///
    /// Returns:
    ///     The number of nodes
    ///
    /// Raises:
    ///     ValueError: If the formula is not in the language, nests deeper
    ///         than `MAX_NESTING` or has more nodes than `MAX_NODES`; the
    ///         message quotes the formula, cut after 80 characters, and
    ///         names the first part of it that is not
    fn check(&self) -> PyResult<usize> {
        self.inner
            .check()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// The tree of the formula, the one the JIT compiles, as JSON: an object
    /// per node, keyed by its kind (`Constant`, `Variable`, `Chain`, `Power`,
    /// `UnaryOp`, `FunctionCall`, `Conditional`); the module `tree` of
    /// `commol_core::math_expression` documents the shape
    ///
    /// Returns:
    ///     The JSON of the tree
    ///
    /// Raises:
    ///     ValueError: If the formula is not in the language, as `check`
    fn tree(&self) -> PyResult<String> {
        self.inner
            .tree_json()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// The variables the formula reads, in the order it first reads them,
    /// without the engine variables: the names a model must define
    #[getter]
    fn variables(&self) -> Vec<String> {
        self.inner.get_variables()
    }
}

/// Register all core types with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::ranges::register(m)?;
    m.add_class::<PyMathExpression>()?;
    let functions: Vec<&str> = commol_core::FUNCTIONS
        .iter()
        .map(|function| function.name)
        .collect();
    m.add("FUNCTIONS", functions)?;
    Ok(())
}
