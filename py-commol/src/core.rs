//! Python bindings for commol-core types.
//!
//! The Python package builds a model in Python, serializes it to JSON and
//! reads it here, so a type of the model is made only from JSON.

use pyo3::prelude::*;

/// Wrapper for commol_core::Model
#[pyclass(name = "Model")]
pub struct PyModel {
    pub inner: commol_core::Model,
}

#[pymethods]
impl PyModel {
    /// Load model from JSON string
    #[staticmethod]
    fn from_json(json_str: String) -> PyResult<Self> {
        let model: commol_core::Model = serde_json::from_str(&json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner: model })
    }
}

/// Wrapper for commol_core::Process
#[pyclass(name = "Process", from_py_object)]
#[derive(Clone)]
pub struct PyProcess {
    pub inner: commol_core::Process,
}

#[pymethods]
impl PyProcess {
    /// Load a process from JSON
    ///
    /// Raises:
    ///     ValueError: If the JSON is not a process
    #[staticmethod]
    fn from_json(json_str: String) -> PyResult<Self> {
        serde_json::from_str(&json_str)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

/// Wrapper for commol_core::Aggregate
#[pyclass(name = "Aggregate", from_py_object)]
#[derive(Clone)]
pub struct PyAggregate {
    pub inner: commol_core::Aggregate,
}

#[pymethods]
impl PyAggregate {
    /// Load an aggregate structure from JSON, with its `kind`
    ///
    /// Raises:
    ///     ValueError: If the JSON is not a structure, or the structure is
    ///         not aggregate
    #[staticmethod]
    fn from_json(json_str: String) -> PyResult<Self> {
        let structure: commol_core::ModelStructure = serde_json::from_str(&json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        match structure {
            commol_core::ModelStructure::Aggregate(inner) => Ok(Self { inner }),
            // coverage: an arm for the test model kind: Python loads only the JSON of an Aggregate
            // (engine_state.py, _rust_structure)
            #[cfg(feature = "test-kind")]
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "The structure is a {} structure, not an aggregate structure",
                other.kind().name()
            ))),
        }
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
    m.add_class::<PyModel>()?;
    m.add_class::<PyProcess>()?;
    m.add_class::<PyAggregate>()?;
    m.add_class::<PyMathExpression>()?;
    let functions: Vec<&str> = commol_core::FUNCTIONS
        .iter()
        .map(|function| function.name)
        .collect();
    m.add("FUNCTIONS", functions)?;
    Ok(())
}
