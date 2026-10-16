//! Python bindings for commol-core types.
//!
//! This module provides Python-accessible wrappers around core Rust types.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Type alias for stratified rates: Vec<(conditions, rate_expression)>
/// where conditions is Vec<(stratification_name, category_value, optional_to_category)>
type StratifiedRatesInput = Vec<(Vec<(String, String, Option<String>)>, String)>;

/// Wrapper for commol_core::Model
#[pyclass(name = "Model")]
pub struct PyModel {
    pub inner: commol_core::Model,
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (name, process, structure, parameters, engine, description=None, version=None))]
    fn new(
        name: String,
        process: PyProcess,
        structure: PyAggregate,
        parameters: Vec<PyParameter>,
        engine: &str,
        description: Option<String>,
        version: Option<String>,
    ) -> PyResult<Self> {
        let engine_type = commol_core::EngineType::from_name(engine)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(Self {
            inner: commol_core::Model {
                name,
                description,
                version,
                parameters: parameters.into_iter().map(|p| p.inner).collect(),
                engine: commol_core::ModelEngine::new(engine_type),
                process: process.inner,
                structure: commol_core::ModelStructure::Aggregate(structure.inner),
            },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(name={:?}, kind={}, engine={})",
            self.inner.name,
            self.inner.structure.kind().name(),
            self.inner.engine.engine_type().name()
        )
    }

    /// Load model from JSON file
    #[staticmethod]
    fn from_json_file(path: String) -> PyResult<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let model: commol_core::Model = serde_json::from_reader(file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner: model })
    }

    /// Load model from JSON string
    #[staticmethod]
    fn from_json(json_str: String) -> PyResult<Self> {
        let model: commol_core::Model = serde_json::from_str(&json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner: model })
    }

    /// Save model to JSON file
    fn to_json_file(&self, path: String) -> PyResult<()> {
        let file = std::fs::File::create(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        serde_json::to_writer_pretty(file, &self.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }

    /// Convert model to JSON string
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
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
    #[new]
    #[pyo3(signature = (bins, stratifications, transitions=None, accumulators=None))]
    fn new(
        bins: Vec<PyBin>,
        stratifications: Vec<PyStratification>,
        transitions: Option<Vec<PyTransition>>,
        accumulators: Option<Vec<PyAccumulator>>,
    ) -> Self {
        Self {
            inner: commol_core::Process {
                bins: bins.into_iter().map(|ds| ds.inner).collect(),
                accumulators: accumulators
                    .unwrap_or_default()
                    .into_iter()
                    .map(|accumulator| accumulator.inner)
                    .collect(),
                stratifications: stratifications.into_iter().map(|s| s.inner).collect(),
                transitions: transitions
                    .unwrap_or_default()
                    .into_iter()
                    .map(|t| t.inner)
                    .collect(),
            },
        }
    }

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
    #[new]
    fn new(initial_conditions: PyInitialConditions) -> Self {
        Self {
            inner: commol_core::Aggregate {
                variant: None,
                initial_conditions: initial_conditions.inner,
                neighbors: None,
            },
        }
    }

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
            #[cfg(feature = "test-kind")]
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "The structure is a {} structure, not an aggregate structure",
                other.kind().name()
            ))),
        }
    }
}

/// Wrapper for commol_core::Accumulator
#[pyclass(name = "Accumulator", from_py_object)]
#[derive(Clone)]
pub struct PyAccumulator {
    pub inner: commol_core::types::Accumulator,
}

#[pymethods]
impl PyAccumulator {
    #[new]
    #[pyo3(signature = (id, name=None))]
    fn new(id: String, name: Option<String>) -> Self {
        Self {
            inner: commol_core::types::Accumulator {
                id: id.clone(),
                name: name.unwrap_or(id),
            },
        }
    }
}

/// Wrapper for commol_core::Bin
#[pyclass(name = "Bin", from_py_object)]
#[derive(Clone)]
pub struct PyBin {
    pub inner: commol_core::Bin,
}

#[pymethods]
impl PyBin {
    #[new]
    #[pyo3(signature = (id, name=None))]
    fn new(id: String, name: Option<String>) -> Self {
        Self {
            inner: commol_core::Bin {
                id: id.clone(),
                name: name.unwrap_or(id),
            },
        }
    }
}

/// Wrapper for commol_core::Stratification
#[pyclass(name = "Stratification", from_py_object)]
#[derive(Clone)]
pub struct PyStratification {
    pub inner: commol_core::Stratification,
}

#[pymethods]
impl PyStratification {
    #[new]
    fn new(id: String, categories: Vec<String>) -> Self {
        Self {
            inner: commol_core::Stratification {
                id,
                categories,
                conditions: None,
            },
        }
    }
}

/// Wrapper for commol_core::InitialConditions
#[pyclass(name = "InitialConditions", from_py_object)]
#[derive(Clone)]
pub struct PyInitialConditions {
    pub inner: commol_core::InitialConditions,
}

#[pymethods]
impl PyInitialConditions {
    #[new]
    #[pyo3(signature = (total, bin_fractions, stratification_fractions=None, compartment_fractions=None))]
    fn new(
        total: f64,
        bin_fractions: Vec<(String, Option<f64>)>,
        stratification_fractions: Option<HashMap<String, HashMap<String, f64>>>,
        compartment_fractions: Option<Vec<(String, f64)>>,
    ) -> Self {
        Self {
            inner: commol_core::InitialConditions {
                total,
                bin_fractions: bin_fractions.into_iter().collect(),
                stratification_fractions: stratification_fractions
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(stratification, fractions)| {
                        (stratification, fractions.into_iter().collect())
                    })
                    .collect(),
                compartment_fractions: compartment_fractions
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
            },
        }
    }

    /// The sum of every compartment at the start.
    #[getter]
    fn total(&self) -> f64 {
        self.inner.total
    }

    /// Declared bin fractions as ``(bin, fraction)`` pairs.
    #[getter]
    fn bin_fractions(&self) -> Vec<(String, Option<f64>)> {
        self.inner
            .bin_fractions
            .iter()
            .map(|(bin, fraction)| (bin.clone(), *fraction))
            .collect()
    }

    /// Declared stratification fractions as ``(stratification, categories)`` pairs.
    #[getter]
    fn stratification_fractions(&self) -> Vec<(String, Vec<(String, f64)>)> {
        self.inner
            .stratification_fractions
            .iter()
            .map(|(stratification, fractions)| {
                (
                    stratification.clone(),
                    fractions
                        .iter()
                        .map(|(category, fraction)| (category.clone(), *fraction))
                        .collect(),
                )
            })
            .collect()
    }

    /// Per-compartment initial-condition overrides as ``(name, fraction)`` pairs.
    #[getter]
    fn compartment_fractions(&self) -> Vec<(String, f64)> {
        self.inner
            .compartment_fractions
            .iter()
            .map(|(compartment, fraction)| (compartment.clone(), *fraction))
            .collect()
    }
}

/// Wrapper for commol_core::Transition
#[pyclass(name = "Transition", from_py_object)]
#[derive(Clone)]
pub struct PyTransition {
    pub inner: commol_core::Transition,
}

#[pymethods]
impl PyTransition {
    #[new]
    #[pyo3(signature = (id, source, target, rate=None, stratified_rates=None, per_compartment=None, accumulators=None))]
    fn new(
        id: String,
        source: Vec<String>,
        target: Vec<String>,
        rate: Option<String>,
        stratified_rates: Option<StratifiedRatesInput>,
        per_compartment: Option<bool>,
        accumulators: Option<Vec<String>>,
    ) -> Self {
        let rate = rate.map(commol_core::RateMathExpression::from_string);

        let stratified_rates = stratified_rates.map(|rates| {
            rates
                .into_iter()
                .map(|(conditions, rate_str)| commol_core::StratifiedRate {
                    conditions: conditions
                        .into_iter()
                        .map(|(stratification, category, to)| {
                            commol_core::StratificationCondition {
                                stratification,
                                category: Some(category),
                                to,
                            }
                        })
                        .collect(),
                    rate: rate_str,
                    absolute: None,
                })
                .collect()
        });

        Self {
            inner: commol_core::Transition {
                id,
                source,
                target,
                accumulators: accumulators.unwrap_or_default(),
                rate,
                stratified_rates,
                per_compartment,
                impulse_at: None,
                impulse_every: None,
            },
        }
    }
}

/// Wrapper for commol_core::Parameter
#[pyclass(name = "Parameter", from_py_object)]
#[derive(Clone)]
pub struct PyParameter {
    pub inner: commol_core::Parameter,
}

#[pymethods]
impl PyParameter {
    #[new]
    #[pyo3(signature = (id, value, description=None))]
    fn new(id: String, value: PyParameterValue, description: Option<String>) -> Self {
        let param_value = match value {
            PyParameterValue::Constant(v) => commol_core::ParameterValue::Constant(v),
            PyParameterValue::Formula(f) => commol_core::ParameterValue::Formula(f),
        };

        Self {
            inner: commol_core::Parameter {
                id,
                value: Some(param_value),
                description,
            },
        }
    }

    fn __repr__(&self) -> String {
        let value_str = match &self.inner.value {
            Some(commol_core::ParameterValue::Constant(v)) => format!("{}", v),
            Some(commol_core::ParameterValue::Formula(f)) => format!("'{}'", f),
            Some(commol_core::ParameterValue::Series { data, mode }) => {
                format!("Series(len={}, mode={:?})", data.len(), mode)
            }
            None => "None".to_string(),
        };
        format!("Parameter(id='{}', value={})", self.inner.id, value_str)
    }
}

/// Python-side representation of parameter values
#[derive(FromPyObject)]
pub enum PyParameterValue {
    Constant(f64),
    Formula(String),
}

/// Wrapper for commol_core::MathExpression
#[pyclass(name = "MathExpression", from_py_object)]
#[derive(Clone)]
pub struct PyMathExpression {
    expression: String,
}

#[pymethods]
impl PyMathExpression {
    #[new]
    fn new(expression: String) -> Self {
        Self { expression }
    }

    /// Validate the mathematical expression syntax
    fn validate(&self) -> PyResult<()> {
        let math_expr = commol_core::MathExpression::new(self.expression.clone());
        // Try to validate the expression - this will fail if syntax is invalid
        math_expr
            .validate()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("MathExpression('{}')", self.expression)
    }
}

/// Register all core types with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyProcess>()?;
    m.add_class::<PyAggregate>()?;
    m.add_class::<PyBin>()?;
    m.add_class::<PyAccumulator>()?;
    m.add_class::<PyStratification>()?;
    m.add_class::<PyInitialConditions>()?;
    m.add_class::<PyTransition>()?;
    m.add_class::<PyParameter>()?;
    m.add_class::<PyMathExpression>()?;
    Ok(())
}
