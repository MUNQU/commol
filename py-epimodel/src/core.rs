//! Python bindings for epimodel-core types.
//!
//! This module provides Python-accessible wrappers around core Rust types.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Wrapper for epimodel_core::Model
#[pyclass(name = "Model")]
pub struct PyModel {
    pub inner: epimodel_core::Model,
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (name, population, dynamics, parameters, description=None, version=None))]
    fn new(
        name: String,
        population: PyPopulation,
        dynamics: PyDynamics,
        parameters: Vec<PyParameter>,
        description: Option<String>,
        version: Option<String>,
    ) -> Self {
        Self {
            inner: epimodel_core::Model {
                name,
                description,
                version,
                population: population.inner,
                parameters: parameters.into_iter().map(|p| p.inner).collect(),
                dynamics: dynamics.inner,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(disease_states={}, transitions={})",
            self.inner.population.disease_states.len(),
            self.inner.dynamics.transitions.len()
        )
    }

    /// Load model from JSON file
    #[staticmethod]
    fn from_json_file(path: String) -> PyResult<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let model: epimodel_core::Model = serde_json::from_reader(file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner: model })
    }

    /// Load model from JSON string
    #[staticmethod]
    fn from_json(json_str: String) -> PyResult<Self> {
        let model: epimodel_core::Model = serde_json::from_str(&json_str)
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

/// Wrapper for epimodel_core::Population
#[pyclass(name = "Population")]
#[derive(Clone)]
pub struct PyPopulation {
    pub inner: epimodel_core::Population,
}

#[pymethods]
impl PyPopulation {
    #[new]
    #[pyo3(signature = (disease_states, stratifications, initial_conditions, transitions=None))]
    fn new(
        disease_states: Vec<PyDiseaseState>,
        stratifications: Vec<PyStratification>,
        initial_conditions: PyInitialConditions,
        transitions: Option<Vec<PyTransition>>,
    ) -> Self {
        Self {
            inner: epimodel_core::Population {
                disease_states: disease_states.into_iter().map(|ds| ds.inner).collect(),
                stratifications: stratifications.into_iter().map(|s| s.inner).collect(),
                transitions: transitions
                    .unwrap_or_default()
                    .into_iter()
                    .map(|t| t.inner)
                    .collect(),
                initial_conditions: initial_conditions.inner,
            },
        }
    }
}

/// Wrapper for epimodel_core::DiseaseState
#[pyclass(name = "DiseaseState")]
#[derive(Clone)]
pub struct PyDiseaseState {
    pub inner: epimodel_core::DiseaseState,
}

#[pymethods]
impl PyDiseaseState {
    #[new]
    #[pyo3(signature = (id, name=None))]
    fn new(id: String, name: Option<String>) -> Self {
        Self {
            inner: epimodel_core::DiseaseState {
                id: id.clone(),
                name: name.unwrap_or(id),
            },
        }
    }
}

/// Wrapper for epimodel_core::Stratification
#[pyclass(name = "Stratification")]
#[derive(Clone)]
pub struct PyStratification {
    pub inner: epimodel_core::Stratification,
}

#[pymethods]
impl PyStratification {
    #[new]
    fn new(id: String, categories: Vec<String>) -> Self {
        Self {
            inner: epimodel_core::Stratification { id, categories },
        }
    }
}

/// Wrapper for epimodel_core::InitialConditions
#[pyclass(name = "InitialConditions")]
#[derive(Clone)]
pub struct PyInitialConditions {
    pub inner: epimodel_core::InitialConditions,
}

#[pymethods]
impl PyInitialConditions {
    #[new]
    fn new(
        population_size: u64,
        disease_state_fractions: Vec<(String, f64)>,
        stratification_fractions: Option<HashMap<String, HashMap<String, f64>>>,
    ) -> Self {
        let disease_state_fractions: Vec<epimodel_core::DiseaseStateFraction> =
            disease_state_fractions
                .into_iter()
                .map(
                    |(disease_state, fraction)| epimodel_core::DiseaseStateFraction {
                        disease_state,
                        fraction,
                    },
                )
                .collect();

        let stratification_fractions = stratification_fractions
            .unwrap_or_default()
            .into_iter()
            .map(
                |(stratification, fractions)| epimodel_core::StratificationFractions {
                    stratification,
                    fractions: fractions
                        .into_iter()
                        .map(
                            |(category, fraction)| epimodel_core::StratificationFraction {
                                category,
                                fraction,
                            },
                        )
                        .collect(),
                },
            )
            .collect();

        Self {
            inner: epimodel_core::InitialConditions {
                population_size,
                disease_state_fractions,
                stratification_fractions,
            },
        }
    }
}

/// Wrapper for epimodel_core::Dynamics
#[pyclass(name = "Dynamics")]
#[derive(Clone)]
pub struct PyDynamics {
    pub inner: epimodel_core::Dynamics,
}

#[pymethods]
impl PyDynamics {
    #[new]
    #[pyo3(signature = (typology, transitions))]
    fn new(typology: &str, transitions: Vec<PyTransition>) -> PyResult<Self> {
        let typology = match typology.to_lowercase().as_str() {
            "differenceequations" | "difference_equations" | "difference" => {
                epimodel_core::ModelTypes::DifferenceEquations
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid typology '{}'. Must be 'DifferenceEquations', 'difference_equations', or 'difference'",
                    typology
                )));
            }
        };

        Ok(Self {
            inner: epimodel_core::Dynamics {
                typology,
                transitions: transitions.into_iter().map(|t| t.inner).collect(),
            },
        })
    }
}

/// Wrapper for epimodel_core::Transition
#[pyclass(name = "Transition")]
#[derive(Clone)]
pub struct PyTransition {
    pub inner: epimodel_core::Transition,
}

#[pymethods]
impl PyTransition {
    #[new]
    #[pyo3(signature = (id, source, target, rate=None, stratified_rates=None))]
    fn new(
        id: String,
        source: Vec<String>,
        target: Vec<String>,
        rate: Option<String>,
        stratified_rates: Option<Vec<(Vec<(String, String)>, String)>>,
    ) -> Self {
        let rate = rate.map(epimodel_core::RateMathExpression::from_string);

        let stratified_rates = stratified_rates.map(|rates| {
            rates
                .into_iter()
                .map(|(conditions, rate_str)| epimodel_core::StratifiedRate {
                    conditions: conditions
                        .into_iter()
                        .map(
                            |(stratification, category)| epimodel_core::StratificationCondition {
                                stratification,
                                category,
                            },
                        )
                        .collect(),
                    rate: rate_str,
                })
                .collect()
        });

        Self {
            inner: epimodel_core::Transition {
                id,
                source,
                target,
                rate,
                stratified_rates,
                condition: None,
            },
        }
    }
}

/// Wrapper for epimodel_core::Parameter
#[pyclass(name = "Parameter")]
#[derive(Clone)]
pub struct PyParameter {
    pub inner: epimodel_core::Parameter,
}

#[pymethods]
impl PyParameter {
    #[new]
    fn new(id: String, value: f64, description: Option<String>) -> Self {
        Self {
            inner: epimodel_core::Parameter {
                id,
                value,
                description,
            },
        }
    }
}

/// Wrapper for epimodel_core::MathExpression
#[pyclass(name = "MathExpression")]
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
        let math_expr = epimodel_core::MathExpression::new(self.expression.clone());
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
    m.add_class::<PyPopulation>()?;
    m.add_class::<PyDiseaseState>()?;
    m.add_class::<PyStratification>()?;
    m.add_class::<PyInitialConditions>()?;
    m.add_class::<PyDynamics>()?;
    m.add_class::<PyTransition>()?;
    m.add_class::<PyParameter>()?;
    m.add_class::<PyMathExpression>()?;
    Ok(())
}
