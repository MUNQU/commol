use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use pyo3::{prelude::*, types::PyType};

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LogicOperator {
    #[serde(rename = "and")]
    And,
    #[serde(rename = "or")]
    Or,
    #[serde(rename = "eq")]
    Eq,
    #[serde(rename = "neq")]
    Neq,
    #[serde(rename = "gt")]
    Gt,
    #[serde(rename = "get")]
    Get,
    #[serde(rename = "lt")]
    Lt,
    #[serde(rename = "let")]
    Let,
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelTypes {
    #[serde(rename = "DifferenceEquations")]
    DifferenceEquations,
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VariablePrefixes {
    #[serde(rename = "state")]
    State,
    #[serde(rename = "strat")]
    Strat,
}

#[cfg_attr(feature = "python", derive(FromPyObject))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RuleValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[cfg(feature = "python")]
mod python_impls {
    use super::RuleValue;
    use pyo3::prelude::*;

    impl<'py> IntoPyObject<'py> for RuleValue {
        type Target = PyAny;
        type Output = Bound<'py, Self::Target>;
        type Error = std::convert::Infallible;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            match self {
                RuleValue::String(s) => Ok(s.into_pyobject(py)?.into_any()),
                RuleValue::Int(i) => Ok(i.into_pyobject(py)?.into_any()),
                RuleValue::Float(f) => Ok(f.into_pyobject(py)?.into_any()),
                RuleValue::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any()),
            }
        }
    }

    impl<'a, 'py> IntoPyObject<'py> for &'a RuleValue {
        type Target = PyAny;
        type Output = Bound<'py, Self::Target>;
        type Error = std::convert::Infallible;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            match self {
                RuleValue::String(s) => Ok(s.into_pyobject(py)?.into_any()),
                RuleValue::Int(i) => Ok(i.into_pyobject(py)?.into_any()),
                RuleValue::Float(f) => Ok(f.into_pyobject(py)?.into_any()),
                RuleValue::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any()),
            }
        }
    }
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    pub variable: String,
    pub operator: LogicOperator,
    pub value: RuleValue,
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Condition {
    pub logic: LogicOperator,
    pub rules: Vec<Rule>,
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiseaseState {
    pub id: String,
    pub name: String,
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stratification {
    pub id: String,
    pub categories: Vec<String>,
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transition {
    pub id: String,
    pub source: Vec<String>,
    pub target: Vec<String>,
    pub rate: Option<String>,
    pub condition: Option<Condition>,
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dynamics {
    pub typology: ModelTypes,
    pub transitions: Vec<Transition>,
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parameter {
    pub id: String,
    pub value: f64,
    pub description: Option<String>,
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitialConditions {
    pub population_size: u64,
    pub disease_state_fraction: HashMap<String, f64>,
    pub stratification_fractions: HashMap<String, HashMap<String, f64>>,
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Population {
    pub disease_states: Vec<DiseaseState>,
    pub stratifications: Vec<Stratification>,
    pub transitions: Vec<Transition>,
    pub initial_conditions: InitialConditions,
}

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    pub description: Option<String>,
    pub version: Option<String>,
    pub population: Population,
    pub parameters: Vec<Parameter>,
    pub dynamics: Dynamics,
}

#[cfg(feature = "python")]
#[pymethods]
impl Model {
    #[classmethod]
    #[pyo3(name = "from_json")]
    fn py_from_json(_cls: &Bound<'_, PyType>, json_data: &str) -> PyResult<Self> {
        let model: Self = serde_json::from_str(json_data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(model)
    }
}
