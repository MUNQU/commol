use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use pyo3::prelude::*;

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

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    pub variable: String,
    pub operator: LogicOperator,
    pub value: serde_json::Value,
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
#[pymodule]
fn epimodel_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<Population>()?;
    m.add_class::<DiseaseState>()?;
    m.add_class::<Stratification>()?;
    m.add_class::<Dynamics>()?;
    m.add_class::<Transition>()?;
    m.add_class::<Parameter>()?;
    m.add_class::<InitialConditions>()?;
    m.add_class::<Condition>()?;
    m.add_class::<Rule>()?;
    m.add_class::<LogicOperator>()?;
    m.add_class::<ModelTypes>()?;
    m.add_class::<VariablePrefixes>()?;
    Ok(())
}
