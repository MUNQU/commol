use serde::{Deserialize, Serialize};

use super::dynamics::Dynamics;
use super::parameters::Parameter;
use super::population::Population;

/// Top-level model structure containing all model components
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    pub description: Option<String>,
    pub version: Option<String>,
    pub population: Population,
    pub parameters: Vec<Parameter>,
    pub dynamics: Dynamics,
}
