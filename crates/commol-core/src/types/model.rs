use serde::{Deserialize, Serialize};

use super::dynamics::Dynamics;
use super::parameters::Parameter;
use super::population::Population;

/// A complete model definition, as Python serializes it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    /// Name of the model.
    pub name: String,
    /// Free-text description.
    pub description: Option<String>,
    /// Version label.
    pub version: Option<String>,
    /// Bins, accumulators, stratifications and initial conditions.
    pub population: Population,
    /// Parameters, in declaration order.
    pub parameters: Vec<Parameter>,
    /// Typology and transitions.
    pub dynamics: Dynamics,
}
