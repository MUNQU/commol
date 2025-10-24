use serde::{Deserialize, Serialize};

use super::dynamics::Transition;

/// A disease state (compartment) in the model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiseaseState {
    pub id: String,
    pub name: String,
}

/// A stratification dimension with its categories
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stratification {
    pub id: String,
    pub categories: Vec<String>,
}

/// Specifies the fraction of population in a particular disease state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiseaseStateFraction {
    pub disease_state: String,
    pub fraction: f64,
}

/// Specifies the fraction of population in a stratification category
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratificationFraction {
    pub category: String,
    pub fraction: f64,
}

/// Groups stratification fractions for a specific stratification dimension
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratificationFractions {
    pub stratification: String,
    pub fractions: Vec<StratificationFraction>,
}

/// Initial conditions for the population model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitialConditions {
    pub population_size: u64,
    pub disease_state_fractions: Vec<DiseaseStateFraction>,
    pub stratification_fractions: Vec<StratificationFractions>,
}

/// Complete population structure including disease states, stratifications, and initial conditions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Population {
    pub disease_states: Vec<DiseaseState>,
    pub stratifications: Vec<Stratification>,
    pub transitions: Vec<Transition>,
    pub initial_conditions: InitialConditions,
}
