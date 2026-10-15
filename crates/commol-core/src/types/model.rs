use serde::{Deserialize, Serialize};

use super::compartmental::CompartmentalStructure;
use super::engine::ModelEngine;
use super::parameters::Parameter;
use super::structure::ModelStructure;

/// A complete model definition, as Python serializes it.
///
/// The header (name, parameters, engine) is the same for every model kind;
/// [`Self::structure`] is the part of its kind.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    /// Name of the model.
    pub name: String,
    /// Free-text description.
    pub description: Option<String>,
    /// Version label.
    pub version: Option<String>,
    /// Parameters, in declaration order.
    pub parameters: Vec<Parameter>,
    /// The engine that runs the model, and its settings.
    pub engine: ModelEngine,
    /// The part of the model that depends on its kind.
    pub structure: ModelStructure,
}

impl Model {
    /// Check that the engine of the model runs the kind of its structure.
    ///
    /// # Errors
    ///
    /// Returns a message that names the engine, the kind it runs and the kind
    /// of the model.
    pub fn check_kind(&self) -> Result<(), String> {
        self.engine.engine_type().check_kind(self.structure.kind())
    }

    /// The compartmental structure of the model.
    ///
    /// # Errors
    ///
    /// Returns a message that names the kind of the model when it is not
    /// compartmental.
    pub fn compartmental(&self) -> Result<&CompartmentalStructure, String> {
        match &self.structure {
            ModelStructure::Compartmental(structure) => Ok(structure),
            #[cfg(feature = "test-kind")]
            other => Err(format!(
                "Model '{}' is a {} model, not a compartmental model",
                self.name,
                other.kind().name()
            )),
        }
    }

    /// The compartmental structure of the model, to change it.
    ///
    /// # Errors
    ///
    /// Returns a message that names the kind of the model when it is not
    /// compartmental.
    pub fn compartmental_mut(&mut self) -> Result<&mut CompartmentalStructure, String> {
        match &mut self.structure {
            ModelStructure::Compartmental(structure) => Ok(structure),
            #[cfg(feature = "test-kind")]
            other => Err(format!(
                "Model '{}' is a {} model, not a compartmental model",
                self.name,
                other.kind().name()
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model_json() -> serde_json::Value {
        json!({
            "name": "m",
            "parameters": [],
            "engine": {"type": "DifferenceEquations"},
            "structure": {
                "kind": "compartmental",
                "bins": [{"id": "A", "name": "A"}],
                "stratifications": [],
                "transitions": [],
                "initial_conditions": {
                    "total": 10.0,
                    "bin_fractions": [{"bin": "A", "fraction": 1.0}],
                    "stratification_fractions": []
                }
            }
        })
    }

    #[test]
    fn a_model_reads_back_with_its_structure() {
        let model: Model = serde_json::from_value(model_json()).unwrap();
        assert_eq!(model.structure.kind().name(), "compartmental");
        assert_eq!(model.compartmental().unwrap().bins[0].id, "A");
        assert_eq!(model.check_kind(), Ok(()));
        let again: Model = serde_json::from_value(serde_json::to_value(&model).unwrap()).unwrap();
        assert_eq!(again.compartmental().unwrap().bins.len(), 1);
        assert_eq!(
            serde_json::to_value(&again).unwrap()["structure"]["kind"],
            "compartmental"
        );
    }

    #[test]
    fn a_structure_of_an_unknown_kind_does_not_read() {
        let mut json = model_json();
        json["structure"]["kind"] = json!("network");
        let error = serde_json::from_value::<Model>(json)
            .unwrap_err()
            .to_string();
        assert!(error.contains("network"), "{error}");
    }
}
