use serde::{Deserialize, Serialize};

use super::aggregate::Aggregate;
use super::engine::ModelEngine;
use super::parameters::Parameter;
use super::process::Process;
use super::structure::ModelStructure;

/// A complete model definition, as Python serializes it.
///
/// The header (name, parameters, engine) is the same for every model kind.
/// [`Self::process`] is written once for every kind; [`Self::structure`] is
/// how the kind lays it out.
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
    /// What changes and how: bins, accumulators, stratifications and
    /// transitions.
    #[serde(default)]
    pub process: Process,
    /// How the process is laid out, which changes between kinds.
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

    /// Check that no two parameters share an id, and the ids of the process
    /// ([`Process::check_ids`]).
    ///
    /// # Errors
    ///
    /// Returns a message that names the kind of object and the first id it
    /// finds twice.
    pub fn check_ids(&self) -> Result<(), String> {
        for (position, parameter) in self.parameters.iter().enumerate() {
            if self.parameters[..position]
                .iter()
                .any(|earlier| earlier.id == parameter.id)
            {
                return Err(format!(
                    "Two parameters share the id '{}'; give each parameter its own id",
                    parameter.id
                ));
            }
        }
        self.process.check_ids()
    }

    /// The aggregate structure of the model.
    ///
    /// # Errors
    ///
    /// Returns a message that names the kind of the model when it is not
    /// aggregate.
    pub fn aggregate(&self) -> Result<&Aggregate, String> {
        match &self.structure {
            ModelStructure::Aggregate(structure) => Ok(structure),
            #[cfg(feature = "test-kind")]
            other => Err(format!(
                "Model '{}' is a {} model, not an aggregate model",
                self.name,
                other.kind().name()
            )),
        }
    }

    /// The aggregate structure of the model, to change it.
    ///
    /// # Errors
    ///
    /// Returns a message that names the kind of the model when it is not
    /// aggregate.
    pub fn aggregate_mut(&mut self) -> Result<&mut Aggregate, String> {
        match &mut self.structure {
            ModelStructure::Aggregate(structure) => Ok(structure),
            #[cfg(feature = "test-kind")]
            other => Err(format!(
                "Model '{}' is a {} model, not an aggregate model",
                self.name,
                other.kind().name()
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::aggregate::Neighbors;
    use serde_json::json;

    fn model_json() -> serde_json::Value {
        json!({
            "name": "m",
            "parameters": [],
            "engine": {"type": "DifferenceEquations"},
            "process": {
                "bins": [{"id": "A", "name": "A"}],
                "stratifications": [],
                "transitions": []
            },
            "structure": {
                "kind": "aggregate",
                "variant": null,
                "bin_amounts": {"A": 10.0},
                "stratification_fractions": {},
                "neighbors": {"kind": "complete"}
            }
        })
    }

    #[cfg(feature = "test-kind")]
    #[test]
    fn a_test_model_has_no_aggregate_structure() {
        let mut json = model_json();
        json["engine"] = json!({"type": "TestEngine"});
        json["structure"] = json!({"kind": "test", "outputs": []});
        let mut model: Model = serde_json::from_value(json).unwrap();
        let expected = "Model 'm' is a test model, not an aggregate model";

        assert_eq!(model.aggregate().unwrap_err(), expected);
        assert_eq!(model.aggregate_mut().unwrap_err(), expected);
    }

    #[test]
    fn a_model_reads_back_with_its_structure() {
        let model: Model = serde_json::from_value(model_json()).unwrap();
        assert_eq!(model.structure.kind().name(), "aggregate");
        assert_eq!(model.process.bins[0].id, "A");
        let aggregate = model.aggregate().unwrap();
        assert_eq!(aggregate.initial_conditions.bin_amounts["A"], Some(10.0));
        assert_eq!(aggregate.neighbors, Some(Neighbors::Complete));
        assert_eq!(model.check_kind(), Ok(()));
        let json = serde_json::to_value(&model).unwrap();
        assert_eq!(json["structure"]["kind"], "aggregate");
        assert_eq!(json["structure"]["bin_amounts"]["A"], 10.0);
        assert_eq!(json["process"]["bins"][0]["id"], "A");
        let again: Model = serde_json::from_value(json).unwrap();
        assert_eq!(again.process.bins.len(), 1);
        assert_eq!(
            again.aggregate().unwrap().neighbors,
            Some(Neighbors::Complete)
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

#[cfg(test)]
mod aggregate_tests {
    use super::*;
    use serde_json::json;

    fn process() -> Process {
        serde_json::from_value(json!({
            "bins": [{"id": "A", "name": "A"}],
            "stratifications": [{"id": "group", "categories": ["g1", "g2"]}]
        }))
        .unwrap()
    }

    fn neighbors(value: serde_json::Value) -> crate::types::aggregate::Neighbors {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn groups_read_back_with_their_weights() {
        let groups = neighbors(json!({
            "kind": "groups",
            "stratification": "group",
            "weights": {"g1": {"g1": 1.0, "g2": 0.5}, "g2": {"g1": 0.5, "g2": 1.0}}
        }));
        assert_eq!(groups.check(&process()), Ok(()));
        assert_eq!(groups.weight("g1", "g2"), 0.5);
        let identity = neighbors(json!({"kind": "groups", "stratification": "group"}));
        assert_eq!(identity.weight("g1", "g1"), 1.0);
        assert_eq!(identity.weight("g1", "g2"), 0.0);
    }

    #[test]
    fn groups_of_an_unknown_stratification_name_it() {
        let groups = neighbors(json!({"kind": "groups", "stratification": "level"}));
        let error = groups.check(&process()).unwrap_err();
        assert!(
            error.contains("level") && error.contains("group"),
            "{error}"
        );
    }

    #[test]
    fn weights_need_every_pair_of_categories_and_no_negative_weight() {
        let missing = neighbors(json!({
            "kind": "groups", "stratification": "group",
            "weights": {"g1": {"g1": 1.0}, "g2": {"g1": 0.5, "g2": 1.0}}
        }));
        let error = missing.check(&process()).unwrap_err();
        assert!(error.contains("'g1'"), "{error}");
        let negative = neighbors(json!({
            "kind": "groups", "stratification": "group",
            "weights": {"g1": {"g1": 1.0, "g2": -0.5}, "g2": {"g1": 0.5, "g2": 1.0}}
        }));
        let error = negative.check(&process()).unwrap_err();
        assert!(error.contains("-0.5"), "{error}");
    }

    #[test]
    fn an_unknown_variant_does_not_read() {
        let error = serde_json::from_value::<crate::types::aggregate::Aggregate>(json!({
            "variant": "no_such_variant",
            "bin_amounts": {}
        }))
        .unwrap_err()
        .to_string();
        assert!(error.contains("no_such_variant"), "{error}");
    }
}
