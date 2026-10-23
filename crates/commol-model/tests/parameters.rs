//! The parameters a caller can set on a compiled model: a constant or a
//! parameter to calibrate, never an id the model does not have or a parameter
//! whose formula or series the engine writes at each step.

use commol_core::Model;
use commol_model::CompiledModel;

/// A → B at the rate `k1`, from A = 900 and B = 100, with the constant `k1`,
/// the parameter to calibrate `k2`, the formula `k3 = 2 * k1` and the series
/// `k4`.
fn model() -> Model {
    let json = r#"{"name":"m",
        "parameters":[{"id":"k1","value":0.1},{"id":"k2","value":null},
        {"id":"k3","value":"2 * k1"},
        {"id":"k4","value":{"data":[[0,0.1],[5,0.2]],"mode":"step_function"}}],
        "engine":{"type":"DifferenceEquations"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k1 + k3 * 0 + k4 * 0"}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":900.0,"B":100.0}}}"#;
    serde_json::from_str(json).expect("valid model JSON")
}

fn compiled() -> CompiledModel {
    CompiledModel::try_from_model(&model()).unwrap()
}

#[test]
fn a_constant_and_a_parameter_to_calibrate_can_be_set() {
    let mut model = compiled();
    assert_eq!(model.set_parameter("k1", 0.2), Ok(()));
    assert_eq!(model.set_parameter("k2", 0.3), Ok(()));
}

#[test]
fn an_unknown_parameter_cannot_be_set() {
    let error = compiled().set_parameter("k9", 0.1).unwrap_err();
    assert!(error.contains("'k9'"), "{error}");
    assert!(error.contains("k1, k2"), "{error}");
}

#[test]
fn a_formula_parameter_cannot_be_set() {
    let error = compiled().set_parameter("k3", 0.1).unwrap_err();
    assert!(error.contains("'k3'"), "{error}");
    assert!(error.contains("its formula sets it"), "{error}");
}

#[test]
fn a_series_parameter_cannot_be_set() {
    let error = compiled().set_parameter("k4", 0.1).unwrap_err();
    assert!(error.contains("'k4'"), "{error}");
    assert!(error.contains("its series sets it"), "{error}");
}

#[test]
fn a_variable_of_the_engine_is_not_a_parameter() {
    for id in ["t", "total"] {
        let error = compiled().set_parameter(id, 0.1).unwrap_err();
        assert!(error.contains(&format!("'{id}'")), "{error}");
    }
}
