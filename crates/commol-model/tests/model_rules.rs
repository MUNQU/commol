//! The rules of a model that the engine checks when it builds one, whatever
//! path brings it: unique ids, the categories of a stratification, a process
//! with bins and transitions, the bin amounts, and the values a run needs.

use commol_core::Model;
use commol_model::CompiledModel;
use serde_json::{Value, json};

/// A → B at `k1 * A`, with `k1` of 0.1, bins A and B of 90 and 10, and the
/// stratification `group` of `g1` and `g2`, split half and half.
fn model_json() -> Value {
    json!({
        "name": "m",
        "parameters": [{"id": "k1", "value": 0.1}],
        "engine": {"type": "DifferenceEquations"},
        "process": {
            "bins": [{"id": "A", "name": "A"}, {"id": "B", "name": "B"}],
            "accumulators": [{"id": "n", "name": "n"}],
            "stratifications": [{"id": "group", "categories": ["g1", "g2"]}],
            "transitions": [{"id": "ab", "source": ["A"], "target": ["B"], "rate": "k1",
                             "accumulators": ["n"]}]
        },
        "structure": {
            "kind": "aggregate",
            "bin_amounts": {"A": 90.0, "B": 10.0},
            "stratification_fractions": {"group": {"g1": 0.5, "g2": 0.5}}
        }
    })
}

fn build(json: Value) -> Result<CompiledModel, String> {
    let model: Model = serde_json::from_value(json).expect("valid model JSON");
    CompiledModel::try_from_model(&model)
}

/// The JSON of the model with the item `item` added to the list at `path`.
fn with_one_more(path: &[&str], item: Value) -> Value {
    let mut json = model_json();
    let mut list = &mut json;
    for key in path {
        list = &mut list[key];
    }
    list.as_array_mut().expect("a list").push(item);
    json
}

#[test]
fn a_repeated_id_is_refused() {
    let cases = [
        (
            with_one_more(&["process", "bins"], json!({"id": "A", "name": "A2"})),
            "Two bins share the id 'A'; give each bin its own id",
        ),
        (
            with_one_more(
                &["process", "accumulators"],
                json!({"id": "n", "name": "n2"}),
            ),
            "Two accumulators share the id 'n'; give each accumulator its own id",
        ),
        (
            with_one_more(
                &["process", "stratifications"],
                json!({"id": "group", "categories": ["h1"]}),
            ),
            "Two stratifications share the id 'group'; give each stratification its own id",
        ),
        (
            with_one_more(
                &["process", "transitions"],
                json!({"id": "ab", "source": ["B"], "target": ["A"], "rate": "k1"}),
            ),
            "Two transitions share the id 'ab'; give each transition its own id",
        ),
        (
            with_one_more(&["parameters"], json!({"id": "k1", "value": 0.2})),
            "Two parameters share the id 'k1'; give each parameter its own id",
        ),
    ];

    for (json, message) in cases {
        assert_eq!(build(json).err().as_deref(), Some(message));
    }
}

#[test]
fn the_categories_of_a_stratification_are_checked() {
    let with_categories = |categories: Value| {
        let mut json = model_json();
        json["process"]["stratifications"][0]["categories"] = categories;
        json
    };

    assert_eq!(
        build(with_categories(json!(["g1", "g1"]))).err().as_deref(),
        Some("Stratification 'group' names the category 'g1' twice; name each category once")
    );
    assert_eq!(
        build(with_categories(json!([]))).err().as_deref(),
        Some("Stratification 'group' has no category; give it at least one")
    );
}

#[test]
fn a_process_without_bins_or_transitions_is_refused() {
    let mut without_transitions = model_json();
    without_transitions["process"]["transitions"] = json!([]);
    let mut without_bins = without_transitions.clone();
    without_bins["process"]["bins"] = json!([]);
    without_bins["structure"]["bin_amounts"] = json!({});

    assert_eq!(
        build(without_transitions).err().as_deref(),
        Some("The process has no transition; an aggregate model needs at least one transition")
    );
    assert_eq!(
        build(without_bins).err().as_deref(),
        Some("The process has no bin; an aggregate model needs at least one bin")
    );
}

#[test]
fn the_bin_amounts_are_the_bins() {
    let with_amounts = |amounts: Value| {
        let mut json = model_json();
        json["structure"]["bin_amounts"] = amounts;
        json
    };

    assert_eq!(
        build(with_amounts(json!({"A": 90.0}))).err().as_deref(),
        Some(
            "Initial bin amounts keys must exactly match bin ids. Missing ids: [\"B\"], \
             Extra ids: []."
        )
    );
    assert_eq!(
        build(with_amounts(json!({"A": 90.0, "B": 10.0, "Z": 1.0})))
            .err()
            .as_deref(),
        Some(
            "Initial bin amounts keys must exactly match bin ids. Missing ids: [], Extra ids: \
             [\"Z\"]."
        )
    );
}

#[test]
fn a_run_needs_every_value() {
    let mut json = model_json();
    json["parameters"][0]["value"] = Value::Null;
    json["structure"]["bin_amounts"]["B"] = Value::Null;
    let mut compiled = build(json).unwrap();
    let state = compiled.initial_state().to_vec();
    let mut rates = vec![0.0; 2];

    assert_eq!(
        compiled.check_values().err().as_deref(),
        Some(
            "The model cannot run: the parameters [\"k1\"] and the amounts of the bins [\"B\"] \
             have no value. Give them a value, or calibrate them and write the calibrated \
             values back"
        )
    );
    assert_eq!(
        compiled
            .transition_rates(0.0, 0.0, &state, &mut rates)
            .err()
            .as_deref(),
        Some("The rate of transition 'ab' cannot be evaluated: the parameter 'k1' has no value")
    );

    compiled.set_parameter("k1", 0.1).unwrap();
    assert_eq!(
        compiled.check_values().err().as_deref(),
        Some(
            "The model cannot run: the amounts of the bins [\"B\"] have no value. Give them a \
             value, or calibrate them and write the calibrated values back"
        )
    );

    let index = |name: &str| {
        compiled
            .compartments()
            .iter()
            .position(|compartment| compartment == name)
            .unwrap()
    };
    let (b_g1, b_g2) = (index("B_g1"), index("B_g2"));
    compiled.set_initial_condition(b_g2, 5.0).unwrap();
    assert_eq!(
        compiled.check_values().err().as_deref(),
        Some(
            "The model cannot run: the amounts of the compartments [\"B_g1\"] have no value. \
             Give them a value, or calibrate them and write the calibrated values back"
        )
    );

    compiled.set_initial_condition(b_g1, 5.0).unwrap();
    assert_eq!(compiled.check_values(), Ok(()));
}
