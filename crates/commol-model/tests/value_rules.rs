//! The rules of the values of a model ([`commol_model::rules`]): the
//! compilation of a model rejects every value outside its rule with the
//! message of the check of that rule, which the Python object of the value
//! also calls.

use commol_core::{Model, ParameterValue, SeriesMode};
use commol_model::{CompiledModel, rules};

/// A → B at the rate `k`, from A = 800 and B = 200, with the series `s` of
/// the points (1, 10) and (5, 30).
fn model() -> Model {
    serde_json::from_str(
        r#"{"name":"m",
        "parameters":[{"id":"k","value":0.3},
        {"id":"s","value":{"data":[[1,10],[5,30]],"mode":"linear"}}],
        "engine":{"type":"DifferentialEquations"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k"}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":800.0,"B":200.0}}}"#,
    )
    .expect("valid model JSON")
}

fn compile_error(model: &Model) -> String {
    CompiledModel::try_from_model(model)
        .err()
        .expect("the model has a value outside its rule")
}

fn bin_amount(amount: f64) -> Model {
    let mut model = model();
    model
        .aggregate_mut()
        .unwrap()
        .initial_conditions
        .bin_amounts["B"] = Some(amount);
    model
}

fn compartment_amount(amount: f64) -> Model {
    let mut model = model();
    model
        .aggregate_mut()
        .unwrap()
        .initial_conditions
        .compartment_amounts
        .insert("A".to_string(), amount);
    model
}

fn impulse(at: Option<Vec<f64>>, every: Option<(f64, f64)>) -> Model {
    let mut model = model();
    let transition = &mut model.process.transitions[0];
    transition.impulse_at = at;
    transition.impulse_every = every;
    model
}

/// The model with the stratification `group` of the categories `g1` and
/// `g2`, of the fractions `g1` and `g2`.
fn fractions(g1: f64, g2: f64) -> Model {
    let mut model = model();
    model.process.stratifications.push(
        serde_json::from_str(r#"{"id":"group","categories":["g1","g2"]}"#)
            .expect("valid stratification JSON"),
    );
    model
        .aggregate_mut()
        .unwrap()
        .initial_conditions
        .stratification_fractions
        .insert(
            "group".to_string(),
            [("g1".to_string(), g1), ("g2".to_string(), g2)]
                .into_iter()
                .collect(),
        );
    model
}

fn series(data: Vec<(f64, f64)>) -> Model {
    let mut model = model();
    model.parameters[1].value = Some(ParameterValue::Series {
        data,
        mode: SeriesMode::Linear,
    });
    model
}

#[test]
fn every_value_outside_its_rule_is_rejected() {
    let both = (Some(vec![1.0]), Some((2.0, 0.0)));
    let cases = [
        (
            bin_amount(-1.0),
            rules::check_bin_amount("B", -1.0),
            "Bin 'B': initial amount is a finite number of at least 0, got -1",
        ),
        (
            bin_amount(f64::NAN),
            rules::check_bin_amount("B", f64::NAN),
            "Bin 'B': initial amount is a finite number of at least 0, got NaN",
        ),
        (
            compartment_amount(f64::INFINITY),
            rules::check_compartment_amount("A", f64::INFINITY),
            "Compartment 'A': initial amount is a finite number of at least 0, got inf",
        ),
        (
            impulse(both.0.clone(), both.1),
            rules::check_impulse("ab", both.0.as_deref(), both.1),
            "Transition 'ab': set impulse_at or impulse_every, not both",
        ),
        (
            impulse(Some(vec![]), None),
            rules::check_impulse("ab", Some(&[]), None),
            "Transition 'ab': impulse_at has no position; give at least one",
        ),
        (
            impulse(Some(vec![1.0, -1.0]), None),
            rules::check_impulse("ab", Some(&[1.0, -1.0]), None),
            "Transition 'ab': impulse position is a finite number of at least 0, got -1",
        ),
        (
            impulse(None, Some((0.0, 0.0))),
            rules::check_impulse("ab", None, Some((0.0, 0.0))),
            "Transition 'ab': impulse period is a finite number greater than 0, got 0",
        ),
        (
            impulse(None, Some((1.0, f64::INFINITY))),
            rules::check_impulse("ab", None, Some((1.0, f64::INFINITY))),
            "Transition 'ab': impulse offset is a finite number of at least 0, got inf",
        ),
        (
            fractions(1.5, -0.5),
            rules::check_fractions("group", [("g1", 1.5), ("g2", -0.5)]),
            "Stratification 'group', category 'g1': fraction is a number from 0 to 1, got 1.5",
        ),
        (
            fractions(0.5, f64::NAN),
            rules::check_fractions("group", [("g1", 0.5), ("g2", f64::NAN)]),
            "Stratification 'group', category 'g2': fraction is a number from 0 to 1, got NaN",
        ),
        (
            fractions(0.5, 0.4),
            rules::check_fractions("group", [("g1", 0.5), ("g2", 0.4)]),
            "Stratification 'group': the fractions of its categories sum to 1 within 1e-6, got 0.9",
        ),
        (
            series(vec![]),
            rules::check_series(&[]),
            "a series has at least one point, got none",
        ),
        (
            series(vec![(1.0, 0.0), (-1.0, 0.0)]),
            rules::check_series(&[(1.0, 0.0), (-1.0, 0.0)]),
            "series position is a finite number of at least 0, got -1",
        ),
        (
            series(vec![(1.0, f64::NAN)]),
            rules::check_series(&[(1.0, f64::NAN)]),
            "series value is a finite number, got NaN",
        ),
        (
            series(vec![(2.0, 0.0), (1.0, 0.0), (2.0, 1.0)]),
            rules::check_series(&[(2.0, 0.0), (1.0, 0.0), (2.0, 1.0)]),
            "a series has one point per position, got two at 2",
        ),
    ];
    for (model, check, message) in cases {
        assert_eq!(check, Err(message.to_string()));
        let compiled = if message.contains("series") {
            format!("Series parameter 's': {message}")
        } else {
            message.to_string()
        };
        assert_eq!(compile_error(&model), compiled);
    }
}

#[test]
fn a_value_inside_its_rule_is_accepted() {
    assert_eq!(rules::check_bin_amount("B", 0.0), Ok(()));
    assert_eq!(rules::check_compartment_amount("A", 3.5), Ok(()));
    assert_eq!(rules::check_impulse("ab", None, None), Ok(()));
    assert_eq!(rules::check_impulse("ab", Some(&[0.0, 2.5]), None), Ok(()));
    assert_eq!(rules::check_impulse("ab", None, Some((0.5, 0.0))), Ok(()));
    assert_eq!(rules::check_series(&[(0.0, -3.0), (2.0, 1.0)]), Ok(()));
    assert_eq!(
        rules::check_fractions("group", [("g1", 1.0), ("g2", 0.0)]),
        Ok(())
    );
    assert_eq!(
        rules::check_fractions("group", [("g1", 0.6000005), ("g2", 0.4)]),
        Ok(())
    );
    assert!(CompiledModel::try_from_model(&fractions(0.25, 0.75)).is_ok());
}
