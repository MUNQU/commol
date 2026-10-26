//! The port of a setting: a union tagged by `kind`, a struct with range
//! rules, and the path that an error names.

use commol_core::json::{self, Value};
use commol_core::{Count, Real, Rule, checked, tagged};
use serde::{Deserialize, Serialize};

const STEPS: Count = Count::at_least("steps", 1);
const SCALE: Real = Real::positive_or_infinite("scale");

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
struct Walk {
    steps: u64,
    scale: f64,
}

checked!(Walk, "Walk", [Rule::Count(STEPS), Rule::Real(SCALE)]);

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", rename_all = "snake_case")]
enum Move {
    Stay,
    Walk(Walk),
}

tagged!(Move);

const HEIGHT: Count = Count::at_least("height", 1);
const JUMP: commol_core::Ranges = commol_core::Ranges {
    object: "Jump",
    rules: &[Rule::Count(HEIGHT)],
};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", rename_all = "snake_case", deny_unknown_fields)]
enum Leap {
    Jump { height: u64 },
    Rest,
}

tagged!(Leap, ranges = {"jump" => JUMP});

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    moves: Vec<Move>,
}

fn error(text: &str) -> String {
    json::read::<Plan>(text).expect_err("the plan is not valid")
}

#[test]
fn a_union_reads_each_variant_by_its_kind() {
    let plan: Plan =
        json::read(r#"{"moves": [{"kind": "stay"}, {"kind": "walk", "steps": 2, "scale": 0.5}]}"#)
            .unwrap();

    assert_eq!(
        plan.moves,
        [
            Move::Stay,
            Move::Walk(Walk {
                steps: 2,
                scale: 0.5
            })
        ]
    );
}

#[test]
fn what_is_written_reads_back() {
    let plan = Plan {
        moves: vec![
            Move::Walk(Walk {
                steps: 3,
                scale: f64::INFINITY,
            }),
            Move::Stay,
        ],
    };

    let text = json::write(&plan);

    assert_eq!(json::read::<Plan>(&text).unwrap(), plan);
    let value: Value = json5::from_str(&text).unwrap();
    let Value::Map(fields) = value else {
        panic!("a plan is an object")
    };
    assert_eq!(fields.len(), 1);
}

#[test]
fn a_rule_names_the_path_of_its_field_through_a_union() {
    assert_eq!(
        error(r#"{"moves": [{"kind": "stay"}, {"kind": "walk", "steps": 0, "scale": 1}]}"#),
        "moves[1].steps: steps is a whole number of at least 1, got 0"
    );
    assert_eq!(
        error(r#"{"moves": [{"kind": "walk", "steps": 1, "scale": NaN}]}"#),
        "moves[0].scale: scale is a number greater than 0, got NaN"
    );
}

#[test]
fn a_count_below_zero_gets_the_message_of_its_rule() {
    assert_eq!(
        error(r#"{"moves": [{"kind": "walk", "steps": -1, "scale": 1}]}"#),
        "moves[0].steps: steps is a whole number of at least 1, got -1"
    );
}

#[test]
fn an_unknown_kind_is_an_error_at_the_tag() {
    let message = error(r#"{"moves": [{"kind": "run"}]}"#);

    assert!(
        message.starts_with("moves[0].kind: unknown variant `run`"),
        "{message}"
    );
}

#[test]
fn an_unknown_field_is_named() {
    let message = error(r#"{"moves": [{"kind": "walk", "steps": 1, "scale": 1, "pace": 2}]}"#);

    assert_eq!(
        message,
        "moves[0].pace: unknown field `pace`, expected `steps` or `scale`"
    );
}

#[test]
fn a_missing_field_is_named() {
    let message = error(r#"{"moves": [{"kind": "walk", "steps": 1}]}"#);

    assert_eq!(message, "moves[0]: missing field `scale`");
}

#[test]
fn a_union_needs_its_tag() {
    assert_eq!(
        error(r#"{"moves": [{"steps": 1}]}"#),
        "moves[0]: missing field `kind`"
    );
}

#[test]
fn a_text_that_is_not_json_is_named() {
    assert!(error("{moves: [").starts_with("Not JSON"));
}

#[test]
fn the_json_written_quotes_each_key_and_names_each_number_that_is_not_finite() {
    let value = Value::Map(vec![
        ("a \"b\"".to_owned(), Value::Null),
        ("c".to_owned(), Value::Bool(true)),
        (
            "d".to_owned(),
            Value::List(vec![
                Value::Int(-3),
                Value::Float(0.5),
                Value::Float(f64::NAN),
                Value::Float(f64::INFINITY),
                Value::Float(f64::NEG_INFINITY),
            ]),
        ),
        ("e".to_owned(), Value::String("f".to_owned())),
    ]);

    assert_eq!(
        json::write(&value),
        r#"{"a \"b\"":null,"c":true,"d":[-3,0.5,NaN,Infinity,-Infinity],"e":"f"}"#
    );
}

#[test]
fn a_union_checks_the_rules_of_the_variant_of_its_kind() {
    let leap = |text: &str| json::read::<Vec<Leap>>(text);

    assert_eq!(
        leap(r#"[{"kind": "jump", "height": 2}, {"kind": "rest"}]"#).unwrap(),
        [Leap::Jump { height: 2 }, Leap::Rest]
    );
    assert_eq!(
        leap(r#"[{"kind": "jump", "height": 0}]"#).unwrap_err(),
        "[0].height: height is a whole number of at least 1, got 0"
    );
    assert_eq!(
        leap(r#"[{"height": 0}]"#).unwrap_err(),
        "[0]: missing field `kind`"
    );
}
