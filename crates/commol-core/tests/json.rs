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
    let value: Value = serde_json::from_str(&text).unwrap();
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
        error(r#"{"moves": [{"kind": "walk", "steps": 1, "scale": "NaN"}]}"#),
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
fn the_json_written_quotes_each_key_and_writes_each_number_that_is_not_finite_as_its_text() {
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
        r#"{"a \"b\"":null,"c":true,"d":[-3,0.5,"NaN","Infinity","-Infinity"],"e":"f"}"#
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

#[test]
fn a_union_needs_an_object_whose_kind_is_a_name() {
    assert_eq!(
        error(r#"{"moves": [3]}"#),
        "moves[0]: expected an object with a kind"
    );
    assert_eq!(
        error(r#"{"moves": [{"kind": 1}]}"#),
        "moves[0].kind: the kind is the name of a variant"
    );
}

#[test]
fn two_values_are_equal_when_their_numbers_are() {
    assert_eq!(Value::Float(0.5), Value::Float(0.5));
    assert_eq!(Value::Int(1), Value::Float(1.0));
    assert_ne!(Value::Null, Value::Bool(false));
    assert_eq!(Value::Float(0.5).field("kind"), None);
}

#[test]
fn a_whole_number_beyond_64_bits_keeps_its_value() {
    let big = Value::Int(i128::MAX);

    assert_eq!(
        json::from_value::<i128, json::Error>(big.clone()),
        Ok(i128::MAX)
    );
    assert_eq!(
        json::from_value::<Value, json::Error>(big.clone()),
        Ok(big.clone())
    );
    assert_eq!(serde_json::to_string(&big).unwrap(), i128::MAX.to_string());
}

#[test]
fn a_value_is_only_what_json_holds() {
    let bytes = serde::de::value::BytesDeserializer::<json::Error>::new(b"x");

    let error = Value::deserialize(bytes).unwrap_err();

    assert!(
        error.to_string().contains("expected a JSON value"),
        "{error}"
    );
}

#[derive(Debug, PartialEq, Deserialize)]
enum Shape {
    Dot,
    Line(f64),
}

#[derive(Debug, PartialEq, Deserialize)]
struct Length(f64);

#[test]
fn an_enum_reads_a_name_or_an_object_of_one_field() {
    assert_eq!(
        json::from_value::<Shape, json::Error>(Value::String("Dot".to_string())),
        Ok(Shape::Dot)
    );
    assert_eq!(
        json::from_value::<Shape, json::Error>(Value::Map(vec![(
            "Line".to_string(),
            Value::Float(2.0)
        )])),
        Ok(Shape::Line(2.0))
    );
    let error = json::from_value::<Shape, json::Error>(Value::Bool(true)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expected a name or an object of one field, got true"),
        "{error}"
    );
}

#[test]
fn a_newtype_reads_its_value() {
    assert_eq!(
        json::from_value::<Length, json::Error>(Value::Float(2.0)),
        Ok(Length(2.0))
    );
}

#[test]
fn a_setting_is_read_only_from_json() {
    let bytes = || serde::de::value::BytesDeserializer::<json::Error>::new(b"x");

    assert!(<Walk as Deserialize>::deserialize(bytes()).is_err());
    assert!(<Leap as Deserialize>::deserialize(bytes()).is_err());
}

#[test]
fn pairs_are_an_object() {
    let error = json::read::<json::Pairs<f64>>("[1]").unwrap_err();

    assert!(error.contains("expected an object"), "{error}");
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Reading {
    label: String,
    value: f64,
    bound: Option<f64>,
}

#[test]
fn a_number_that_is_not_finite_is_read_from_its_text_where_a_number_is_asked() {
    for (text, value) in [
        ("Infinity", f64::INFINITY),
        ("-Infinity", f64::NEG_INFINITY),
    ] {
        let reading: Reading = json::read(&format!(
            r#"{{"label": "a", "value": "{text}", "bound": "{text}"}}"#
        ))
        .unwrap();

        assert_eq!(reading.value, value);
        assert_eq!(reading.bound, Some(value));
    }
    let reading: Reading = json::read(r#"{"label": "a", "value": "NaN", "bound": null}"#).unwrap();
    assert!(reading.value.is_nan());
    let message =
        json::read::<Reading>(r#"{"label": "a", "value": "x", "bound": null}"#).unwrap_err();
    assert!(
        message.starts_with("value: invalid type: string"),
        "{message}"
    );
}

#[derive(Debug, PartialEq, Deserialize)]
struct Single {
    value: f32,
}

#[test]
fn a_single_precision_number_reads_the_text_of_a_number_that_is_not_finite() {
    let single: Single = json::read(r#"{"value": "-Infinity"}"#).unwrap();

    assert_eq!(single.value, f32::NEG_INFINITY);
}

#[test]
fn the_text_of_a_number_that_is_not_finite_stays_text_where_a_string_is_asked() {
    let reading: Reading = json::read(r#"{"label": "NaN", "value": 1.0, "bound": null}"#).unwrap();

    assert_eq!(reading.label, "NaN");
    let written = json::write(&Reading {
        label: "NaN".to_string(),
        value: f64::NAN,
        bound: Some(f64::INFINITY),
    });
    assert_eq!(
        written,
        r#"{"label":"NaN","value":"NaN","bound":"Infinity"}"#
    );
}

#[test]
fn a_rule_reads_the_text_of_a_number_that_is_not_finite() {
    let message = json::read::<Walk>(r#"{"steps": 1, "scale": "-Infinity"}"#).unwrap_err();

    assert!(message.starts_with("scale: scale is"), "{message}");
    assert!(message.ends_with("got -inf"), "{message}");
}

#[derive(Serialize)]
struct Unit;

#[derive(Serialize)]
struct Pair(i8, u8);

#[derive(Serialize)]
struct Tag(char);

#[derive(Serialize)]
enum Shapes {
    Point(i16, u16),
    Box { width: f32 },
}

#[derive(Serialize)]
struct Every {
    small: (i32, u32, i64),
    unit: Unit,
    nothing: (),
    pair: Pair,
    tag: Tag,
    shapes: Vec<Shapes>,
    map: std::collections::BTreeMap<String, u64>,
    whole: i128,
}

#[test]
fn every_shape_of_serde_writes_its_json() {
    let every = Every {
        small: (-1, 2, -3),
        unit: Unit,
        nothing: (),
        pair: Pair(-4, 5),
        tag: Tag('x'),
        shapes: vec![Shapes::Point(-6, 7), Shapes::Box { width: 0.5 }],
        map: [("k".to_string(), 8)].into(),
        whole: 9,
    };

    assert_eq!(
        json::write(&every),
        r#"{"small":[-1,2,-3],"unit":null,"nothing":null,"pair":[-4,5],"tag":"x","shapes":[{"Point":[-6,7]},{"Box":{"width":0.5}}],"map":{"k":8},"whole":9}"#
    );
}

#[derive(Serialize)]
struct Bytes(#[serde(with = "serde_bytes_like")] Vec<u8>);

mod serde_bytes_like {
    pub fn serialize<S: serde::Serializer>(value: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(value)
    }
}

#[test]
#[should_panic(expected = "JSON has no bytes")]
fn bytes_are_not_json() {
    json::write(&Bytes(vec![1]));
}

#[test]
#[should_panic(expected = "a key of JSON is a string")]
fn a_key_of_json_is_a_string() {
    json::write(&std::collections::BTreeMap::from([(1, 2)]));
}
