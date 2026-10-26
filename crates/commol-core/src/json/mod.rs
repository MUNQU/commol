//! The port of a setting: the JSON of its Python model, read into the type
//! the engine owns, with the path of the field that an error is about.
//!
//! A setting of the Python API crosses to the engine as the JSON that
//! `model_dump()` of its pydantic model writes. JSON has no number that is
//! not finite, so a number such as an infinite weight crosses as the text
//! `"Infinity"`, `"-Infinity"` or `"NaN"`, as pydantic writes it with
//! `ser_json_inf_nan="strings"`. A reader reads such a text as a number only
//! where its type asks for a number, so a label `"NaN"` stays text, and
//! [`write()`] writes the same three texts. An error names the path of its field,
//! such as `optimization_config.inertia_factor: inertia_factor is a number
//! greater than 0, got -1`.
//!
//! - [`read`] reads a setting, [`write()`] writes it back.
//! - [`tagged!`](crate::tagged) gives a union of classes, tagged by `kind`
//!   as each of its pydantic classes is, a reader that keeps the path into
//!   the variant.
//! - [`checked!`](crate::checked) gives a struct whose settings have range
//!   rules a reader that checks each rule, with the path of its field.
//! - [`from_value`] reads a variant of a union whose members differ by JSON
//!   type, such as a number or a name, with the path into the variant.
//!
//! A reader that reads its input twice holds it in a [`Value`] and reads the
//! value with a path of its own; its error carries that path to the reader
//! above it ([`nested`]), which joins the two.

mod serializer;
mod value;

use serde::de::{self, DeserializeOwned};
use serde_path_to_error::{Path, Segment, Track};
pub use value::{Error, Value, non_finite, non_finite_text};

use crate::range::{Ranges, Rule, Setting};

#[doc(hidden)]
pub mod reexport {
    pub use serde;
}

/// The mark around the path that an error of a reader carries to the reader
/// above it.
const MARK: char = '\u{1d}';

/// The deserializer of a [`Value`] that records the path of each field.
pub type Tracked<'a> = serde_path_to_error::Deserializer<'a, 'a, Value>;

/// Read `text`, the JSON of a setting, into `T`.
///
/// # Errors
///
/// Returns the message of the first error, after the path of its field when
/// it has one: `optimization_config.inertia_factor: <message>`.
pub fn read<T: DeserializeOwned>(text: &str) -> Result<T, String> {
    let value: Value = serde_json::from_str(text).map_err(|error| format!("Not JSON: {error}"))?;
    let mut track = Track::new();
    T::deserialize(Tracked::new(value, &mut track))
        .map_err(|error| located(&track.path(), &error.to_string()))
}

/// The JSON of `value`, with the texts `"Infinity"`, `"-Infinity"` and
/// `"NaN"` for the numbers that are not finite, as [`read`] reads them.
///
/// # Panics
///
/// Panics when `value` is not plain data, such as a map whose keys are not
/// strings: every setting is plain data.
pub fn write<T: serde::Serialize>(value: &T) -> String {
    to_value(value).to_string()
}

/// The [`Value`] of `value`.
///
/// # Panics
///
/// Panics when `value` is not plain data, as [`write()`] does.
pub fn to_value<T: serde::Serialize>(value: &T) -> Value {
    value
        .serialize(serializer::ValueSerializer)
        .expect("a setting is plain data that JSON writes")
}

/// Read `value` into `T`, with an error that carries the path into `value`
/// to the reader above.
///
/// # Errors
///
/// Returns the error of `T`, with its path ([`nested`]).
pub fn from_value<T: DeserializeOwned, E: de::Error>(value: Value) -> Result<T, E> {
    tracked(value, |tracked| T::deserialize(tracked))
}

/// Read `value` with `read`, which takes the deserializer that records the
/// path, with an error that carries that path to the reader above.
///
/// # Errors
///
/// Returns the error of `read`, with its path ([`nested`]).
pub fn tracked<T, E: de::Error>(
    value: Value,
    read: impl for<'a> FnOnce(Tracked<'a>) -> Result<T, Error>,
) -> Result<T, E> {
    let mut track = Track::new();
    read(Tracked::new(value, &mut track))
        .map_err(|error| nested(&path_text(&track.path()), &error.to_string()))
}

/// An error at `path` below the reader that returns it, with `message`.
pub fn nested<E: de::Error>(path: &str, message: &str) -> E {
    let (inner, message) = split(message);
    E::custom(format!("{MARK}{}{MARK}{message}", join(path, inner)))
}

/// The message of an error at `path`, with the path a reader below carried
/// in `message`.
pub fn located(path: &Path, message: &str) -> String {
    let (inner, message) = split(message);
    let full = join(&path_text(path), inner);
    if full.is_empty() {
        message.to_string()
    } else {
        format!("{full}: {message}")
    }
}

/// The path that `message` carries, and the message without it.
fn split(message: &str) -> (&str, &str) {
    message
        .strip_prefix(MARK)
        .and_then(|rest| rest.split_once(MARK))
        .unwrap_or(("", message))
}

/// `inner` below `outer`: `a.b`, or `a[0]` for an item of a list.
fn join(outer: &str, inner: &str) -> String {
    match (outer.is_empty(), inner.is_empty()) {
        (true, _) => inner.to_string(),
        (_, true) => outer.to_string(),
        _ if inner.starts_with('[') => format!("{outer}{inner}"),
        _ => format!("{outer}.{inner}"),
    }
}

/// The fields and the items of `path`, such as `observed_data[2].value`;
/// the variant of an enum is not a field of the JSON, so it is left out.
fn path_text(path: &Path) -> String {
    path.iter()
        .fold(String::new(), |text, segment| match segment {
            Segment::Seq { index } => format!("{text}[{index}]"),
            Segment::Map { key } => join(&text, key),
            Segment::Enum { .. } | Segment::Unknown => text,
        })
}

/// The setting of `value`, as a range rule reads it.
fn setting(value: &Value) -> Setting {
    match value {
        Value::Int(value) => Setting::Whole(*value),
        Value::Float(value) => Setting::Number(*value),
        Value::String(text) => non_finite(text).map_or(Setting::Other, Setting::Number),
        Value::List(items) => Setting::List(items.iter().map(setting).collect()),
        _ => Setting::Other,
    }
}

/// Check each field of `value`, an object, with the rule of its name.
///
/// # Errors
///
/// Returns the message of the first rule a field breaks, at the path of the
/// field.
pub fn check_rules<E: de::Error>(value: &Value, rules: &[Rule]) -> Result<(), E> {
    rules.iter().try_for_each(|rule| {
        value
            .field(rule.name())
            .map_or(Ok(()), |field| rule.check(&setting(field)))
            .map_err(|message| nested(rule.name(), &message))
    })
}

/// Check the fields of `value`, an object of a union tagged by `kind`, with
/// the rules of its `kind` in `ranges`. A `kind` with no rules, and a value
/// with no `kind`, which [`read_variant`] rejects, pass.
///
/// # Errors
///
/// Returns the message of the first rule a field breaks, at the path of the
/// field.
pub fn check_variant<E: de::Error>(value: &Value, ranges: &[(&str, Ranges)]) -> Result<(), E> {
    let Some(Value::String(kind)) = value.field("kind") else {
        return Ok(());
    };
    ranges
        .iter()
        .find(|(name, _)| name == kind)
        .map_or(Ok(()), |(_, ranges)| check_rules(value, ranges.rules))
}

/// The object of a union tagged by `kind`, as an enum reads it: `{variant:
/// fields}`, or `variant` alone when it has no other field.
///
/// # Errors
///
/// Returns an error when `value` is not an object or has no `kind`, and an
/// error at `kind` when the tag is not a name.
pub fn untag<E: de::Error>(value: Value) -> Result<Value, E> {
    let Value::Map(mut fields) = value else {
        return Err(E::custom("expected an object with a kind"));
    };
    let Some(index) = fields.iter().position(|(key, _)| key == "kind") else {
        return Err(E::missing_field("kind"));
    };
    let Value::String(kind) = fields.remove(index).1 else {
        return Err(nested("kind", "the kind is the name of a variant"));
    };
    Ok(if fields.is_empty() {
        Value::String(kind)
    } else {
        Value::Map(vec![(kind, Value::Map(fields))])
    })
}

/// The object of a union tagged by `kind` from the form an enum writes:
/// `{variant: fields}` or `variant` alone, the tag first.
///
/// # Panics
///
/// Panics when `value` is not a form an enum writes.
pub fn tag(value: Value) -> Value {
    match value {
        Value::String(variant) => Value::Map(vec![("kind".to_string(), Value::String(variant))]),
        Value::Map(mut entries) if entries.len() == 1 => {
            let (variant, fields) = entries.remove(0);
            let Value::Map(fields) = fields else {
                // coverage: invariant, serde writes the fields of a variant as an object
                panic!("a variant of a tagged union is an object, got {fields:?}");
            };
            let mut tagged = vec![("kind".to_string(), Value::String(variant))];
            tagged.extend(fields);
            Value::Map(tagged)
        }
        // coverage: invariant, serde writes a variant as a name or an object of one field
        other => panic!("an enum writes a name or an object, got {other:?}"),
    }
}

/// Read `value`, an object of a union tagged by `kind`, with `read`, the
/// reader of the enum of the union ([`tagged!`](crate::tagged)).
///
/// # Errors
///
/// Returns the error of the variant with its path, or an error at `kind`
/// when the tag names no variant.
pub fn read_variant<T, E: de::Error>(
    value: Value,
    read: impl for<'a> FnOnce(Tracked<'a>) -> Result<T, Error>,
) -> Result<T, E> {
    let value = untag(value)?;
    let mut track = Track::new();
    let result = read(Tracked::new(value, &mut track));
    result.map_err(|error| {
        let path = track.path();
        let message = error.to_string();
        if path
            .iter()
            .any(|segment| matches!(segment, Segment::Enum { .. }))
        {
            nested(&path_text(&path), &message)
        } else {
            nested("kind", &message)
        }
    })
}

/// Give a union of classes tagged by `kind` its reader and its writer.
///
/// The enum derives `Serialize` and `Deserialize` with
/// `#[serde(remote = "Self", rename_all = "snake_case")]`: each variant is
/// the name of its `kind`, a unit variant for a class with no other field
/// and a newtype or struct variant otherwise. The reader keeps the path of
/// an error into the variant, which the internal tag of serde loses.
///
/// With `ranges = {"kind" => RANGES, ...}`, the reader checks the rules of
/// the variant of each `kind` on the JSON before it reads the variant, as
/// [`checked!`](crate::checked) does for a struct.
#[macro_export]
macro_rules! tagged {
    ($type:ty) => {
        $crate::tagged!($type, ranges = {});
    };
    ($type:ty, ranges = { $($kind:literal => $ranges:expr),* $(,)? }) => {
        impl $crate::json::reexport::serde::Serialize for $type {
            fn serialize<S: $crate::json::reexport::serde::Serializer>(
                &self,
                serializer: S,
            ) -> Result<S::Ok, S::Error> {
                struct External<'a>(&'a $type);
                impl $crate::json::reexport::serde::Serialize for External<'_> {
                    fn serialize<S: $crate::json::reexport::serde::Serializer>(
                        &self,
                        serializer: S,
                    ) -> Result<S::Ok, S::Error> {
                        <$type>::serialize(self.0, serializer)
                    }
                }
                $crate::json::reexport::serde::Serialize::serialize(
                    &$crate::json::tag($crate::json::to_value(&External(self))),
                    serializer,
                )
            }
        }

        impl<'de> $crate::json::reexport::serde::Deserialize<'de> for $type {
            fn deserialize<D: $crate::json::reexport::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> Result<Self, D::Error> {
                let value = <$crate::json::Value as $crate::json::reexport::serde::Deserialize>::deserialize(
                    deserializer,
                )?;
                let ranges: &[(&str, $crate::Ranges)] = &[$(($kind, $ranges)),*];
                $crate::json::check_variant::<D::Error>(&value, ranges)?;
                $crate::json::read_variant(value, |tracked| <$type>::deserialize(tracked))
            }
        }
    };
}

/// Give a struct whose settings have range rules its reader, its writer and
/// its rules, `RANGES`, as the Python API names its class.
///
/// The struct derives `Serialize` and `Deserialize` with
/// `#[serde(remote = "Self", deny_unknown_fields)]`. The reader checks each
/// rule on the JSON before it reads the fields, so a count of `-1` gets the
/// message of its rule, at the path of its field. With `check = function`,
/// the reader then checks the struct it read with `function`, of type
/// `fn(&Self) -> Result<(), String>`, for a rule that joins its fields.
#[macro_export]
macro_rules! checked {
    ($type:ty, $object:literal, [$($rule:expr),* $(,)?]) => {
        $crate::checked!($type, $object, [$($rule),*], check = |_: &$type| Ok(()));
    };
    ($type:ty, $object:literal, [$($rule:expr),* $(,)?], check = $check:expr) => {
        impl $type {
            /// The range rule of each setting, under the name of the class
            /// of the Python API.
            pub const RANGES: $crate::Ranges = $crate::Ranges {
                object: $object,
                rules: &[$($rule),*],
            };
        }

        impl $crate::json::reexport::serde::Serialize for $type {
            fn serialize<S: $crate::json::reexport::serde::Serializer>(
                &self,
                serializer: S,
            ) -> Result<S::Ok, S::Error> {
                <$type>::serialize(self, serializer)
            }
        }

        impl<'de> $crate::json::reexport::serde::Deserialize<'de> for $type {
            fn deserialize<D: $crate::json::reexport::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> Result<Self, D::Error> {
                let value = <$crate::json::Value as $crate::json::reexport::serde::Deserialize>::deserialize(
                    deserializer,
                )?;
                $crate::json::check_rules::<D::Error>(&value, Self::RANGES.rules)?;
                let read: Self = $crate::json::tracked(value, |tracked| <$type>::deserialize(tracked))?;
                let check: fn(&$type) -> Result<(), String> = $check;
                check(&read).map_err(<D::Error as $crate::json::reexport::serde::de::Error>::custom)?;
                Ok(read)
            }
        }
    };
}

/// The fields of a JSON object in the order of its text, such as a
/// `dict[str, float]` of the Python API: a map whose order a reader keeps.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Pairs<T>(pub Vec<(String, T)>);

impl<T> Pairs<T> {
    /// The value of `key`, if the object has it.
    pub fn get(&self, key: &str) -> Option<&T> {
        self.0
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }
}

impl<T: serde::Serialize> serde::Serialize for Pairs<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        self.0
            .iter()
            .try_for_each(|(key, value)| map.serialize_entry(key, value))?;
        map.end()
    }
}

impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for Pairs<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PairsVisitor<T>(std::marker::PhantomData<T>);
        impl<'de, T: serde::Deserialize<'de>> de::Visitor<'de> for PairsVisitor<T> {
            type Value = Pairs<T>;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an object")
            }
            fn visit_map<A: de::MapAccess<'de>>(
                self,
                mut entries: A,
            ) -> Result<Pairs<T>, A::Error> {
                let mut pairs = Vec::new();
                while let Some(entry) = entries.next_entry::<String, T>()? {
                    pairs.push(entry);
                }
                Ok(Pairs(pairs))
            }
        }
        deserializer.deserialize_map(PairsVisitor(std::marker::PhantomData))
    }
}
