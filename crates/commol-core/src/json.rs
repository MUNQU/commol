//! The port of a setting: the JSON of its Python model, read into the type
//! the engine owns, with the path of the field that an error is about.
//!
//! A setting of the Python API crosses to the engine as the JSON that
//! `model_dump()` of its pydantic model writes. JSON has no number that is
//! not finite, so a number such as an infinite weight crosses as one of the
//! constants of JSON5, `Infinity`, `-Infinity` or `NaN`; [`read`] reads
//! JSON5, which is JSON with them. An error names the path of its field,
//! such as `optimization_config.inertia_factor: inertia_factor is a number
//! greater than 0, got -1`.
//!
//! - [`read`] reads a setting, [`write`] writes it back.
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

use serde::de::{self, DeserializeOwned};
use serde_path_to_error::{Path, Segment, Track};
pub use serde_value::Value;
use serde_value::{DeserializerError, ValueDeserializer};

use crate::range::{Rule, Setting};

#[doc(hidden)]
pub mod reexport {
    pub use serde;
    pub use serde_path_to_error;
    pub use serde_value;
}

/// The mark around the path that an error of a reader carries to the reader
/// above it.
const MARK: char = '\u{1d}';

/// The deserializer of a [`Value`] that records the path of each field.
pub type Tracked<'a> =
    serde_path_to_error::Deserializer<'a, 'a, ValueDeserializer<DeserializerError>>;

/// Read `text`, the JSON of a setting, into `T`.
///
/// # Errors
///
/// Returns the message of the first error, after the path of its field when
/// it has one: `optimization_config.inertia_factor: <message>`.
pub fn read<T: DeserializeOwned>(text: &str) -> Result<T, String> {
    let value: Value = json5::from_str(text).map_err(|error| format!("Not JSON: {error}"))?;
    let mut track = Track::new();
    T::deserialize(Tracked::new(ValueDeserializer::new(value), &mut track))
        .map_err(|error| located(&track.path(), &error.to_string()))
}

/// The JSON of `value`, with `Infinity`, `-Infinity` and `NaN` for the
/// numbers that are not finite, as [`read`] reads them.
///
/// # Panics
///
/// Panics when `value` is not plain data, such as a map whose keys are not
/// strings: every setting is plain data.
pub fn write<T: serde::Serialize>(value: &T) -> String {
    json5::to_string(value).expect("a setting is plain data that JSON writes")
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
    read: impl for<'a> FnOnce(Tracked<'a>) -> Result<T, DeserializerError>,
) -> Result<T, E> {
    let mut track = Track::new();
    read(Tracked::new(ValueDeserializer::new(value), &mut track))
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

/// The setting of a number of `value`, as a range rule reads it.
fn setting(value: &Value) -> Setting {
    match value {
        Value::U8(v) => Setting::Whole(i128::from(*v)),
        Value::U16(v) => Setting::Whole(i128::from(*v)),
        Value::U32(v) => Setting::Whole(i128::from(*v)),
        Value::U64(v) => Setting::Whole(i128::from(*v)),
        Value::I8(v) => Setting::Whole(i128::from(*v)),
        Value::I16(v) => Setting::Whole(i128::from(*v)),
        Value::I32(v) => Setting::Whole(i128::from(*v)),
        Value::I64(v) => Setting::Whole(i128::from(*v)),
        Value::F32(v) => Setting::Number(f64::from(*v)),
        Value::F64(v) => Setting::Number(*v),
        Value::Seq(items) => Setting::List(items.iter().map(setting).collect()),
        Value::Option(Some(inner)) | Value::Newtype(inner) => setting(inner),
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
    let Value::Map(fields) = value else {
        return Ok(());
    };
    rules.iter().try_for_each(|rule| {
        fields
            .get(&Value::String(rule.name().to_string()))
            .map_or(Ok(()), |field| rule.check(&setting(field)))
            .map_err(|message| nested(rule.name(), &message))
    })
}

/// The object of a union tagged by `kind`, as an enum reads it: `{variant:
/// fields}`, or `variant` alone when it has no other field.
///
/// # Errors
///
/// Returns an error at `kind` when the object has no `kind`, or it is not a
/// name.
pub fn untag<E: de::Error>(value: Value) -> Result<Value, E> {
    let Value::Map(mut fields) = value else {
        return Err(E::invalid_type(
            de::Unexpected::Other("not an object"),
            &"an object with a kind",
        ));
    };
    let kind = match fields.remove(&Value::String("kind".to_string())) {
        Some(Value::String(kind)) => kind,
        Some(_) => return Err(nested("kind", "kind is the name of a variant")),
        None => return Err(E::missing_field("kind")),
    };
    Ok(if fields.is_empty() {
        Value::String(kind)
    } else {
        Value::Map([(Value::String(kind), Value::Map(fields))].into())
    })
}

/// The object of a union tagged by `kind` from the form an enum writes:
/// `{variant: fields}` or `variant` alone.
///
/// # Panics
///
/// Panics when `value` is not a form an enum writes.
pub fn tag(value: Value) -> Value {
    let key = Value::String("kind".to_string());
    match value {
        Value::String(_) => Value::Map([(key, value)].into()),
        Value::Map(entries) => {
            let (variant, fields) = entries
                .into_iter()
                .next()
                .expect("an enum writes its variant");
            let mut fields = match fields {
                Value::Map(fields) => fields,
                Value::Unit => std::collections::BTreeMap::new(),
                other => panic!("a variant of a tagged union is an object, got {other:?}"),
            };
            fields.insert(key, variant);
            Value::Map(fields)
        }
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
    read: impl for<'a> FnOnce(Tracked<'a>) -> Result<T, DeserializerError>,
) -> Result<T, E> {
    let value = untag(value)?;
    let mut track = Track::new();
    let result = read(Tracked::new(ValueDeserializer::new(value), &mut track));
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
#[macro_export]
macro_rules! tagged {
    ($type:ty) => {
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
                let value = $crate::json::reexport::serde_value::to_value(External(self))
                    .map_err(<S::Error as $crate::json::reexport::serde::ser::Error>::custom)?;
                $crate::json::reexport::serde::Serialize::serialize(
                    &$crate::json::tag(value),
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
/// message of its rule, at the path of its field.
#[macro_export]
macro_rules! checked {
    ($type:ty, $object:literal, [$($rule:expr),* $(,)?]) => {
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
                $crate::json::tracked(value, |tracked| <$type>::deserialize(tracked))
            }
        }
    };
}
