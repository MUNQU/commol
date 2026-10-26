//! A JSON value held in memory, in the order of its text, so that a reader
//! can read it twice: once to check it, once into its type.

use std::fmt::{self, Display};

use serde::de::value::{MapAccessDeserializer, MapDeserializer, SeqDeserializer};
use serde::de::{self, IntoDeserializer, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};

/// A JSON value. A number keeps whether it is whole, and a number that is
/// not finite is a [`Value::Float`]. Two numbers are equal when their values
/// are, `1` and `1.0` too, as JSON reads them.
#[derive(Clone, Debug)]
pub enum Value {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A whole number.
    Int(i128),
    /// Any other number, `Infinity`, `-Infinity` and `NaN` included.
    Float(f64),
    /// A string.
    String(String),
    /// A list.
    List(Vec<Value>),
    /// An object, its fields in the order of the text.
    Map(Vec<(String, Value)>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a == b,
            (Self::Int(a), Self::Float(b)) | (Self::Float(b), Self::Int(a)) => *a as f64 == *b,
            (Self::String(a), Self::String(b)) => a == b,
            (Self::List(a), Self::List(b)) => a == b,
            (Self::Map(a), Self::Map(b)) => a == b,
            _ => false,
        }
    }
}

impl Value {
    /// The field `name` of an object, if it has one.
    pub fn field(&self, name: &str) -> Option<&Value> {
        match self {
            Self::Map(fields) => fields
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value),
            _ => None,
        }
    }
}

/// The JSON of a value, with no spaces: each key quoted, as every JSON
/// reader reads it, and `Infinity`, `-Infinity` or `NaN` for a number that is
/// not finite.
impl Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let quoted = |text: &str| serde_json::to_string(text).expect("JSON writes every string");
        match self {
            Self::Null => f.write_str("null"),
            Self::Bool(value) => write!(f, "{value}"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Float(value) if value.is_nan() => f.write_str("NaN"),
            Self::Float(value) if value.is_infinite() => f.write_str(if *value > 0.0 {
                "Infinity"
            } else {
                "-Infinity"
            }),
            Self::Float(value) => {
                f.write_str(&serde_json::to_string(value).expect("JSON writes every finite number"))
            }
            Self::String(value) => f.write_str(&quoted(value)),
            Self::List(items) => {
                f.write_str("[")?;
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        f.write_str(",")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("]")
            }
            Self::Map(fields) => {
                f.write_str("{")?;
                for (index, (key, value)) in fields.iter().enumerate() {
                    if index > 0 {
                        f.write_str(",")?;
                    }
                    write!(f, "{}:{value}", quoted(key))?;
                }
                f.write_str("}")
            }
        }
    }
}

/// The error of a reader of a [`Value`].
#[derive(Debug, Clone, PartialEq)]
pub struct Error(String);

impl Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl de::Error for Error {
    fn custom<T: Display>(message: T) -> Self {
        Self(message.to_string())
    }
}

impl serde::ser::Error for Error {
    fn custom<T: Display>(message: T) -> Self {
        Self(message.to_string())
    }
}

/// Reads any JSON value into a [`Value`].
struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Int(i128::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Int(i128::from(value)))
    }

    fn visit_i128<E>(self, value: i128) -> Result<Value, E> {
        Ok(Value::Int(value))
    }

    fn visit_u128<E>(self, value: u128) -> Result<Value, E> {
        // A whole number beyond i128 is read as the nearest number.
        Ok(i128::try_from(value).map_or(Value::Float(value as f64), Value::Int))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Value, E> {
        Ok(Value::Float(value))
    }

    fn visit_str<E>(self, value: &str) -> Result<Value, E> {
        Ok(Value::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> Result<Value, E> {
        Ok(Value::String(value))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D: de::Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        de::Deserialize::deserialize(deserializer)
    }

    fn visit_seq<A: de::SeqAccess<'de>>(self, mut items: A) -> Result<Value, A::Error> {
        let mut list = Vec::new();
        while let Some(item) = items.next_element()? {
            list.push(item);
        }
        Ok(Value::List(list))
    }

    fn visit_map<A: de::MapAccess<'de>>(self, mut entries: A) -> Result<Value, A::Error> {
        let mut fields = Vec::new();
        while let Some(entry) = entries.next_entry::<String, Value>()? {
            fields.push(entry);
        }
        Ok(Value::Map(fields))
    }
}

impl<'de> de::Deserialize<'de> for Value {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ValueVisitor)
    }
}

impl serde::Serialize for Value {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Int(value) => match (u64::try_from(*value), i64::try_from(*value)) {
                (Ok(value), _) => serializer.serialize_u64(value),
                (_, Ok(value)) => serializer.serialize_i64(value),
                _ => serializer.serialize_i128(*value),
            },
            Self::Float(value) => serializer.serialize_f64(*value),
            Self::String(value) => serializer.serialize_str(value),
            Self::List(items) => {
                let mut list = serializer.serialize_seq(Some(items.len()))?;
                items
                    .iter()
                    .try_for_each(|item| list.serialize_element(item))?;
                list.end()
            }
            Self::Map(fields) => {
                let mut map = serializer.serialize_map(Some(fields.len()))?;
                fields
                    .iter()
                    .try_for_each(|(key, value)| map.serialize_entry(key, value))?;
                map.end()
            }
        }
    }
}

impl<'de> IntoDeserializer<'de, Error> for Value {
    type Deserializer = Self;

    fn into_deserializer(self) -> Self {
        self
    }
}

impl<'de> de::Deserializer<'de> for Value {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Self::Null => visitor.visit_unit(),
            Self::Bool(value) => visitor.visit_bool(value),
            Self::Int(value) => match (u64::try_from(value), i64::try_from(value)) {
                (Ok(value), _) => visitor.visit_u64(value),
                (_, Ok(value)) => visitor.visit_i64(value),
                _ => visitor.visit_i128(value),
            },
            Self::Float(value) => visitor.visit_f64(value),
            Self::String(value) => visitor.visit_string(value),
            Self::List(items) => {
                let mut items = SeqDeserializer::new(items.into_iter());
                let value = visitor.visit_seq(&mut items)?;
                items.end()?;
                Ok(value)
            }
            Self::Map(fields) => {
                let mut fields = MapDeserializer::new(fields.into_iter());
                let value = visitor.visit_map(&mut fields)?;
                fields.end()?;
                Ok(value)
            }
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Self::Null => visitor.visit_none(),
            value => visitor.visit_some(value),
        }
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        match self {
            Self::String(variant) => visitor.visit_enum(variant.into_deserializer()),
            Self::Map(fields) if fields.len() == 1 => visitor.visit_enum(
                MapAccessDeserializer::new(MapDeserializer::new(fields.into_iter())),
            ),
            other => Err(de::Error::invalid_type(other.unexpected(), &visitor)),
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_newtype_struct(self)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes
        byte_buf unit unit_struct seq tuple tuple_struct map struct identifier
        ignored_any
    }
}

impl Value {
    /// What `self` is, for the message of a value of the wrong type.
    fn unexpected(&self) -> de::Unexpected<'_> {
        match self {
            Self::Null => de::Unexpected::Unit,
            Self::Bool(value) => de::Unexpected::Bool(*value),
            Self::Int(value) => i64::try_from(*value).map_or(
                de::Unexpected::Other("a whole number"),
                de::Unexpected::Signed,
            ),
            Self::Float(value) => de::Unexpected::Float(*value),
            Self::String(value) => de::Unexpected::Str(value),
            Self::List(_) => de::Unexpected::Seq,
            Self::Map(_) => de::Unexpected::Map,
        }
    }
}
