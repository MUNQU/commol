//! The [`Value`] of a value that serde writes: what [`to_value`](super::to_value)
//! returns. A number that is not finite stays a number, so a reader reads
//! the value back as it was.

use serde::ser::{self, Serialize};

use super::value::{Error, Value};

/// Writes a value of serde into a [`Value`].
pub(super) struct ValueSerializer;

impl ser::Serializer for ValueSerializer {
    type Ok = Value;
    type Error = Error;
    type SerializeSeq = List;
    type SerializeTuple = List;
    type SerializeTupleStruct = List;
    type SerializeTupleVariant = Variant<List>;
    type SerializeMap = Object;
    type SerializeStruct = Object;
    type SerializeStructVariant = Variant<Object>;

    fn serialize_bool(self, value: bool) -> Result<Value, Error> {
        Ok(Value::Bool(value))
    }

    fn serialize_i8(self, value: i8) -> Result<Value, Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_i16(self, value: i16) -> Result<Value, Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_i32(self, value: i32) -> Result<Value, Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_i64(self, value: i64) -> Result<Value, Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_i128(self, value: i128) -> Result<Value, Error> {
        Ok(Value::Int(value))
    }

    fn serialize_u8(self, value: u8) -> Result<Value, Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_u16(self, value: u16) -> Result<Value, Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_u32(self, value: u32) -> Result<Value, Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_u64(self, value: u64) -> Result<Value, Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_f32(self, value: f32) -> Result<Value, Error> {
        self.serialize_f64(f64::from(value))
    }

    fn serialize_f64(self, value: f64) -> Result<Value, Error> {
        Ok(Value::Float(value))
    }

    fn serialize_char(self, value: char) -> Result<Value, Error> {
        Ok(Value::String(value.to_string()))
    }

    fn serialize_str(self, value: &str) -> Result<Value, Error> {
        Ok(Value::String(value.to_string()))
    }

    fn serialize_bytes(self, _value: &[u8]) -> Result<Value, Error> {
        Err(ser::Error::custom("JSON has no bytes"))
    }

    fn serialize_none(self) -> Result<Value, Error> {
        Ok(Value::Null)
    }

    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<Value, Error> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Value, Error> {
        Ok(Value::Null)
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Value, Error> {
        Ok(Value::Null)
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
    ) -> Result<Value, Error> {
        self.serialize_str(variant)
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Value, Error> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Value, Error> {
        Ok(Value::Map(vec![(
            variant.to_string(),
            value.serialize(self)?,
        )]))
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<List, Error> {
        Ok(List(Vec::new()))
    }

    fn serialize_tuple(self, len: usize) -> Result<List, Error> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_struct(self, _name: &'static str, len: usize) -> Result<List, Error> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Variant<List>, Error> {
        Ok(Variant(variant, self.serialize_seq(Some(len))?))
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Object, Error> {
        Ok(Object::default())
    }

    fn serialize_struct(self, _name: &'static str, len: usize) -> Result<Object, Error> {
        self.serialize_map(Some(len))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Variant<Object>, Error> {
        Ok(Variant(variant, self.serialize_map(Some(len))?))
    }
}

/// The items of a list, a tuple or a tuple struct.
pub(super) struct List(Vec<Value>);

impl List {
    fn push<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        self.0.push(value.serialize(ValueSerializer)?);
        Ok(())
    }
}

impl ser::SerializeSeq for List {
    type Ok = Value;
    type Error = Error;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<Value, Error> {
        Ok(Value::List(self.0))
    }
}

impl ser::SerializeTuple for List {
    type Ok = Value;
    type Error = Error;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<Value, Error> {
        Ok(Value::List(self.0))
    }
}

impl ser::SerializeTupleStruct for List {
    type Ok = Value;
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }

    fn end(self) -> Result<Value, Error> {
        Ok(Value::List(self.0))
    }
}

/// The fields of an object, a map or a struct, in the order they are
/// written.
#[derive(Default)]
pub(super) struct Object {
    fields: Vec<(String, Value)>,
    key: Option<String>,
}

impl Object {
    fn insert<T: ?Sized + Serialize>(&mut self, key: String, value: &T) -> Result<(), Error> {
        self.fields.push((key, value.serialize(ValueSerializer)?));
        Ok(())
    }
}

impl ser::SerializeMap for Object {
    type Ok = Value;
    type Error = Error;

    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), Error> {
        let Value::String(key) = key.serialize(ValueSerializer)? else {
            return Err(ser::Error::custom("a key of JSON is a string"));
        };
        self.key = Some(key);
        Ok(())
    }

    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        let key = self
            .key
            .take()
            .expect("serde writes a key before its value");
        self.insert(key, value)
    }

    fn end(self) -> Result<Value, Error> {
        Ok(Value::Map(self.fields))
    }
}

impl ser::SerializeStruct for Object {
    type Ok = Value;
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.insert(key.to_string(), value)
    }

    fn end(self) -> Result<Value, Error> {
        Ok(Value::Map(self.fields))
    }
}

/// A variant with its fields: `{variant: fields}`.
pub(super) struct Variant<T>(&'static str, T);

impl ser::SerializeTupleVariant for Variant<List> {
    type Ok = Value;
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        self.1.push(value)
    }

    fn end(self) -> Result<Value, Error> {
        Ok(Value::Map(vec![(
            self.0.to_string(),
            Value::List(self.1.0),
        )]))
    }
}

impl ser::SerializeStructVariant for Variant<Object> {
    type Ok = Value;
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.1.insert(key.to_string(), value)
    }

    fn end(self) -> Result<Value, Error> {
        Ok(Value::Map(vec![(
            self.0.to_string(),
            Value::Map(self.1.fields),
        )]))
    }
}
