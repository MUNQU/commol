//! The settings of each method as the Python API writes them: the JSON of
//! the `model_dump()` of its pydantic class, read with
//! [`commol_core::json::read`].
//!
//! Each type here has the fields of its Python class, with the same names,
//! so the value a reader reads writes back to the same JSON, and its range
//! rules under the name of that class (`RANGES`). A field that holds a
//! result of Python with behavior in the engine, such as the sample of
//! `SplitConformal`, is not part of the JSON: it crosses as its handle, and
//! the type has no field for it.
//!
//! | Module | Python classes |
//! |---|---|
//! | [`resampling`] | the bootstraps, `BlockJackknife`, the observation noises |
//! | [`smc`] | `GibbsPosterior`, `BayesianPosterior`, `ABCSMC`, `ABCRejection`, the calibrations of `η`, the distances |
//! | [`local`] | `Laplace`, `Sandwich`, `LaplaceMixture`, its targets |
//! | [`conformal`] | `SplitConformal`, `RollingConformal`, `JackknifePlus`, `EnbPI`, `AdaptiveConformal` |
//! | [`profiles`] | `Threshold`, `PredictionSpec`, `ProfileLikelihood`, `PredictionProfile` |
//! | [`identifiability`] | `StructuralRank`, `Collinearity`, `SampleIdentifiability`, `StartsIdentifiability` |
//! | [`sensitivity`] | `InputSpace`, `Propagation`, `PRCC`, `Morris`, `Sobol`, `PolynomialChaos` |
//!
//! [`PriorsSettings`] reads `Priors`, and [`BlockLength`] the union
//! `int | "auto"` of a block length.

pub mod conformal;
pub mod identifiability;
pub mod local;
pub mod profiles;
pub mod resampling;
pub mod sensitivity;
pub mod smc;

use commol_core::json::{self, Pairs, Value};
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};

use crate::distribution::Family;
use crate::priors::Priors;

/// The priors of the parameters, as the Python class `Priors` writes them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PriorsSettings {
    /// The prior of each parameter by name, or None for uniform on the
    /// bounds of every parameter.
    pub distributions: Option<Pairs<Family>>,
}

impl PriorsSettings {
    /// The priors of the parameters `names` with `bounds`.
    ///
    /// # Errors
    ///
    /// As [`Priors::new`] and [`Priors::uniform_on_bounds`].
    pub fn priors(&self, names: &[String], bounds: &[(f64, f64)]) -> Result<Priors, String> {
        match &self.distributions {
            None => Priors::uniform_on_bounds(names, bounds),
            Some(Pairs(families)) => Priors::new(names, bounds, families),
        }
    }
}

/// The positions per block of a method: a number, or `"auto"` for the
/// length the engine estimates from the residuals of the fit (None).
///
/// The range rule of the number is the rule `block_length` of the method
/// that holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLength(pub Option<usize>);

impl Serialize for BlockLength {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Some(length) => serializer.serialize_u64(length as u64),
            None => serializer.serialize_str("auto"),
        }
    }
}

impl<'de> Deserialize<'de> for BlockLength {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::String(text) if text == "auto" => Ok(Self(None)),
            value @ Value::Int(_) => json::from_value(value).map(|length| Self(Some(length))),
            _ => Err(de::Error::custom(
                "block_length is a whole number of at least 1 or \"auto\"",
            )),
        }
    }
}
