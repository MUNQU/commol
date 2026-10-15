//! The model kind of the tests, compiled only with the `test-kind` feature.
//!
//! Each output starts at a value and decays exponentially at the rate of a
//! parameter: `x(t) = initial * exp(-rate * t)`. The kind exists to test that
//! the generic layers work with a kind that is not compartmental.

use serde::{Deserialize, Serialize};

/// One output of a test model.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecayOutput {
    /// Name of the output.
    pub id: String,
    /// Value at position 0.
    pub initial: f64,
    /// Id of the parameter that is the decay rate.
    pub rate: String,
}

/// The structure of a test model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestStructure {
    /// The outputs, in the order the engine reports them.
    pub outputs: Vec<DecayOutput>,
}
