//! The structure of a model: how its process is laid out, which changes
//! between kinds of model.
//!
//! A [`Model`](super::Model) holds one [`ModelStructure`], tagged in JSON by
//! `kind`. Each engine runs some kinds
//! ([`EngineType::kinds`](super::EngineType::kinds)).

use serde::{Deserialize, Serialize};

use super::aggregate::Aggregate;
use super::engine::ModelKind;
#[cfg(feature = "test-kind")]
use super::test_kind::TestStructure;

/// The structure of a model, one variant per model kind.
// A model holds one structure, so the size of the largest variant costs
// nothing; the test kind exists only in test builds.
#[cfg_attr(feature = "test-kind", allow(clippy::large_enum_variant))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelStructure {
    /// Amounts per compartment: the bins of the process expanded over its
    /// stratifications.
    Aggregate(Aggregate),
    /// The model kind of the tests: outputs that decay exponentially.
    #[cfg(feature = "test-kind")]
    Test(TestStructure),
}

impl ModelStructure {
    /// The kind of the structure.
    pub fn kind(&self) -> ModelKind {
        match self {
            Self::Aggregate(_) => ModelKind::Aggregate,
            #[cfg(feature = "test-kind")]
            Self::Test(_) => ModelKind::Test,
        }
    }
}
