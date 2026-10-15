//! The structure of a model: the part that depends on its kind.
//!
//! A [`Model`](super::Model) holds one [`ModelStructure`], tagged in JSON by
//! `kind`. The engine a model names runs one kind
//! ([`EngineType::kind`](super::EngineType::kind)).

use serde::{Deserialize, Serialize};

use super::compartmental::CompartmentalStructure;
use super::engine::ModelKind;
#[cfg(feature = "test-kind")]
use super::test_kind::TestStructure;

/// The structure of a model, one variant per model kind.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelStructure {
    /// Bins expanded over stratifications, joined by transitions.
    Compartmental(CompartmentalStructure),
    /// The model kind of the tests: outputs that decay exponentially.
    #[cfg(feature = "test-kind")]
    Test(TestStructure),
}

impl ModelStructure {
    /// The kind of the structure.
    pub fn kind(&self) -> ModelKind {
        match self {
            Self::Compartmental(_) => ModelKind::Compartmental,
            #[cfg(feature = "test-kind")]
            Self::Test(_) => ModelKind::Test,
        }
    }
}
