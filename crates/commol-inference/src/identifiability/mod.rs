//! Which parameters the model and the data determine.
//!
//! | Module | Answers |
//! |---|---|
//! | [`structural`] | The numerical rank of the scaled sensitivities at random points, and the combinations a rank deficit leaves |
//! | [`collinearity`](mod@collinearity) | Brun's sensitivity measures and collinearity indices |
//! | [`sample`] | The contraction, correlations and modes of a parameter sample |
//! | [`starts`] | The parameters that differ between starts of equal loss |
//!
//! Every analysis reads derivatives only through a
//! [`SensitivitySource`](crate::sensitivity::SensitivitySource).

pub mod collinearity;
pub mod sample;
pub mod starts;
pub mod structural;

pub use collinearity::{Collinearity, Subset, collinearity, collinearity_index};
pub use sample::{Reference, SampleIdentifiability, sample_identifiability};
pub use starts::{StartsIdentifiability, starts_identifiability};
pub use structural::{
    Combination, PointRank, StructuralAnalysis, StructuralVerdict, structural_rank,
};

use crate::sensitivity::SensitivityError;

/// Whether a failure of the sensitivities is a point to skip.
fn skippable(error: &SensitivityError) -> bool {
    matches!(
        error,
        SensitivityError::Infeasible { .. } | SensitivityError::InfeasiblePoint(_)
    )
}
