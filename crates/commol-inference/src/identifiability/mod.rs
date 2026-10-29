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

pub use collinearity::{
    Collinearity, CollinearityVerdict, Subset, collinearity, collinearity_index,
};
pub use sample::{Reference, SampleIdentifiability, SampleVerdict, sample_identifiability};
pub use starts::{StartsIdentifiability, starts_identifiability};
pub use structural::{
    Combination, PointRank, StructuralAnalysis, StructuralVerdict, TighterRank, structural_rank,
};

use crate::sensitivity::SensitivityError;

/// What the data say about a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PracticalVerdict {
    /// The data determine it.
    Determined,
    /// The data leave it poorly determined.
    PoorlyDetermined,
}

impl PracticalVerdict {
    /// The verdict `PoorlyDetermined` when `poorly` holds, else `Determined`.
    fn of(poorly: bool) -> Self {
        if poorly {
            Self::PoorlyDetermined
        } else {
            Self::Determined
        }
    }

    /// The name of the verdict: `"determined"` or `"poorly_determined"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Determined => "determined",
            Self::PoorlyDetermined => "poorly_determined",
        }
    }
}

/// Whether a failure of the sensitivities is a point to skip.
fn skippable(error: &SensitivityError) -> bool {
    matches!(
        error,
        SensitivityError::Infeasible { .. } | SensitivityError::InfeasiblePoint(_)
    )
}
