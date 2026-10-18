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

use std::fmt;

use crate::sensitivity::SensitivityError;

/// Why an analysis has no result.
#[derive(Debug, Clone, PartialEq)]
pub enum IdentifiabilityError {
    /// Every point of the analysis is infeasible.
    NoFeasiblePoint {
        /// How many points were tried.
        points: usize,
        /// Why each was skipped.
        reasons: Vec<String>,
    },
    /// The sensitivities failed for a reason no point avoids.
    Sensitivity(SensitivityError),
    /// A setting or an input is not valid.
    Invalid(String),
}

impl fmt::Display for IdentifiabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoFeasiblePoint { points, reasons } => {
                let mut distinct: Vec<&String> = Vec::new();
                for reason in reasons {
                    if !distinct.contains(&reason) {
                        distinct.push(reason);
                    }
                }
                let distinct: Vec<&str> = distinct.iter().map(|reason| reason.as_str()).collect();
                write!(
                    f,
                    "Every one of the {points} points is infeasible: {}",
                    distinct.join("; ")
                )
            }
            Self::Sensitivity(error) => write!(f, "{error}"),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for IdentifiabilityError {}

/// Whether a failure of the sensitivities is a point to skip.
fn skippable(error: &SensitivityError) -> bool {
    matches!(
        error,
        SensitivityError::Infeasible { .. } | SensitivityError::InfeasiblePoint(_)
    )
}
