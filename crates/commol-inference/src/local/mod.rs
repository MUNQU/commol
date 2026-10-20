//! Local approximations around a fit: the curvature of the objective on the
//! unbounded scale, the Laplace approximation and the delta method.
//!
//! | Module | Holds |
//! |---|---|
//! | [`approximation`] | [`CurvatureKind`], [`curvature_at`] and [`laplace_approximation`]: the approximations of a problem at a fit |
//! | [`band`] | [`DeltaBand`]: the band of every output by the delta method, [`delta_band`] |
//! | [`curvature`] | [`Curvature`]: the expected curvature (Gauss–Newton) of the objective at a fit, on the unbounded scale of [`crate::transform`] |
//! | [`gaussian`] | [`Gaussian`]: a normal or Student-t distribution on the unbounded scale, its draws mapped back; [`laplace`] |
//!
//! Every matrix lives on the unbounded scale of [`crate::transform::Transform`]:
//! `H_u = D · H_θ · D` with `D = diag(dθ/du)` at the fit, centred on the fit
//! mapped to that scale. A draw maps back inside the bounds.

pub mod approximation;
pub mod band;
pub mod curvature;
pub mod gaussian;

use std::fmt;

pub use approximation::{CurvatureKind, curvature_at, laplace_approximation};
pub use band::{DeltaBand, delta_band};
pub use curvature::Curvature;
pub use gaussian::{Gaussian, laplace};

/// Why a local approximation could not be made.
#[derive(Debug, Clone, PartialEq)]
pub enum LocalError {
    /// The method does not apply to this problem or fit, in words.
    Refused(String),
    /// A run or a derivative failed.
    Failed(String),
}

impl fmt::Display for LocalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(message) | Self::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for LocalError {}
