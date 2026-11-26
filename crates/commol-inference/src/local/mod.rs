//! Local approximations around a fit: the curvature of the objective on the
//! unbounded scale, the Laplace approximation, the delta method, the
//! sandwich and the Laplace mixture.
//!
//! | Module | Holds |
//! |---|---|
//! | [`approximation`] | [`CurvatureKind`], [`curvature_at`] and [`laplace_approximation`]: the approximations of a problem at a fit |
//! | [`band`] | [`DeltaBand`]: the band of every output by the delta method, [`delta_band`] |
//! | [`curvature`] | [`Curvature`]: the expected curvature (Gauss–Newton) of the objective at a fit, or its Hessian, scaled by a unit of each parameter |
//! | [`inverse`] | [`InverseCurvature`]: the inverse of a curvature and its one rule of what is singular |
//! | [`mixture`] | [`laplace_mixture`]: a Laplace approximation at each optimum of the starts, the proposal of an importance sampler |
//! | [`sandwich`](mod@sandwich) | [`sandwich()`]: the covariance of the influences of the blocks of a fit, [`sandwich_approximation`] |
//! | [`shares`] | [`BlockShares`]: the shares of the blocks in the curvature, which give the factor and the degrees of freedom of a sandwich |
//! | [`gaussian`] | [`Gaussian`]: a normal or grouped Student-t distribution on the unbounded scale, its draws mapped back; [`laplace`] |
//!
//! Every approximation lives on the unbounded scale of
//! [`crate::transform::Transform`], centred on the fit mapped to that scale:
//! its curvature is `H_u = D · H_θ · D` with `D = diag(dθ/du)` at the fit. A
//! draw maps back inside the bounds.

pub mod approximation;
pub mod band;
pub mod curvature;
pub mod gaussian;
pub mod inverse;
pub mod mixture;
pub mod sandwich;
pub mod shares;

pub use approximation::{CurvatureKind, curvature_at, laplace_approximation};
pub use band::{DeltaBand, delta_band};
pub use curvature::Curvature;
pub use gaussian::{Gaussian, laplace};
pub use inverse::InverseCurvature;
pub use mixture::{Component, MixtureRun, laplace_mixture};
pub use sandwich::{SandwichRun, sandwich, sandwich_approximation, scaled_block_scores};
pub use shares::BlockShares;
