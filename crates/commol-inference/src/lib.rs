//! # Commol Inference
//!
//! Uncertainty and identifiability methods on top of a calibration problem
//! of `commol-calibration`, and what every method returns or reads.
//!
//! | Module | Holds |
//! |---|---|
//! | [`sample`] | [`ParameterSample`]: weighted draws of the parameters |
//! | [`statistics`] | The one quantile convention, weighted moments and effective size |
//! | [`target`] | [`Target`]: a log density, `−∞` outside the support |
//! | [`predictive`] | [`Predictive`]: the run of every draw, its bands, and every draw's value of observations |
//! | [`scores`] | Coverage, interval scores, CRPS and PIT of a predictive |
//! | [`identifiability`] | Structural rank, collinearity, and the identifiability of a sample and of starts |
//! | [`sensitivity`] | The port of derivatives, [`sensitivity::SensitivitySource`], and finite differences |
//! | [`resampling`] | Bootstraps and the jackknife: fits to reweighted or resampled data, and residual noise |
//! | [`distribution`] | [`Distribution`]: a family of distributions truncated to the bounds of a parameter |
//! | [`priors`] | [`Priors`]: one distribution per parameter |
//! | [`smc`] | Sequential Monte Carlo: the sampler, its kernels, the Gibbs posterior and ABC |
//! | [`transform`] | The unbounded scale of the moves: logit, log or identity by the bounds |
//!
//! Rules: this crate depends on `commol-calibration`, never the reverse; a
//! method touches an engine only through the problem's
//! [`commol_calibration::ForwardModel`], reads observations only through its
//! [`commol_calibration::ObservationOperator`], and draws every random number
//! from a [`commol_random::Generator`].

#![warn(missing_docs)]

pub mod distribution;
pub mod identifiability;
pub mod predictive;
pub mod priors;
pub mod resampling;
pub mod sample;
pub mod scores;
pub mod sensitivity;
pub mod smc;
mod special;
pub mod statistics;
pub mod target;
pub mod transform;

pub use distribution::{Distribution, Family};
pub use predictive::{
    Band, NoNoise, ObservationDraws, ObservationNoise, Predictive, PredictiveError, SeriesBand,
    observation_draws,
};
pub use priors::Priors;
pub use sample::ParameterSample;
pub use target::{Target, discrepancy_in_support};
