//! # Commol Inference
//!
//! Uncertainty, identifiability and sensitivity methods on top of the forward
//! model of `commol-calibration`, and what every method returns or reads.
//!
//! | Module | Holds |
//! |---|---|
//! | [`error`] | [`MethodError`]: the one error of every method, [`ErrorKind::Invalid`] or [`ErrorKind::Failed`] |
//! | [`sample`] | [`ParameterSample`]: weighted draws of the parameters |
//! | [`target`] | [`Target`]: a log density, `−∞` outside the support |
//! | [`predictive`] | [`Predictive`]: the run of every draw, its bands, and every draw's value of observations |
//! | [`scores`] | Coverage, interval scores, CRPS and PIT of a predictive |
//! | [`identifiability`] | Structural rank, collinearity, and the identifiability of a sample and of starts |
//! | [`local`] | Local approximations around a fit: the curvature on the unbounded scale, the Laplace approximation, the delta method, the sandwich and the Laplace mixture |
//! | [`importance`] | Importance sampling of a target from a proposal on the unbounded scale |
//! | [`psis`] | Pareto-smoothed importance sampling: the smoothed weights and `k̂` |
//! | [`sensitivity`] | The port of derivatives, [`sensitivity::SensitivitySource`], and finite differences |
//! | [`resampling`] | Bootstraps and the jackknife: fits to reweighted or resampled data, and residual noise |
//! | [`distribution`] | [`Distribution`]: a family of distributions truncated to the bounds of a parameter |
//! | [`priors`] | [`Priors`]: one distribution per parameter |
//! | [`smc`] | Sequential Monte Carlo: the sampler, its kernels, the Gibbs posterior and ABC |
//! | [`transform`] | The unbounded scale of the moves: logit, log or identity by the bounds |
//! | [`conformal`] | Conformal prediction intervals: split, rolling, jackknife+, EnbPI and adaptive |
//! | [`design`] | [`design::InputSpace`]: independent distributions or a sample, and the designs on it |
//! | [`quantity`] | [`quantity::Quantity`]: the scalars a method reads from each run |
//! | [`batch`] | [`batch::BatchRunner`]: every point of a design through a forward model, reduced to its quantities |
//! | [`global_sensitivity`] | The estimators of global sensitivity: PRCC, Morris, Sobol and polynomial chaos |
//! | [`profiles`] | Profile likelihood of parameters and predictions: the tracer and the thresholds |
//! | [`ranges`] | The range rule of each setting of a method, a prior and a quantity |
//!
//! Rules: this crate depends on `commol-calibration`, never the reverse; a
//! method touches an engine only through a
//! [`commol_calibration::ForwardModel`] (a problem's, or one with no
//! observations), reads observations and quantities only through a
//! [`commol_calibration::ObservationOperator`], draws every random number
//! with the generator and the samplers of [`commol_random`], from a stream of
//! its registry ([`commol_random::Stream`]); every public function of a method returns a [`MethodError`].

#![warn(missing_docs)]

pub mod batch;
pub mod conformal;
pub mod design;
pub mod distribution;
pub mod error;
pub mod global_sensitivity;
pub mod identifiability;
pub mod importance;
pub mod local;
pub mod predictive;
pub mod priors;
pub mod profiles;
pub mod psis;
pub mod quantity;
pub mod ranges;
pub mod resampling;
pub mod sample;
pub mod scores;
pub mod sensitivity;
pub mod settings;
pub mod smc;
mod special;
pub mod target;
pub mod transform;

pub use distribution::{Distribution, Family};
pub use error::{ErrorKind, MethodError};
pub use predictive::{
    Band, DistributionNoise, NanRun, NoNoise, ObservationDraws, ObservationNoise, Predictive,
    PredictiveError, SeriesBand, draw_seeds, observation_draws, share_among_replicates,
};
pub use priors::Priors;
pub use sample::ParameterSample;
pub use target::{Target, discrepancy_in_support};
