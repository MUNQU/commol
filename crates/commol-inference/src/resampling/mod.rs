//! Uncertainty by fitting the model again to reweighted or resampled data,
//! with no noise model and no prior.
//!
//! | Module | Holds |
//! |---|---|
//! | [`blocks`] | [`BlockScheme`]: blocks of consecutive positions across series, and the automatic block length |
//! | [`residuals`] | [`Residuals`] of a fit on a [`ResidualScale`] |
//! | [`resamplers`] | The [`Resampler`] port and its methods: the Bayesian, residual and wild block bootstraps, and the parametric bootstrap of observation models |
//! | [`runner`] | [`ReplicateRunner`]: fits the data of each replicate, in parallel; [`bootstrap`] gathers a sample |
//! | [`jackknife`] | [`block_jackknife`]: one fit per block left out |
//! | [`noise`] | [`ResidualNoise`]: observation noise made of the residuals of a fit |
//!
//! A replicate changes the data only through
//! [`commol_calibration::CalibrationProblem::with_data`]; the engine, the
//! observation operator and the feasibility stay those of the problem.

pub mod blocks;
pub mod jackknife;
pub mod noise;
pub mod resamplers;
pub mod residuals;
pub mod runner;

pub use blocks::{
    AutomaticLength, BlockDrawing, BlockScheme, automatic_block_length, automatic_length,
};
pub use jackknife::{Jackknife, JackknifeStatistics, block_jackknife, jackknife_statistics};
pub use noise::ResidualNoise;
pub use resamplers::{
    BayesianBootstrap, Multipliers, ParametricResampler, Perturbation, Resampler,
    ResidualBootstrap, WildBootstrap,
};
pub use residuals::{ResidualBlocks, ResidualScale, Residuals};
pub use runner::{BootstrapRun, ReplicateOutcome, ReplicateRunner, bootstrap};
