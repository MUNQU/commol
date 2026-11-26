//! The numerical helpers, log densities, weighted statistics and linear
//! algebra of commol.
//!
//! What the model compiler, the calibration and the inference methods use,
//! and no model: a leaf crate, which depends on no crate of commol.
//!
//! - [`dependence`]: the autocovariances of a series with gaps, the robust
//!   portmanteau and the integrated autocorrelation time of Geyer;
//! - [`linear_algebra`]: the spectral radius of a square matrix, which the
//!   model compiler reads, and the inverse of a symmetric matrix with its rule
//!   of what is singular, which the inference methods read;
//! - [`numeric`]: `log Σ exp`, the compensated sum, `x ln y` and `x ln(1 + y)`,
//!   the special functions `ln Γ`, `ln B`, digamma and trigamma, and the log
//!   densities of the normal, the log-normal and the Student t, one copy of
//!   each;
//! - [`special`]: the normal distribution function and its quantile, the
//!   regularized incomplete gamma and beta functions, and the quantiles of
//!   the Student t and of the F of one degree of freedom;
//! - [`statistics`]: the one quantile of commol, and the weighted mean,
//!   covariance, summary and effective size of a sample.
//!
//! Every elementary function comes from `libm`, so a value is the same on
//! every platform; the eigenvalues of [`linear_algebra`] come from
//! `nalgebra`.

#![warn(missing_docs)]

pub mod dependence;
pub mod linear_algebra;
pub mod numeric;
pub mod special;
pub mod statistics;
