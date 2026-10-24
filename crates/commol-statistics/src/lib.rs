//! The numerical helpers, log densities and weighted statistics of commol.
//!
//! What the calibration and the inference methods share, and no model: a
//! leaf crate, which depends on no crate of commol.
//!
//! - [`numeric`]: `log Σ exp`, `x ln y` and `x ln(1 + y)`, the special
//!   functions `ln Γ`, `ln B`, digamma and trigamma, and the log densities of
//!   the normal, the log-normal and the Student t, one copy of each;
//! - [`statistics`]: the one quantile of commol, and the weighted mean,
//!   covariance, summary and effective size of a sample.
//!
//! Every elementary function comes from `libm`, so a value is the same on
//! every platform.

#![warn(missing_docs)]

pub mod numeric;
pub mod statistics;
