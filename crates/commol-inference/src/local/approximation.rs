//! The local approximations of a problem at a fit: which curvature they read,
//! and the Laplace approximation of observation models.

use commol_calibration::CalibrationProblem;
use commol_core::SimulationEngine;

use super::LocalError;
use super::curvature::{Curvature, sensitivity_error};
use super::gaussian::{Gaussian, laplace};
use crate::sensitivity::{Design, FiniteDifferences};

/// The curvature a local approximation reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurvatureKind {
    /// The expected curvature: `Jᵀ W J` and the Fisher information of the
    /// noise arguments ([`Curvature::gauss_newton`]).
    GaussNewton,
    /// The central differences of the objective ([`Curvature::hessian`]).
    Hessian,
}

impl CurvatureKind {
    /// The kind of a name: `"gauss_newton"` or `"hessian"`.
    ///
    /// # Errors
    ///
    /// Returns a message that lists the names when `name` is neither.
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "gauss_newton" => Ok(Self::GaussNewton),
            "hessian" => Ok(Self::Hessian),
            _ => Err(format!(
                "Unknown curvature '{name}'; the curvatures are 'gauss_newton' and 'hessian'"
            )),
        }
    }
}

/// The curvature of the objective of `problem` at `fit`; Gauss–Newton reads
/// finite differences of the observations.
///
/// # Errors
///
/// As [`Curvature::gauss_newton`] and [`Curvature::hessian`].
pub fn curvature_at<E>(
    problem: &CalibrationProblem<E>,
    fit: &[f64],
    kind: CurvatureKind,
) -> Result<Curvature, LocalError>
where
    E: SimulationEngine + Send + Sync,
{
    match kind {
        CurvatureKind::GaussNewton => {
            let source =
                FiniteDifferences::new(problem, Design::Observations).map_err(sensitivity_error)?;
            Curvature::gauss_newton(problem, &source, fit)
        }
        CurvatureKind::Hessian => Curvature::hessian(problem, fit),
    }
}

/// The Laplace approximation of the observation models of `problem` at
/// `fit`: the normal `N(û, H_u⁻¹)` on the unbounded scale, with the
/// curvature it inverts.
///
/// # Errors
///
/// [`LocalError::Refused`] when the problem has a loss, which has no noise
/// scale; otherwise as [`curvature_at`] and [`laplace`].
pub fn laplace_approximation<E>(
    problem: &CalibrationProblem<E>,
    fit: &[f64],
    kind: CurvatureKind,
) -> Result<(Curvature, Gaussian), LocalError>
where
    E: SimulationEngine + Send + Sync,
{
    if problem.discrepancy().likelihood().is_none() {
        return Err(LocalError::Refused(
            "Laplace(...) needs observation models: a loss has no noise scale, so its \
             curvature is no inverse variance. Give observation_models, or use \
             Sandwich(...), which needs no noise model"
                .to_string(),
        ));
    }
    let curvature = curvature_at(problem, fit, kind)?;
    let gaussian = laplace(&curvature)?;
    Ok((curvature, gaussian))
}
