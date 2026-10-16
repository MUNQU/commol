//! The target of a sampler: a log density over parameter space.
//!
//! A constraint, a bound, an infeasible start or an infeasible run is the
//! edge of the support: the log density there is `−∞`. A method never sees a
//! penalty ([`commol_calibration::OptimizationObjective`] is the only place
//! that makes one).

use commol_calibration::{CalibrationProblem, EvaluationError, Workspace};
use commol_core::SimulationEngine;

/// A log density over the parameters of a calibration problem, up to a
/// constant.
pub trait Target {
    /// Number of parameters.
    fn dimension(&self) -> usize;

    /// The log density at `values`: `−∞` outside the support.
    ///
    /// # Errors
    ///
    /// Returns a message when the engine or a constraint cannot be
    /// evaluated, a failure that no point avoids.
    fn log_density(&self, values: &[f64], workspace: &mut Workspace) -> Result<f64, String>;
}

/// The discrepancy of `problem` at `values`, or `None` outside the support:
/// outside the bounds, or a violation of the problem.
///
/// A target built on the discrepancy, such as a Gibbs posterior or a
/// likelihood, reads it through this function so that every method has the
/// same support.
///
/// # Errors
///
/// Returns a message when the engine or a constraint cannot be evaluated.
pub fn discrepancy_in_support<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    values: &[f64],
    workspace: &mut Workspace,
) -> Result<Option<f64>, String> {
    if problem.feasibility().check_bounds(values).is_err() {
        return Ok(None);
    }
    match problem.evaluate(values, workspace) {
        Ok(discrepancy) => Ok(Some(discrepancy)),
        Err(EvaluationError::Violation(_)) => Ok(None),
        Err(EvaluationError::Failed(message)) => Err(message),
    }
}
