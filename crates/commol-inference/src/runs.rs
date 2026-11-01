//! The events of a loop of model runs: the points of a design, the sets of
//! a predictive, the draws of an importance sampler or of the prior of ABC.
//!
//! Each step of such a loop is one run of the model, so a loop sends one
//! [`RunEvent::Began`] with the number of points, then one
//! [`RunEvent::Ran`] per point, from the thread that ran it
//! ([`commol_core::Steps`]). A point that begins once the steps stop does
//! not run, and fails with [`commol_core::STOPPED`] ([`stopped`]).

use commol_calibration::EvaluationError;
use commol_core::{STOPPED, Steps};

/// An event of a loop of model runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunEvent {
    /// The loop begins with `total` points.
    Began {
        /// The number of points.
        total: usize,
    },
    /// Point `index` ran, or failed.
    Ran {
        /// The index of the point.
        index: usize,
    },
}

/// The receiver of the events of a loop of model runs.
pub type RunSteps<'s> = dyn Steps<RunEvent> + 's;

/// The error of a point that a stop kept from running.
pub fn stopped() -> EvaluationError {
    EvaluationError::Failed(STOPPED.to_string())
}
