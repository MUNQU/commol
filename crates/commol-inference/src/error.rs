//! The one error of every method: [`MethodError`], whose every variant is of
//! one [`ErrorKind`].
//!
//! | Kind | When | Python |
//! |---|---|---|
//! | [`ErrorKind::Invalid`] | A setting, an input, the data or the model does not fit the method | `ValueError`; `InfeasibleStateError` for a draw that leaves the values its model allows |
//! | [`ErrorKind::Failed`] | A computation that could start failed: an engine run, an optimizer, every replicate, a sampler that did not reach its target | `RuntimeError` |
//!
//! A structured error that callers read stays a variant of its own
//! ([`MethodError::Prediction`], [`MethodError::Sensitivity`],
//! [`MethodError::NoFeasiblePoint`]), and [`MethodError::kind`] gives its
//! kind by a fixed rule.

use std::fmt;

use commol_calibration::EvaluationError;

use crate::predictive::PredictiveError;
use crate::sensitivity::SensitivityError;

/// The kind of a [`MethodError`]: what the caller did wrong, or what went
/// wrong in the computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// A setting, an input, the data or the model does not fit the method.
    Invalid,
    /// A computation that could start failed.
    Failed,
}

/// Why a method has no result.
#[derive(Debug, Clone, PartialEq)]
pub enum MethodError {
    /// A setting, an input, the data or the model does not fit the method;
    /// the message says why.
    Invalid(String),
    /// A computation that could start failed: an engine run, an optimizer,
    /// every replicate, or a sampler that did not reach its target; the
    /// message says why.
    Failed(String),
    /// The runs of a sample or of a design have no prediction. Its kind is
    /// that of [`PredictiveError`]: [`ErrorKind::Failed`] when a draw's run
    /// fails, [`ErrorKind::Invalid`] otherwise.
    Prediction(PredictiveError),
    /// The derivatives failed. Its kind is [`ErrorKind::Invalid`] for a
    /// stochastic engine, which has no derivatives, and
    /// [`ErrorKind::Failed`] otherwise.
    Sensitivity(SensitivityError),
    /// Every point of an analysis is infeasible; of kind
    /// [`ErrorKind::Invalid`].
    NoFeasiblePoint {
        /// How many points were tried.
        points: usize,
        /// Why each was skipped.
        reasons: Vec<String>,
    },
}

impl MethodError {
    /// The kind of the error, which sets the Python exception.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Invalid(_) | Self::NoFeasiblePoint { .. } => ErrorKind::Invalid,
            Self::Failed(_) => ErrorKind::Failed,
            Self::Prediction(PredictiveError::Draw {
                error: EvaluationError::Failed(_),
                ..
            }) => ErrorKind::Failed,
            Self::Prediction(_) => ErrorKind::Invalid,
            Self::Sensitivity(SensitivityError::Stochastic) => ErrorKind::Invalid,
            Self::Sensitivity(_) => ErrorKind::Failed,
        }
    }

    /// The error of a point that has no run or no value, after `context`:
    /// [`MethodError::Invalid`] when the point is not feasible,
    /// [`MethodError::Failed`] when the engine fails.
    pub fn of_point(context: &str, error: EvaluationError) -> Self {
        let message = format!("{context}: {error}");
        match error {
            EvaluationError::Violation(_) => Self::Invalid(message),
            EvaluationError::Failed(_) => Self::Failed(message),
        }
    }
}

impl From<PredictiveError> for MethodError {
    fn from(error: PredictiveError) -> Self {
        Self::Prediction(error)
    }
}

impl From<SensitivityError> for MethodError {
    fn from(error: SensitivityError) -> Self {
        Self::Sensitivity(error)
    }
}

impl fmt::Display for MethodError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) | Self::Failed(message) => f.write_str(message),
            Self::Prediction(error) => error.fmt(f),
            Self::Sensitivity(error) => error.fmt(f),
            Self::NoFeasiblePoint { points, reasons } => {
                let mut distinct: Vec<&str> = Vec::new();
                for reason in reasons {
                    if !distinct.contains(&reason.as_str()) {
                        distinct.push(reason);
                    }
                }
                write!(
                    f,
                    "Every one of the {points} points is infeasible: {}",
                    distinct.join("; ")
                )
            }
        }
    }
}

impl std::error::Error for MethodError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_structured_error_has_the_kind_of_its_rule() {
        let draw = |error| MethodError::Prediction(PredictiveError::Draw { draw: 0, error });
        assert_eq!(
            draw(EvaluationError::Failed("x".to_string())).kind(),
            ErrorKind::Failed
        );
        assert_eq!(
            draw(EvaluationError::Violation(
                commol_calibration::Violation::NonFinite
            ))
            .kind(),
            ErrorKind::Invalid
        );
        assert_eq!(
            MethodError::Prediction(PredictiveError::Mismatch("x".to_string())).kind(),
            ErrorKind::Invalid
        );
        assert_eq!(
            MethodError::from(SensitivityError::Stochastic).kind(),
            ErrorKind::Invalid
        );
        assert_eq!(
            MethodError::from(SensitivityError::InfeasiblePoint("x".to_string())).kind(),
            ErrorKind::Failed
        );
        let none = MethodError::NoFeasiblePoint {
            points: 3,
            reasons: vec!["a".to_string(), "b".to_string(), "a".to_string()],
        };
        assert_eq!(none.kind(), ErrorKind::Invalid);
        assert_eq!(
            none.to_string(),
            "Every one of the 3 points is infeasible: a; b"
        );
    }
}
