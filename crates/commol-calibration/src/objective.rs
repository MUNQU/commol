//! The optimization objective: the argmin cost function of a calibration
//! problem.
//!
//! It is the only code that turns a [`Violation`] into a penalty. An
//! optimizer needs a number at every point; an inference method sees the
//! same violation as a point outside the support.

use std::cell::RefCell;

use argmin::core::{CostFunction, Error};
use commol_core::SimulationEngine;

use crate::calibration_problem::CalibrationProblem;
use crate::feasibility::{EvaluationError, Violation};
use crate::forward::Workspace;
use crate::types::CalibrationEvaluation;

thread_local! {
    /// The buffers of this thread's evaluations.
    static WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::default());
}

/// One scored point: the values the problem scored (clamped to their
/// bounds), the cost the optimizer sees, and whether it is the discrepancy
/// of a feasible point or a penalty.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    /// The values the problem scored.
    pub applied: Vec<f64>,
    /// The discrepancy, or the penalty of an infeasible point.
    pub cost: f64,
    /// Whether the point is feasible, so `cost` is its discrepancy.
    pub feasible: bool,
}

/// The cost function argmin minimizes: the discrepancy of a feasible point,
/// or a penalty above every discrepancy the data allow.
///
/// Values outside their bounds are clamped first, because optimizers may
/// explore outside them.
pub struct OptimizationObjective<'a, E: SimulationEngine> {
    problem: &'a CalibrationProblem<E>,
    base_penalty: f64,
}

impl<'a, E: SimulationEngine> OptimizationObjective<'a, E> {
    /// The objective of `problem`.
    pub fn new(problem: &'a CalibrationProblem<E>) -> Self {
        let max_observed = problem
            .observed_data()
            .iter()
            .map(|obs| obs.value)
            .fold(0.0f64, f64::max);
        let count = problem.observed_data().len() as f64;
        Self {
            problem,
            base_penalty: (max_observed * max_observed * count * 1000.0).max(1e10),
        }
    }

    /// The problem.
    pub fn problem(&self) -> &'a CalibrationProblem<E> {
        self.problem
    }

    /// The penalty of a violation: `base × (1 + amount)`, where `base` scales
    /// with the largest observed value and is at least `1e10`.
    pub fn penalty(&self, violation: &Violation) -> f64 {
        self.base_penalty * (1.0 + violation.amount())
    }

    /// Score `values`.
    ///
    /// # Errors
    ///
    /// Returns a message when the number of values is wrong, or the engine
    /// or a constraint cannot be evaluated.
    pub fn score(&self, values: &[f64]) -> Result<Score, Error> {
        if values.len() != self.problem.num_parameters() {
            return Err(Error::msg(format!(
                "Expected {} parameters, got {}",
                self.problem.num_parameters(),
                values.len()
            )));
        }
        let applied = self.problem.applied_parameter_values(values);
        let result = WORKSPACE
            .with(|workspace| self.problem.evaluate(&applied, &mut workspace.borrow_mut()));
        match result {
            Ok(cost) => Ok(Score {
                applied,
                cost,
                feasible: true,
            }),
            Err(EvaluationError::Violation(violation)) => Ok(Score {
                applied,
                cost: self.penalty(&violation),
                feasible: false,
            }),
            Err(EvaluationError::Failed(message)) => Err(Error::msg(message)),
        }
    }
}

impl<E: SimulationEngine> CostFunction for OptimizationObjective<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, values: &Self::Param) -> Result<Self::Output, Error> {
        self.score(values).map(|score| score.cost)
    }
}

/// An objective that keeps every feasible evaluation, for
/// [`crate::optimize_with_history`].
pub(crate) struct Recorded<'a, E: SimulationEngine> {
    pub(crate) objective: OptimizationObjective<'a, E>,
    pub(crate) evaluations: RefCell<Vec<CalibrationEvaluation>>,
}

impl<E: SimulationEngine> CostFunction for Recorded<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, values: &Self::Param) -> Result<Self::Output, Error> {
        let score = self.objective.score(values)?;
        if score.feasible {
            self.evaluations.borrow_mut().push(CalibrationEvaluation {
                parameters: score.applied,
                loss: score.cost,
            });
        }
        Ok(score.cost)
    }
}
