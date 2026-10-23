//! The optimization objective: the argmin cost function of a calibration
//! problem, its loss or its negative log-likelihood.
//!
//! It is the only code that turns a [`Violation`] into a penalty. An
//! optimizer needs a number at every point; an inference method sees the
//! same violation as a point outside the support.

use std::borrow::Borrow;
use std::cell::RefCell;
use std::sync::{Mutex, PoisonError};

use argmin::core::{CostFunction, Error, SendAlias, SyncAlias};
use commol_core::{SimulationEngine, inner_levels_in_parallel};
use rayon::prelude::*;

use crate::calibration_problem::CalibrationProblem;
use crate::feasibility::{EvaluationError, Violation};
use crate::forward::Workspace;
use crate::types::CalibrationEvaluation;

thread_local! {
    /// The buffers of this thread's evaluations, while no evaluation of
    /// this thread holds them.
    static WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::default());
}

/// Run `job` with the buffers of this thread, taken out of the thread-local
/// while `job` runs and put back after it. A job that this thread runs in
/// the meantime, such as one that `rayon` steals, takes buffers of its own,
/// so no buffers are borrowed twice.
pub(crate) fn with_workspace<R>(job: impl FnOnce(&mut Workspace) -> R) -> R {
    let mut workspace = WORKSPACE.take();
    let result = job(&mut workspace);
    WORKSPACE.set(workspace);
    result
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
        Self {
            problem,
            base_penalty: problem.discrepancy().penalty_base(),
        }
    }

    /// The penalty of a violation: `base × (1 + amount)`, where `base`
    /// ([`crate::Discrepancy::penalty_base`]) is above the discrepancy of a
    /// poor fit and at least `1e10`.
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
        with_workspace(|workspace| self.score_in(values, workspace))
    }

    /// Score `values` with the buffers of `workspace`, which then holds the
    /// rows of the run of a feasible point. It makes no `rayon` call, so the
    /// thread that runs it steals no other job meanwhile.
    ///
    /// # Errors
    ///
    /// As [`Self::score`].
    pub fn score_in(&self, values: &[f64], workspace: &mut Workspace) -> Result<Score, Error> {
        if values.len() != self.problem.num_parameters() {
            return Err(Error::msg(format!(
                "Expected {} parameters, got {}",
                self.problem.num_parameters(),
                values.len()
            )));
        }
        let applied = self.problem.applied_parameter_values(values);
        match self.problem.evaluate(&applied, workspace) {
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

    /// Score each of `points`, in the order of the points: in parallel when
    /// the inner levels run in parallel ([`inner_levels_in_parallel`]), one
    /// after the other otherwise.
    ///
    /// # Errors
    ///
    /// The error of the first point, in the order of the points, that
    /// fails, whatever the order in which the threads score them.
    pub fn score_all<P>(&self, points: &[P]) -> Result<Vec<Score>, Error>
    where
        P: Borrow<Vec<f64>> + Sync,
        E: Sync,
    {
        if inner_levels_in_parallel() {
            let scores: Vec<Result<Score, Error>> = points
                .par_iter()
                .map(|point| self.score(point.borrow()))
                .collect();
            scores.into_iter().collect()
        } else {
            points
                .iter()
                .map(|point| self.score(point.borrow()))
                .collect()
        }
    }
}

impl<E: SimulationEngine + Sync> CostFunction for OptimizationObjective<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, values: &Self::Param) -> Result<Self::Output, Error> {
        self.score(values).map(|score| score.cost)
    }

    /// The costs of `params` by [`OptimizationObjective::score_all`].
    fn bulk_cost<P>(&self, params: &[P]) -> Result<Vec<Self::Output>, Error>
    where
        P: Borrow<Self::Param> + SyncAlias,
        Self::Output: SendAlias,
        Self: SyncAlias,
    {
        let scores = self.score_all(params)?;
        Ok(scores.into_iter().map(|score| score.cost).collect())
    }

    fn parallelize(&self) -> bool {
        inner_levels_in_parallel()
    }
}

/// An objective that keeps every feasible evaluation, for
/// [`crate::optimize_with_history`].
///
/// The points of a bulk evaluation are scored first, in parallel when the
/// inner levels run in parallel, and then recorded in the order of the
/// points, so the history does not depend on the order in which the
/// threads finish.
pub(crate) struct Recorded<'a, E: SimulationEngine> {
    pub(crate) objective: OptimizationObjective<'a, E>,
    pub(crate) evaluations: Mutex<Vec<CalibrationEvaluation>>,
}

impl<'a, E: SimulationEngine> Recorded<'a, E> {
    /// The objective of `problem`, with no evaluation yet.
    pub(crate) fn new(problem: &'a CalibrationProblem<E>) -> Self {
        Self {
            objective: OptimizationObjective::new(problem),
            evaluations: Mutex::new(Vec::new()),
        }
    }

    /// Keep the feasible scores of `scores`, in their order.
    fn record<'s>(&self, scores: impl IntoIterator<Item = &'s Score>) {
        let mut evaluations = self
            .evaluations
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        evaluations.extend(
            scores
                .into_iter()
                .filter(|score| score.feasible)
                .map(|score| CalibrationEvaluation {
                    parameters: score.applied.clone(),
                    loss: score.cost,
                }),
        );
    }

    /// Every evaluation kept, in the order of the evaluations.
    pub(crate) fn into_evaluations(self) -> Vec<CalibrationEvaluation> {
        self.evaluations
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl<E: SimulationEngine + Sync> CostFunction for Recorded<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, values: &Self::Param) -> Result<Self::Output, Error> {
        let score = self.objective.score(values)?;
        self.record([&score]);
        Ok(score.cost)
    }

    /// The costs of `params` by [`OptimizationObjective::score_all`], with
    /// their feasible scores recorded in the order of `params`.
    fn bulk_cost<P>(&self, params: &[P]) -> Result<Vec<Self::Output>, Error>
    where
        P: Borrow<Self::Param> + SyncAlias,
        Self::Output: SendAlias,
        Self: SyncAlias,
    {
        let scores = self.objective.score_all(params)?;
        self.record(&scores);
        Ok(scores.into_iter().map(|score| score.cost).collect())
    }

    fn parallelize(&self) -> bool {
        inner_levels_in_parallel()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CalibrationParameter, LossConfig, ObservedDataPoint};
    use commol_core::Model;
    use commol_difference::DifferenceEquations;

    /// A → B at the proportional rate `k`, from A = 1000, with B observed at
    /// 2.
    fn problem() -> CalibrationProblem<DifferenceEquations> {
        let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
            "engine":{"type":"DifferenceEquations"},
            "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
            "accumulators":[],"stratifications":[],
            "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k",
            "stratified_rates":null,"per_compartment":null}]},
            "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#;
        let model: Model = serde_json::from_str(json).expect("valid model JSON");
        CalibrationProblem::new(
            DifferenceEquations::try_from_model(&model).unwrap(),
            vec![ObservedDataPoint::new(2.0, "B".to_string(), 190.0)],
            vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
            vec![],
            LossConfig::SumSquaredError,
        )
        .unwrap()
    }

    #[test]
    fn a_score_holds_no_borrow_of_the_workspace() {
        let problem = problem();
        let objective = OptimizationObjective::new(&problem);
        let alone = objective.score(&[0.2]).unwrap();
        let (outer, inner) = with_workspace(|outer| {
            // While a score of this thread holds its buffers, the
            // thread-local is free, and a job that the thread steals scores
            // another point with buffers of its own.
            assert!(WORKSPACE.with(|workspace| workspace.try_borrow_mut().is_ok()));
            let outer_score = objective.score_in(&[0.1], outer).unwrap();
            let inner = objective.score(&[0.2]).unwrap();
            (outer_score, inner)
        });
        assert_eq!(inner, alone);
        // B at 2 is 1000 (1 − 0.9²) = 190 at k = 0.1, the observed value.
        assert!(outer.cost.abs() < 1e-9, "{outer:?}");
        assert!(outer.feasible);
        // The buffers come back to the thread-local after a score.
        assert!(WORKSPACE.with(|workspace| !workspace.borrow().rows.is_empty()));
    }
}
