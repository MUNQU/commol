//! Subproblems: the calibration problem with one quantity held fixed while
//! the problem's own optimizer fits the others.
//!
//! A parameter is held by leaving it out of the values the optimizer
//! searches: the optimizer sees one dimension less, and every point it
//! scores has the fixed value in its place. A prediction `g(θ) = c` is held
//! as an equality constraint by the augmented Lagrangian of the argmin fork,
//! whose inner solver is the problem's optimizer
//! ([`crate::optimize_subproblem`]).

use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;

use argmin::core::{CostFunction, Error, Operator, SendAlias, SyncAlias};
use commol_core::{SimulationEngine, inner_levels_in_parallel};

use crate::calibration_problem::CalibrationProblem;
use crate::forward::{Workspace, replicate_mean};
use crate::objective::{OptimizationObjective, with_workspace};
use crate::observation::ObservationOperator;
use crate::types::{CalibrationResult, ObservedDataPoint};

/// How the augmented Lagrangian holds a fixed prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintSettings {
    /// The largest `|g(θ) − c|` a converged run leaves.
    pub tolerance: f64,
    /// The largest number of outer iterations; each runs the optimizer once.
    pub max_outer_iterations: u64,
    /// The initial multiplier `λ`, such as that of a neighbouring point;
    /// `None` starts from 0.
    pub multiplier: Option<f64>,
    /// The initial penalty `μ`; `None` takes the scaled default of Algencan,
    /// `10·max{1, |f(θ₀)| / max{1, ½ (g(θ₀) − c)²}}`.
    pub penalty: Option<f64>,
}

impl ConstraintSettings {
    /// The settings of a prediction held at `value`: a tolerance of
    /// `1e-8 · max(1, |value|)`, 50 outer iterations, no multiplier and the
    /// default penalty.
    pub fn for_value(value: f64) -> Self {
        Self {
            tolerance: 1e-8 * value.abs().max(1.0),
            max_outer_iterations: 50,
            multiplier: None,
            penalty: None,
        }
    }

    fn check(&self) -> Result<(), String> {
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(format!(
                "The tolerance of a held prediction is finite and positive, got {}",
                self.tolerance
            ));
        }
        if self.max_outer_iterations == 0 {
            return Err("A held prediction needs at least one outer iteration".to_string());
        }
        if let Some(penalty) = self.penalty.filter(|p| !(p.is_finite() && *p > 0.0)) {
            return Err(format!(
                "The penalty of a held prediction is finite and positive, got {penalty}"
            ));
        }
        if let Some(multiplier) = self.multiplier.filter(|m| !m.is_finite()) {
            return Err(format!(
                "The multiplier of a held prediction is finite, got {multiplier}"
            ));
        }
        Ok(())
    }
}

/// A prediction held at a value: the problem observed at the prediction too,
/// and the operator that reads the prediction from its runs.
#[derive(Debug, Clone)]
pub(crate) struct HeldPrediction<E: SimulationEngine> {
    pub(crate) observing: CalibrationProblem<E>,
    pub(crate) operator: ObservationOperator,
    pub(crate) value: f64,
    pub(crate) settings: ConstraintSettings,
}

/// What a subproblem holds fixed.
#[derive(Debug, Clone)]
pub(crate) enum Fixed<E: SimulationEngine> {
    /// The parameter at `index` at `value`.
    Parameter { index: usize, value: f64 },
    /// A prediction, as an equality constraint.
    Prediction(Box<HeldPrediction<E>>),
}

/// A calibration problem with one quantity held fixed.
///
/// [`crate::optimize_subproblem`] fits the other values with the problem's
/// own optimizer.
#[derive(Debug, Clone)]
pub struct Subproblem<'a, E: SimulationEngine> {
    pub(crate) problem: &'a CalibrationProblem<E>,
    pub(crate) fixed: Fixed<E>,
}

impl<'a, E: SimulationEngine> Subproblem<'a, E> {
    /// The problem with the parameter at `index` held at `value`.
    ///
    /// # Errors
    ///
    /// Returns a message when `index` is not a parameter of the problem or
    /// `value` is outside its bounds.
    pub fn fix_parameter(
        problem: &'a CalibrationProblem<E>,
        index: usize,
        value: f64,
    ) -> Result<Self, String> {
        let parameter = problem.parameters().get(index).ok_or_else(|| {
            format!(
                "Parameter {index} does not exist; the problem has {}",
                problem.num_parameters()
            )
        })?;
        if !(parameter.min_bound..=parameter.max_bound).contains(&value) {
            return Err(format!(
                "'{}' cannot be held at {value}, outside its bounds [{}, {}]",
                parameter.id, parameter.min_bound, parameter.max_bound
            ));
        }
        Ok(Self {
            problem,
            fixed: Fixed::Parameter { index, value },
        })
    }

    /// The problem with the prediction of `spec` held at `value`.
    ///
    /// `spec` is read as an observation is (its outputs summed, its window,
    /// its factor), through the [`ObservationOperator`]; its value is not
    /// read. With replicates of a stochastic engine, the prediction is the
    /// mean of the replicates. The settings are
    /// [`ConstraintSettings::for_value`]; [`Self::with_settings`] replaces
    /// them.
    ///
    /// # Errors
    ///
    /// Returns a message when `value` is not finite, the engine rejects the
    /// position of `spec`, or `spec` names an unknown output or factor.
    pub fn fix_prediction(
        problem: &'a CalibrationProblem<E>,
        spec: &ObservedDataPoint,
        value: f64,
    ) -> Result<Self, String> {
        if !value.is_finite() {
            return Err(format!(
                "A prediction is held at a finite value, got {value}"
            ));
        }
        let (observing, operator) = problem.observing(std::slice::from_ref(spec))?;
        Ok(Self {
            problem,
            fixed: Fixed::Prediction(Box::new(HeldPrediction {
                observing,
                operator,
                value,
                settings: ConstraintSettings::for_value(value),
            })),
        })
    }

    /// The same subproblem with other settings of the held prediction.
    ///
    /// # Errors
    ///
    /// Returns a message when the subproblem holds a parameter, or a setting
    /// is not valid: a tolerance or a penalty that is not finite and
    /// positive, no outer iteration, or a multiplier that is not finite.
    pub fn with_settings(mut self, settings: ConstraintSettings) -> Result<Self, String> {
        settings.check()?;
        match &mut self.fixed {
            Fixed::Prediction(held) => {
                held.settings = settings;
                Ok(self)
            }
            Fixed::Parameter { .. } => {
                Err("Constraint settings apply to a held prediction, not a parameter".to_string())
            }
        }
    }
}

/// The prediction of `spec` at `values`, which are inside their bounds: the
/// value [`Subproblem::fix_prediction`] holds, read as an observation is;
/// the mean of the replicates of a stochastic engine.
///
/// # Errors
///
/// Returns a message when `spec` is not valid for the problem, or the values
/// are not feasible or the run fails.
pub fn predict<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    spec: &ObservedDataPoint,
    values: &[f64],
) -> Result<f64, String> {
    let (observing, operator) = problem.observing(std::slice::from_ref(spec))?;
    let mut workspace = Workspace::default();
    observing
        .evaluate(&observing.applied_parameter_values(values), &mut workspace)
        .map_err(|error| error.to_string())?;
    Ok(mean_prediction(&operator, &workspace, values))
}

/// The mean over the replicates in `workspace` of the one observation of
/// `operator`.
fn mean_prediction(operator: &ObservationOperator, workspace: &Workspace, values: &[f64]) -> f64 {
    let replicates: Vec<[f64; 1]> = workspace
        .rows
        .iter()
        .map(|rows| {
            let mut out = [0.0];
            operator.apply(rows, values, &mut out);
            out
        })
        .collect();
    replicate_mean(replicates.iter().map(|out| out.as_slice()), 1)[0]
}

/// The fit of a subproblem.
#[derive(Debug, Clone)]
pub struct SubproblemResult {
    /// The fit, with every parameter in problem order (the fixed one at its
    /// value); `final_loss` is the objective at the best values, the
    /// discrepancy of a feasible point. For a held prediction, `converged`
    /// says that the constraint holds within its tolerance and the last
    /// run of the optimizer converged.
    pub fit: CalibrationResult,
    /// The prediction at the best values, for a held prediction.
    pub prediction: Option<f64>,
    /// The final multiplier `λ` of a held prediction: minus the slope of
    /// the profile of the objective at the held value.
    pub multiplier: Option<f64>,
    /// The final penalty `μ` of a held prediction.
    pub penalty: Option<f64>,
    /// The number of evaluations of the objective; each runs the model once
    /// per replicate.
    pub evaluations: usize,
}

/// The objective of the free values: each point gets the fixed value at
/// `index`.
pub(crate) struct FixedParameter<'a, E: SimulationEngine> {
    pub(crate) objective: OptimizationObjective<'a, E>,
    pub(crate) index: usize,
    pub(crate) value: f64,
    pub(crate) evaluations: AtomicUsize,
}

impl<E: SimulationEngine> FixedParameter<'_, E> {
    /// Every value, the fixed one inserted at its index.
    pub(crate) fn full(&self, free: &[f64]) -> Vec<f64> {
        let mut values = Vec::with_capacity(free.len() + 1);
        values.extend_from_slice(&free[..self.index]);
        values.push(self.value);
        values.extend_from_slice(&free[self.index..]);
        values
    }
}

impl<E: SimulationEngine + Sync> CostFunction for FixedParameter<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, free: &Self::Param) -> Result<Self::Output, Error> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        self.objective.cost(&self.full(free))
    }

    /// The costs of `params` by [`OptimizationObjective::score_all`].
    fn bulk_cost<P>(&self, params: &[P]) -> Result<Vec<Self::Output>, Error>
    where
        P: Borrow<Self::Param> + SyncAlias,
        Self::Output: SendAlias,
        Self: SyncAlias,
    {
        self.evaluations.fetch_add(params.len(), Ordering::Relaxed);
        let points: Vec<Vec<f64>> = params.iter().map(|free| self.full(free.borrow())).collect();
        let scores = self.objective.score_all(&points)?;
        Ok(scores.into_iter().map(|score| score.cost).collect())
    }

    fn parallelize(&self) -> bool {
        inner_levels_in_parallel()
    }
}

/// The last point that each thread scored and its prediction; `None` when
/// the point is not feasible.
type LastPoints = HashMap<ThreadId, (Vec<f64>, Option<f64>)>;

/// The objective and the constraint `g(θ) − c` of a held prediction, both
/// from one run: the constraint of the point that a thread scored last is
/// kept, so the augmented Lagrangian, which asks for both at each point on
/// one thread, runs the model once. A swarm scores its particles on several
/// threads at once when the inner levels run in parallel.
pub(crate) struct PredictionCost<'a, E: SimulationEngine> {
    objective: OptimizationObjective<'a, E>,
    operator: &'a ObservationOperator,
    value: f64,
    last: Mutex<LastPoints>,
    pub(crate) evaluations: AtomicUsize,
}

impl<'a, E: SimulationEngine> PredictionCost<'a, E> {
    pub(crate) fn new(held: &'a HeldPrediction<E>) -> Self {
        Self {
            objective: OptimizationObjective::new(&held.observing),
            operator: &held.operator,
            value: held.value,
            last: Mutex::default(),
            evaluations: AtomicUsize::new(0),
        }
    }

    /// The objective and the prediction at `values`; no prediction at a
    /// point that is not feasible.
    pub(crate) fn evaluate(&self, values: &[f64]) -> Result<(f64, Option<f64>), Error> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        let (cost, prediction) = with_workspace(|workspace| {
            let score = self.objective.score_in(values, workspace)?;
            let prediction = score
                .feasible
                .then(|| mean_prediction(self.operator, workspace, &score.applied));
            Ok::<_, Error>((score.cost, prediction))
        })?;
        self.last_points()
            .insert(std::thread::current().id(), (values.to_vec(), prediction));
        Ok((cost, prediction))
    }

    /// The last point of each thread and its prediction.
    pub(crate) fn last_points(&self) -> MutexGuard<'_, LastPoints> {
        self.last.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<E: SimulationEngine + Sync> CostFunction for PredictionCost<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, values: &Self::Param) -> Result<Self::Output, Error> {
        self.evaluate(values).map(|(cost, _)| cost)
    }

    fn parallelize(&self) -> bool {
        inner_levels_in_parallel()
    }
}

impl<E: SimulationEngine + Sync> Operator for PredictionCost<'_, E> {
    type Param = Vec<f64>;
    type Output = Vec<f64>;

    /// `g(θ) − c`, or 0 at a point that is not feasible, whose cost is
    /// already the penalty of the objective.
    fn apply(&self, values: &Self::Param) -> Result<Self::Output, Error> {
        let kept = match self.last_points().get(&std::thread::current().id()) {
            Some((point, prediction)) if point == values => Some(*prediction),
            _ => None,
        };
        let prediction = match kept {
            Some(prediction) => prediction,
            None => self.evaluate(values)?.1,
        };
        Ok(vec![prediction.map_or(0.0, |g| g - self.value)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CalibrationParameter, LossConfig};
    use commol_core::Model;
    use commol_difference::DifferenceEquations;

    /// A → B at the proportional rate `k`, from A = 1000, with B observed at
    /// 2 and the prediction of B at 5.
    fn held() -> HeldPrediction<DifferenceEquations> {
        let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
            "engine":{"type":"DifferenceEquations"},
            "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
            "accumulators":[],"stratifications":[],
            "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k",
            "stratified_rates":null,"per_compartment":null}]},
            "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#;
        let model: Model = serde_json::from_str(json).expect("valid model JSON");
        let problem = CalibrationProblem::new(
            DifferenceEquations::try_from_model(&model).unwrap(),
            vec![ObservedDataPoint::new(2.0, "B".to_string(), 190.0)],
            vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
            vec![],
            LossConfig::SumSquaredError,
        )
        .unwrap();
        let spec = ObservedDataPoint::new(5.0, "B".to_string(), 0.0);
        let (observing, operator) = problem.observing(&[spec]).unwrap();
        HeldPrediction {
            observing,
            operator,
            value: 400.0,
            settings: ConstraintSettings::for_value(400.0),
        }
    }

    #[test]
    fn the_constraint_of_a_point_not_scored_runs_it_once() {
        let held = held();
        let cost = PredictionCost::new(&held);

        // B at 5 is 1000 (1 − 0.9⁵) = 409.51 at k = 0.1, and B at 2 is the
        // observed 190.
        let constraint = cost.apply(&vec![0.1]).unwrap();
        assert!((constraint[0] - 9.51).abs() < 1e-9, "{constraint:?}");
        assert_eq!(cost.evaluations.load(Ordering::Relaxed), 1);
        // The cost of the same point reads the run again; the constraint of
        // the point just scored does not.
        let loss = cost.cost(&vec![0.1]).unwrap();
        assert!(loss.abs() < 1e-9, "{loss}");
        assert_eq!(cost.apply(&vec![0.1]).unwrap(), constraint);
        assert_eq!(cost.evaluations.load(Ordering::Relaxed), 2);
    }
}
