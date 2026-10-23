//! The calibration problem: the layers that score a point of parameter
//! space, built once.
//!
//! [`CalibrationProblem`] holds the forward model ([`ForwardModel`]), which
//! runs the engine; the observation operator ([`ObservationOperator`]), which
//! turns a run into observations; the feasibility ([`Feasibility`]), which
//! checks bounds and constraints; and the discrepancy ([`Discrepancy`]),
//! which compares predictions with data by a loss or by the likelihood of
//! observation models. It has no interior mutability: the
//! buffers of an evaluation live in a [`Workspace`] the caller owns.

use commol_core::SimulationEngine;

use crate::discrepancy::{Discrepancy, DiscrepancyConfig, TermError};
use crate::feasibility::{EvaluationError, Feasibility, Violation, constraint_positions};
use crate::forward::{ForwardModel, Workspace, schedule_of};
use crate::likelihood::TermDerivatives;
use crate::observation::ObservationOperator;
use crate::types::{CalibrationConstraint, CalibrationParameter, ObservedDataPoint, Replicates};
use crate::values::{ValueWriter, clamp_to_bounds};

/// A calibration problem for any [`SimulationEngine`].
///
/// # Example
///
/// ```rust
/// use commol_calibration::{CalibrationProblem, types::*};
/// use commol_core::Model;
/// use commol_difference::DifferenceEquations;
///
/// // A → B at the rate k1, from A = 900 and B = 100.
/// let json = r#"{"name":"m","parameters":[{"id":"k1","value":0.1}],
///     "engine":{"type":"DifferenceEquations"},
///     "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
///     "accumulators":[],"stratifications":[],
///     "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k1"}]},
///     "structure":{"kind":"aggregate","bin_amounts":{"A":900.0,"B":100.0}}}"#;
/// let model: Model = serde_json::from_str(json).unwrap();
/// let engine = DifferenceEquations::try_from_model(&model).unwrap();
/// let observed_data = vec![
///     ObservedDataPoint::new(10.0, "B".to_string(), 690.0),
///     ObservedDataPoint::new(20.0, "B".to_string(), 890.0),
/// ];
/// let problem = CalibrationProblem::new(
///     engine.clone(),
///     observed_data.clone(),
///     vec![CalibrationParameter::new("k1".to_string(), 0.0, 1.0)],
///     vec![],
///     LossConfig::SumSquaredError,
/// )
/// .unwrap();
/// assert_eq!(problem.num_parameters(), 1);
///
/// // The engine rejects a parameter that the model does not have.
/// let error = CalibrationProblem::new(
///     engine,
///     observed_data,
///     vec![CalibrationParameter::new("k9".to_string(), 0.0, 1.0)],
///     vec![],
///     LossConfig::SumSquaredError,
/// )
/// .err()
/// .unwrap();
/// assert!(error.contains("'k9'"));
/// ```
#[derive(Debug, Clone)]
pub struct CalibrationProblem<E: SimulationEngine> {
    observed_data: Vec<ObservedDataPoint>,
    forward: ForwardModel<E>,
    feasibility: Feasibility,
    discrepancy: Discrepancy,
}

impl<E: SimulationEngine> CalibrationProblem<E> {
    /// Create a new calibration problem
    ///
    /// # Arguments
    ///
    /// * `base_engine` - The simulation engine to calibrate (cloned for each evaluation)
    /// * `observed_data` - Vector of observed data points
    /// * `parameters` - Parameters to calibrate with their bounds
    /// * `constraints` - Constraints on parameters and/or compartment values
    /// * `discrepancy` - A loss ([`crate::LossConfig`]), or one observation
    ///   model per series label ([`DiscrepancyConfig::Likelihood`])
    ///
    /// # Errors
    ///
    /// Returns a message when there is no observation or parameter, a
    /// position is one the engine rejects, an observation names an unknown
    /// output or factor, the engine does not resolve a value of its model
    /// kind, a constraint is invalid, or the observation models do not fit
    /// the observations and parameters ([`crate::Likelihood::new`]).
    pub fn new(
        base_engine: E,
        observed_data: Vec<ObservedDataPoint>,
        parameters: Vec<CalibrationParameter>,
        constraints: Vec<CalibrationConstraint>,
        discrepancy: impl Into<DiscrepancyConfig>,
    ) -> Result<Self, String> {
        if observed_data.is_empty() {
            return Err("No observed data provided".to_string());
        }
        check_positions(&base_engine, &observed_data, &constraints)?;
        if parameters.is_empty() {
            return Err("No calibration parameters provided".to_string());
        }

        let schedule = schedule_of(&observed_data, constraint_positions(&constraints));

        let output_names = base_engine.output_names();
        let observation =
            ObservationOperator::new(&observed_data, &parameters, &output_names, &schedule)?;
        let values = ValueWriter::new(&base_engine, &parameters, None)?;
        let feasibility = Feasibility::new(
            &parameters,
            &constraints,
            &output_names,
            base_engine.state_outputs(),
            base_engine.output_sums(),
            &schedule,
        )?;
        let discrepancy = Discrepancy::new(discrepancy.into(), &observed_data, &parameters)?;

        Ok(Self {
            forward: ForwardModel::new(base_engine, parameters, values, schedule, observation),
            observed_data,
            feasibility,
            discrepancy,
        })
    }

    /// Enable or disable per-series observation normalization.
    ///
    /// When enabled, each residual is multiplied by `1 / rms` of its series'
    /// observed values before the loss, so that series measured on very
    /// different scales contribute comparably. Disabled by default.
    ///
    /// # Errors
    ///
    /// Returns a message when it is enabled on a problem with observation
    /// models, which set the scale of each series themselves.
    pub fn with_observation_normalization(mut self, enabled: bool) -> Result<Self, String> {
        self.discrepancy.set_normalization(enabled)?;
        Ok(self)
    }

    /// Run `replicates` at each evaluation, for a stochastic engine.
    ///
    /// Each evaluation runs one replicate per seed, the same seeds at every
    /// evaluation (common random numbers), so the loss is a deterministic
    /// function of the parameters and the optimizers keep working. The
    /// observations are compared with the summary of the replicates
    /// ([`crate::LossConfig::replicate_loss`]). `None` runs the engine once.
    ///
    /// # Errors
    ///
    /// Returns a message when the problem has observation models: a
    /// stochastic engine needs a loss.
    pub fn with_replicates(mut self, replicates: Option<Replicates>) -> Result<Self, String> {
        self.discrepancy
            .set_summary(replicates.as_ref().map(|r| r.summary))?;
        self.forward.set_replicates(replicates);
        Ok(self)
    }

    /// Name the id that takes what the calibrated initial conditions leave of
    /// the sum the model kind keeps; the engine resolves it.
    ///
    /// For an aggregate model it is a compartment or bin that takes the sum
    /// of the amounts the model states less every other compartment, so the
    /// compartments the values and the remainder cover keep their declared
    /// sum. Without a remainder, calibrated initial conditions are applied as
    /// given. With one, a combination that would leave the remainder
    /// negative is infeasible.
    ///
    /// # Errors
    ///
    /// Returns the engine's message when the id does not resolve, for example
    /// when it is not a compartment, bin or bin prefix, or overlaps a
    /// calibrated bin or compartment.
    pub fn with_initial_condition_remainder(mut self, remainder: &str) -> Result<Self, String> {
        let values = ValueWriter::new(
            self.forward.base_engine(),
            self.forward.parameters(),
            Some(remainder),
        )?;
        self.forward.set_values(values);
        Ok(self)
    }

    /// A copy of the problem that compares predictions with other observed
    /// values and weights, one of each per observation in problem order.
    ///
    /// Everything else stays: the engine, the parameters, the observation
    /// operator, the feasibility, the loss or the observation models and
    /// their settings. The discrepancy is built as [`Self::new`] builds it,
    /// so with normalization on its factors come from the new values. A
    /// resampling method makes each replicate with it.
    ///
    /// # Errors
    ///
    /// Returns a message that names the observation when a value is not
    /// finite or lies outside the support of its observation model, or a
    /// weight is negative or not finite, or when the numbers of values or
    /// weights are not the number of observations, or every weight is
    /// zero.
    pub fn with_data(&self, values: &[f64], weights: &[f64]) -> Result<Self, String> {
        let n = self.observed_data.len();
        if values.len() != n || weights.len() != n {
            return Err(format!(
                "Other data need one value and one weight per observation: {n} \
                 observations, {} values, {} weights",
                values.len(),
                weights.len()
            ));
        }
        for ((obs, &value), &weight) in self.observed_data.iter().zip(values).zip(weights) {
            if !value.is_finite() {
                return Err(format!(
                    "Observation '{}' at {} gets the value {value}; a value must be finite",
                    obs.label, obs.at
                ));
            }
            if !(weight.is_finite() && weight >= 0.0) {
                return Err(format!(
                    "Observation '{}' at {} gets the weight {weight}; a weight must be \
                     finite and not negative",
                    obs.label, obs.at
                ));
            }
        }
        if weights.iter().all(|&weight| weight == 0.0) {
            return Err("Every weight is zero; at least one observation needs weight".to_string());
        }
        let mut problem = self.clone();
        for ((obs, &value), &weight) in problem.observed_data.iter_mut().zip(values).zip(weights) {
            obs.value = value;
            obs.weight = weight;
        }
        problem.discrepancy = self.discrepancy.with_data(&problem.observed_data)?;
        Ok(problem)
    }

    /// A copy of the problem whose runs also report at the positions of
    /// `extra` and the starts of their windows, and the operator that reads
    /// `extra` from the rows of those runs.
    ///
    /// The copy scores every point as the problem does: the observations,
    /// the discrepancy and the constraints stay. Its runs leave the
    /// predictions of `extra` in [`Workspace::rows`], which the operator
    /// reads (with the values of the parameters, which hold the factors).
    ///
    /// # Errors
    ///
    /// Returns a message when the engine rejects a position of `extra`, or
    /// an entry of `extra` names an unknown output or factor.
    pub fn observing(
        &self,
        extra: &[ObservedDataPoint],
    ) -> Result<(Self, ObservationOperator), String> {
        check_positions(self.forward.base_engine(), extra, &[])?;
        let schedule = schedule_of(extra, self.forward.schedule().iter().copied());
        let operator = ObservationOperator::new(
            extra,
            self.parameters(),
            &self.forward.base_engine().output_names(),
            &schedule,
        )?;
        let forward = self.forward.on_schedule(&self.observed_data, schedule);
        let feasibility = self.feasibility.on_schedule(forward.schedule());
        Ok((
            Self {
                forward,
                feasibility,
                ..self.clone()
            },
            operator,
        ))
    }

    /// The forward model.
    pub fn forward(&self) -> &ForwardModel<E> {
        &self.forward
    }

    /// The observation operator.
    pub fn observation(&self) -> &ObservationOperator {
        self.forward.observation()
    }

    /// The bounds and constraints.
    pub fn feasibility(&self) -> &Feasibility {
        &self.feasibility
    }

    /// The observations.
    pub fn observed_data(&self) -> &[ObservedDataPoint] {
        &self.observed_data
    }

    /// The parameters, in problem order.
    pub fn parameters(&self) -> &[CalibrationParameter] {
        self.forward.parameters()
    }

    /// Number of parameters being calibrated.
    pub fn num_parameters(&self) -> usize {
        self.parameters().len()
    }

    /// Parameter names, in problem order.
    pub fn parameter_names(&self) -> Vec<String> {
        self.parameters().iter().map(|p| p.id.clone()).collect()
    }

    /// Initial values: each guess, or the midpoint of its bounds.
    pub fn initial_parameters(&self) -> Vec<f64> {
        self.parameters()
            .iter()
            .map(|p| p.initial_value())
            .collect()
    }

    /// Parameter bounds as `(min, max)`.
    pub fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.parameters()
            .iter()
            .map(|p| (p.min_bound, p.max_bound))
            .collect()
    }

    /// The values the objective scores for a parameter vector: each value
    /// clamped to its bounds.
    pub fn applied_parameter_values(&self, values: &[f64]) -> Vec<f64> {
        clamp_to_bounds(self.parameters(), values)
    }

    /// The discrepancy at `values`, which are inside their bounds.
    ///
    /// The steps: the parameter constraints, the run of each replicate at
    /// the schedule, the constraints at positions on every replicate, the
    /// observation operator and the discrepancy.
    ///
    /// # Errors
    ///
    /// A violation when the point is not feasible; a failure when the engine
    /// or a constraint cannot be evaluated.
    pub fn evaluate(
        &self,
        values: &[f64],
        workspace: &mut Workspace,
    ) -> Result<f64, EvaluationError> {
        self.feasibility.check_parameters(values)?;
        self.forward.run_schedule(values, workspace)?;
        if self.feasibility.reads_rows() {
            let mut amount = 0.0;
            for rows in &workspace.rows {
                amount += self.feasibility.row_violation(values, rows);
            }
            if amount > 0.0 {
                return Err(Violation::Constraint { amount }.into());
            }
        }
        self.forward.observe(values, workspace);
        let discrepancy = self.discrepancy.of(values, &mut workspace.predictions)?;
        if discrepancy.is_finite() {
            Ok(discrepancy)
        } else {
            Err(Violation::NonFinite.into())
        }
    }

    /// The weighted log density `wᵢ · log p(yᵢ | μᵢ)` of each observation
    /// at `values`, which are inside their bounds; they sum to the
    /// log-likelihood. The steps are those of [`Self::evaluate`].
    ///
    /// # Errors
    ///
    /// As [`Self::evaluate`], and a failure when the problem has a loss
    /// instead of observation models.
    pub fn pointwise_log_likelihood(
        &self,
        values: &[f64],
        workspace: &mut Workspace,
    ) -> Result<Vec<f64>, EvaluationError> {
        let likelihood = self.discrepancy.likelihood().ok_or_else(|| {
            EvaluationError::Failed(
                "The pointwise log-likelihood needs observation models; the problem has a                  loss"
                    .to_string(),
            )
        })?;
        self.predict_once(values, None, workspace)?;
        Ok(likelihood.pointwise(values, &workspace.predictions)?)
    }

    /// The predicted observations of one run at `values`, which are inside
    /// their bounds: one replicate seeded with `seed`, whatever the
    /// problem's replicates, or the one run of the engine without a seed.
    /// The steps are those of [`Self::evaluate`] but the discrepancy; the
    /// predictions are in `workspace.predictions`.
    ///
    /// # Errors
    ///
    /// As [`Self::evaluate`].
    pub fn predict_once(
        &self,
        values: &[f64],
        seed: Option<u64>,
        workspace: &mut Workspace,
    ) -> Result<(), EvaluationError> {
        self.feasibility.check_parameters(values)?;
        self.forward.run_once(values, seed, workspace)?;
        if self.feasibility.reads_rows() {
            let amount = self.feasibility.row_violation(values, &workspace.rows[0]);
            if amount > 0.0 {
                return Err(Violation::Constraint { amount }.into());
            }
        }
        self.forward.observe(values, workspace);
        Ok(())
    }

    /// The derivatives of each observation's term of the discrepancy at the
    /// parameter `values`, given the predictions of one run
    /// ([`Discrepancy::term_derivatives`]).
    ///
    /// # Errors
    ///
    /// Returns a message for a loss with no curvature, or that names the
    /// observation whose mean lies outside its observation model.
    pub fn term_derivatives(
        &self,
        values: &[f64],
        predictions: &[f64],
    ) -> Result<Vec<TermDerivatives>, String> {
        self.discrepancy
            .term_derivatives(values, predictions)
            .map_err(|error| match error {
                TermError::NoCurvature(message) => message,
                TermError::MeanOutside(index) => {
                    let obs = &self.observed_data[index];
                    format!(
                        "Observation '{}' at {} has the mean {}, outside the means its \
                         observation model allows",
                        obs.label, obs.at, predictions[index]
                    )
                }
            })
    }

    /// The discrepancy: the loss or the likelihood, the observed values,
    /// their weights and normalization.
    pub fn discrepancy(&self) -> &Discrepancy {
        &self.discrepancy
    }

    /// An engine clone with the values applied exactly as the objective
    /// applies them (clamped to their bounds), including calibrated initial
    /// conditions.
    ///
    /// # Errors
    ///
    /// Returns a message when the number of values is wrong, the engine
    /// rejects a value, or the values make the state infeasible.
    pub fn configured_engine(&self, values: &[f64]) -> Result<E, String> {
        if values.len() != self.num_parameters() {
            return Err(format!(
                "Expected {} parameters, got {}",
                self.num_parameters(),
                values.len()
            ));
        }
        self.forward
            .configure(&self.applied_parameter_values(values))
            .map_err(|error| match error {
                EvaluationError::Violation(Violation::InfeasibleStart { message, .. }) => message,
                error => error.to_string(),
            })
    }
}

/// Check every position of a calibration problem with the engine.
///
/// The engine decides which positions its model type accepts
/// ([`SimulationEngine::check_position`]); this function names the object
/// that holds the first one it rejects: the observation (at its position or
/// at the start of its window) or the constraint.
///
/// # Errors
///
/// Returns a message that names the observation or constraint and gives the
/// engine's reason, or that names an observation whose window is not a
/// finite, positive length.
pub fn check_positions<E: SimulationEngine>(
    engine: &E,
    observed_data: &[ObservedDataPoint],
    constraints: &[CalibrationConstraint],
) -> Result<(), String> {
    for obs in observed_data {
        engine
            .check_position(obs.at)
            .map_err(|reason| format!("Observation '{}': {reason}", obs.label))?;
        if let Some(window) = obs.window {
            if !(window.is_finite() && window > 0.0) {
                return Err(format!(
                    "Observation '{}' at {} has the window {window}; a window \
                     must be finite and positive",
                    obs.label, obs.at
                ));
            }
            let start = crate::observation::window_start(obs.at, window);
            engine.check_position(start).map_err(|reason| {
                format!(
                    "Observation '{}' at {} with the window {window} starts its \
                     window at {start}: {reason}",
                    obs.label, obs.at
                )
            })?;
        }
    }
    for constraint in constraints {
        for &position in constraint.at.iter().flatten() {
            engine
                .check_position(position)
                .map_err(|reason| format!("Constraint '{}': {reason}", constraint.id))?;
        }
    }
    Ok(())
}
