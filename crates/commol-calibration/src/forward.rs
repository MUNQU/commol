//! The forward model: the parameters of a problem written into an engine and
//! run.
//!
//! This is the only code that clones, resets, writes values to or runs the
//! engine of a calibration problem. A stochastic engine runs one replicate
//! per seed of its [`Replicates`], the same seeds at every evaluation.

use commol_core::{SimulationEngine, SimulationError, ValuePlan};

use crate::feasibility::{EvaluationError, Violation};
use crate::observation::ObservationOperator;
use crate::types::{CalibrationParameter, ObservedDataPoint, Replicates};
use crate::values::ValueWriter;

/// The buffers of one thread's evaluations, reused from one to the next.
#[derive(Debug, Default, Clone)]
pub struct Workspace {
    /// Rows of each replicate's run at the schedule.
    pub rows: Vec<Vec<Vec<f64>>>,
    /// Predicted observations, replicate after replicate: the prediction of
    /// observation `i` of replicate `r` is at `r × n_observations + i`.
    pub predictions: Vec<f64>,
}

/// An engine, the parameters a problem writes into it, and the positions a
/// run reports at.
#[derive(Debug, Clone)]
pub struct ForwardModel<E: SimulationEngine> {
    base_engine: E,
    parameters: Vec<CalibrationParameter>,
    values: ValueWriter<E>,
    /// Sorted, deduplicated positions every evaluation runs at.
    schedule: Vec<f64>,
    observation: ObservationOperator,
    replicates: Option<Replicates>,
}

impl<E: SimulationEngine> ForwardModel<E> {
    pub(crate) fn new(
        base_engine: E,
        parameters: Vec<CalibrationParameter>,
        values: ValueWriter<E>,
        schedule: Vec<f64>,
        observation: ObservationOperator,
    ) -> Self {
        Self {
            base_engine,
            parameters,
            values,
            schedule,
            observation,
            replicates: None,
        }
    }

    pub(crate) fn set_replicates(&mut self, replicates: Option<Replicates>) {
        self.replicates = replicates;
    }

    pub(crate) fn set_values(&mut self, values: ValueWriter<E>) {
        self.values = values;
    }

    /// The same model observed at `observed_data` instead: the engine, the
    /// parameters, the values and the replicates stay, the schedule and the
    /// observation operator are those of the new observations.
    ///
    /// # Errors
    ///
    /// A message when the engine rejects a position, or an observation names
    /// an unknown output or factor.
    pub fn with_observations(&self, observed_data: &[ObservedDataPoint]) -> Result<Self, String> {
        crate::calibration_problem::check_positions(&self.base_engine, observed_data, &[])?;
        let mut schedule: Vec<f64> = observed_data
            .iter()
            .flat_map(|obs| std::iter::once(obs.at).chain(obs.window.map(|window| obs.at - window)))
            .collect();
        schedule.sort_by(f64::total_cmp);
        schedule.dedup();
        let observation = ObservationOperator::new(
            observed_data,
            &self.parameters,
            &self.base_engine.output_names(),
            &schedule,
        )?;
        Ok(Self {
            schedule,
            observation,
            ..self.clone()
        })
    }

    /// The engine every run starts from.
    pub fn base_engine(&self) -> &E {
        &self.base_engine
    }

    /// The parameters, in problem order.
    pub fn parameters(&self) -> &[CalibrationParameter] {
        &self.parameters
    }

    /// The operator that turns the rows of a run at the schedule into
    /// observations.
    pub fn observation(&self) -> &ObservationOperator {
        &self.observation
    }

    /// The replicates of a stochastic engine, or `None`.
    pub fn replicates(&self) -> Option<&Replicates> {
        self.replicates.as_ref()
    }

    /// Number of runs per evaluation: one per replicate, or one.
    pub fn replicate_count(&self) -> usize {
        self.replicates.as_ref().map_or(1, |r| r.seeds.len())
    }

    /// The engine's message for a starting state `excess` away from
    /// feasible.
    pub(crate) fn plan_infeasibility(&self, excess: f64) -> String {
        self.values.plan().infeasibility(excess)
    }

    /// A clone of the base engine, reset, with `values` written into it.
    ///
    /// The values are written as given: callers check or clamp the bounds.
    ///
    /// # Errors
    ///
    /// [`Violation::InfeasibleStart`] when the values make the starting
    /// state infeasible, or a failure when the engine rejects a value or the
    /// number of values is wrong.
    pub fn configure(&self, values: &[f64]) -> Result<E, EvaluationError> {
        if values.len() != self.parameters.len() {
            return Err(EvaluationError::Failed(format!(
                "Expected {} parameters, got {}",
                self.parameters.len(),
                values.len()
            )));
        }
        let mut engine = self.base_engine.clone();
        engine.reset();
        let excess = self
            .values
            .apply(&mut engine, &self.parameters, values)
            .map_err(EvaluationError::Failed)?;
        if excess > 0.0 {
            return Err(Violation::InfeasibleStart { excess }.into());
        }
        Ok(engine)
    }

    /// Run at `positions` into `rows`, one entry per replicate.
    fn run_into(
        &self,
        configured: &mut E,
        positions: &[f64],
        rows: &mut Vec<Vec<Vec<f64>>>,
    ) -> Result<(), EvaluationError> {
        let count = self.replicate_count();
        rows.resize_with(count, Vec::new);
        match &self.replicates {
            None => configured
                .run_at_positions_into_buffer(positions, &mut rows[0])
                .map_err(simulation_error)?,
            Some(replicates) => {
                for (&seed, replicate_rows) in replicates.seeds.iter().zip(rows.iter_mut()) {
                    let mut engine = configured.clone();
                    engine.set_seed(seed);
                    engine.reset();
                    engine
                        .run_at_positions_into_buffer(positions, replicate_rows)
                        .map_err(simulation_error)?;
                }
            }
        }
        let finite = rows
            .iter()
            .flatten()
            .all(|row| row.iter().all(|value| value.is_finite()));
        if finite {
            Ok(())
        } else {
            Err(Violation::NonFinite.into())
        }
    }

    /// Run at the schedule into `workspace.rows`, one entry per replicate.
    ///
    /// # Errors
    ///
    /// A violation when the start or the run is infeasible or a value is not
    /// finite; a failure when the engine fails.
    pub fn run_schedule(
        &self,
        values: &[f64],
        workspace: &mut Workspace,
    ) -> Result<(), EvaluationError> {
        let mut engine = self.configure(values)?;
        self.run_into(&mut engine, &self.schedule, &mut workspace.rows)
    }

    /// Run once at the schedule into `workspace.rows`, whatever the
    /// replicates: with `seed`, as one replicate of a stochastic engine
    /// seeded with it; without, as the engine is.
    ///
    /// # Errors
    ///
    /// As [`Self::run_schedule`].
    pub fn run_once(
        &self,
        values: &[f64],
        seed: Option<u64>,
        workspace: &mut Workspace,
    ) -> Result<(), EvaluationError> {
        let mut engine = self.configure(values)?;
        if let Some(seed) = seed {
            engine.set_seed(seed);
            engine.reset();
        }
        workspace.rows.resize_with(1, Vec::new);
        engine
            .run_at_positions_into_buffer(&self.schedule, &mut workspace.rows[0])
            .map_err(simulation_error)?;
        if workspace.rows[0]
            .iter()
            .all(|row| row.iter().all(|v| v.is_finite()))
        {
            Ok(())
        } else {
            Err(Violation::NonFinite.into())
        }
    }

    /// Write the predicted observations of the rows in `workspace` into
    /// `workspace.predictions`, replicate after replicate.
    pub fn observe(&self, values: &[f64], workspace: &mut Workspace) {
        let n = self.observation.count();
        workspace.predictions.resize(n * workspace.rows.len(), 0.0);
        for (rows, out) in workspace
            .rows
            .iter()
            .zip(workspace.predictions.chunks_mut(n.max(1)))
        {
            self.observation.apply(rows, values, out);
        }
    }

    /// One predicted value per observation and replicate, replicate after
    /// replicate.
    ///
    /// # Errors
    ///
    /// As [`Self::run_schedule`].
    pub fn run_observed<'w>(
        &self,
        values: &[f64],
        workspace: &'w mut Workspace,
    ) -> Result<&'w [f64], EvaluationError> {
        self.run_schedule(values, workspace)?;
        self.observe(values, workspace);
        Ok(&workspace.predictions)
    }

    /// The rows of a run at `positions`, one entry per replicate.
    ///
    /// # Errors
    ///
    /// As [`Self::run_schedule`].
    pub fn run_at(
        &self,
        values: &[f64],
        positions: &[f64],
    ) -> Result<Vec<Vec<Vec<f64>>>, EvaluationError> {
        let mut engine = self.configure(values)?;
        let mut rows = Vec::new();
        self.run_into(&mut engine, positions, &mut rows)?;
        Ok(rows)
    }
}

fn simulation_error(error: SimulationError) -> EvaluationError {
    match error {
        // The parameters leave the values the model allows: a point to
        // avoid, not a reason to stop.
        SimulationError::Infeasible { .. } => Violation::InfeasibleRun.into(),
        SimulationError::Failed(error) => {
            EvaluationError::Failed(format!("Simulation failed: {error}"))
        }
    }
}
