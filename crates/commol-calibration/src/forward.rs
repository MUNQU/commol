//! The forward model: the parameters of a problem written into an engine and
//! run.
//!
//! This is the only code that clones, resets, writes values to or runs the
//! engine of a calibration problem. A stochastic engine runs one replicate
//! per seed of its [`Replicates`], the same seeds at every evaluation.

use commol_core::{SimulationEngine, SimulationError, ValuePlan};

use crate::feasibility::{EvaluationError, Violation};
use crate::observation::ObservationOperator;
use crate::types::{
    CalibrationParameter, CalibrationParameterType, ObservedDataPoint, ReplicateSummary, Replicates,
};
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

    /// A forward model with no observations, for a method that asks what a
    /// model does rather than how it fits data: the values of `parameters`
    /// reach the engine as a calibration writes them, with `remainder` for
    /// the initial conditions, and with a stochastic engine
    /// ([`SimulationEngine::is_stochastic`]) each run is `replicates`
    /// replicates seeded from `seed`, the same seeds at every run. A
    /// deterministic engine runs once, and reads neither. Its schedule is
    /// empty: it runs at the positions it is given ([`Self::run_at`]).
    ///
    /// # Errors
    ///
    /// Returns a message when there is no parameter, a parameter is a factor
    /// or a noise parameter (both belong to observations), `replicates` is 0
    /// ([`crate::ranges::REPLICATES`]), the engine is stochastic and
    /// `replicates` or `seed` is not set, or the engine does not resolve a
    /// value of its model kind or the remainder.
    pub fn unobserved(
        base_engine: E,
        parameters: Vec<CalibrationParameter>,
        remainder: Option<&str>,
        replicates: Option<usize>,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        if parameters.is_empty() {
            return Err("No parameters provided".to_string());
        }
        for parameter in &parameters {
            let what = match parameter.parameter_type {
                CalibrationParameterType::Factor => "a factor, which multiplies an observation",
                CalibrationParameterType::Noise => {
                    "a noise parameter, which belongs to an observation model"
                }
                CalibrationParameterType::Parameter
                | CalibrationParameterType::InitialCondition => continue,
            };
            return Err(format!(
                "'{}' is {what}; a model with no observations has none",
                parameter.id
            ));
        }
        if let Some(count) = replicates {
            crate::ranges::REPLICATES.check(count)?;
        }
        let replicates = if base_engine.is_stochastic() {
            let engine = the_engine(&base_engine);
            let count = replicates.ok_or_else(|| {
                format!(
                    "{engine} is stochastic: set replicates, how many replicates each point of \
                     a design runs."
                )
            })?;
            let seed = seed.ok_or_else(|| {
                format!(
                    "{engine} is stochastic: set seed, so the replicates and the analysis are \
                     reproducible."
                )
            })?;
            Some(Replicates::new(count, seed, ReplicateSummary::Mean)?)
        } else {
            None
        };
        let values = ValueWriter::new(&base_engine, &parameters, remainder)?;
        let observation =
            ObservationOperator::new(&[], &parameters, &base_engine.output_names(), &[])?;
        let mut model = Self::new(base_engine, parameters, values, Vec::new(), observation);
        model.replicates = replicates;
        Ok(model)
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
        let schedule = schedule_of(observed_data, []);
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

    /// The same model run at `schedule`, sorted and deduplicated, which holds
    /// every position of `observed_data` and the start of every window; the
    /// observation operator reads `observed_data` from those runs.
    ///
    /// # Panics
    ///
    /// Panics when an observation names an unknown output or factor: the
    /// observations of a problem built their operator on these outputs and
    /// parameters already, and the schedule does not change what it accepts.
    pub(crate) fn on_schedule(
        &self,
        observed_data: &[ObservedDataPoint],
        schedule: Vec<f64>,
    ) -> Self {
        let observation = ObservationOperator::new(
            observed_data,
            &self.parameters,
            &self.base_engine.output_names(),
            &schedule,
        )
        .expect("the observations of the problem built their operator already");
        Self {
            schedule,
            observation,
            ..self.clone()
        }
    }

    /// The sorted, deduplicated positions every evaluation runs at.
    pub fn schedule(&self) -> &[f64] {
        &self.schedule
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

    /// A clone of the base engine, reset, with `values` written into it.
    ///
    /// The values are written as given: callers check or clamp the bounds.
    ///
    /// # Errors
    ///
    /// [`Violation::InfeasibleStart`], with the plan of values' message, when
    /// the values make the starting state infeasible, or a failure when the engine rejects a value or the
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
            let message = self.values.plan().infeasibility(excess);
            return Err(Violation::InfeasibleStart { excess, message }.into());
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
        workspace.rows.resize_with(1, Vec::new);
        self.run_once_into(values, &self.schedule, seed, &mut workspace.rows[0])
    }

    /// The rows of one run at `positions`, whatever the replicates: with
    /// `seed`, as one replicate of a stochastic engine seeded with it;
    /// without, as the engine is.
    ///
    /// # Errors
    ///
    /// As [`Self::run_schedule`].
    pub fn run_once_at(
        &self,
        values: &[f64],
        positions: &[f64],
        seed: Option<u64>,
    ) -> Result<Vec<Vec<f64>>, EvaluationError> {
        let mut rows = Vec::new();
        self.run_once_into(values, positions, seed, &mut rows)?;
        Ok(rows)
    }

    fn run_once_into(
        &self,
        values: &[f64],
        positions: &[f64],
        seed: Option<u64>,
        rows: &mut Vec<Vec<f64>>,
    ) -> Result<(), EvaluationError> {
        let mut engine = self.configure(values)?;
        if let Some(seed) = seed {
            engine.set_seed(seed);
            engine.reset();
        }
        engine
            .run_at_positions_into_buffer(positions, rows)
            .map_err(simulation_error)?;
        if rows.iter().all(|row| row.iter().all(|v| v.is_finite())) {
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

/// The positions a run reports at to read `observations`, and `extra`
/// positions: each position and the start of each window, sorted, once.
pub fn schedule_of(
    observations: &[ObservedDataPoint],
    extra: impl IntoIterator<Item = f64>,
) -> Vec<f64> {
    let mut schedule: Vec<f64> = observations
        .iter()
        .flat_map(|obs| std::iter::once(obs.at).chain(obs.window_start()))
        .chain(extra)
        .collect();
    schedule.sort_by(f64::total_cmp);
    schedule.dedup();
    schedule
}

/// The mean over `replicates` of each of `count` predictions: entry `i` is
/// the mean of entry `i` of every replicate, summed in replicate order.
/// With no replicate, each mean is NaN.
pub fn replicate_mean<'a>(
    replicates: impl IntoIterator<Item = &'a [f64]>,
    count: usize,
) -> Vec<f64> {
    let mut sums = vec![0.0; count];
    let mut runs = 0usize;
    for replicate in replicates {
        for (sum, value) in sums.iter_mut().zip(replicate) {
            *sum += value;
        }
        runs += 1;
    }
    sums.into_iter().map(|sum| sum / runs as f64).collect()
}

/// How a message names `engine`: "The StochasticSteps engine", by its entry
/// of the list of engines, or "The engine" for an engine off the list.
pub(crate) fn the_engine<E: SimulationEngine>(engine: &E) -> String {
    let name = engine
        .engine_type()
        .map_or(String::new(), |entry| format!("{} ", entry.name()));
    format!("The {name}engine")
}

fn simulation_error(error: SimulationError) -> EvaluationError {
    match error {
        // The parameters leave the values the model allows: a point to
        // avoid, not a reason to stop.
        SimulationError::Infeasible { message, .. } => Violation::InfeasibleRun { message }.into(),
        SimulationError::Failed(error) => {
            EvaluationError::Failed(format!("Simulation failed: {error}"))
        }
    }
}
