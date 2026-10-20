//! The interface every simulation engine implements, whatever the kind of its
//! model.

use rayon::prelude::*;
use std::fmt::{self, Debug, Display};

use crate::types::ModelValueType;

/// Outputs of a run and the positions on the axis they were reported at.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    /// Positions on the axis, ascending; `rows[i]` is the state at `positions[i]`.
    pub positions: Vec<f64>,
    /// One output row per position, in the order of `output_names()`.
    pub rows: Vec<Vec<f64>>,
}

/// Why a run stopped.
#[derive(Debug, Clone, PartialEq)]
pub enum SimulationError {
    /// The state left the values the model allows, such as an amount below
    /// zero. Calibration treats such a run as infeasible, not as a failure.
    Infeasible {
        /// The position on the axis where the state left them.
        position: f64,
        /// What left them, for the user.
        message: String,
    },
    /// Any other error, such as a rate that cannot be evaluated.
    Failed(String),
}

impl Display for SimulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Infeasible { message, .. } | Self::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for SimulationError {}

impl From<String> for SimulationError {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}

/// Check that positions are sorted, unique, not before `from`, and each one
/// accepted by `accept`.
///
/// Engines call it before they run, so the rule on which positions a model
/// type accepts lives in `accept` and nowhere else.
///
/// # Errors
///
/// Returns the message of `accept` for the first position it rejects, or a
/// message that names the first pair out of order or the first position
/// before `from`.
pub fn check_positions(
    positions: &[f64],
    from: f64,
    accept: impl Fn(f64) -> Result<(), String>,
) -> Result<(), String> {
    for &position in positions {
        accept(position)?;
    }
    if let Some(pair) = positions.windows(2).find(|pair| pair[0] >= pair[1]) {
        return Err(format!(
            "Positions must be sorted and unique, got {} before {}",
            pair[0], pair[1]
        ));
    }
    if let Some(&first) = positions.first()
        && first < from
    {
        return Err(format!(
            "Position {first} is before the current position {from}"
        ));
    }
    Ok(())
}

/// A value of the model kind that calibration sets, besides parameters: its
/// type and its id, such as an initial condition of bin `A`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelValue {
    /// The type of the value.
    pub value_type: ModelValueType,
    /// The id the engine resolves, such as a bin or a compartment.
    pub id: String,
}

/// Values of a model kind that an engine has resolved, ready to write
/// ([`SimulationEngine::plan_values`]).
pub trait ValuePlan: Clone + Debug + Send + Sync {
    /// Why the state is infeasible, for a positive excess returned by
    /// [`SimulationEngine::apply_values`].
    fn infeasibility(&self, excess: f64) -> String;
}

/// Common interface that all model engines must implement
/// to be compatible with calibration and other analysis tools.
///
/// Calibration and optimization only use this trait, so they work with any
/// engine that implements it, whatever the kind of its model. A position on
/// the axis is an `f64` everywhere; each engine decides which positions its
/// model type accepts ([`Self::check_position`]). The values of its model
/// kind that calibration sets, such as initial conditions, are resolved and
/// written by the engine ([`Self::plan_values`], [`Self::apply_values`]).
///
/// # Example Implementation
///
/// ```rust,ignore
/// use commol_core::SimulationEngine;
///
/// #[derive(Clone)]
/// struct MyModel {
///     // ... model fields
/// }
///
/// impl SimulationEngine for MyModel {
///     type ValuePlan = MyValuePlan;
///
///     fn check_position(&self, position: f64) -> Result<(), String> {
///         // Implementation
///     }
///
///     // ... implement other required methods
/// }
/// ```
pub trait SimulationEngine: Clone {
    /// The values of the model kind, resolved by [`Self::plan_values`].
    type ValuePlan: ValuePlan;

    /// Check that the engine can report at a position on the axis.
    ///
    /// # Errors
    ///
    /// Returns why the position is not accepted, without naming the object
    /// that holds it: the caller adds that name. A difference engine accepts
    /// only non-negative integers; a differential engine any finite,
    /// non-negative position.
    fn check_position(&self, position: f64) -> Result<(), String>;

    /// Run from the current position through the last of `positions`,
    /// recording the outputs at each of them into a reusable buffer.
    ///
    /// The output rows are aligned with `positions`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Infeasible`] when the state leaves the
    /// values the model allows, and [`SimulationError::Failed`] when a
    /// position is rejected by [`Self::check_position`], when the positions
    /// are not sorted and unique or one is before the current position (see
    /// [`check_positions`]), or when the simulation fails for another reason.
    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError>;

    /// Run through the last of `positions` and return the outputs at each of
    /// them. See [`Self::run_at_positions_into_buffer`].
    ///
    /// # Errors
    ///
    /// See [`Self::run_at_positions_into_buffer`].
    fn run_at(&mut self, positions: &[f64]) -> Result<Vec<Vec<f64>>, SimulationError> {
        let mut buffer = Vec::with_capacity(positions.len());
        self.run_at_positions_into_buffer(positions, &mut buffer)?;
        Ok(buffer)
    }

    /// The default grid of a run of `num_steps` steps: the current position
    /// and the `num_steps` positions one unit apart after it.
    fn grid(&self, num_steps: u32) -> Vec<f64> {
        let start = self.current_position();
        (0..=num_steps).map(|k| start + f64::from(k)).collect()
    }

    /// Run on the default grid of `num_steps` steps ([`Self::grid`]).
    ///
    /// # Errors
    ///
    /// See [`Self::run_at_positions_into_buffer`].
    fn run(&mut self, num_steps: u32) -> Result<Trajectory, SimulationError> {
        let positions = self.grid(num_steps);
        let rows = self.run_at(&positions)?;
        Ok(Trajectory { positions, rows })
    }

    /// Advance the state by one unit of the axis.
    ///
    /// # Errors
    ///
    /// See [`Self::run_at_positions_into_buffer`].
    fn step(&mut self) -> Result<(), SimulationError>;

    /// Names of the outputs, in the order of each row a run returns.
    fn output_names(&self) -> Vec<String>;

    /// How many leading outputs are the state of the model. An expression
    /// reads their sum as `total`; the outputs after them, such as accumulators,
    /// do not count towards it.
    fn state_outputs(&self) -> usize;

    /// Named sums of outputs an expression may read besides the output
    /// names, as `(name, output indices)`: for example the total of a bin
    /// over its categories. None by default.
    fn output_sums(&self) -> Vec<(String, Vec<usize>)> {
        Vec::new()
    }

    /// The relative accuracy of the outputs of a run: the size of the
    /// relative error a run may carry. Finite differences set their step
    /// from it. Machine epsilon by default, for an engine whose outputs are
    /// exact up to rounding; an engine that integrates with a tolerance
    /// returns that tolerance.
    fn output_accuracy(&self) -> f64 {
        f64::EPSILON
    }

    /// The absolute accuracy of the outputs of a run: the error a run may
    /// carry on an output near 0, beside [`Self::output_accuracy`] times its
    /// size. 0 by default; an engine that integrates with an absolute
    /// tolerance returns it.
    fn output_absolute_accuracy(&self) -> f64 {
        0.0
    }

    /// Reset the simulation to its initial state.
    ///
    /// After calling this method:
    /// - `current_position()` should return 0.0
    /// - the state is the initial state, with the values last written by
    ///   [`Self::apply_values`]
    /// - Parameters remain unchanged (use `set_parameter()` to modify them)
    ///
    /// This is essential for calibration workflows where the same model
    /// is simulated multiple times with different parameters.
    ///
    /// A stochastic engine also restores the generator of its seed
    /// ([`Self::set_seed`]), so a run after a reset repeats the run before it.
    fn reset(&mut self);

    /// Seed the random numbers of the engine, and restore its generator.
    ///
    /// A stochastic engine draws every random number from one generator
    /// seeded here (`commol_random::Generator`), and [`Self::reset`] restores
    /// it, so a run is reproducible from its seed. The default does nothing:
    /// a deterministic engine draws no random number, and gives the same run
    /// for every seed.
    fn set_seed(&mut self, _seed: u64) {}

    /// Update a parameter value by its identifier.
    ///
    /// # Arguments
    ///
    /// * `parameter_id` - The unique identifier of the parameter (e.g., "k1", "k2")
    /// * `value` - The new value to assign
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error message if the parameter doesn't exist
    /// or the value is invalid.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// engine.set_parameter("k1", 0.3)?;
    /// engine.set_parameter("k2", 0.1)?;
    /// ```
    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String>;

    /// The current position on the axis.
    ///
    /// Should return 0.0 immediately after construction or `reset()`.
    fn current_position(&self) -> f64;

    /// Resolve values of the model kind that calibration sets.
    ///
    /// # Arguments
    ///
    /// * `values` - The type and id of each value, in the order
    ///   [`Self::apply_values`] takes their numbers.
    /// * `remainder` - An id that takes what the values leave of a total the
    ///   kind keeps, if any.
    ///
    /// # Errors
    ///
    /// Returns a message when the kind does not accept a type of value, or an
    /// id or the remainder does not resolve.
    fn plan_values(
        &self,
        values: &[ModelValue],
        remainder: Option<&str>,
    ) -> Result<Self::ValuePlan, String>;

    /// Write values of the model kind into the state the engine starts from.
    ///
    /// # Arguments
    ///
    /// * `plan` - Values resolved by [`Self::plan_values`].
    /// * `values` - The number of each value, parallel to the plan.
    ///
    /// # Returns
    ///
    /// How far the state is from feasible: `0.0` for a feasible state.
    /// [`ValuePlan::infeasibility`] says why a positive excess is not.
    ///
    /// # Errors
    ///
    /// Returns a message when the engine rejects a value.
    fn apply_values(&mut self, plan: &Self::ValuePlan, values: &[f64]) -> Result<f64, String>;
}

/// Run one replicate per seed of a stochastic engine, in parallel.
///
/// Each replicate clones `engine`, seeds it ([`SimulationEngine::set_seed`]),
/// resets it and runs it at `positions`, so replicate `i` gives the same rows
/// however many replicates run and in whatever order the threads take them.
///
/// # Errors
///
/// Returns the error of the first replicate, in seed order, that fails.
pub fn run_replicates<E: SimulationEngine + Send + Sync>(
    engine: &E,
    positions: &[f64],
    seeds: &[u64],
) -> Result<Vec<Vec<Vec<f64>>>, SimulationError> {
    seeds
        .par_iter()
        .map(|&seed| {
            let mut replicate = engine.clone();
            replicate.set_seed(seed);
            replicate.reset();
            replicate.run_at(positions)
        })
        .collect()
}
