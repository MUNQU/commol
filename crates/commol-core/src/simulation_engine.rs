//! The interface every simulation engine implements, whatever the kind of its
//! model.

use rayon::prelude::*;
use std::fmt::{self, Debug, Display};

use crate::Count;
use crate::types::ModelValueType;

/// The rule of the steps of a run: a whole number from 0 to the largest
/// `u32`, the type of [`SimulationEngine::run`].
pub const NUM_STEPS: Count = Count::between("num_steps", 0, u32::MAX as i128);

/// The rule of the number of replicates of a stochastic run
/// ([`run_replicates`]): at least 1.
pub const REPLICATE_COUNT: Count = Count::at_least("count", 1);

/// The rule of a seed: a whole number of 64 bits, from 0 to 2^64 - 1.
pub const SEED: Count = Count::between("seed", 0, u64::MAX as i128);

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

/// The accuracy of the outputs of a run ([`SimulationEngine::accuracy`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Accuracy {
    /// The size of the relative error a run may carry.
    pub relative: f64,
    /// The error a run may carry on an output near 0, beside `relative`
    /// times its size.
    pub absolute: f64,
}

impl Accuracy {
    /// Outputs exact up to rounding: machine epsilon, and 0 near 0.
    pub const EXACT: Self = Self {
        relative: f64::EPSILON,
        absolute: 0.0,
    };
}

/// The settings that set the accuracy of an engine, as a user writes them
/// ([`SimulationEngine::accuracy_settings`]), such as `ODESolver.rtol`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccuracySettings {
    /// The setting of [`Accuracy::relative`].
    pub relative: &'static str,
    /// The setting of [`Accuracy::absolute`].
    pub absolute: &'static str,
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
/// # Example
///
/// An engine of one output that keeps the value of its one parameter:
///
/// ```rust
/// use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan, check_positions};
///
/// #[derive(Clone)]
/// struct Constant {
///     value: f64,
///     position: f64,
/// }
///
/// /// The engine sets no value of a model kind.
/// #[derive(Clone, Debug)]
/// struct NoValues;
///
/// impl ValuePlan for NoValues {
///     fn infeasibility(&self, excess: f64) -> String {
///         format!("The state is {excess} from feasible")
///     }
/// }
///
/// impl SimulationEngine for Constant {
///     type ValuePlan = NoValues;
///
///     fn check_position(&self, position: f64) -> Result<(), String> {
///         if position.is_finite() && position >= 0.0 {
///             Ok(())
///         } else {
///             Err(format!("{position} is not a finite, non-negative position"))
///         }
///     }
///
///     fn run_at_positions_into_buffer(
///         &mut self,
///         positions: &[f64],
///         buffer: &mut Vec<Vec<f64>>,
///     ) -> Result<(), SimulationError> {
///         check_positions(positions, self.position, |position| {
///             self.check_position(position)
///         })?;
///         buffer.clear();
///         buffer.extend(positions.iter().map(|_| vec![self.value]));
///         if let Some(&last) = positions.last() {
///             self.position = last;
///         }
///         Ok(())
///     }
///
///     fn step(&mut self) -> Result<(), SimulationError> {
///         self.position += 1.0;
///         Ok(())
///     }
///
///     fn output_names(&self) -> Vec<String> {
///         vec!["A".to_string()]
///     }
///
///     fn state_outputs(&self) -> usize {
///         1
///     }
///
///     fn reset(&mut self) {
///         self.position = 0.0;
///     }
///
///     fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
///         if parameter_id != "k1" {
///             return Err(format!("The model has no parameter '{parameter_id}'"));
///         }
///         self.value = value;
///         Ok(())
///     }
///
///     fn current_position(&self) -> f64 {
///         self.position
///     }
///
///     fn plan_values(
///         &self,
///         values: &[ModelValue],
///         _remainder: Option<&str>,
///     ) -> Result<NoValues, String> {
///         match values.first() {
///             None => Ok(NoValues),
///             Some(value) => Err(format!("The engine sets no value '{}'", value.id)),
///         }
///     }
///
///     fn apply_values(&mut self, _plan: &NoValues, _values: &[f64]) -> Result<f64, String> {
///         Ok(0.0)
///     }
/// }
///
/// let mut engine = Constant { value: 2.0, position: 0.0 };
/// engine.set_parameter("k1", 3.0).unwrap();
/// assert!(engine.set_parameter("k9", 1.0).is_err());
/// assert_eq!(engine.run(2).unwrap().rows, vec![vec![3.0]; 3]);
/// ```
pub trait SimulationEngine: Clone {
    /// The values of the model kind, resolved by [`Self::plan_values`].
    type ValuePlan: ValuePlan;

    /// Check that the engine can report at a position on the axis.
    ///
    /// # Errors
    ///
    /// Returns why the position is not accepted, without naming the object
    /// that holds it: the caller adds that name. An engine of the list
    /// returns the rule of its entry ([`crate::EngineType::check_position`]):
    /// only integer steps for an engine whose rate is per step, any finite,
    /// non-negative position for the others.
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

    /// The matrices between categories of the model, each with the current
    /// value of its scales ([`Self::set_parameter`] changes them). A
    /// calibration constraint reads their entries and spectral radius at
    /// the values of each point. None by default.
    fn matrices(&self) -> Vec<crate::ScaledMatrix> {
        Vec::new()
    }

    /// The accuracy of the outputs of a run. Finite differences set their
    /// step from it. [`Accuracy::EXACT`] by default, for an engine whose
    /// outputs are exact up to rounding; an engine that integrates with
    /// tolerances returns them. A solver holds its tolerances at each step,
    /// not over a run, so an error built on them is an estimate, not a
    /// bound.
    fn accuracy(&self) -> Accuracy {
        Accuracy::EXACT
    }

    /// The settings that set [`Self::accuracy`], or `None` when no setting
    /// sets it (the default).
    fn accuracy_settings(&self) -> Option<AccuracySettings> {
        None
    }

    /// The same engine with its outputs at `accuracy`, both parts finite and
    /// greater than 0: its state, position and parameters stay. `None` when
    /// no setting sets its accuracy ([`Self::accuracy_settings`], the
    /// default); an engine that integrates with tolerances returns itself
    /// with those tolerances.
    fn at_accuracy(&self, _accuracy: Accuracy) -> Option<Self> {
        None
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
    ///
    /// The new seed acts from the current position
    /// ([`Self::current_position`]): the state there stays, and every draw
    /// after it comes from the new generator. A stochastic engine first
    /// settles at the position what it drew for later, such as a pending
    /// wait or leap, so two copies of one state with new seeds share
    /// nothing after it.
    fn set_seed(&mut self, _seed: u64) {}

    /// The entry of the list of engines that the engine runs
    /// ([`crate::EngineType`]), which names it in messages. None by default:
    /// an engine off the list, such as an engine of a test.
    fn engine_type(&self) -> Option<crate::EngineType> {
        None
    }

    /// Whether the engine draws random numbers ([`Self::set_seed`]), so one
    /// run is one draw of the model and a calibration or an analysis of it
    /// runs replicates on seeds its caller sets: the rule of its entry of the
    /// list ([`crate::EngineType::is_stochastic`]). An engine off the list is
    /// deterministic: it gives the same run for every seed.
    fn is_stochastic(&self) -> bool {
        self.engine_type()
            .is_some_and(crate::EngineType::is_stochastic)
    }

    /// Update a parameter value by its identifier.
    ///
    /// The new value acts from the current position
    /// ([`Self::current_position`]), never before it: a stochastic engine
    /// first settles at the position what it drew with the old value, such
    /// as a pending wait or leap. Calibration sets its parameters through
    /// this method, and checks each one with it when it is built.
    ///
    /// # Arguments
    ///
    /// * `parameter_id` - The unique identifier of the parameter (e.g., "k1", "k2")
    /// * `value` - The new value to assign
    ///
    /// # Errors
    ///
    /// Returns a message that names the parameter when the engine does not
    /// let a caller set it: an id that is not a parameter of the model, or a
    /// parameter whose value the engine writes itself, such as one an
    /// engine of the aggregate kind computes from its formula or its series
    /// at each step. The engine is unchanged after an error.
    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String>;

    /// Check that the model has every value a run needs: a parameter or a
    /// value of the kind left for calibration has one, from
    /// [`Self::set_parameter`] or [`Self::apply_values`]. An engine checks it
    /// before it runs; a caller checks it to refuse a run before it starts.
    /// Every value is given by default.
    ///
    /// # Errors
    ///
    /// Returns a message that names every value the model lacks.
    fn check_values(&self) -> Result<(), String> {
        Ok(())
    }

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
