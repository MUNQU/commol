//! The interface every simulation engine implements, whatever the kind of its
//! model.

use std::collections::HashMap;
use std::fmt::Debug;

use crate::types::ModelValueType;

/// Outputs of a run and the positions on the axis they were reported at.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    /// Positions on the axis, ascending; `rows[i]` is the state at `positions[i]`.
    pub positions: Vec<f64>,
    /// One output row per position, in the order of `output_names()`.
    pub rows: Vec<Vec<f64>>,
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
/// use commol_core::{SimulationEngine};
/// use std::collections::HashMap;
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
    /// Returns a message when a position is rejected by
    /// [`Self::check_position`], when the positions are not sorted and unique
    /// or one is before the current position (see [`check_positions`]), or
    /// when the simulation fails.
    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String>;

    /// Run through the last of `positions` and return the outputs at each of
    /// them. See [`Self::run_at_positions_into_buffer`].
    ///
    /// # Errors
    ///
    /// See [`Self::run_at_positions_into_buffer`].
    fn run_at(&mut self, positions: &[f64]) -> Result<Vec<Vec<f64>>, String> {
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
    /// Returns an error string if the simulation fails (e.g., invalid parameters,
    /// numerical instability, etc.)
    fn run(&mut self, num_steps: u32) -> Result<Trajectory, String> {
        let positions = self.grid(num_steps);
        let rows = self.run_at(&positions)?;
        Ok(Trajectory { positions, rows })
    }

    /// Advance the state by one unit of the axis.
    ///
    /// # Errors
    ///
    /// Returns an error string if the step fails
    fn step(&mut self) -> Result<(), String>;

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
    fn reset(&mut self);

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

    /// Get all current parameter values.
    ///
    /// Returns a reference to a hashmap mapping parameter IDs to their current values.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let params = engine.get_parameters();
    /// assert_eq!(params.get("k1"), Some(&0.3));
    /// ```
    fn get_parameters(&self) -> &HashMap<String, f64>;

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
