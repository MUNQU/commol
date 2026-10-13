//! The interface every simulation engine implements, and the labels of its outputs.

use std::collections::HashMap;

/// Source and applied stratification categories of one simulation output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputLabels {
    /// Bin or accumulator the output expands from.
    pub source: String,
    /// `(stratification id, category)` pairs applied to the output, in
    /// stratification declaration order.
    pub categories: Vec<(String, String)>,
}

impl OutputLabels {
    /// Names the output answers to as a prefix: its source, then its source
    /// followed by each leading category (`A`, `A_g1`, `A_g1_s2`). The last
    /// one is the output's own name.
    pub fn prefixes(&self) -> Vec<String> {
        let mut prefix = self.source.clone();
        let mut prefixes = vec![prefix.clone()];
        for (_, category) in &self.categories {
            prefix.push('_');
            prefix.push_str(category);
            prefixes.push(prefix.clone());
        }
        prefixes
    }

    /// Category applied to the output for a stratification, if any.
    pub fn category(&self, stratification: &str) -> Option<&str> {
        self.categories
            .iter()
            .find(|(id, _)| id == stratification)
            .map(|(_, category)| category.as_str())
    }
}

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

/// Common interface that all model engines must implement
/// to be compatible with calibration and other analysis tools.
///
/// Calibration and optimization only use this trait, so they work with any
/// engine that implements it. A position on the axis is an `f64` everywhere;
/// each engine decides which positions its model type accepts
/// ([`Self::check_position`]).
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
///     fn check_position(&self, position: f64) -> Result<(), String> {
///         // Implementation
///     }
///
///     // ... implement other required methods
/// }
/// ```
pub trait SimulationEngine: Clone {
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

    /// Get the names of all compartments in order.
    ///
    /// The order must match the column order in the rows returned by `run()`
    /// and `population()`.
    ///
    /// # Example
    ///
    /// For an SIR model: `vec!["S", "I", "R"]`
    /// For a stratified model: `vec!["S_child", "S_adult", "I_child", "I_adult", ...]`
    fn compartments(&self) -> Vec<String>;

    /// Get the source and applied stratification categories of each output.
    ///
    /// The order must match `output_names()`, so the compartments come first.
    /// Calibration uses these labels to resolve initial conditions and
    /// aggregate names without parsing output names.
    fn output_labels(&self) -> Vec<OutputLabels>;

    /// Get the names of all simulation output columns.
    ///
    /// By default this is the same as `compartments()`. Engines with tracked
    /// non-population outputs, such as accumulators, should append those names here.
    fn output_names(&self) -> Vec<String> {
        self.compartments()
    }

    /// Get the current population state across all compartments.
    ///
    /// Returns a vector where `result[i]` is the population in compartment `i`,
    /// matching the order from `compartments()`.
    fn population(&self) -> Vec<f64>;

    /// Reset the simulation to its initial state.
    ///
    /// After calling this method:
    /// - `current_position()` should return 0.0
    /// - `population()` should return initial conditions
    /// - Parameters remain unchanged (use `set_parameter()` to modify them)
    ///
    /// This is essential for calibration workflows where the same model
    /// is simulated multiple times with different parameters.
    fn reset(&mut self);

    /// Update a parameter value by its identifier.
    ///
    /// # Arguments
    ///
    /// * `parameter_id` - The unique identifier of the parameter (e.g., "beta", "gamma")
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
    /// engine.set_parameter("beta", 0.3)?;
    /// engine.set_parameter("gamma", 0.1)?;
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
    /// assert_eq!(params.get("beta"), Some(&0.3));
    /// ```
    fn get_parameters(&self) -> &HashMap<String, f64>;

    /// The current position on the axis.
    ///
    /// Should return 0.0 immediately after construction or `reset()`.
    fn current_position(&self) -> f64;

    /// Set the initial condition for a specific compartment.
    ///
    /// This method allows modifying the initial population value for a compartment,
    /// which is useful for calibration workflows where initial conditions need to be
    /// optimized alongside parameters.
    ///
    /// # Arguments
    ///
    /// * `compartment_index` - The index of the compartment (matching the order from `compartments()`)
    /// * `value` - The new initial population value
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error message if the compartment index is invalid
    /// or the value is invalid (e.g., negative).
    ///
    /// # Note
    ///
    /// This method updates the stored initial conditions. The current population state
    /// is also updated to reflect the new initial value. After calling this method,
    /// `reset()` will restore the population to these new initial conditions.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // For an SIR model with compartments ["S", "I", "R"]
    /// engine.set_initial_condition(1, 10.0)?;  // Set initial I to 10
    /// engine.reset();  // Reset to new initial conditions
    /// ```
    fn set_initial_condition(&mut self, compartment_index: usize, value: f64)
    -> Result<(), String>;
}
