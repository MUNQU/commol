use std::collections::HashMap;

/// Common interface that all model engines must implement
/// to be compatible with calibration and other analysis tools.
///
/// This trait provides a model-agnostic abstraction layer that allows
/// different model types (DifferenceEquations, NetworkModel, StochasticModel, etc.)
/// to work seamlessly with the same calibration and optimization code.
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
///     fn run(&mut self, num_steps: u32) -> Result<Vec<Vec<f64>>, String> {
///         // Implementation
///     }
///
///     // ... implement other required methods
/// }
/// ```
pub trait SimulationEngine: Clone {
    /// Run the simulation for a given number of steps.
    ///
    /// Returns a matrix where:
    /// - First dimension (rows): time steps (0 to num_steps inclusive)
    /// - Second dimension (columns): compartment values
    ///
    /// # Arguments
    ///
    /// * `num_steps` - Number of simulation steps to execute
    ///
    /// # Returns
    ///
    /// A vector of vectors where `result[t][c]` is the population in compartment `c`
    /// at time step `t`. The result includes the initial state (t=0) plus all steps,
    /// so the length is `num_steps + 1`.
    ///
    /// # Errors
    ///
    /// Returns an error string if the simulation fails (e.g., invalid parameters,
    /// numerical instability, etc.)
    fn run(&mut self, num_steps: u32) -> Result<Vec<Vec<f64>>, String>;

    /// Execute a single simulation step, advancing the model state by one time unit.
    ///
    /// # Errors
    ///
    /// Returns an error string if the step fails
    fn step(&mut self) -> Result<(), String>;

    /// Get the names of all compartments in order.
    ///
    /// The order must match the column order in the matrices returned by `run()`
    /// and `population()`.
    ///
    /// # Example
    ///
    /// For an SIR model: `vec!["S", "I", "R"]`
    /// For a stratified model: `vec!["S_child", "S_adult", "I_child", "I_adult", ...]`
    fn compartments(&self) -> Vec<String>;

    /// Get the current population state across all compartments.
    ///
    /// Returns a vector where `result[i]` is the population in compartment `i`,
    /// matching the order from `compartments()`.
    fn population(&self) -> Vec<f64>;

    /// Reset the simulation to its initial state.
    ///
    /// After calling this method:
    /// - `current_step()` should return 0.0
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

    /// Get the current simulation time step.
    ///
    /// For discrete-time models, this is typically an integer counter.
    /// For continuous-time models, this represents the current time value.
    ///
    /// Should return 0.0 immediately after construction or `reset()`.
    fn current_step(&self) -> f64;

    /// Run the simulation for a given number of steps, writing results into a pre-allocated buffer.
    ///
    /// This is a performance optimization for calibration that avoids repeated allocations.
    /// The buffer should be pre-allocated with capacity for `num_steps + 1` rows and
    /// `num_compartments` columns.
    ///
    /// # Arguments
    ///
    /// * `num_steps` - Number of simulation steps to execute
    /// * `buffer` - Pre-allocated buffer to write results into. Will be cleared and resized if needed.
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, error string otherwise
    ///
    /// # Default Implementation
    ///
    /// The default implementation calls `run()` and copies the result. Engines should
    /// override this for better performance.
    fn run_into_buffer(
        &mut self,
        num_steps: u32,
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        let results = self.run(num_steps)?;
        buffer.clear();
        buffer.extend(results);
        Ok(())
    }

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
