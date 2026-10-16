//! Calibration problem definition and implementation

use argmin::core::{CostFunction, Error};
use commol_core::{
    MathExpression, MathExpressionContext, SimulationEngine, SimulationError, ENGINE_VARIABLES,
};
use std::marker::PhantomData;
use std::sync::RwLock;

use crate::types::{
    CalibrationConstraint, CalibrationEvaluation, CalibrationParameter, CalibrationParameterType,
    LossConfig, ObservedDataPoint,
};
use crate::values::{clamp_to_bounds, configure_engine, ValueWriter};

/// Generic calibration problem that works with any SimulationEngine implementation.
///
/// This struct is model-agnostic and can work with DifferenceEquations,
/// NetworkModel, or any other future model type that implements SimulationEngine.
///
/// # Type Parameters
///
/// * `E` - The simulation engine type (must implement `SimulationEngine`)
///
/// # Example
///
/// ```rust,ignore
/// use commol_calibration::{CalibrationProblem, types::*};
/// use commol_difference::DifferenceEquations;
///
/// let engine = DifferenceEquations::from_model(&model);
/// let observed_data = vec![
///     ObservedDataPoint::new(10.0, "B".to_string(), 501.0),  // position 10, compartment B, value=501
///     ObservedDataPoint::new(20.0, "B".to_string(), 823.0),
/// ];
/// let params = vec![
///     CalibrationParameter::new("k1".to_string(), 0.0, 1.0),
///     CalibrationParameter::new("k2".to_string(), 0.0, 0.5),
/// ];
///
/// let problem = CalibrationProblem::new(
///     engine,
///     observed_data,
///     params,
///     LossConfig::SumSquaredError,
/// ).unwrap();
/// ```
pub struct CalibrationProblem<E: SimulationEngine> {
    /// Base engine used as template (cloned for each evaluation)
    base_engine: E,

    /// Observed data points to fit against
    observed_data: Vec<ObservedDataPoint>,

    /// Indices of observed outputs in the engine's output vector.
    /// Each observation may sum one or more outputs before comparison.
    observed_output_indices: Vec<Vec<usize>>,

    /// Sorted, deduplicated observation positions for sparse loss evaluation
    observed_positions: Vec<f64>,

    /// Row index in `observed_positions` for each observed data point
    observed_row_indices: Vec<usize>,

    /// Optional previous row index for windowed cumulative-output observations.
    observed_previous_row_indices: Vec<Option<usize>>,

    /// Row indices in `observed_positions` for each constraint at positions.
    constraint_row_indices: Vec<Option<Vec<usize>>>,

    /// Parameters to calibrate with their bounds
    parameters: Vec<CalibrationParameter>,

    /// How the parameters and the values of the model kind reach the engine.
    values: ValueWriter<E>,

    /// Factor parameter indices for observed data points (parallel to observed_data vec)
    /// None if no factor is applied, Some(param_idx) if a factor parameter should be applied
    observed_factor_indices: Vec<Option<usize>>,

    /// Per-series normalization factors (parallel to observed_data vec).
    ///
    /// Each entry is `1 / rms` of its series' observed values, used by the
    /// [`LossConfig::NormalizedSSE`] loss so that series of very different
    /// magnitudes contribute comparably. Always `1.0` for series whose observed
    /// values are all (near) zero. Ignored by every other loss configuration.
    observed_normalization: Vec<f64>,

    /// Loss function configuration
    loss_config: LossConfig,

    /// Pre-allocated buffer for simulation results (reused across evaluations)
    /// Wrapped in RwLock to allow thread-safe mutation in cost() method
    result_buffer: RwLock<Vec<Vec<f64>>>,

    /// Constraints on parameters and/or compartment values
    constraints: Vec<CalibrationConstraint>,

    /// Compiled constraint expressions (cached for performance)
    compiled_constraints: Vec<MathExpression>,

    /// Evaluation history tracker (wrapped in RwLock for thread-safe interior mutability)
    evaluations: RwLock<Vec<CalibrationEvaluation>>,

    /// Indices of parameter-only constraints (no positions, evaluated once before simulation)
    parameter_constraint_indices: Vec<usize>,

    /// Indices of constraints at positions (evaluated at those positions during simulation)
    position_constraint_indices: Vec<usize>,

    /// Named sums of outputs a constraint may read
    /// ([`SimulationEngine::output_sums`]), such as the total of a bin over
    /// its categories.
    output_sums: Vec<(String, Vec<usize>)>,

    /// Outputs after the state ([`SimulationEngine::state_outputs`]), such as
    /// accumulators, as `(name, row index)`. These are injected by name during
    /// constraint evaluation so they resolve without counting towards `total`.
    non_state_outputs: Vec<(String, usize)>,

    /// Phantom data for type parameter
    _phantom: PhantomData<E>,
}

// Manual Clone implementation because RwLock doesn't implement Clone
impl<E: SimulationEngine> Clone for CalibrationProblem<E> {
    fn clone(&self) -> Self {
        // Pre-allocate result buffer for the clone
        let buffer_capacity = self.observed_positions.len();
        let result_buffer = Vec::with_capacity(buffer_capacity);

        Self {
            base_engine: self.base_engine.clone(),
            observed_data: self.observed_data.clone(),
            observed_output_indices: self.observed_output_indices.clone(),
            observed_positions: self.observed_positions.clone(),
            observed_row_indices: self.observed_row_indices.clone(),
            observed_previous_row_indices: self.observed_previous_row_indices.clone(),
            constraint_row_indices: self.constraint_row_indices.clone(),
            parameters: self.parameters.clone(),
            values: self.values.clone(),
            observed_factor_indices: self.observed_factor_indices.clone(),
            observed_normalization: self.observed_normalization.clone(),
            loss_config: self.loss_config,
            result_buffer: RwLock::new(result_buffer),
            constraints: self.constraints.clone(),
            compiled_constraints: self.compiled_constraints.clone(),
            evaluations: RwLock::new(Vec::new()), // Each clone gets fresh evaluation history
            parameter_constraint_indices: self.parameter_constraint_indices.clone(),
            position_constraint_indices: self.position_constraint_indices.clone(),
            output_sums: self.output_sums.clone(),
            non_state_outputs: self.non_state_outputs.clone(),
            _phantom: PhantomData,
        }
    }
}

/// Compute per-observation normalization factors (parallel to `observed_data`).
///
/// Observations are grouped by series (`compartment` label) and each factor is
/// `1 / rms` of that series' observed values, so a residual equal to the series'
/// typical magnitude contributes comparably across series. Series whose values
/// are all (near) zero get a factor of `1.0`. When `enabled` is `false`, every
/// factor is `1.0`, leaving all loss metrics unchanged.
///
/// This must stay consistent with the Python-side `series_normalization_factors`
/// used during fit-gated ensemble selection.
fn compute_observation_normalization(
    observed_data: &[ObservedDataPoint],
    enabled: bool,
) -> Vec<f64> {
    if !enabled {
        return vec![1.0; observed_data.len()];
    }

    let mut series_sum_squares: std::collections::HashMap<&str, (f64, usize)> =
        std::collections::HashMap::new();
    for obs in observed_data {
        let entry = series_sum_squares
            .entry(obs.label.as_str())
            .or_insert((0.0, 0));
        entry.0 += obs.value * obs.value;
        entry.1 += 1;
    }

    observed_data
        .iter()
        .map(|obs| {
            let (sum_squares, count) = series_sum_squares[obs.label.as_str()];
            let rms = if count > 0 {
                (sum_squares / count as f64).sqrt()
            } else {
                0.0
            };
            if rms > f64::EPSILON {
                1.0 / rms
            } else {
                1.0
            }
        })
        .collect()
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
    /// * `loss_config` - Loss function to use
    ///
    /// # Returns
    ///
    /// Returns `Ok(CalibrationProblem)` if successful, or an error if:
    /// - Compartment names in observed data are invalid
    /// - No observed data provided
    /// - No calibration parameters provided
    /// - Constraint expressions are invalid or reference unknown variables
    /// - Outputs are referenced in constraints without positions (`at`)
    pub fn new(
        base_engine: E,
        observed_data: Vec<ObservedDataPoint>,
        parameters: Vec<CalibrationParameter>,
        constraints: Vec<CalibrationConstraint>,
        loss_config: LossConfig,
    ) -> Result<Self, String> {
        // Validate inputs
        if observed_data.is_empty() {
            return Err("No observed data provided".to_string());
        }
        check_positions(&base_engine, &observed_data, &constraints)?;

        if parameters.is_empty() {
            return Err("No calibration parameters provided".to_string());
        }

        // Build output name to index mapping for observations. Outputs include
        // state compartments and outputs outside the total, such as accumulators.
        let output_names = base_engine.output_names();
        let output_map: std::collections::HashMap<&str, usize> = output_names
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.as_str(), idx))
            .collect();

        // Named sums of outputs, so a constraint can name, for example, a bare
        // bin and get its total across categories. Indices address full output
        // rows, matching the rows that constraints at positions are evaluated
        // against.
        let output_sums = base_engine.output_sums();
        let output_sum_names: std::collections::HashSet<&str> =
            output_sums.iter().map(|(name, _)| name.as_str()).collect();

        // Validate observed output names and convert to indices.
        let mut observed_output_indices = Vec::with_capacity(observed_data.len());
        for obs in &observed_data {
            let observed_outputs: Vec<&str> = obs.outputs.iter().map(String::as_str).collect();
            if observed_outputs.is_empty() {
                return Err(format!(
                    "Observation '{}' at {} has an empty list of outputs",
                    obs.label, obs.at
                ));
            }

            let mut indices = Vec::with_capacity(observed_outputs.len());
            for output_name in observed_outputs {
                match output_map.get(output_name) {
                    Some(&idx) => indices.push(idx),
                    None => {
                        return Err(format!(
                            "Invalid observed output name '{}' for observation '{}' \
                            (available: {})",
                            output_name,
                            obs.label,
                            output_names.join(", ")
                        ));
                    }
                }
            }
            observed_output_indices.push(indices);
        }

        // Ask the engine to resolve the values of its model kind
        let values = ValueWriter::new(&base_engine, &parameters, None)?;

        // Build parameter ID to index mapping for factor lookups
        let param_id_map: std::collections::HashMap<&str, usize> = parameters
            .iter()
            .enumerate()
            .map(|(idx, param)| (param.id.as_str(), idx))
            .collect();

        // Build factor parameter indices for observed data
        let mut observed_factor_indices = Vec::with_capacity(observed_data.len());
        for obs in &observed_data {
            if let Some(ref factor_id) = obs.factor_id {
                match param_id_map.get(factor_id.as_str()) {
                    Some(&param_idx) => {
                        // Verify this parameter is actually a Factor type
                        if parameters[param_idx].parameter_type != CalibrationParameterType::Factor
                        {
                            return Err(format!(
                                "Parameter '{}' referenced as factor_id but is not a Factor parameter",
                                factor_id
                            ));
                        }
                        observed_factor_indices.push(Some(param_idx));
                    }
                    None => {
                        return Err(format!(
                            "Invalid factor_id '{}' referenced in observed data (not found in parameters)",
                            factor_id
                        ));
                    }
                }
            } else {
                observed_factor_indices.push(None);
            }
        }

        // Observation normalization is disabled by default (factors are all
        // 1.0, a no-op in every loss metric). Callers opt in via
        // `with_observation_normalization`.
        let observed_normalization = compute_observation_normalization(&observed_data, false);

        // Compile and validate constraints
        let compiled_constraints: Vec<MathExpression> = constraints
            .iter()
            .map(|c| MathExpression::new(c.expression.clone()))
            .collect();

        // Validate constraints
        let param_ids: std::collections::HashMap<&str, usize> = parameters
            .iter()
            .enumerate()
            .map(|(idx, p)| (p.id.as_str(), idx))
            .collect();

        let mut parameter_constraint_indices = Vec::new();
        let mut position_constraint_indices = Vec::new();
        let mut simulation_positions: Vec<f64> = observed_data
            .iter()
            .flat_map(|obs| std::iter::once(obs.at).chain(obs.window.map(|window| obs.at - window)))
            .collect();

        for (idx, constraint) in constraints.iter().enumerate() {
            let expr = &compiled_constraints[idx];

            // Validate expression syntax
            if let Err(e) = expr.validate() {
                return Err(format!(
                    "Constraint '{}' has invalid expression: {:?}",
                    constraint.id, e
                ));
            }

            let variables = expr.get_variables();

            // Validate that all variables are either parameters, compartments, or special constants
            for var in &variables {
                // Skip special variables
                if ENGINE_VARIABLES.contains(&var.as_str()) {
                    continue;
                }

                let is_parameter = param_ids.contains_key(var.as_str());
                // Constraints resolve against every simulation output, so they can
                // reference accumulators as well as state compartments, plus any
                // aggregate prefix of either.
                let is_compartment = output_map.contains_key(var.as_str())
                    || output_sum_names.contains(var.as_str());

                if !is_parameter && !is_compartment {
                    return Err(format!(
                        "Constraint '{}' references unknown variable '{}' (not a parameter or simulation output)",
                        constraint.id, var
                    ));
                }

                // Compartments can only be used in constraints at positions
                if is_compartment && constraint.at.is_none() {
                    return Err(format!(
                        "Constraint '{}' references output '{}' but has no positions (`at`)",
                        constraint.id, var
                    ));
                }
            }

            // Add sparse constraint positions to the simulation schedule.
            if let Some(ref positions) = constraint.at {
                simulation_positions.extend(positions.iter().copied());
                position_constraint_indices.push(idx);
            } else {
                parameter_constraint_indices.push(idx);
            }
        }

        simulation_positions.sort_by(f64::total_cmp);
        simulation_positions.dedup();
        let position_index = |position: f64| {
            simulation_positions
                .binary_search_by(|step| step.total_cmp(&position))
                .ok()
        };

        let observed_row_indices: Vec<usize> = observed_data
            .iter()
            .map(|obs| position_index(obs.at).expect("observed position must be present"))
            .collect();

        let observed_previous_row_indices: Vec<Option<usize>> = observed_data
            .iter()
            .map(|obs| {
                obs.window.map(|window| {
                    position_index(obs.at - window).expect("window start must be present")
                })
            })
            .collect();

        let constraint_row_indices: Vec<Option<Vec<usize>>> = constraints
            .iter()
            .map(|constraint| {
                constraint.at.as_ref().map(|positions| {
                    positions
                        .iter()
                        .map(|position| {
                            position_index(*position).expect("constraint position must be present")
                        })
                        .collect()
                })
            })
            .collect();

        // Pre-allocate result buffer for performance
        let buffer_capacity = simulation_positions.len();
        let result_buffer = Vec::with_capacity(buffer_capacity);

        // Outputs past the state, such as accumulators; keep their row indices
        // so constraints can name them individually.
        let non_state_outputs: Vec<(String, usize)> = output_names
            .iter()
            .enumerate()
            .skip(base_engine.state_outputs())
            .map(|(idx, name)| (name.clone(), idx))
            .collect();

        Ok(Self {
            base_engine,
            observed_data,
            observed_output_indices,
            observed_positions: simulation_positions,
            observed_row_indices,
            observed_previous_row_indices,
            constraint_row_indices,
            parameters,
            values,
            observed_factor_indices,
            observed_normalization,
            loss_config,
            result_buffer: RwLock::new(result_buffer),
            constraints,
            compiled_constraints,
            evaluations: RwLock::new(Vec::new()),
            parameter_constraint_indices,
            position_constraint_indices,
            output_sums,
            non_state_outputs,
            _phantom: PhantomData,
        })
    }

    /// Enable or disable per-series observation normalization.
    ///
    /// When enabled, each residual is multiplied by `1 / rms` of its series'
    /// observed values before the configured loss metric is applied, so that
    /// series measured on very different scales contribute comparably to the
    /// loss. This is orthogonal to the choice of metric and applies to all of them.
    ///
    /// Disabled by default. When disabled, the factors are all `1.0`, leaving
    /// every metric unchanged.
    pub fn with_observation_normalization(mut self, enabled: bool) -> Self {
        self.observed_normalization =
            compute_observation_normalization(&self.observed_data, enabled);
        self
    }

    /// Name the id that takes what the calibrated initial conditions leave of
    /// the total the model kind keeps; the engine resolves it.
    ///
    /// For an aggregate model it is a compartment or bin that absorbs the
    /// difference to the total. Without a remainder, calibrated initial
    /// conditions are applied as given and only a sum above the total is
    /// penalized. With one, the sum of the initial amounts always equals the
    /// total, and a combination that would leave the remainder negative is
    /// penalized.
    ///
    /// # Errors
    ///
    /// Returns the engine's message when the id does not resolve, for example
    /// when it is not a compartment, bin or bin prefix, or overlaps a
    /// calibrated bin or compartment.
    pub fn with_initial_condition_remainder(mut self, remainder: &str) -> Result<Self, String> {
        self.values = ValueWriter::new(&self.base_engine, &self.parameters, Some(remainder))?;
        Ok(self)
    }

    /// Get the number of parameters being calibrated
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// Get parameter names in order
    pub fn parameter_names(&self) -> Vec<String> {
        self.parameters.iter().map(|p| p.id.clone()).collect()
    }

    /// Get initial parameter values
    pub fn initial_parameters(&self) -> Vec<f64> {
        self.parameters.iter().map(|p| p.initial_value()).collect()
    }

    /// Get parameter bounds as (min, max) tuples
    pub fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.parameters
            .iter()
            .map(|p| (p.min_bound, p.max_bound))
            .collect()
    }

    /// Get all recorded objective function evaluations
    pub fn get_evaluations(&self) -> Vec<CalibrationEvaluation> {
        self.evaluations
            .read()
            .expect("Failed to acquire read lock on evaluations")
            .clone()
    }

    /// Clear the evaluation history (useful when reusing the problem)
    pub fn clear_evaluations(&self) {
        self.evaluations
            .write()
            .expect("Failed to acquire write lock on evaluations")
            .clear();
    }

    /// Get parameter types for external use
    pub fn get_parameter_types(&self) -> Vec<CalibrationParameterType> {
        self.parameters.iter().map(|p| p.parameter_type).collect()
    }

    /// Values the objective applies for a parameter vector: each value
    /// clamped to its bounds.
    ///
    /// Optimizers may explore outside the bounds; this is the vector that was
    /// actually scored, so it is the one to report.
    pub fn applied_parameter_values(&self, param_values: &[f64]) -> Vec<f64> {
        clamp_to_bounds(&self.parameters, param_values)
    }

    /// Calculate loss from a sparse result matrix whose rows are aligned with
    /// `self.observed_positions`.
    fn calculate_sparse_loss(&self, observed_results: &[Vec<f64>], param_values: &[f64]) -> f64 {
        let observation_iter = || {
            self.observed_data
                .iter()
                .zip(&self.observed_output_indices)
                .zip(&self.observed_factor_indices)
                .zip(&self.observed_row_indices)
                .zip(&self.observed_previous_row_indices)
                .zip(&self.observed_normalization)
                .filter_map(
                    |(
                        ((((obs, output_indices), &factor_idx), &row_idx), &previous_row_idx),
                        &normalization,
                    )| {
                        observed_results.get(row_idx).map(|step_data| {
                            let current: f64 =
                                output_indices.iter().map(|&idx| step_data[idx]).sum();
                            let predicted = if let Some(previous_row_idx) = previous_row_idx {
                                let previous: f64 = observed_results
                                    .get(previous_row_idx)
                                    .map(|row| {
                                        output_indices
                                            .iter()
                                            .filter_map(|&idx| row.get(idx))
                                            .copied()
                                            .sum()
                                    })
                                    .unwrap_or(0.0);
                                current - previous
                            } else {
                                current
                            };
                            let scaled_predicted = if let Some(param_idx) = factor_idx {
                                predicted * param_values[param_idx]
                            } else {
                                predicted
                            };
                            (obs, scaled_predicted, normalization)
                        })
                    },
                )
        };

        // `normalization` is the per-series factor (1.0 unless observation
        // normalization is enabled). It scales every residual, so series of very
        // different magnitudes contribute comparably to any loss. The weight is
        // applied by `LossConfig::term`.
        self.loss_config
            .aggregate(observation_iter().map(|(obs, predicted, normalization)| {
                ((obs.value - predicted) * normalization, obs.weight)
            }))
    }

    /// Clamp parameter values to their defined bounds
    ///
    /// This is necessary because some optimization algorithms
    /// can explore outside the bounds during their search process. By clamping,
    /// we ensure the simulation always receives valid parameter values while
    /// still allowing the optimizer to explore the parameter space freely.
    fn clamp_to_bounds(&self, param_values: &[f64]) -> Vec<f64> {
        clamp_to_bounds(&self.parameters, param_values)
    }

    /// Validate parameter vector length
    fn validate_parameter_count(&self, param_values: &[f64]) -> Result<(), String> {
        if param_values.len() != self.parameters.len() {
            return Err(format!(
                "Expected {} parameters, got {}",
                self.parameters.len(),
                param_values.len()
            ));
        }
        Ok(())
    }

    /// Build an engine clone with calibration parameters applied exactly as in
    /// objective evaluation, including calibrated initial conditions.
    pub fn configured_engine(&self, param_values: &[f64]) -> Result<E, String> {
        configure_engine(
            &self.base_engine,
            &self.parameters,
            &self.values,
            param_values,
        )
        .map(|(engine, _)| engine)
    }

    /// Calculate base penalty value for constraint violations
    ///
    /// This is used to scale penalties relative to the loss function magnitude
    fn calculate_base_penalty(&self) -> f64 {
        let max_observed = self
            .observed_data
            .iter()
            .map(|obs| obs.value)
            .fold(0.0f64, |a, b| a.max(b));
        let num_obs = self.observed_data.len() as f64;
        (max_observed * max_observed * num_obs * 1000.0).max(1e10)
    }

    /// Evaluate parameter-only constraints (no positions)
    ///
    /// These constraints can only reference calibration parameters, not compartment values.
    /// Evaluated once before simulation starts.
    ///
    /// Returns the total penalty from violated constraints, or 0.0 if all are satisfied.
    fn evaluate_parameter_constraints(&self, param_values: &[f64]) -> Result<f64, String> {
        if self.parameter_constraint_indices.is_empty() {
            return Ok(0.0);
        }

        // Create context with parameter values
        let mut context = MathExpressionContext::new();
        for (idx, param) in self.parameters.iter().enumerate() {
            context.set_parameter(param.id.clone(), param_values[idx]);
        }

        let mut total_penalty = 0.0;

        for &constraint_idx in &self.parameter_constraint_indices {
            let constraint = &self.constraints[constraint_idx];
            let expr = &self.compiled_constraints[constraint_idx];

            match expr.evaluate(&mut context) {
                Ok(value) => {
                    if value < 0.0 {
                        // Constraint violated
                        let violation_magnitude = -value;

                        // Linear penalty scaled by weight
                        let penalty = constraint.weight * violation_magnitude;

                        total_penalty += penalty;
                    }
                    // value >= 0 means constraint satisfied, no penalty
                }
                Err(e) => {
                    return Err(format!(
                        "Parameter constraint '{}' evaluation failed: {:?}",
                        constraint.id, e
                    ));
                }
            }
        }

        Ok(total_penalty)
    }

    /// Evaluate constraints at positions (with `at`)
    ///
    /// These constraints can reference both calibration parameters and compartment values.
    /// Evaluated at each specified position after simulation completes.
    ///
    /// Returns the total penalty from violated constraints, or 0.0 if all are satisfied.
    fn evaluate_position_constraints(
        &self,
        param_values: &[f64],
        simulation_results: &[Vec<f64>],
    ) -> Result<f64, String> {
        if self.position_constraint_indices.is_empty() {
            return Ok(0.0);
        }

        // Create context with parameter values
        let mut context = MathExpressionContext::new();

        // Only the state outputs are registered, so `total` stays the sum of the
        // state. Outputs after it, such as accumulators, are injected by name
        // below, alongside the sums.
        let mut state_names = self.base_engine.output_names();
        state_names.truncate(self.base_engine.state_outputs());
        context.init_compartments(state_names);

        // Set parameter values
        for (idx, param) in self.parameters.iter().enumerate() {
            context.set_parameter(param.id.clone(), param_values[idx]);
        }

        let mut total_penalty = 0.0;

        for &constraint_idx in &self.position_constraint_indices {
            let constraint = &self.constraints[constraint_idx];
            let expr = &self.compiled_constraints[constraint_idx];

            let positions = constraint
                .at
                .as_ref()
                .expect("a constraint at positions has positions");
            let row_indices = self.constraint_row_indices[constraint_idx]
                .as_ref()
                .expect("a constraint at positions has row indices");

            for (&position, &row_idx) in positions.iter().zip(row_indices) {
                // Get compartment values at this position
                if let Some(step_data) = simulation_results.get(row_idx) {
                    // Update context with compartment values at this position
                    context.set_compartments_by_index(step_data);
                    // Outputs after the state are not registered (they must not
                    // count towards `total`), so inject them by name.
                    for (name, idx) in &self.non_state_outputs {
                        context.set_parameter_str(name, step_data[*idx]);
                    }
                    // Inject the sums, overriding any calibration parameter with
                    // the same name.
                    for (name, indices) in &self.output_sums {
                        let total: f64 = indices.iter().map(|&i| step_data[i]).sum();
                        context.set_parameter_str(name, total);
                    }
                    context.set_step(position);

                    match expr.evaluate(&mut context) {
                        Ok(value) => {
                            if value < 0.0 {
                                // Constraint violated at this position
                                let violation_magnitude = -value;

                                // Linear penalty scaled by weight
                                let penalty = constraint.weight * violation_magnitude;

                                total_penalty += penalty;
                            }
                        }
                        Err(e) => {
                            return Err(format!(
                                "Constraint '{}' evaluation failed at position {}: {:?}",
                                constraint.id, position, e
                            ));
                        }
                    }
                } else {
                    return Err(format!(
                        "Position {} not found in simulation results for constraint '{}'",
                        position, constraint.id
                    ));
                }
            }
        }

        Ok(total_penalty)
    }
}

/// Implement argmin's CostFunction trait - model-agnostic implementation
///
/// This works with any model type that implements SimulationEngine.
impl<E: SimulationEngine> CostFunction for CalibrationProblem<E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param_values: &Self::Param) -> Result<Self::Output, Error> {
        // Validate parameter count
        self.validate_parameter_count(param_values)
            .map_err(Error::msg)?;

        // Clamp parameters to bounds (handles optimizers that explore outside bounds)
        let clamped_params = self.clamp_to_bounds(param_values);

        // Evaluate parameter-only constraints before running simulation
        let param_constraint_penalty = self
            .evaluate_parameter_constraints(&clamped_params)
            .map_err(Error::msg)?;

        if param_constraint_penalty > 0.0 {
            // Parameter constraints violated - skip simulation and return penalty
            let base_penalty = self.calculate_base_penalty();
            return Ok(base_penalty + param_constraint_penalty * base_penalty);
        }

        // Clone the base engine (works for any model type)
        let mut engine = self.base_engine.clone();

        // Reset engine to initial conditions
        engine.reset();

        // Update parameters and the values of the model kind
        let excess = self
            .values
            .apply(&mut engine, &self.parameters, &clamped_params)
            .map_err(Error::msg)?;

        if excess > 0.0 {
            // The values make the starting state infeasible
            let base_penalty = self.calculate_base_penalty();
            return Ok(base_penalty * (1.0 + excess * 100.0));
        }

        // Run simulation using pre-allocated buffer to avoid allocations
        let mut buffer = self
            .result_buffer
            .write()
            .expect("Failed to acquire write lock on result_buffer");

        match engine.run_at_positions_into_buffer(&self.observed_positions, &mut buffer) {
            Ok(()) => {}
            // The parameters leave the values the model allows: a point to
            // avoid, not a reason to stop.
            Err(SimulationError::Infeasible { .. }) => return Ok(self.calculate_base_penalty()),
            Err(SimulationError::Failed(error)) => {
                return Err(Error::msg(format!("Simulation failed: {error}")));
            }
        }

        // Check for numerical instability (NaN or infinity values)
        let has_invalid_values = buffer
            .iter()
            .any(|step| step.iter().any(|&value| !value.is_finite()));

        if has_invalid_values {
            let base_penalty = self.calculate_base_penalty();
            return Ok(base_penalty);
        }

        let loss = if self.position_constraint_indices.is_empty() {
            self.calculate_sparse_loss(&buffer, &clamped_params)
        } else {
            // Evaluate constraints at positions using simulation results
            let position_constraint_penalty = self
                .evaluate_position_constraints(&clamped_params, &buffer)
                .map_err(Error::msg)?;

            if position_constraint_penalty > 0.0 {
                // Compartment value constraints violated - return penalty
                let base_penalty = self.calculate_base_penalty();
                return Ok(base_penalty + position_constraint_penalty * base_penalty);
            }

            // Calculate and return loss (use corrected params for scale parameters)
            self.calculate_sparse_loss(&buffer, &clamped_params)
        };

        // Check if loss itself is invalid (defensive programming)
        if !loss.is_finite() {
            let base_penalty = self.calculate_base_penalty();
            return Ok(base_penalty);
        }

        // Record this evaluation in the history (use corrected params)
        let evaluation = CalibrationEvaluation {
            parameters: clamped_params.clone(),
            loss,
            predictions: vec![], // Predictions are generated later in Python
        };
        self.evaluations
            .write()
            .expect("Failed to acquire write lock on evaluations")
            .push(evaluation);

        Ok(loss)
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
            let start = obs.at - window;
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
