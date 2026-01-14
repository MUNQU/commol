//! Calibration problem definition and implementation

use argmin::core::{CostFunction, Error};
use commol_core::{MathExpression, MathExpressionContext, SimulationEngine};
use std::marker::PhantomData;
use std::sync::RwLock;

use crate::types::{
    CalibrationConstraint, CalibrationEvaluation, CalibrationParameter, CalibrationParameterType,
    LossConfig, ObservedDataPoint,
};

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
///     ObservedDataPoint::new(10, "I".to_string(), 501.0),  // time=10, compartment I, value=501
///     ObservedDataPoint::new(20, "I".to_string(), 823.0),
/// ];
/// let params = vec![
///     CalibrationParameter::new("beta".to_string(), 0.0, 1.0),
///     CalibrationParameter::new("gamma".to_string(), 0.0, 0.5),
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

    /// Indices of observed compartments in the engine's compartment vector
    observed_compartment_indices: Vec<usize>,

    /// Parameters to calibrate with their bounds
    parameters: Vec<CalibrationParameter>,

    /// Compartment indices for initial condition parameters (parallel to parameters vec)
    /// None for Parameter type, Some(index) for InitialCondition type
    parameter_compartment_indices: Vec<Option<usize>>,

    /// Scale parameter indices for observed data points (parallel to observed_data vec)
    /// None if no scale is applied, Some(param_idx) if a scale parameter should be applied
    observed_scale_indices: Vec<Option<usize>>,

    /// Loss function configuration
    loss_config: LossConfig,

    /// Maximum time step in observed data (cached for performance)
    max_time_step: u32,

    /// Pre-allocated buffer for simulation results (reused across evaluations)
    /// Wrapped in RwLock to allow thread-safe mutation in cost() method
    result_buffer: RwLock<Vec<Vec<f64>>>,

    /// Initial population size for converting fractions to absolute values
    /// From the model's defined initial_population_size
    initial_population_size: f64,

    /// Constraints on parameters and/or compartment values
    constraints: Vec<CalibrationConstraint>,

    /// Compiled constraint expressions (cached for performance)
    compiled_constraints: Vec<MathExpression>,

    /// Evaluation history tracker (wrapped in RwLock for thread-safe interior mutability)
    evaluations: RwLock<Vec<CalibrationEvaluation>>,

    /// Indices of parameter-only constraints (no time_steps, evaluated once before simulation)
    parameter_constraint_indices: Vec<usize>,

    /// Indices of time-dependent constraints (with time_steps, evaluated at specified times during simulation)
    time_dependent_constraint_indices: Vec<usize>,

    /// Phantom data for type parameter
    _phantom: PhantomData<E>,
}

// Manual Clone implementation because RwLock doesn't implement Clone
impl<E: SimulationEngine> Clone for CalibrationProblem<E> {
    fn clone(&self) -> Self {
        // Pre-allocate result buffer for the clone
        let buffer_capacity = (self.max_time_step + 1) as usize;
        let result_buffer = Vec::with_capacity(buffer_capacity);

        Self {
            base_engine: self.base_engine.clone(),
            observed_data: self.observed_data.clone(),
            observed_compartment_indices: self.observed_compartment_indices.clone(),
            parameters: self.parameters.clone(),
            parameter_compartment_indices: self.parameter_compartment_indices.clone(),
            observed_scale_indices: self.observed_scale_indices.clone(),
            loss_config: self.loss_config,
            max_time_step: self.max_time_step,
            result_buffer: RwLock::new(result_buffer),
            initial_population_size: self.initial_population_size,
            constraints: self.constraints.clone(),
            compiled_constraints: self.compiled_constraints.clone(),
            evaluations: RwLock::new(Vec::new()), // Each clone gets fresh evaluation history
            parameter_constraint_indices: self.parameter_constraint_indices.clone(),
            time_dependent_constraint_indices: self.time_dependent_constraint_indices.clone(),
            _phantom: PhantomData,
        }
    }
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
    /// - Compartments are referenced in constraints without time_steps specified
    pub fn new(
        base_engine: E,
        observed_data: Vec<ObservedDataPoint>,
        parameters: Vec<CalibrationParameter>,
        constraints: Vec<CalibrationConstraint>,
        loss_config: LossConfig,
        initial_population_size: u64,
    ) -> Result<Self, String> {
        // Validate inputs
        if observed_data.is_empty() {
            return Err("No observed data provided".to_string());
        }

        if parameters.is_empty() {
            return Err("No calibration parameters provided".to_string());
        }

        // Build compartment name to index mapping
        let compartments = base_engine.compartments();
        let compartment_map: std::collections::HashMap<&str, usize> = compartments
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.as_str(), idx))
            .collect();

        // Validate compartment names and convert to indices
        let mut observed_compartment_indices = Vec::with_capacity(observed_data.len());
        for obs in &observed_data {
            match compartment_map.get(obs.compartment.as_str()) {
                Some(&idx) => observed_compartment_indices.push(idx),
                None => {
                    return Err(format!(
                        "Invalid compartment name '{}' (available: {})",
                        obs.compartment,
                        compartments.join(", ")
                    ));
                }
            }
        }

        // Build compartment indices for calibration parameters
        let mut parameter_compartment_indices = Vec::with_capacity(parameters.len());
        for param in &parameters {
            match param.parameter_type {
                CalibrationParameterType::Parameter => {
                    // No compartment index needed for regular parameters
                    parameter_compartment_indices.push(None);
                }
                CalibrationParameterType::InitialCondition => {
                    // Look up compartment index by bin ID
                    match compartment_map.get(param.id.as_str()) {
                        Some(&idx) => parameter_compartment_indices.push(Some(idx)),
                        None => {
                            return Err(format!(
                                "Invalid bin ID '{}' for initial condition calibration
                                (available: {})",
                                param.id,
                                compartments.join(", ")
                            ));
                        }
                    }
                }
                CalibrationParameterType::Scale => {
                    // No compartment index needed for scale parameters
                    parameter_compartment_indices.push(None);
                }
            }
        }

        // Build parameter ID to index mapping for scale lookups
        let param_id_map: std::collections::HashMap<&str, usize> = parameters
            .iter()
            .enumerate()
            .map(|(idx, param)| (param.id.as_str(), idx))
            .collect();

        // Build scale parameter indices for observed data
        let mut observed_scale_indices = Vec::with_capacity(observed_data.len());
        for obs in &observed_data {
            if let Some(ref scale_id) = obs.scale_id {
                match param_id_map.get(scale_id.as_str()) {
                    Some(&param_idx) => {
                        // Verify this parameter is actually a Scale type
                        if parameters[param_idx].parameter_type != CalibrationParameterType::Scale {
                            return Err(format!(
                                "Parameter '{}' referenced as scale_id but is not a Scale parameter",
                                scale_id
                            ));
                        }
                        observed_scale_indices.push(Some(param_idx));
                    }
                    None => {
                        return Err(format!(
                            "Invalid scale_id '{}' referenced in observed data (not found in parameters)",
                            scale_id
                        ));
                    }
                }
            } else {
                observed_scale_indices.push(None);
            }
        }

        // Find maximum time step to avoid recomputing in each cost evaluation
        let max_time_step = observed_data
            .iter()
            .map(|obs| obs.time_step)
            .max()
            .unwrap_or(100);

        // Pre-allocate result buffer for performance
        let buffer_capacity = (max_time_step + 1) as usize;
        let result_buffer = Vec::with_capacity(buffer_capacity);

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
        let mut time_dependent_constraint_indices = Vec::new();

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
                if var == "N" || var == "step" || var == "t" || var == "pi" || var == "e" {
                    continue;
                }

                let is_parameter = param_ids.contains_key(var.as_str());
                let is_compartment = compartment_map.contains_key(var.as_str());

                if !is_parameter && !is_compartment {
                    return Err(format!(
                        "Constraint '{}' references unknown variable '{}' (not a parameter or compartment)",
                        constraint.id, var
                    ));
                }

                // Compartments can only be used in time-dependent constraints
                if is_compartment && constraint.time_steps.is_none() {
                    return Err(format!(
                        "Constraint '{}' references compartment '{}' but has no time_steps specified",
                        constraint.id, var
                    ));
                }
            }

            // Validate time steps are within simulation range
            if let Some(ref time_steps) = constraint.time_steps {
                for &ts in time_steps {
                    if ts > max_time_step {
                        return Err(format!(
                            "Constraint '{}' has time step {} exceeding max observed time step {}",
                            constraint.id, ts, max_time_step
                        ));
                    }
                }
                time_dependent_constraint_indices.push(idx);
            } else {
                parameter_constraint_indices.push(idx);
            }
        }

        Ok(Self {
            base_engine,
            observed_data,
            observed_compartment_indices,
            parameters,
            parameter_compartment_indices,
            observed_scale_indices,
            loss_config,
            max_time_step,
            result_buffer: RwLock::new(result_buffer),
            initial_population_size: initial_population_size as f64,
            constraints,
            compiled_constraints,
            evaluations: RwLock::new(Vec::new()),
            parameter_constraint_indices,
            time_dependent_constraint_indices,
            _phantom: PhantomData,
        })
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

    /// Calculate auto-corrected parameter values for initial conditions
    ///
    /// This is the single source of truth for auto-IC calculation logic.
    /// When calibrating initial conditions, one IC parameter may be auto-calculated
    /// to ensure all fractions sum to 1.0.
    ///
    /// # Arguments
    /// * `param_values` - Raw parameter values from the optimizer
    ///
    /// # Returns
    /// Tuple of (corrected_params, fixed_initial_conditions_sum, calibrated_initial_conditions_sum_excluding_auto)
    fn calculate_auto_corrected_parameters(&self, param_values: &[f64]) -> (Vec<f64>, f64, f64) {
        // Identify which compartments are being calibrated
        let num_compartments = self.base_engine.compartments().len();
        let mut calibrated_compartments = vec![false; num_compartments];

        for (param, compartment_idx) in self
            .parameters
            .iter()
            .zip(&self.parameter_compartment_indices)
        {
            if param.parameter_type == CalibrationParameterType::InitialCondition {
                if let Some(idx) = compartment_idx {
                    calibrated_compartments[*idx] = true;
                }
            }
        }

        // Calculate sum of fixed (non-calibrated) initial condition fractions
        let current_population = self.base_engine.population();
        let fixed_initial_conditions_sum: f64 = current_population
            .iter()
            .enumerate()
            .filter(|(idx, _)| !calibrated_compartments[*idx])
            .map(|(_, &val)| val / self.initial_population_size)
            .sum();

        // Determine which initial condition parameter should be auto-calculated
        let initial_conditions_params_indices: Vec<usize> = self
            .parameters
            .iter()
            .enumerate()
            .filter(|(_, param)| param.parameter_type == CalibrationParameterType::InitialCondition)
            .map(|(idx, _)| idx)
            .collect();

        let num_initial_conditions_params = initial_conditions_params_indices.len();
        let all_compartments_are_initial_conditions =
            num_initial_conditions_params == num_compartments;

        // Auto-calculate the last initial condition parameter if:
        // 1. All compartments are initial conditions AND there are 2+ initial condition parameters, OR
        // 2. There's exactly 1 initial condition parameter with fixed compartments (sum constraint)
        let auto_calc_initial_conditions_idx = if (num_initial_conditions_params >= 2
            && all_compartments_are_initial_conditions)
            || (num_initial_conditions_params == 1 && !all_compartments_are_initial_conditions)
        {
            initial_conditions_params_indices.last().copied()
        } else {
            None
        };

        // Calculate sum of calibrated initial conditions (excluding auto-calculated one)
        let calibrated_initial_conditions_sum: f64 = param_values
            .iter()
            .enumerate()
            .filter(|(idx, _)| {
                self.parameters[*idx].parameter_type == CalibrationParameterType::InitialCondition
                    && Some(*idx) != auto_calc_initial_conditions_idx
            })
            .map(|(_, value)| value)
            .sum();

        // Apply auto-calculation if needed
        let mut corrected_params = param_values.to_vec();
        if let Some(idx) = auto_calc_initial_conditions_idx {
            let auto_calculated_value =
                (1.0 - fixed_initial_conditions_sum - calibrated_initial_conditions_sum).max(0.0);
            corrected_params[idx] = auto_calculated_value;
        }

        (
            corrected_params,
            fixed_initial_conditions_sum,
            calibrated_initial_conditions_sum,
        )
    }

    /// Get parameter types for external use
    pub fn get_parameter_types(&self) -> Vec<CalibrationParameterType> {
        self.parameters.iter().map(|p| p.parameter_type).collect()
    }

    /// Get information needed for parameter correction (used by python_observer)
    ///
    /// Returns (fixed_initial_conditions_sum, auto_calc_initial_conditions_idx, param_types)
    /// This is a convenience method that extracts metadata without requiring parameter values.
    pub fn get_parameter_fix_info(&self) -> (f64, Option<usize>, Vec<CalibrationParameterType>) {
        // Calculate fixed initial conditions sum using empty params (we only need the structure)
        let dummy_params = vec![0.0; self.parameters.len()];
        let (_, fixed_initial_conditions_sum, _) =
            self.calculate_auto_corrected_parameters(&dummy_params);

        // Determine auto-calc index
        let num_compartments = self.base_engine.compartments().len();
        let initial_conditions_params_indices: Vec<usize> = self
            .parameters
            .iter()
            .enumerate()
            .filter(|(_, param)| param.parameter_type == CalibrationParameterType::InitialCondition)
            .map(|(idx, _)| idx)
            .collect();

        let num_initial_conditions_params = initial_conditions_params_indices.len();
        let all_compartments_are_initial_conditions =
            num_initial_conditions_params == num_compartments;

        let auto_calc_initial_conditions_idx = if (num_initial_conditions_params >= 2
            && all_compartments_are_initial_conditions)
            || (num_initial_conditions_params == 1 && !all_compartments_are_initial_conditions)
        {
            initial_conditions_params_indices.last().copied()
        } else {
            None
        };

        let param_types = self.get_parameter_types();

        (
            fixed_initial_conditions_sum,
            auto_calc_initial_conditions_idx,
            param_types,
        )
    }

    /// Fix auto-calculated initial condition parameters in the result
    ///
    /// This is a public wrapper around calculate_auto_corrected_parameters
    /// for use by optimizers that need to correct final results.
    ///
    /// # Arguments
    /// * `param_values` - Parameter values from the optimizer
    ///
    /// # Returns
    /// Corrected parameter values with auto-calculated ICs fixed
    pub fn fix_auto_calculated_parameters(&self, param_values: Vec<f64>) -> Vec<f64> {
        let (corrected, _, _) = self.calculate_auto_corrected_parameters(&param_values);
        corrected
    }

    /// Calculate loss between simulation results and observed data
    fn calculate_loss(&self, simulation_results: &[Vec<f64>], param_values: &[f64]) -> f64 {
        let observation_iter = || {
            self.observed_data
                .iter()
                .zip(&self.observed_compartment_indices)
                .zip(&self.observed_scale_indices)
                .filter_map(|((obs, &compartment_idx), &scale_idx)| {
                    let time_idx = obs.time_step as usize;
                    simulation_results.get(time_idx).map(|step_data| {
                        let predicted = step_data[compartment_idx];
                        // Apply scale if present
                        let scaled_predicted = if let Some(param_idx) = scale_idx {
                            predicted * param_values[param_idx]
                        } else {
                            predicted
                        };
                        (obs, scaled_predicted)
                    })
                })
        };

        match self.loss_config {
            LossConfig::SumSquaredError | LossConfig::WeightedSSE => observation_iter()
                .map(|(obs, predicted)| {
                    let error = (obs.value - predicted) * obs.weight;
                    error * error
                })
                .sum(),

            LossConfig::RootMeanSquaredError => {
                let (sum_squared_error, count) = observation_iter()
                    .map(|(obs, predicted)| {
                        let error = obs.value - predicted;
                        error * error
                    })
                    .fold((0.0, 0), |(sum, count), error| (sum + error, count + 1));

                if count > 0 {
                    (sum_squared_error / count as f64).sqrt()
                } else {
                    0.0
                }
            }

            LossConfig::MeanAbsoluteError => {
                let (total_error, count) = observation_iter()
                    .map(|(obs, predicted)| (obs.value - predicted).abs())
                    .fold((0.0, 0), |(sum, count), error| (sum + error, count + 1));

                if count > 0 {
                    total_error / count as f64
                } else {
                    0.0
                }
            }
        }
    }

    /// Clamp parameter values to their defined bounds
    ///
    /// This is necessary because some optimization algorithms
    /// can explore outside the bounds during their search process. By clamping,
    /// we ensure the simulation always receives valid parameter values while
    /// still allowing the optimizer to explore the parameter space freely.
    fn clamp_to_bounds(&self, param_values: &[f64]) -> Vec<f64> {
        param_values
            .iter()
            .zip(&self.parameters)
            .map(|(value, param)| value.clamp(param.min_bound, param.max_bound))
            .collect()
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

    /// Evaluate parameter-only constraints (no time_steps specified)
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

    /// Evaluate time-dependent constraints (with time_steps specified)
    ///
    /// These constraints can reference both calibration parameters and compartment values.
    /// Evaluated at each specified time step after simulation completes.
    ///
    /// Returns the total penalty from violated constraints, or 0.0 if all are satisfied.
    fn evaluate_time_dependent_constraints(
        &self,
        param_values: &[f64],
        simulation_results: &[Vec<f64>],
    ) -> Result<f64, String> {
        if self.time_dependent_constraint_indices.is_empty() {
            return Ok(0.0);
        }

        // Create context with parameter values
        let mut context = MathExpressionContext::new();

        // Initialize compartment names
        let compartment_names = self.base_engine.compartments();
        context.init_compartments(compartment_names);

        // Set parameter values
        for (idx, param) in self.parameters.iter().enumerate() {
            context.set_parameter(param.id.clone(), param_values[idx]);
        }

        let mut total_penalty = 0.0;

        for &constraint_idx in &self.time_dependent_constraint_indices {
            let constraint = &self.constraints[constraint_idx];
            let expr = &self.compiled_constraints[constraint_idx];

            let time_steps = constraint
                .time_steps
                .as_ref()
                .expect("Time-dependent constraint must have time_steps");

            for &time_step in time_steps {
                let time_idx = time_step as usize;

                // Get compartment values at this time step
                if let Some(step_data) = simulation_results.get(time_idx) {
                    // Update context with compartment values at this time step
                    context.set_compartments_by_index(step_data);
                    context.set_step(time_step as f64);

                    match expr.evaluate(&mut context) {
                        Ok(value) => {
                            if value < 0.0 {
                                // Constraint violated at this time step
                                let violation_magnitude = -value;

                                // Linear penalty scaled by weight
                                let penalty = constraint.weight * violation_magnitude;

                                total_penalty += penalty;
                            }
                        }
                        Err(e) => {
                            return Err(format!(
                                "Time-dependent constraint '{}' evaluation failed at step {}: {:?}",
                                constraint.id, time_step, e
                            ));
                        }
                    }
                } else {
                    return Err(format!(
                        "Time step {} not found in simulation results for constraint '{}'",
                        time_step, constraint.id
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

        // Apply auto-correction for initial condition parameters
        let (corrected_params, fixed_initial_conditions_sum, calibrated_initial_conditions_sum) =
            self.calculate_auto_corrected_parameters(&clamped_params);

        // Validate that fixed + calibrated fractions don't exceed 1.0
        if fixed_initial_conditions_sum + calibrated_initial_conditions_sum > 1.0 {
            // Invalid parameter combination: would result in negative last initial condition
            let base_penalty = self.calculate_base_penalty();
            let excess = fixed_initial_conditions_sum + calibrated_initial_conditions_sum - 1.0;
            let penalty = base_penalty * (1.0 + excess * 100.0);
            return Ok(penalty);
        }

        // Update parameters and initial conditions using corrected values
        for ((value, param), compartment_idx) in corrected_params
            .iter()
            .zip(&self.parameters)
            .zip(&self.parameter_compartment_indices)
        {
            match param.parameter_type {
                CalibrationParameterType::Parameter => {
                    // Set model parameter
                    engine.set_parameter(&param.id, *value).map_err(|e| {
                        Error::msg(format!("Failed to set parameter '{}': {}", param.id, e))
                    })?;
                }
                CalibrationParameterType::InitialCondition => {
                    let idx =
                        compartment_idx.expect("InitialCondition must have compartment index");

                    // Use corrected value (auto-calculation already applied)
                    let fraction = *value;
                    let absolute_population = fraction * self.initial_population_size;
                    engine
                        .set_initial_condition(idx, absolute_population)
                        .map_err(|e| {
                            Error::msg(format!(
                                "Failed to set initial condition for '{}': {}",
                                param.id, e
                            ))
                        })?;
                }
                CalibrationParameterType::Scale => {
                    // Scale parameters are not applied to the engine
                    // They are used in loss calculation
                }
            }
        }

        // Run simulation using pre-allocated buffer to avoid allocations
        let mut buffer = self
            .result_buffer
            .write()
            .expect("Failed to acquire write lock on result_buffer");
        engine
            .run_into_buffer(self.max_time_step, &mut buffer)
            .map_err(|e| Error::msg(format!("Simulation failed: {}", e)))?;

        // Check for numerical instability (NaN or infinity values)
        let has_invalid_values = buffer
            .iter()
            .any(|step| step.iter().any(|&value| !value.is_finite()));

        if has_invalid_values {
            // Return a penalty value proportional to the worst-case realistic loss
            // Calculate penalty as: (max observed value)^2 * num_observations * penalty_factor
            // This ensures the penalty is large enough to discourage invalid parameters
            // but not so large that it causes numerical issues
            let base_penalty = self.calculate_base_penalty();
            return Ok(base_penalty);
        }

        // Evaluate time-dependent constraints using simulation results
        let time_constraint_penalty = self
            .evaluate_time_dependent_constraints(&clamped_params, &buffer)
            .map_err(Error::msg)?;

        if time_constraint_penalty > 0.0 {
            // Compartment value constraints violated - return penalty
            let base_penalty = self.calculate_base_penalty();
            return Ok(base_penalty + time_constraint_penalty * base_penalty);
        }

        // Calculate and return loss (use corrected params for scale parameters)
        let loss = self.calculate_loss(&buffer, &corrected_params);

        // Check if loss itself is invalid (defensive programming)
        if !loss.is_finite() {
            let base_penalty = self.calculate_base_penalty();
            return Ok(base_penalty);
        }

        // Record this evaluation in the history (use corrected params)
        let evaluation = CalibrationEvaluation {
            parameters: corrected_params.clone(),
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
