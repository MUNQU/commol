//! Calibration problem definition and implementation

use argmin::core::{CostFunction, Error};
use commol_core::{MathExpression, MathExpressionContext, SimulationEngine};
use std::marker::PhantomData;

use crate::types::{
    CalibrationConstraint, CalibrationParameter, CalibrationParameterType, LossConfig,
    ObservedDataPoint,
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
    /// Wrapped in RefCell to allow mutation in cost() method
    result_buffer: std::cell::RefCell<Vec<Vec<f64>>>,

    /// Initial population size for converting fractions to absolute values
    /// From the model's defined initial_population_size
    initial_population_size: f64,

    /// Constraints on parameters and/or compartment values
    constraints: Vec<CalibrationConstraint>,

    /// Compiled constraint expressions (cached for performance)
    compiled_constraints: Vec<MathExpression>,

    /// Indices of parameter-only constraints (no time_steps, evaluated once before simulation)
    parameter_constraint_indices: Vec<usize>,

    /// Indices of time-dependent constraints (with time_steps, evaluated at specified times during simulation)
    time_dependent_constraint_indices: Vec<usize>,

    /// Phantom data for type parameter
    _phantom: PhantomData<E>,
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
            result_buffer: std::cell::RefCell::new(result_buffer),
            initial_population_size: initial_population_size as f64,
            constraints,
            compiled_constraints,
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

    /// Helper method to determine which compartments are being calibrated as ICs
    fn get_calibrated_compartments(&self) -> Vec<bool> {
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

        calibrated_compartments
    }

    /// Calculate the sum of fixed (non-calibrated) initial condition fractions
    fn calculate_fixed_ic_sum(&self, calibrated_compartments: &[bool]) -> f64 {
        let current_population = self.base_engine.population();
        current_population
            .iter()
            .enumerate()
            .filter(|(idx, _)| !calibrated_compartments[*idx])
            .map(|(_, &val)| val / self.initial_population_size)
            .sum()
    }

    /// Determine which IC parameter (if any) should be auto-calculated
    ///
    /// Returns the index of the parameter that should be auto-calculated,
    /// or None if no auto-calculation is needed.
    fn get_auto_calculated_ic_index(&self) -> Option<usize> {
        let num_compartments = self.base_engine.compartments().len();

        let ic_params_indices: Vec<usize> = self
            .parameters
            .iter()
            .enumerate()
            .filter(|(_, param)| param.parameter_type == CalibrationParameterType::InitialCondition)
            .map(|(idx, _)| idx)
            .collect();

        let num_ic_params = ic_params_indices.len();
        let all_compartments_are_ics = num_ic_params == num_compartments;

        // Auto-calculate the last IC parameter if:
        // 1. All compartments are ICs AND there are 2+ IC parameters, OR
        // 2. There's exactly 1 IC parameter with fixed compartments (sum constraint)
        if (num_ic_params >= 2 && all_compartments_are_ics)
            || (num_ic_params == 1 && !all_compartments_are_ics)
        {
            ic_params_indices.last().copied()
        } else {
            None
        }
    }

    /// Get information needed to fix auto-calculated parameters
    ///
    /// Returns (fixed_ic_sum, auto_calc_ic_idx, param_types) where:
    /// - fixed_ic_sum: Sum of fractions for fixed (non-calibrated) compartments
    /// - auto_calc_ic_idx: Index of the IC parameter to auto-calculate (if any)
    /// - param_types: Vector of parameter types for each parameter
    pub fn get_parameter_fix_info(&self) -> (f64, Option<usize>, Vec<CalibrationParameterType>) {
        let calibrated_compartments = self.get_calibrated_compartments();
        let fixed_ic_sum = self.calculate_fixed_ic_sum(&calibrated_compartments);
        let auto_calc_ic_idx = self.get_auto_calculated_ic_index();
        let param_types: Vec<CalibrationParameterType> =
            self.parameters.iter().map(|p| p.parameter_type).collect();

        (fixed_ic_sum, auto_calc_ic_idx, param_types)
    }

    /// Fix auto-calculated initial condition parameters in the result
    ///
    /// Some IC parameters may be auto-calculated to ensure fractions sum to 1.0.
    /// This method replaces those auto-calculated values with their correct values
    /// based on the constraint.
    ///
    /// # Arguments
    /// * `param_values` - Parameter values from the optimizer
    ///
    /// # Returns
    /// Corrected parameter values with auto-calculated ICs fixed
    pub fn fix_auto_calculated_parameters(&self, mut param_values: Vec<f64>) -> Vec<f64> {
        let calibrated_compartments = self.get_calibrated_compartments();
        let fixed_ic_sum = self.calculate_fixed_ic_sum(&calibrated_compartments);
        let auto_calc_ic_idx = self.get_auto_calculated_ic_index();

        // If there's an auto-calculated parameter, compute and set its correct value
        if let Some(idx) = auto_calc_ic_idx {
            // Calculate sum of other calibrated ICs (excluding the auto-calculated one)
            let calibrated_ic_sum: f64 = param_values
                .iter()
                .enumerate()
                .filter(|(param_idx, _)| {
                    self.parameters[*param_idx].parameter_type
                        == CalibrationParameterType::InitialCondition
                        && *param_idx != idx
                })
                .map(|(_, value)| value)
                .sum();

            // Auto-calculated value ensures fractions sum to 1.0
            let auto_calculated_value = (1.0 - fixed_ic_sum - calibrated_ic_sum).max(0.0);
            param_values[idx] = auto_calculated_value;
        }

        param_values
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

        // Calculate sum of fixed initial conditions (those not being calibrated)
        let num_compartments = engine.compartments().len();
        let mut calibrated_compartments = vec![false; num_compartments];

        // Mark which compartments are being calibrated
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

        // Get current (fixed) initial conditions for non-calibrated compartments
        let current_population = engine.population();
        let fixed_ic_sum: f64 = current_population
            .iter()
            .enumerate()
            .filter(|(idx, _)| !calibrated_compartments[*idx])
            .map(|(_, &val)| val / self.initial_population_size)
            .sum();

        // Determine which IC parameter (if any) should be auto-calculated to ensure sum = 1.0
        // Logic:
        // - If there's only 1 IC parameter being calibrated, it should be auto-calculated
        //   to ensure fixed + calibrated = 1.0 (unless all compartments are being calibrated)
        // - If there are 2+ IC parameters, the last one is auto-calculated to ensure sum = 1.0
        let ic_params_indices: Vec<usize> = self
            .parameters
            .iter()
            .enumerate()
            .filter(|(_, param)| param.parameter_type == CalibrationParameterType::InitialCondition)
            .map(|(idx, _)| idx)
            .collect();

        let num_ic_params = ic_params_indices.len();
        let num_total_compartments = num_compartments;

        // Auto-calculate the last IC parameter if:
        // 1. There are IC parameters to calibrate, and
        // 2. Either:
        //    a) There are 2+ IC parameters (always auto-calculate last), or
        //    b) There's 1 IC parameter and it's not the only compartment (has fixed compartments)
        let last_ic_param_idx = if num_ic_params > 0 {
            let has_fixed_compartments = num_ic_params < num_total_compartments;
            if num_ic_params >= 2 || (num_ic_params == 1 && has_fixed_compartments) {
                ic_params_indices.last().copied()
            } else {
                None
            }
        } else {
            None
        };

        // Calculate sum of calibrated initial conditions (excluding the last one)
        let calibrated_ic_sum: f64 = clamped_params
            .iter()
            .enumerate()
            .filter(|(idx, _)| {
                self.parameters[*idx].parameter_type == CalibrationParameterType::InitialCondition
                    && Some(*idx) != last_ic_param_idx
            })
            .map(|(_, value)| value)
            .sum();

        // Validate that fixed + calibrated fractions don't exceed 1.0
        // This ensures the last initial condition can be calculated as a non-negative value
        if fixed_ic_sum + calibrated_ic_sum > 1.0 {
            // Invalid parameter combination: would result in negative last initial condition
            // Return penalty proportional to the excess
            let base_penalty = self.calculate_base_penalty();
            // Scale penalty by how much we exceeded 1.0
            let excess = fixed_ic_sum + calibrated_ic_sum - 1.0;
            let penalty = base_penalty * (1.0 + excess * 100.0);
            return Ok(penalty);
        }

        // Update parameters and initial conditions
        for (param_idx, ((value, param), compartment_idx)) in clamped_params
            .iter()
            .zip(&self.parameters)
            .zip(&self.parameter_compartment_indices)
            .enumerate()
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

                    // Calculate the actual fraction to use
                    let fraction = if Some(param_idx) == last_ic_param_idx {
                        // Last initial condition: calculate as remainder
                        // fraction = 1.0 - fixed_sum - calibrated_sum (excluding this one)
                        let remaining = 1.0 - fixed_ic_sum - calibrated_ic_sum;
                        // Ensure non-negative
                        remaining.max(0.0)
                    } else {
                        // Regular calibrated initial condition
                        *value
                    };

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
        let mut buffer = self.result_buffer.borrow_mut();
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

        // Calculate and return loss
        let loss = self.calculate_loss(&buffer, &clamped_params);

        // Check if loss itself is invalid (defensive programming)
        if !loss.is_finite() {
            let base_penalty = self.calculate_base_penalty();
            return Ok(base_penalty);
        }

        Ok(loss)
    }
}
