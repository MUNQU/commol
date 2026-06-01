//! Calibration problem definition and implementation

use argmin::core::{CostFunction, Error};
use commol_core::{MathExpression, MathExpressionContext, SimulationEngine};
use std::marker::PhantomData;
use std::sync::RwLock;

use crate::types::{
    CalibrationConstraint, CalibrationEvaluation, CalibrationParameter, CalibrationParameterType,
    LossConfig, ObservedDataPoint,
};

#[derive(Clone)]
enum InitialConditionTarget {
    PopulationFraction { entries: Vec<(usize, f64)> },
    PairedCategoryFraction { pairs: Vec<(usize, usize)> },
}

impl InitialConditionTarget {
    fn exact(index: usize) -> Self {
        Self::PopulationFraction {
            entries: vec![(index, 1.0)],
        }
    }

    fn aggregate(indices: Vec<usize>, population: &[f64]) -> Self {
        let total: f64 = indices.iter().map(|&idx| population[idx]).sum();
        let weight = if total > 0.0 {
            None
        } else {
            Some(1.0 / indices.len() as f64)
        };

        Self::PopulationFraction {
            entries: indices
                .into_iter()
                .map(|idx| {
                    let entry_weight = weight.unwrap_or(population[idx] / total);
                    (idx, entry_weight)
                })
                .collect(),
        }
    }

    fn paired_category(
        category: &str,
        compartment_map: &std::collections::HashMap<&str, usize>,
    ) -> Result<Option<Self>, String> {
        let mut pairs = Vec::new();
        for (&name, &category_idx) in compartment_map {
            let tokens: Vec<&str> = name.split('_').collect();
            for (token_idx, token) in tokens.iter().enumerate() {
                // Token 0 is the bin id. Category calibration targets only
                // stratification suffix tokens.
                if token_idx == 0 {
                    continue;
                }
                if *token != category {
                    continue;
                }

                let mut complement_indices = Vec::new();
                for (&candidate_name, &candidate_idx) in compartment_map {
                    let candidate_tokens: Vec<&str> = candidate_name.split('_').collect();
                    if candidate_tokens.len() != tokens.len()
                        || candidate_tokens[token_idx] == category
                    {
                        continue;
                    }

                    let same_except_category = tokens
                        .iter()
                        .enumerate()
                        .all(|(idx, token)| idx == token_idx || candidate_tokens[idx] == *token);
                    if same_except_category {
                        complement_indices.push(candidate_idx);
                    }
                }

                match complement_indices.as_slice() {
                    [complement_idx] => pairs.push((category_idx, *complement_idx)),
                    [] => return Ok(None),
                    _ => {
                        return Err(format!(
                            "Initial condition category '{}' is ambiguous for compartment '{}': \
                             expected exactly one complementary category, found {}",
                            category,
                            name,
                            complement_indices.len()
                        ));
                    }
                }
            }
        }

        pairs.sort_unstable();
        Ok((!pairs.is_empty()).then_some(Self::PairedCategoryFraction { pairs }))
    }

    fn covered_population_compartments(&self) -> Vec<usize> {
        match self {
            Self::PopulationFraction { entries } => entries.iter().map(|(idx, _)| *idx).collect(),
            Self::PairedCategoryFraction { .. } => Vec::new(),
        }
    }

    fn is_population_fraction(&self) -> bool {
        matches!(self, Self::PopulationFraction { .. })
    }
}

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

    /// Indices of observed outputs in the engine's output vector.
    /// Each observation may sum one or more outputs before comparison.
    observed_output_indices: Vec<Vec<usize>>,

    /// Sorted, deduplicated observation time steps for sparse loss evaluation
    observed_time_steps: Vec<u32>,

    /// Row index in `observed_time_steps` for each observed data point
    observed_time_indices: Vec<usize>,

    /// Optional previous row index for windowed cumulative-output observations.
    observed_previous_time_indices: Vec<Option<usize>>,

    /// Row indices in `observed_time_steps` for each time-dependent constraint.
    constraint_time_indices: Vec<Option<Vec<usize>>>,

    /// Parameters to calibrate with their bounds
    parameters: Vec<CalibrationParameter>,

    /// Initial-condition targets for calibration parameters (parallel to parameters vec).
    /// None for non-initial-condition parameters.
    parameter_initial_condition_targets: Vec<Option<InitialConditionTarget>>,

    /// Scale parameter indices for observed data points (parallel to observed_data vec)
    /// None if no scale is applied, Some(param_idx) if a scale parameter should be applied
    observed_scale_indices: Vec<Option<usize>>,

    /// Loss function configuration
    loss_config: LossConfig,

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

    /// Pre-computed prefix → compartment-index mappings for constraint evaluation.
    ///
    /// A prefix referenced in a constraint expression resolves to the total
    /// population of all compartments that share that prefix. This covers full bin
    /// names (aggregating over all strata) as well as partial stratification
    /// specifications (aggregating over the remaining unspecified strata),
    /// consistent with how such names behave everywhere else in the model.
    bin_aggregates: Vec<(String, Vec<usize>)>,

    /// Phantom data for type parameter
    _phantom: PhantomData<E>,
}

// Manual Clone implementation because RwLock doesn't implement Clone
impl<E: SimulationEngine> Clone for CalibrationProblem<E> {
    fn clone(&self) -> Self {
        // Pre-allocate result buffer for the clone
        let buffer_capacity = self.observed_time_steps.len();
        let result_buffer = Vec::with_capacity(buffer_capacity);

        Self {
            base_engine: self.base_engine.clone(),
            observed_data: self.observed_data.clone(),
            observed_output_indices: self.observed_output_indices.clone(),
            observed_time_steps: self.observed_time_steps.clone(),
            observed_time_indices: self.observed_time_indices.clone(),
            observed_previous_time_indices: self.observed_previous_time_indices.clone(),
            constraint_time_indices: self.constraint_time_indices.clone(),
            parameters: self.parameters.clone(),
            parameter_initial_condition_targets: self.parameter_initial_condition_targets.clone(),
            observed_scale_indices: self.observed_scale_indices.clone(),
            loss_config: self.loss_config,
            result_buffer: RwLock::new(result_buffer),
            initial_population_size: self.initial_population_size,
            constraints: self.constraints.clone(),
            compiled_constraints: self.compiled_constraints.clone(),
            evaluations: RwLock::new(Vec::new()), // Each clone gets fresh evaluation history
            parameter_constraint_indices: self.parameter_constraint_indices.clone(),
            time_dependent_constraint_indices: self.time_dependent_constraint_indices.clone(),
            bin_aggregates: self.bin_aggregates.clone(),
            _phantom: PhantomData,
        }
    }
}

/// Build a mapping from partial compartment name prefixes to the indices of all
/// compartments that share that prefix.
///
/// Every prefix formed by taking a compartment name up to any of its underscore
/// separators is a candidate key. A key is retained only when two or more
/// compartments share it, making it useful for aggregation. This covers both
/// full bin names (aggregating over all strata) and partial stratification
/// specifications (aggregating over the remaining unspecified strata).
fn compute_bin_aggregates(compartment_names: &[String]) -> Vec<(String, Vec<usize>)> {
    let mut map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    for (idx, name) in compartment_names.iter().enumerate() {
        let mut pos = 0;
        while let Some(rel) = name[pos..].find('_') {
            let prefix = name[..pos + rel].to_string();
            map.entry(prefix).or_default().push(idx);
            pos += rel + 1;
        }
    }
    let mut result: Vec<(String, Vec<usize>)> =
        map.into_iter().filter(|(_, v)| v.len() > 1).collect();
    result.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    result
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

        // Build output name to index mapping for observations. Outputs include
        // state compartments and non-population outputs such as accumulators.
        let output_names = base_engine.output_names();
        let output_map: std::collections::HashMap<&str, usize> = output_names
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.as_str(), idx))
            .collect();

        // Validate observed output names and convert to indices.
        let mut observed_output_indices = Vec::with_capacity(observed_data.len());
        for obs in &observed_data {
            if let Some(window_steps) = obs.window_steps {
                if window_steps > obs.time_step {
                    return Err(format!(
                        "Observation '{}' at step {} has window_steps={} before simulation start",
                        obs.compartment, obs.time_step, window_steps
                    ));
                }
            }

            let observed_outputs: Vec<&str> = obs.compartments.as_ref().map_or_else(
                || vec![obs.compartment.as_str()],
                |compartments| compartments.iter().map(String::as_str).collect(),
            );
            if observed_outputs.is_empty() {
                return Err(format!(
                    "Observation '{}' at step {} has an empty compartments list",
                    obs.compartment, obs.time_step
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
                            obs.compartment,
                            output_names.join(", ")
                        ));
                    }
                }
            }
            observed_output_indices.push(indices);
        }

        // Build compartment indices for calibration parameters
        let compartments = base_engine.compartments();
        let compartment_map: std::collections::HashMap<&str, usize> = compartments
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.as_str(), idx))
            .collect();
        let initial_population = base_engine.population();
        let mut parameter_initial_condition_targets = Vec::with_capacity(parameters.len());
        for param in &parameters {
            match param.parameter_type {
                CalibrationParameterType::Parameter => {
                    // No initial-condition target needed for regular parameters
                    parameter_initial_condition_targets.push(None);
                }
                CalibrationParameterType::InitialCondition => {
                    // Initial conditions may target an exact expanded compartment
                    // (for example A_cat0) or an aggregate bin (for example A).
                    if let Some(&idx) = compartment_map.get(param.id.as_str()) {
                        parameter_initial_condition_targets
                            .push(Some(InitialConditionTarget::exact(idx)));
                    } else {
                        let prefix = format!("{}_", param.id);
                        let aggregate_indices: Vec<usize> = compartments
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, name)| {
                                (name == &param.id || name.starts_with(&prefix)).then_some(idx)
                            })
                            .collect();

                        if aggregate_indices.is_empty() {
                            if let Some(target) = InitialConditionTarget::paired_category(
                                &param.id,
                                &compartment_map,
                            )? {
                                parameter_initial_condition_targets.push(Some(target));
                                continue;
                            }

                            return Err(format!(
                                "Invalid bin or compartment ID '{}' for initial condition calibration
                                (available: {})",
                                param.id,
                                compartments.join(", ")
                            ));
                        }

                        parameter_initial_condition_targets.push(Some(
                            InitialConditionTarget::aggregate(
                                aggregate_indices,
                                &initial_population,
                            ),
                        ));
                    }
                }
                CalibrationParameterType::Scale => {
                    // No initial-condition target needed for scale parameters
                    parameter_initial_condition_targets.push(None);
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
        let mut simulation_time_steps: Vec<u32> = observed_data
            .iter()
            .flat_map(|obs| {
                std::iter::once(obs.time_step).chain(
                    obs.window_steps
                        .map(|window_steps| obs.time_step - window_steps),
                )
            })
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

            // Add sparse constraint time steps to the simulation schedule.
            if let Some(ref time_steps) = constraint.time_steps {
                simulation_time_steps.extend(time_steps.iter().copied());
                time_dependent_constraint_indices.push(idx);
            } else {
                parameter_constraint_indices.push(idx);
            }
        }

        simulation_time_steps.sort_unstable();
        simulation_time_steps.dedup();

        let observed_time_indices: Vec<usize> = observed_data
            .iter()
            .map(|obs| {
                simulation_time_steps
                    .binary_search(&obs.time_step)
                    .expect("observed time step must be present")
            })
            .collect();

        let observed_previous_time_indices: Vec<Option<usize>> = observed_data
            .iter()
            .map(|obs| {
                obs.window_steps.map(|window_steps| {
                    simulation_time_steps
                        .binary_search(&(obs.time_step - window_steps))
                        .expect("window previous time step must be present")
                })
            })
            .collect();

        let constraint_time_indices: Vec<Option<Vec<usize>>> = constraints
            .iter()
            .map(|constraint| {
                constraint.time_steps.as_ref().map(|time_steps| {
                    time_steps
                        .iter()
                        .map(|time_step| {
                            simulation_time_steps
                                .binary_search(time_step)
                                .expect("constraint time step must be present")
                        })
                        .collect()
                })
            })
            .collect();

        // Pre-allocate result buffer for performance
        let buffer_capacity = simulation_time_steps.len();
        let result_buffer = Vec::with_capacity(buffer_capacity);

        let bin_aggregates = compute_bin_aggregates(&compartments);

        Ok(Self {
            base_engine,
            observed_data,
            observed_output_indices,
            observed_time_steps: simulation_time_steps,
            observed_time_indices,
            observed_previous_time_indices,
            constraint_time_indices,
            parameters,
            parameter_initial_condition_targets,
            observed_scale_indices,
            loss_config,
            result_buffer: RwLock::new(result_buffer),
            initial_population_size: initial_population_size as f64,
            constraints,
            compiled_constraints,
            evaluations: RwLock::new(Vec::new()),
            parameter_constraint_indices,
            time_dependent_constraint_indices,
            bin_aggregates,
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

        for (param, target) in self
            .parameters
            .iter()
            .zip(&self.parameter_initial_condition_targets)
        {
            if param.parameter_type == CalibrationParameterType::InitialCondition {
                if let Some(target) = target {
                    for idx in target.covered_population_compartments() {
                        calibrated_compartments[idx] = true;
                    }
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
            .filter(|(idx, param)| {
                param.parameter_type == CalibrationParameterType::InitialCondition
                    && self.parameter_initial_condition_targets[*idx]
                        .as_ref()
                        .is_some_and(InitialConditionTarget::is_population_fraction)
            })
            .map(|(idx, _)| idx)
            .collect();

        let num_initial_conditions_params = initial_conditions_params_indices.len();
        let all_compartments_are_initial_conditions = calibrated_compartments
            .iter()
            .all(|is_calibrated| *is_calibrated);

        // Auto-calculate the last population-fraction initial condition when
        // there is enough information for a remainder. Fixed compartments stay
        // out of the optimizer and contribute through fixed_initial_conditions_sum.
        let auto_calc_initial_conditions_idx = if num_initial_conditions_params >= 2
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
                    && self.parameter_initial_condition_targets[*idx]
                        .as_ref()
                        .is_some_and(InitialConditionTarget::is_population_fraction)
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
            .filter(|(idx, param)| {
                param.parameter_type == CalibrationParameterType::InitialCondition
                    && self.parameter_initial_condition_targets[*idx]
                        .as_ref()
                        .is_some_and(InitialConditionTarget::is_population_fraction)
            })
            .map(|(idx, _)| idx)
            .collect();

        let num_initial_conditions_params = initial_conditions_params_indices.len();
        let mut calibrated_compartments = vec![false; num_compartments];
        for target in self.parameter_initial_condition_targets.iter().flatten() {
            for idx in target.covered_population_compartments() {
                calibrated_compartments[idx] = true;
            }
        }
        let all_compartments_are_initial_conditions = calibrated_compartments
            .iter()
            .all(|is_calibrated| *is_calibrated);

        let auto_calc_initial_conditions_idx = if num_initial_conditions_params >= 2
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

    /// Calculate loss from a sparse result matrix whose rows are aligned with
    /// `self.observed_time_steps`.
    fn calculate_sparse_loss(&self, observed_results: &[Vec<f64>], param_values: &[f64]) -> f64 {
        let observation_iter = || {
            self.observed_data
                .iter()
                .zip(&self.observed_output_indices)
                .zip(&self.observed_scale_indices)
                .zip(&self.observed_time_indices)
                .zip(&self.observed_previous_time_indices)
                .filter_map(
                    |((((obs, output_indices), &scale_idx), &time_idx), &previous_time_idx)| {
                        observed_results.get(time_idx).map(|step_data| {
                            let current: f64 =
                                output_indices.iter().map(|&idx| step_data[idx]).sum();
                            let predicted = if let Some(previous_time_idx) = previous_time_idx {
                                let previous: f64 = observed_results
                                    .get(previous_time_idx)
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
                            let scaled_predicted = if let Some(param_idx) = scale_idx {
                                predicted * param_values[param_idx]
                            } else {
                                predicted
                            };
                            (obs, scaled_predicted)
                        })
                    },
                )
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

    /// Build an engine clone with calibration parameters applied exactly as in
    /// objective evaluation, including calibrated initial conditions.
    pub fn configured_engine(&self, param_values: &[f64]) -> Result<E, String> {
        self.validate_parameter_count(param_values)?;
        let clamped_params = self.clamp_to_bounds(param_values);
        let (corrected_params, fixed_initial_conditions_sum, calibrated_initial_conditions_sum) =
            self.calculate_auto_corrected_parameters(&clamped_params);

        if fixed_initial_conditions_sum + calibrated_initial_conditions_sum > 1.0 {
            return Err(
                "Invalid initial-condition fractions: fixed + calibrated values exceed 1.0"
                    .to_string(),
            );
        }

        let mut engine = self.base_engine.clone();
        engine.reset();

        for ((value, param), target) in corrected_params
            .iter()
            .zip(&self.parameters)
            .zip(&self.parameter_initial_condition_targets)
        {
            match param.parameter_type {
                CalibrationParameterType::Parameter => {
                    engine
                        .set_parameter(&param.id, *value)
                        .map_err(|e| format!("Failed to set parameter '{}': {}", param.id, e))?;
                }
                CalibrationParameterType::InitialCondition => {
                    let target = target
                        .as_ref()
                        .expect("InitialCondition must have a target");

                    match target {
                        InitialConditionTarget::PopulationFraction { entries } => {
                            for &(idx, weight) in entries {
                                engine
                                    .set_initial_condition(
                                        idx,
                                        *value * weight * self.initial_population_size,
                                    )
                                    .map_err(|e| {
                                        format!(
                                            "Failed to set initial condition for '{}': {}",
                                            param.id, e
                                        )
                                    })?;
                            }
                        }
                        InitialConditionTarget::PairedCategoryFraction { pairs } => {
                            let current_population = engine.population();
                            for &(category_idx, complement_idx) in pairs {
                                let pair_total = current_population[category_idx]
                                    + current_population[complement_idx];
                                engine
                                    .set_initial_condition(category_idx, *value * pair_total)
                                    .map_err(|e| {
                                        format!(
                                            "Failed to set initial condition for '{}': {}",
                                            param.id, e
                                        )
                                    })?;
                                engine
                                    .set_initial_condition(
                                        complement_idx,
                                        (1.0 - *value) * pair_total,
                                    )
                                    .map_err(|e| {
                                        format!(
                                            "Failed to set initial condition for '{}': {}",
                                            param.id, e
                                        )
                                    })?;
                            }
                        }
                    }
                }
                CalibrationParameterType::Scale => {}
            }
        }

        Ok(engine)
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
            let time_indices = self.constraint_time_indices[constraint_idx]
                .as_ref()
                .expect("Time-dependent constraint must have row indices");

            for (&time_step, &time_idx) in time_steps.iter().zip(time_indices) {
                // Get compartment values at this time step
                if let Some(step_data) = simulation_results.get(time_idx) {
                    // Update context with compartment values at this time step
                    context.set_compartments_by_index(step_data);
                    // Inject bin totals so bare bin names resolve to the sum across
                    // all strata, overriding any calibration parameter with the same name.
                    for (bin_name, indices) in &self.bin_aggregates {
                        let total: f64 = indices.iter().map(|&i| step_data[i]).sum();
                        context.set_parameter_str(bin_name, total);
                    }
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
        for ((value, param), target) in corrected_params
            .iter()
            .zip(&self.parameters)
            .zip(&self.parameter_initial_condition_targets)
        {
            match param.parameter_type {
                CalibrationParameterType::Parameter => {
                    // Set model parameter
                    engine.set_parameter(&param.id, *value).map_err(|e| {
                        Error::msg(format!("Failed to set parameter '{}': {}", param.id, e))
                    })?;
                }
                CalibrationParameterType::InitialCondition => {
                    let target = target
                        .as_ref()
                        .expect("InitialCondition must have a target");

                    let fraction = *value;
                    match target {
                        InitialConditionTarget::PopulationFraction { entries } => {
                            // Use corrected value (auto-calculation already applied).
                            for &(idx, weight) in entries {
                                let absolute_population =
                                    fraction * weight * self.initial_population_size;
                                engine
                                    .set_initial_condition(idx, absolute_population)
                                    .map_err(|e| {
                                        Error::msg(format!(
                                            "Failed to set initial condition for '{}': {}",
                                            param.id, e
                                        ))
                                    })?;
                            }
                        }
                        InitialConditionTarget::PairedCategoryFraction { pairs } => {
                            let current_population = engine.population();
                            for &(category_idx, complement_idx) in pairs {
                                let pair_total = current_population[category_idx]
                                    + current_population[complement_idx];
                                engine
                                    .set_initial_condition(category_idx, fraction * pair_total)
                                    .map_err(|e| {
                                        Error::msg(format!(
                                            "Failed to set initial condition for '{}': {}",
                                            param.id, e
                                        ))
                                    })?;
                                engine
                                    .set_initial_condition(
                                        complement_idx,
                                        (1.0 - fraction) * pair_total,
                                    )
                                    .map_err(|e| {
                                        Error::msg(format!(
                                            "Failed to set initial condition for '{}': {}",
                                            param.id, e
                                        ))
                                    })?;
                            }
                        }
                    }
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
            .run_at_steps_into_buffer(&self.observed_time_steps, &mut buffer)
            .map_err(|e| Error::msg(format!("Simulation failed: {}", e)))?;

        // Check for numerical instability (NaN or infinity values)
        let has_invalid_values = buffer
            .iter()
            .any(|step| step.iter().any(|&value| !value.is_finite()));

        if has_invalid_values {
            let base_penalty = self.calculate_base_penalty();
            return Ok(base_penalty);
        }

        let loss = if self.time_dependent_constraint_indices.is_empty() {
            self.calculate_sparse_loss(&buffer, &corrected_params)
        } else {
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
            self.calculate_sparse_loss(&buffer, &corrected_params)
        };

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
