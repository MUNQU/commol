//! Calibration problem definition and implementation

use argmin::core::{CostFunction, Error};
use commol_core::{MathExpression, MathExpressionContext, SimulationEngine};
use std::marker::PhantomData;
use std::sync::RwLock;

use crate::initial_conditions::{
    clamp_to_bounds, configure_engine, is_infeasible, InitialConditionPlan,
};
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

    /// Resolved initial-condition targets and remainder of the parameters.
    initial_conditions: InitialConditionPlan,

    /// Scale parameter indices for observed data points (parallel to observed_data vec)
    /// None if no scale is applied, Some(param_idx) if a scale parameter should be applied
    observed_scale_indices: Vec<Option<usize>>,

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

    /// Outputs that are not population compartments (accumulators), as
    /// `(name, row index)`. These are injected by name during constraint
    /// evaluation so they resolve without counting towards `N`.
    non_population_outputs: Vec<(String, usize)>,

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
            initial_conditions: self.initial_conditions.clone(),
            observed_scale_indices: self.observed_scale_indices.clone(),
            observed_normalization: self.observed_normalization.clone(),
            loss_config: self.loss_config,
            result_buffer: RwLock::new(result_buffer),
            initial_population_size: self.initial_population_size,
            constraints: self.constraints.clone(),
            compiled_constraints: self.compiled_constraints.clone(),
            evaluations: RwLock::new(Vec::new()), // Each clone gets fresh evaluation history
            parameter_constraint_indices: self.parameter_constraint_indices.clone(),
            time_dependent_constraint_indices: self.time_dependent_constraint_indices.clone(),
            bin_aggregates: self.bin_aggregates.clone(),
            non_population_outputs: self.non_population_outputs.clone(),
            _phantom: PhantomData,
        }
    }
}

/// Build a mapping from output name prefixes to the indices of every output
/// that shares that prefix.
///
/// A prefix is an output's source followed by any number of its leading
/// categories (`A`, `A_g1`), short of its full name, and is built from the
/// labels rather than by splitting names, so a bin `A` never covers the
/// outputs of a bin `A_x`. A key is retained only when two or more outputs
/// share it, making it useful for aggregation. This covers both full bin names
/// (aggregating over all strata) and partial stratification specifications
/// (aggregating over the remaining unspecified strata).
fn compute_bin_aggregates(labels: &[commol_core::OutputLabels]) -> Vec<(String, Vec<usize>)> {
    let mut map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    for (idx, label) in labels.iter().enumerate() {
        let mut prefixes = label.prefixes();
        prefixes.pop();
        for prefix in prefixes {
            map.entry(prefix).or_default().push(idx);
        }
    }
    let mut result: Vec<(String, Vec<usize>)> =
        map.into_iter().filter(|(_, v)| v.len() > 1).collect();
    result.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    result
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
            .entry(obs.compartment.as_str())
            .or_insert((0.0, 0));
        entry.0 += obs.value * obs.value;
        entry.1 += 1;
    }

    observed_data
        .iter()
        .map(|obs| {
            let (sum_squares, count) = series_sum_squares[obs.compartment.as_str()];
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

        // Aggregate prefixes over every output, so a constraint can name a bare bin
        // or a bare accumulator and get the total across strata. Indices address
        // full output rows, matching the rows that time-dependent constraints are
        // evaluated against.
        let output_aggregates = compute_bin_aggregates(&base_engine.output_labels());
        let output_aggregate_names: std::collections::HashSet<&str> = output_aggregates
            .iter()
            .map(|(name, _)| name.as_str())
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

        // Resolve the initial-condition target of every calibration parameter
        let compartments = base_engine.compartments();
        let initial_conditions = InitialConditionPlan::new(&base_engine, &parameters)?;

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
                // Constraints resolve against every simulation output, so they can
                // reference accumulators as well as state compartments, plus any
                // aggregate prefix of either.
                let is_compartment = output_map.contains_key(var.as_str())
                    || output_aggregate_names.contains(var.as_str());

                if !is_parameter && !is_compartment {
                    return Err(format!(
                        "Constraint '{}' references unknown variable '{}' (not a parameter or simulation output)",
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

        let bin_aggregates = output_aggregates;
        // Outputs past the population block are accumulators; keep their row
        // indices so constraints can name them individually.
        let non_population_outputs: Vec<(String, usize)> = output_names
            .iter()
            .enumerate()
            .skip(compartments.len())
            .map(|(idx, name)| (name.clone(), idx))
            .collect();

        Ok(Self {
            base_engine,
            observed_data,
            observed_output_indices,
            observed_time_steps: simulation_time_steps,
            observed_time_indices,
            observed_previous_time_indices,
            constraint_time_indices,
            parameters,
            initial_conditions,
            observed_scale_indices,
            observed_normalization,
            loss_config,
            result_buffer: RwLock::new(result_buffer),
            initial_population_size: initial_population_size as f64,
            constraints,
            compiled_constraints,
            evaluations: RwLock::new(Vec::new()),
            parameter_constraint_indices,
            time_dependent_constraint_indices,
            bin_aggregates,
            non_population_outputs,
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

    /// Name the compartment or bin that absorbs the difference to the
    /// population size once the calibrated initial conditions are applied.
    ///
    /// Without a remainder, calibrated initial conditions are applied as
    /// given and only a total above the population size is penalized. With
    /// one, the initial population always equals the population size, and a
    /// combination that would leave the remainder negative is penalized.
    ///
    /// # Errors
    ///
    /// Returns a message when the id is not a compartment, bin or bin prefix,
    /// or overlaps a calibrated bin or compartment.
    pub fn with_initial_condition_remainder(mut self, remainder: &str) -> Result<Self, String> {
        self.initial_conditions = self.initial_conditions.with_remainder(
            &self.base_engine,
            &self.parameters,
            remainder,
        )?;
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
    /// `self.observed_time_steps`.
    fn calculate_sparse_loss(&self, observed_results: &[Vec<f64>], param_values: &[f64]) -> f64 {
        let observation_iter = || {
            self.observed_data
                .iter()
                .zip(&self.observed_output_indices)
                .zip(&self.observed_scale_indices)
                .zip(&self.observed_time_indices)
                .zip(&self.observed_previous_time_indices)
                .zip(&self.observed_normalization)
                .filter_map(
                    |(
                        ((((obs, output_indices), &scale_idx), &time_idx), &previous_time_idx),
                        &normalization,
                    )| {
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
                            (obs, scaled_predicted, normalization)
                        })
                    },
                )
        };

        // `normalization` is the per-series factor (1.0 unless observation
        // normalization is enabled). It is applied to every metric so that
        // series of very different magnitudes contribute comparably to any error
        // measurement, not just to a single dedicated loss variant.
        match self.loss_config {
            LossConfig::SumSquaredError | LossConfig::WeightedSSE => observation_iter()
                .map(|(obs, predicted, normalization)| {
                    let error = (obs.value - predicted) * obs.weight * normalization;
                    error * error
                })
                .sum(),

            LossConfig::RootMeanSquaredError => {
                let (sum_squared_error, count) = observation_iter()
                    .map(|(obs, predicted, normalization)| {
                        let error = (obs.value - predicted) * normalization;
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
                    .map(|(obs, predicted, normalization)| {
                        ((obs.value - predicted) * normalization).abs()
                    })
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
            &self.initial_conditions,
            self.initial_population_size,
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

        // Only population compartments are registered, so `N` stays the population
        // total. Accumulators live further along each result row and are injected by
        // name below, alongside the aggregates.
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
                    // Accumulators are not registered as compartments (they must not
                    // count towards `N`), so inject them by name.
                    for (name, idx) in &self.non_population_outputs {
                        context.set_parameter_str(name, step_data[*idx]);
                    }
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

        // Update parameters and initial conditions
        let excess = self
            .initial_conditions
            .apply(
                &mut engine,
                &self.parameters,
                self.initial_population_size,
                &clamped_params,
            )
            .map_err(Error::msg)?;

        if is_infeasible(excess) {
            // The initial population exceeds the population size
            let base_penalty = self.calculate_base_penalty();
            return Ok(base_penalty * (1.0 + excess * 100.0));
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
            self.calculate_sparse_loss(&buffer, &clamped_params)
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

#[cfg(test)]
mod aggregate_tests {
    use super::*;
    use commol_core::OutputLabels;

    fn label(source: &str, categories: &[(&str, &str)]) -> OutputLabels {
        OutputLabels {
            source: source.to_string(),
            categories: categories
                .iter()
                .map(|(id, category)| (id.to_string(), category.to_string()))
                .collect(),
        }
    }

    #[test]
    fn aggregates_follow_labels_not_name_prefixes() {
        let labels = vec![
            label("A", &[("group", "g1"), ("sub", "s1")]),
            label("A", &[("group", "g1"), ("sub", "s2")]),
            label("A", &[("group", "g2"), ("sub", "s1")]),
            label("A_x", &[("group", "g1"), ("sub", "s1")]),
            label("A_x", &[("group", "g1"), ("sub", "s2")]),
        ];

        let aggregates: std::collections::HashMap<String, Vec<usize>> =
            compute_bin_aggregates(&labels).into_iter().collect();

        assert_eq!(aggregates["A"], vec![0, 1, 2]);
        assert_eq!(aggregates["A_g1"], vec![0, 1]);
        assert_eq!(aggregates["A_x"], vec![3, 4]);
        assert_eq!(aggregates["A_x_g1"], vec![3, 4]);
        assert!(!aggregates.contains_key("A_g2"));
    }
}
