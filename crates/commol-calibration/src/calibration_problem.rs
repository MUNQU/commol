//! Calibration problem definition and implementation

use argmin::core::{CostFunction, Error};
use commol_core::SimulationEngine;
use std::marker::PhantomData;

use crate::types::{CalibrationParameter, LossConfig, ObservedDataPoint};

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

    /// Loss function configuration
    loss_config: LossConfig,

    /// Maximum time step in observed data (cached for performance)
    max_time_step: u32,

    /// Pre-allocated buffer for simulation results (reused across evaluations)
    /// Wrapped in RefCell to allow mutation in cost() method
    result_buffer: std::cell::RefCell<Vec<Vec<f64>>>,

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
    /// * `loss_config` - Loss function to use
    ///
    /// # Returns
    ///
    /// Returns `Ok(CalibrationProblem)` if successful, or an error if:
    /// - Compartment names in observed data are invalid
    /// - No observed data provided
    /// - No calibration parameters provided
    pub fn new(
        base_engine: E,
        observed_data: Vec<ObservedDataPoint>,
        parameters: Vec<CalibrationParameter>,
        loss_config: LossConfig,
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

        // Find maximum time step to avoid recomputing in each cost evaluation
        let max_time_step = observed_data
            .iter()
            .map(|obs| obs.time_step)
            .max()
            .unwrap_or(100);

        // Pre-allocate result buffer for performance
        let buffer_capacity = (max_time_step + 1) as usize;
        let result_buffer = Vec::with_capacity(buffer_capacity);

        Ok(Self {
            base_engine,
            observed_data,
            observed_compartment_indices,
            parameters,
            loss_config,
            max_time_step,
            result_buffer: std::cell::RefCell::new(result_buffer),
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

    /// Calculate loss between simulation results and observed data
    fn calculate_loss(&self, simulation_results: &[Vec<f64>]) -> f64 {
        let observation_iter = || {
            self.observed_data
                .iter()
                .zip(&self.observed_compartment_indices)
                .filter_map(|(obs, &compartment_idx)| {
                    let time_idx = obs.time_step as usize;
                    simulation_results
                        .get(time_idx)
                        .map(|step_data| (obs, step_data[compartment_idx]))
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
    /// This is necessary because some optimization algorithms (like Nelder-Mead)
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

        // Clone the base engine (works for any model type)
        let mut engine = self.base_engine.clone();

        // Reset engine to initial conditions
        engine.reset();

        // Update parameters with clamped values
        for (value, param) in clamped_params.iter().zip(&self.parameters) {
            engine.set_parameter(&param.id, *value).map_err(|e| {
                Error::msg(format!("Failed to set parameter '{}': {}", param.id, e))
            })?;
        }

        // Run simulation using pre-allocated buffer to avoid allocations
        let mut buffer = self.result_buffer.borrow_mut();
        engine
            .run_into_buffer(self.max_time_step, &mut buffer)
            .map_err(|e| Error::msg(format!("Simulation failed: {}", e)))?;

        // Calculate and return loss
        let loss = self.calculate_loss(&buffer);

        Ok(loss)
    }
}
