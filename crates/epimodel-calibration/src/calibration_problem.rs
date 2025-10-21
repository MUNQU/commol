//! Calibration problem definition and implementation

use argmin::core::{CostFunction, Error};
use epimodel_core::SimulationEngine;
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
/// use epimodel_calibration::{CalibrationProblem, types::*};
/// use epimodel_difference::DifferenceEquations;
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
    /// Template engine (will be cloned for each evaluation)
    template_engine: E,

    /// Observed data to fit against
    observed_data: Vec<ObservedDataPoint>,

    /// Compartment name to index mapping (computed once during construction)
    compartment_indices: Vec<usize>,

    /// Parameters to calibrate
    calibration_params: Vec<CalibrationParameter>,

    /// Loss function configuration
    loss_config: LossConfig,

    /// Maximum time step in observed data (cached for performance)
    max_time_step: u32,

    /// Phantom data for type parameter
    _phantom: PhantomData<E>,
}

impl<E: SimulationEngine> CalibrationProblem<E> {
    /// Create a new calibration problem
    ///
    /// # Arguments
    ///
    /// * `template_engine` - The simulation engine to calibrate (will be cloned for each evaluation)
    /// * `observed_data` - Vector of observed data points
    /// * `calibration_params` - Parameters to calibrate with their bounds
    /// * `loss_config` - Loss function to use
    ///
    /// # Returns
    ///
    /// Returns `Ok(CalibrationProblem)` if successful, or an error if:
    /// - Compartment indices in observed data are invalid
    /// - No observed data provided
    /// - No calibration parameters provided
    pub fn new(
        template_engine: E,
        observed_data: Vec<ObservedDataPoint>,
        calibration_params: Vec<CalibrationParameter>,
        loss_config: LossConfig,
    ) -> Result<Self, String> {
        // Validate inputs
        if observed_data.is_empty() {
            return Err("No observed data provided".to_string());
        }

        if calibration_params.is_empty() {
            return Err("No calibration parameters provided".to_string());
        }

        // Build compartment name to index mapping
        let compartments = template_engine.compartments();
        let compartment_map: std::collections::HashMap<&str, usize> = compartments
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.as_str(), idx))
            .collect();

        // Validate compartment names and convert to indices
        let mut compartment_indices = Vec::with_capacity(observed_data.len());
        for obs in &observed_data {
            match compartment_map.get(obs.compartment.as_str()) {
                Some(&idx) => compartment_indices.push(idx),
                None => {
                    return Err(format!(
                        "Invalid compartment name '{}' (available compartments: {})",
                        obs.compartment,
                        compartments.join(", ")
                    ));
                }
            }
        }

        // Pre-compute maximum time step to avoid recomputing in every cost evaluation
        let max_time_step = observed_data
            .iter()
            .map(|obs| obs.time_step)
            .max()
            .unwrap_or(100);

        Ok(Self {
            template_engine,
            observed_data,
            compartment_indices,
            calibration_params,
            loss_config,
            max_time_step,
            _phantom: PhantomData,
        })
    }

    /// Get the number of parameters being calibrated
    pub fn num_parameters(&self) -> usize {
        self.calibration_params.len()
    }

    /// Get parameter names
    pub fn parameter_names(&self) -> Vec<String> {
        self.calibration_params
            .iter()
            .map(|p| p.id.clone())
            .collect()
    }

    /// Get initial parameter guesses
    pub fn initial_parameters(&self) -> Vec<f64> {
        self.calibration_params
            .iter()
            .map(|p| p.get_initial_guess())
            .collect()
    }

    /// Get parameter bounds as (min, max) tuples
    pub fn get_parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.calibration_params
            .iter()
            .map(|p| (p.min_bound, p.max_bound))
            .collect()
    }

    /// Calculate loss between simulation results and observed data
    fn calculate_loss(&self, simulation_results: &[Vec<f64>]) -> f64 {
        match self.loss_config {
            LossConfig::SumSquaredError | LossConfig::WeightedSSE => {
                let mut total_error = 0.0;
                for (obs, &compartment_idx) in
                    self.observed_data.iter().zip(&self.compartment_indices)
                {
                    let step_idx = obs.time_step as usize;
                    if step_idx < simulation_results.len() {
                        let predicted = simulation_results[step_idx][compartment_idx];
                        let error = (obs.value - predicted) * obs.weight;
                        total_error += error * error;
                    }
                }
                total_error
            }

            LossConfig::RootMeanSquaredError => {
                let mut sum_squared_error = 0.0;
                let mut count = 0;
                for (obs, &compartment_idx) in
                    self.observed_data.iter().zip(&self.compartment_indices)
                {
                    let step_idx = obs.time_step as usize;
                    if step_idx < simulation_results.len() {
                        let predicted = simulation_results[step_idx][compartment_idx];
                        let error = obs.value - predicted;
                        sum_squared_error += error * error;
                        count += 1;
                    }
                }
                if count > 0 {
                    (sum_squared_error / count as f64).sqrt()
                } else {
                    0.0
                }
            }

            LossConfig::MeanAbsoluteError => {
                let mut total_error = 0.0;
                let mut count = 0;
                for (obs, &compartment_idx) in
                    self.observed_data.iter().zip(&self.compartment_indices)
                {
                    let step_idx = obs.time_step as usize;
                    if step_idx < simulation_results.len() {
                        let predicted = simulation_results[step_idx][compartment_idx];
                        total_error += (obs.value - predicted).abs();
                        count += 1;
                    }
                }
                if count > 0 {
                    total_error / count as f64
                } else {
                    0.0
                }
            }
        }
    }

    /// Validate parameter bounds
    fn validate_params(&self, params: &[f64]) -> Result<(), String> {
        if params.len() != self.calibration_params.len() {
            return Err(format!(
                "Expected {} parameters, got {}",
                self.calibration_params.len(),
                params.len()
            ));
        }

        for (i, (param_value, param_config)) in
            params.iter().zip(&self.calibration_params).enumerate()
        {
            if !param_config.is_valid(*param_value) {
                return Err(format!(
                    "Parameter '{}' (index {}) value {} out of bounds [{}, {}]",
                    param_config.id, i, param_value, param_config.min_bound, param_config.max_bound
                ));
            }
        }

        Ok(())
    }
}

/// Implement argmin's CostFunction trait - COMPLETELY MODEL AGNOSTIC
///
/// This implementation works with ANY model type that implements SimulationEngine.
impl<E: SimulationEngine> CostFunction for CalibrationProblem<E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        // Validate parameter bounds
        self.validate_params(params).map_err(|e| Error::msg(e))?;

        // Clone the template engine (works for ANY model type)
        let mut engine = self.template_engine.clone();

        // Reset engine to initial conditions
        engine.reset();

        // Update parameters
        for (param_value, param_config) in params.iter().zip(&self.calibration_params) {
            engine
                .set_parameter(&param_config.id, *param_value)
                .map_err(|e| {
                    Error::msg(format!(
                        "Failed to set parameter '{}': {}",
                        param_config.id, e
                    ))
                })?;
        }

        // Run simulation
        let results = engine
            .run(self.max_time_step)
            .map_err(|e| Error::msg(format!("Simulation failed: {}", e)))?;

        // 5. Calculate loss (model-agnostic)
        let loss = self.calculate_loss(&results);

        Ok(loss)
    }
}
