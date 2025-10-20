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
///     ObservedDataPoint::new(10, 1, 501.0),  // time=10, compartment I (index 1), value=501
///     ObservedDataPoint::new(20, 1, 823.0),
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

    /// Parameters to calibrate
    calibration_params: Vec<CalibrationParameter>,

    /// Loss function configuration
    loss_config: LossConfig,

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

        // Validate that compartment indices are valid
        let num_compartments = template_engine.compartments().len();
        for obs in &observed_data {
            if obs.compartment_index >= num_compartments {
                return Err(format!(
                    "Invalid compartment index {} (only {} compartments exist)",
                    obs.compartment_index, num_compartments
                ));
            }
        }

        Ok(Self {
            template_engine,
            observed_data,
            calibration_params,
            loss_config,
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
                for obs in &self.observed_data {
                    let step_idx = obs.time_step as usize;
                    if step_idx < simulation_results.len() {
                        let predicted = simulation_results[step_idx][obs.compartment_index];
                        let error = (obs.value - predicted) * obs.weight;
                        total_error += error * error;
                    }
                }
                total_error
            }

            LossConfig::RootMeanSquaredError => {
                let mut sum_squared_error = 0.0;
                let mut count = 0;
                for obs in &self.observed_data {
                    let step_idx = obs.time_step as usize;
                    if step_idx < simulation_results.len() {
                        let predicted = simulation_results[step_idx][obs.compartment_index];
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
                for obs in &self.observed_data {
                    let step_idx = obs.time_step as usize;
                    if step_idx < simulation_results.len() {
                        let predicted = simulation_results[step_idx][obs.compartment_index];
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

            LossConfig::NegativeLogLikelihoodPoisson => {
                let mut nll = 0.0;
                for obs in &self.observed_data {
                    let step_idx = obs.time_step as usize;
                    if step_idx < simulation_results.len() {
                        let lambda = simulation_results[step_idx][obs.compartment_index];
                        // Poisson NLL: -log(P(k|λ)) = λ - k*log(λ) + log(k!)
                        // We ignore the constant term log(k!)
                        if lambda > 0.0 {
                            nll += lambda - obs.value * lambda.ln();
                        } else {
                            // Penalty for invalid lambda (should be > 0 for Poisson)
                            nll += 1e10;
                        }
                    }
                }
                nll
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
        // 1. Validate parameter bounds
        self.validate_params(params).map_err(|e| Error::msg(e))?;

        // 2. Clone the template engine (works for ANY model type)
        let mut engine = self.template_engine.clone();

        // 3. Update parameters (generic method)
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

        // 4. Run simulation (generic method)
        let max_step = self
            .observed_data
            .iter()
            .map(|obs| obs.time_step)
            .max()
            .unwrap_or(100);

        let results = engine
            .run(max_step)
            .map_err(|e| Error::msg(format!("Simulation failed: {}", e)))?;

        // 5. Calculate loss (model-agnostic)
        let loss = self.calculate_loss(&results);

        Ok(loss)
    }
}
