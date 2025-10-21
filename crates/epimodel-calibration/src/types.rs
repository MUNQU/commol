//! Type definitions for calibration

use serde::{Deserialize, Serialize};

/// Represents an observed data point to fit against
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedDataPoint {
    /// Time step at which this observation was made
    pub time_step: u32,

    /// Name of the compartment being observed
    pub compartment: String,

    /// Observed value (e.g., number of infected individuals)
    pub value: f64,

    /// Weight for this observation (default 1.0)
    /// Higher weights give more importance to this data point in the loss function
    pub weight: f64,
}

impl ObservedDataPoint {
    /// Create a new observed data point with default weight of 1.0
    pub fn new(time_step: u32, compartment: String, value: f64) -> Self {
        Self {
            time_step,
            compartment,
            value,
            weight: 1.0,
        }
    }

    /// Create a new observed data point with a custom weight
    pub fn with_weight(time_step: u32, compartment: String, value: f64, weight: f64) -> Self {
        Self {
            time_step,
            compartment,
            value,
            weight,
        }
    }
}

/// Parameter to be calibrated with its bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationParameter {
    /// Parameter identifier (must match model parameter ID)
    pub id: String,

    /// Minimum allowed value
    pub min_bound: f64,

    /// Maximum allowed value
    pub max_bound: f64,

    /// Optional initial guess (if None, will use midpoint of bounds)
    pub initial_guess: Option<f64>,
}

impl CalibrationParameter {
    /// Create a new calibration parameter
    pub fn new(id: String, min_bound: f64, max_bound: f64) -> Self {
        Self {
            id,
            min_bound,
            max_bound,
            initial_guess: None,
        }
    }

    /// Create a new calibration parameter with initial guess
    pub fn with_initial_guess(
        id: String,
        min_bound: f64,
        max_bound: f64,
        initial_guess: f64,
    ) -> Self {
        Self {
            id,
            min_bound,
            max_bound,
            initial_guess: Some(initial_guess),
        }
    }

    /// Get the initial value, or midpoint of bounds if not specified
    pub fn initial_value(&self) -> f64 {
        self.initial_guess
            .unwrap_or_else(|| (self.min_bound + self.max_bound) / 2.0)
    }

    /// Check if a value is within the parameter bounds
    pub fn is_within_bounds(&self, value: f64) -> bool {
        value >= self.min_bound && value <= self.max_bound
    }
}

/// Configuration for loss function calculation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LossConfig {
    /// Sum of squared errors: Σ(observed - predicted)²
    SumSquaredError,

    /// Root mean squared error: √(Σ(observed - predicted)² / n)
    RootMeanSquaredError,

    /// Mean absolute error: Σ|observed - predicted| / n
    MeanAbsoluteError,

    /// Weighted sum of squared errors (uses observation weights)
    WeightedSSE,
}

impl Default for LossConfig {
    fn default() -> Self {
        LossConfig::SumSquaredError
    }
}

impl std::fmt::Display for LossConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LossConfig::SumSquaredError => write!(f, "Sum Squared Error"),
            LossConfig::RootMeanSquaredError => write!(f, "Root Mean Squared Error"),
            LossConfig::MeanAbsoluteError => write!(f, "Mean Absolute Error"),
            LossConfig::WeightedSSE => write!(f, "Weighted Sum Squared Error"),
        }
    }
}

/// Result from a calibration run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Best parameter values found
    pub best_parameters: Vec<f64>,

    /// Parameter names (in same order as best_parameters)
    pub parameter_names: Vec<String>,

    /// Final loss value achieved
    pub final_loss: f64,

    /// Number of iterations performed
    pub iterations: usize,

    /// Whether the optimization converged
    pub converged: bool,

    /// Termination reason
    pub termination_reason: String,
}

impl CalibrationResult {
    /// Get parameters as a HashMap for easy lookup
    pub fn parameters_map(&self) -> std::collections::HashMap<String, f64> {
        self.parameter_names
            .iter()
            .zip(self.best_parameters.iter())
            .map(|(name, value)| (name.clone(), *value))
            .collect()
    }
}
