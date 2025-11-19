//! Type definitions for calibration

use serde::{Deserialize, Serialize};

/// Type of value being calibrated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationParameterType {
    /// Model parameter (e.g., beta, gamma)
    Parameter,
    /// Initial population in a compartment (e.g., initial I value)
    InitialCondition,
    /// Scaling factor for observed data (multiplies model output before comparison)
    Scale,
}

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

    /// Optional scale parameter ID to apply to model output before comparison
    /// If provided, the model's predicted value will be multiplied by this scale parameter
    /// before computing the loss. Useful when observed data is in different units or
    /// there's an unknown proportionality constant.
    pub scale_id: Option<String>,
}

impl ObservedDataPoint {
    /// Create a new observed data point with default weight of 1.0
    pub fn new(time_step: u32, compartment: String, value: f64) -> Self {
        Self {
            time_step,
            compartment,
            value,
            weight: 1.0,
            scale_id: None,
        }
    }

    /// Create a new observed data point with a custom weight
    pub fn with_weight(time_step: u32, compartment: String, value: f64, weight: f64) -> Self {
        Self {
            time_step,
            compartment,
            value,
            weight,
            scale_id: None,
        }
    }

    /// Create a new observed data point with a scale parameter
    pub fn with_scale(time_step: u32, compartment: String, value: f64, scale_id: String) -> Self {
        Self {
            time_step,
            compartment,
            value,
            weight: 1.0,
            scale_id: Some(scale_id),
        }
    }

    /// Create a new observed data point with both weight and scale
    pub fn with_weight_and_scale(
        time_step: u32,
        compartment: String,
        value: f64,
        weight: f64,
        scale_id: String,
    ) -> Self {
        Self {
            time_step,
            compartment,
            value,
            weight,
            scale_id: Some(scale_id),
        }
    }
}

/// Parameter to be calibrated with its bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationParameter {
    /// Parameter identifier (parameter ID for parameters, bin ID for initial conditions)
    pub id: String,

    /// Type of value being calibrated
    pub parameter_type: CalibrationParameterType,

    /// Minimum allowed value
    pub min_bound: f64,

    /// Maximum allowed value
    pub max_bound: f64,

    /// Optional initial guess (if None, will use midpoint of bounds)
    pub initial_guess: Option<f64>,
}

impl CalibrationParameter {
    /// Create a new calibration parameter (defaults to Parameter type)
    pub fn new(id: String, min_bound: f64, max_bound: f64) -> Self {
        Self {
            id,
            parameter_type: CalibrationParameterType::Parameter,
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
            parameter_type: CalibrationParameterType::Parameter,
            min_bound,
            max_bound,
            initial_guess: Some(initial_guess),
        }
    }

    /// Create a new calibration parameter with explicit type
    pub fn with_type(
        id: String,
        parameter_type: CalibrationParameterType,
        min_bound: f64,
        max_bound: f64,
    ) -> Self {
        Self {
            id,
            parameter_type,
            min_bound,
            max_bound,
            initial_guess: None,
        }
    }

    /// Create a new calibration parameter with type and initial guess
    pub fn with_type_and_guess(
        id: String,
        parameter_type: CalibrationParameterType,
        min_bound: f64,
        max_bound: f64,
        initial_guess: f64,
    ) -> Self {
        Self {
            id,
            parameter_type,
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum LossConfig {
    /// Sum of squared errors: sum(observed - predicted) ** 2
    #[default]
    SumSquaredError,

    /// Root mean squared error: sqrt(sum(observed - predicted) ** 2 / n)
    RootMeanSquaredError,

    /// Mean absolute error: sum(abs(observed - predicted)) / n
    MeanAbsoluteError,

    /// Weighted sum of squared errors (uses observation weights)
    WeightedSSE,
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
