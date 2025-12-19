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

/// A constraint on calibration parameters and/or compartment values defined as a mathematical expression
///
/// Constraints are mathematical expressions that must evaluate to >= 0 for the
/// constraint to be satisfied. When the expression evaluates to < 0, the constraint
/// is violated and a penalty is applied.
///
/// # Constraint Types
///
/// **Parameter-only constraints** (`time_steps: None`):
/// - Evaluated once before simulation starts
/// - Can only reference calibration parameter IDs
/// - Use for relationships between parameters
///
/// **Time-dependent constraints** (`time_steps: Some(vec)`):
/// - Evaluated at specified time steps during simulation
/// - Can reference both parameter IDs AND compartment names
/// - Use for constraining compartment values or dynamic relationships
///
/// # Examples
///
/// ## Parameter-only constraints:
/// ```rust,ignore
/// // R0 = beta/gamma must be <= 5
/// CalibrationConstraint::new(
///     "r0_bound".to_string(),
///     "5.0 - beta/gamma".to_string(),
/// )
///
/// // beta must be >= gamma
/// CalibrationConstraint::new(
///     "ordering".to_string(),
///     "beta - gamma".to_string(),
/// )
///
/// // Sum of parameters <= 1.0
/// CalibrationConstraint::new(
///     "sum_bound".to_string(),
///     "1.0 - (param1 + param2 + param3)".to_string(),
/// )
/// ```
///
/// ## Time-dependent (compartment value) constraints:
/// ```rust,ignore
/// // Infected compartment must stay <= 500 at specific time steps
/// CalibrationConstraint::new(
///     "max_infected".to_string(),
///     "500.0 - I".to_string(),
/// )
/// .with_time_steps(vec![10, 20, 30, 40])
///
/// // Total population must stay <= 1000
/// CalibrationConstraint::new(
///     "population_bound".to_string(),
///     "1000.0 - (S + I + R)".to_string(),
/// )
/// .with_time_steps(vec![0, 5, 10, 15, 20])
///
/// // Ratio of compartments with parameter reference
/// CalibrationConstraint::new(
///     "ratio_bound".to_string(),
///     "threshold - I/S".to_string(),
/// )
/// .with_time_steps(vec![10, 20, 30])
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConstraint {
    /// Unique identifier for this constraint
    pub id: String,

    /// Mathematical expression that must evaluate >= 0 for constraint satisfaction
    ///
    /// The expression can reference calibration parameters by their IDs.
    /// When time_steps is specified, the expression can also reference
    /// compartment values (S, I, R, etc.) at those time steps.
    ///
    /// Examples:
    /// - "5.0 - beta/gamma" → R0 = beta/gamma must be <= 5
    /// - "beta - gamma" → beta >= gamma
    /// - "1.0 - (param1 + param2)" → param1 + param2 <= 1.0
    /// - "500.0 - I" (with time_steps) → Infected compartment <= 500 at specified times
    pub expression: String,

    /// Human-readable description (optional, for diagnostics)
    pub description: Option<String>,

    /// Penalty weight multiplier (default 1.0)
    ///
    /// Higher values make this constraint more important relative to others.
    /// The penalty for violating this constraint is: weight * violation^2
    pub weight: f64,

    /// Optional time steps at which to evaluate this constraint
    ///
    /// - If None: Constraint is evaluated once before simulation using parameter values only
    /// - If Some(vec): Constraint is evaluated at each specified time step during simulation,
    ///   and can reference both parameters and compartment values
    pub time_steps: Option<Vec<u32>>,
}

impl CalibrationConstraint {
    /// Create a new calibration constraint
    pub fn new(id: String, expression: String) -> Self {
        Self {
            id,
            expression,
            description: None,
            weight: 1.0,
            time_steps: None,
        }
    }

    /// Set the description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Set the weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Set the time steps
    pub fn with_time_steps(mut self, time_steps: Vec<u32>) -> Self {
        self.time_steps = Some(time_steps);
        self
    }

    /// Check if this is a parameter-only constraint
    pub fn is_parameter_only(&self) -> bool {
        self.time_steps.is_none()
    }

    /// Check if this is a time-dependent constraint
    pub fn is_time_dependent(&self) -> bool {
        self.time_steps.is_some()
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

/// A single evaluation during calibration optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationEvaluation {
    /// Parameter values for this evaluation
    pub parameters: Vec<f64>,

    /// Loss value for this parameter set
    pub loss: f64,

    /// Model predictions at each time step for all compartments
    /// predictions[time_step][compartment_idx]
    pub predictions: Vec<Vec<f64>>,
}

/// Result from a calibration run with history of all evaluations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResultWithHistory {
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

    /// All unique evaluations performed during optimization (deduplicated)
    pub evaluations: Vec<CalibrationEvaluation>,
}

impl CalibrationResultWithHistory {
    /// Convert to a standard CalibrationResult (without history)
    pub fn to_standard_result(&self) -> CalibrationResult {
        CalibrationResult {
            best_parameters: self.best_parameters.clone(),
            parameter_names: self.parameter_names.clone(),
            final_loss: self.final_loss,
            iterations: self.iterations,
            converged: self.converged,
            termination_reason: self.termination_reason.clone(),
        }
    }

    /// Get parameters as a HashMap for easy lookup
    pub fn parameters_map(&self) -> std::collections::HashMap<String, f64> {
        self.parameter_names
            .iter()
            .zip(self.best_parameters.iter())
            .map(|(name, value)| (name.clone(), *value))
            .collect()
    }
}
