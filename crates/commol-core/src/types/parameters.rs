use serde::{Deserialize, Serialize};

/// Prefixes of a `Rule` variable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VariablePrefixes {
    /// `state:<bin>`.
    #[serde(rename = "state")]
    State,
    /// `strat:<stratification>`.
    #[serde(rename = "strat")]
    Strat,
}

/// How a [`ParameterValue::TimeSeries`] is interpolated between listed steps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SeriesMode {
    /// Non-zero only at the listed steps; returns 0.0 at all other steps.
    /// Use for discrete events.
    #[serde(rename = "pulse")]
    Pulse,
    /// Returns the value of the most recent listed step at or before the current
    /// step (zero-order hold); returns 0.0 before the first listed step.
    /// Use for values that change in steps.
    #[serde(rename = "step_function")]
    StepFunction,
    /// Linearly interpolates between adjacent listed steps; returns 0.0 outside
    /// the listed range.
    /// Use for smooth empirical schedules.
    #[serde(rename = "linear")]
    Linear,
}

/// Represents different types of parameter values
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParameterValue {
    /// Constant numeric value
    Constant(f64),
    /// Formula expression that can reference other parameters or special variables
    Formula(String),
    /// Empirical time series: a sorted list of (step, value) pairs evaluated
    /// according to the chosen [`SeriesMode`].  Efficient O(log N) lookup —
    /// avoids embedding large lookup tables as formula strings.
    TimeSeries {
        /// (step, value) pairs.  Need not be pre-sorted; the engine sorts them
        /// at build time.
        data: Vec<(u64, f64)>,
        /// How the steps between the pairs are read.
        mode: SeriesMode,
    },
}

impl ParameterValue {
    /// Check if this is a constant value
    pub fn is_constant(&self) -> bool {
        matches!(self, ParameterValue::Constant(_))
    }

    /// Check if this is a formula
    pub fn is_formula(&self) -> bool {
        matches!(self, ParameterValue::Formula(_))
    }

    /// Check if this is a time series
    pub fn is_time_series(&self) -> bool {
        matches!(self, ParameterValue::TimeSeries { .. })
    }

    /// Get the constant value if this is a constant, otherwise None
    pub fn as_constant(&self) -> Option<f64> {
        match self {
            ParameterValue::Constant(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the formula string if this is a formula, otherwise None
    pub fn as_formula(&self) -> Option<&str> {
        match self {
            ParameterValue::Formula(s) => Some(s),
            _ => None,
        }
    }
}

impl From<f64> for ParameterValue {
    fn from(value: f64) -> Self {
        ParameterValue::Constant(value)
    }
}

impl From<String> for ParameterValue {
    fn from(value: String) -> Self {
        ParameterValue::Formula(value)
    }
}

impl From<&str> for ParameterValue {
    fn from(value: &str) -> Self {
        ParameterValue::Formula(value.to_string())
    }
}

/// A parameter definition with its value and optional description
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parameter {
    /// Identifier used in expressions.
    pub id: String,
    /// Parameter value - None indicates the parameter needs calibration
    pub value: Option<ParameterValue>,
    /// Free-text description.
    pub description: Option<String>,
}

impl Parameter {
    /// Create a new parameter with a constant value
    pub fn new_constant(id: String, value: f64, description: Option<String>) -> Self {
        Self {
            id,
            value: Some(ParameterValue::Constant(value)),
            description,
        }
    }

    /// Create a new parameter with a formula
    pub fn new_formula(id: String, formula: String, description: Option<String>) -> Self {
        Self {
            id,
            value: Some(ParameterValue::Formula(formula)),
            description,
        }
    }

    /// Create a new parameter with None value (needs calibration)
    pub fn new_uncalibrated(id: String, description: Option<String>) -> Self {
        Self {
            id,
            value: None,
            description,
        }
    }

    /// Check if this parameter has a constant value
    pub fn is_constant(&self) -> bool {
        self.value.as_ref().is_some_and(|v| v.is_constant())
    }

    /// Check if this parameter has a formula
    pub fn is_formula(&self) -> bool {
        self.value.as_ref().is_some_and(|v| v.is_formula())
    }

    /// Check if this parameter needs calibration (value is None)
    pub fn needs_calibration(&self) -> bool {
        self.value.is_none()
    }
}
