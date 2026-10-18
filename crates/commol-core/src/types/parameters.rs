use serde::{Deserialize, Serialize};

/// How a [`ParameterValue::Series`] is read between its points.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SeriesMode {
    /// Non-zero only on the step of a listed point, read at `step`; 0.0 at
    /// every other step. Its points must be integers. Use for discrete
    /// events.
    #[serde(rename = "pulse")]
    Pulse,
    /// Returns the value of the last point at or before `t` (zero-order
    /// hold); returns 0.0 before the first point. Use for values that change
    /// at given positions.
    #[serde(rename = "step_function")]
    StepFunction,
    /// Interpolates linearly in `t` between adjacent points; returns 0.0
    /// outside the listed range. Use for smooth empirical schedules.
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
    /// Empirical series: a list of (position, value) points evaluated
    /// according to the chosen [`SeriesMode`].  Efficient O(log N) lookup —
    /// avoids embedding large lookup tables as formula strings.
    Series {
        /// (position, value) points.  Need not be pre-sorted; the engine sorts
        /// them at build time.
        data: Vec<(f64, f64)>,
        /// How the positions between the points are read.
        mode: SeriesMode,
    },
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
