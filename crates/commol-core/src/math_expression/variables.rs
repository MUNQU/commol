//! The engine variables: the names every formula can read.

/// The total: the sum of every compartment at the current position,
/// accumulators excluded.
pub const SPECIAL_VAR_TOTAL: &str = "total";
/// The prefix of a subgroup total: `total_<cat>…` is the sum of the
/// compartments with the categories `<cat>…`.
pub const SUBGROUP_TOTAL_PREFIX: &str = "total_";
/// The step that holds the current position: `floor(t)`.
pub const SPECIAL_VAR_STEP: &str = "step";
/// The current position on the axis. An engine that reports only at integer
/// steps (`DifferenceEquations`, `StochasticSteps`) reads it at each step, so
/// there it equals `step`; `DifferentialEquations` and `StochasticEvents` read
/// it anywhere.
pub const SPECIAL_VAR_T: &str = "t";
/// The constant π.
pub const SPECIAL_VAR_PI: &str = "pi";
/// The constant e.
pub const SPECIAL_VAR_E: &str = "e";
/// Every variable the engine provides in every expression. No parameter, bin
/// or accumulator may take one of these ids.
pub const ENGINE_VARIABLES: [&str; 5] = [
    SPECIAL_VAR_TOTAL,
    SPECIAL_VAR_STEP,
    SPECIAL_VAR_T,
    SPECIAL_VAR_PI,
    SPECIAL_VAR_E,
];
