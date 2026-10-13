use serde::{Deserialize, Serialize};

use crate::math_expression::RateMathExpression;

/// Model typologies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelTypes {
    /// Discrete time: each step applies every flow once.
    #[serde(rename = "DifferenceEquations")]
    DifferenceEquations,
    /// Continuous time: the engine integrates `dX/dt = Σ flows` with an ODE
    /// solver (`commol-differential`).
    #[serde(rename = "DifferentialEquations")]
    DifferentialEquations,
}

/// Integration method of a differential-equations model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OdeMethod {
    /// Explicit Tsitouras 5(4) Runge–Kutta method, for non-stiff models.
    #[default]
    #[serde(rename = "tsit45")]
    Tsit45,
    /// Backward differentiation formulas, for stiff models.
    #[serde(rename = "bdf")]
    Bdf,
}

/// Solver settings of a differential-equations model.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OdeSolver {
    /// Integration method.
    #[serde(default)]
    pub method: OdeMethod,
    /// Relative tolerance of each state component.
    #[serde(default = "OdeSolver::default_rtol")]
    pub rtol: f64,
    /// Absolute tolerance of each state component.
    #[serde(default = "OdeSolver::default_atol")]
    pub atol: f64,
}

impl OdeSolver {
    /// Default relative tolerance.
    pub const DEFAULT_RTOL: f64 = 1e-6;
    /// Default absolute tolerance.
    pub const DEFAULT_ATOL: f64 = 1e-6;

    fn default_rtol() -> f64 {
        Self::DEFAULT_RTOL
    }

    fn default_atol() -> f64 {
        Self::DEFAULT_ATOL
    }
}

impl Default for OdeSolver {
    fn default() -> Self {
        Self {
            method: OdeMethod::default(),
            rtol: Self::DEFAULT_RTOL,
            atol: Self::DEFAULT_ATOL,
        }
    }
}

/// Condition that specifies a stratification category for rate matching.
///
/// When `to` is set, the matched category in the source compartment is replaced
/// with the `to` category in the target compartment name. This enables
/// cross-category transitions within the same bin (for example `g1` → `g2`).
///
/// When `category` is `None`, the condition acts as a target-only override:
/// it does not filter source compartments but still contributes a `to` category
/// to the computed target compartment name. This is required when the source
/// compartment lacks a stratification that the target compartment must have
/// (e.g., an `A_cat0` compartment routing into an `A_cat1_sub0` compartment).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratificationCondition {
    /// Id of the stratification.
    pub stratification: String,
    /// Category the source compartment must have; `None` filters nothing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Category the target compartment gets instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}

/// Rate that applies to a specific stratification condition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratifiedRate {
    /// Conditions that must all match the source compartment.
    pub conditions: Vec<StratificationCondition>,
    /// The rate expression.
    pub rate: String,
    /// Forces the rate to be absolute (`true`) or per-capita (`false`); `None` infers it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub absolute: Option<bool>,
}

/// A flow between bins, or between the categories of one stratification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transition {
    /// Identifier.
    pub id: String,
    /// Bins (or one category) that lose the flow; empty for an inflow.
    pub source: Vec<String>,
    /// Bins (or one category) that gain the flow; empty for an outflow.
    pub target: Vec<String>,
    /// Ids of the accumulators that add up the flow.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accumulators: Vec<String>,
    /// The rate, or the fallback rate when `stratified_rates` is set.
    pub rate: Option<RateMathExpression>,
    /// Rates for compartments with some categories.
    pub stratified_rates: Option<Vec<StratifiedRate>>,
    /// When true, base compartment names in the rate expression are replaced
    /// with the specific stratified compartment name for each expanded flow.
    pub per_compartment: Option<bool>,
    /// Positions on the axis where the transition is an impulse: at each one,
    /// the rate is a fraction (per-capita) or an amount (absolute) moved at
    /// once, and the transition moves nothing between them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impulse_at: Option<Vec<f64>>,
    /// `(period, offset)`: the transition is an impulse at `offset`,
    /// `offset + period`, `offset + 2 * period`, …
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impulse_every: Option<(f64, f64)>,
}

impl Transition {
    /// Whether the transition is an impulse rather than a continuous flow.
    pub fn is_impulse(&self) -> bool {
        self.impulse_at.is_some() || self.impulse_every.is_some()
    }
}

/// Model dynamics specification including type and transitions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dynamics {
    /// The typology of the model.
    pub typology: ModelTypes,
    /// The transitions the engine plans and simulates.
    pub transitions: Vec<Transition>,
    /// Solver settings of a differential-equations model; `None` uses the
    /// defaults. A difference-equations model has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver: Option<OdeSolver>,
}
