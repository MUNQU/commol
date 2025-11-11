//! Core data types for the difference equations engine.

use commol_core::{MathExpressionContext, RateMathExpression};

/// Pre-computed transition flow information for performance
#[derive(Clone)]
pub(crate) struct TransitionFlow {
    pub(crate) source_index: usize,
    pub(crate) target_index: usize,
    pub(crate) rate_expression: RateMathExpression,
    /// Whether the rate references compartment variables (absolute vs per-capita)
    pub(crate) references_compartments: bool,
}

/// Pre-computed subpopulation mapping for stratifications
#[derive(Clone)]
pub(crate) struct SubpopulationMapping {
    /// Compartment indices that contribute to this subpopulation total
    pub(crate) contributing_compartment_indices: Vec<usize>,
    /// Parameter name for this subpopulation (e.g., "N_young")
    pub(crate) parameter_name: String,
}

/// Difference equations simulation engine.
///
/// This struct represents a compiled compartment model using difference equations
/// for discrete-time simulation. It pre-computes transition flows and stratification
/// mappings for efficient simulation.
#[derive(Clone)]
pub struct DifferenceEquations {
    pub(crate) compartments: Vec<String>,
    pub(crate) population: Vec<f64>,
    pub(crate) expression_context: MathExpressionContext,
    pub(crate) current_step: f64,
    /// Store initial state for reset functionality
    pub(crate) initial_population: Vec<f64>,
    /// Pre-computed transition flows for performance
    pub(crate) transition_flows: Vec<TransitionFlow>,
    /// Reusable buffer for compartment flows to avoid allocations
    pub(crate) compartment_flows: Vec<f64>,
    /// Pre-computed subpopulation mappings for stratifications
    pub(crate) subpopulation_mappings: Vec<SubpopulationMapping>,
    /// Parameters defined as formulas that need to be evaluated each step
    pub(crate) formula_parameters: Vec<(String, RateMathExpression)>,
}
