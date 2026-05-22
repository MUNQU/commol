//! Core data types for the difference equations engine.

use commol_core::{MathExpressionContext, RateMathExpression, SeriesMode};

/// Pre-computed transition flow information for performance
#[derive(Clone)]
pub(crate) struct TransitionFlow {
    pub(crate) source_index: usize,
    pub(crate) target_index: usize,
    pub(crate) rate_expression: RateMathExpression,
    /// Whether the rate references compartment variables (absolute vs per-capita)
    pub(crate) references_compartments: bool,
}

/// Pre-computed time-series parameter for O(log N) step lookup.
#[derive(Clone)]
pub(crate) struct TimeSeriesParameter {
    pub(crate) parameter_name: String,
    /// Sorted by step for binary search.
    data: Vec<(u64, f64)>,
    mode: SeriesMode,
}

impl TimeSeriesParameter {
    pub(crate) fn new(parameter_name: String, mut data: Vec<(u64, f64)>, mode: SeriesMode) -> Self {
        data.sort_unstable_by_key(|(step, _)| *step);
        Self {
            parameter_name,
            data,
            mode,
        }
    }

    pub(crate) fn evaluate(&self, step: u64) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        match self.mode {
            SeriesMode::Pulse => match self.data.binary_search_by_key(&step, |(s, _)| *s) {
                Ok(idx) => self.data[idx].1,
                Err(_) => 0.0,
            },
            SeriesMode::StepFunction => {
                let pos = self.data.partition_point(|(s, _)| *s <= step);
                if pos == 0 { 0.0 } else { self.data[pos - 1].1 }
            }
            SeriesMode::Linear => {
                let first_step = self.data[0].0;
                let last_step = self.data[self.data.len() - 1].0;
                if step < first_step || step > last_step {
                    return 0.0;
                }
                let pos = self.data.partition_point(|(s, _)| *s <= step);
                if pos == 0 {
                    return self.data[0].1;
                }
                if pos >= self.data.len() {
                    return self.data[self.data.len() - 1].1;
                }
                let (s0, v0) = self.data[pos - 1];
                let (s1, v1) = self.data[pos];
                if s0 == s1 {
                    return v0;
                }
                let t = (step - s0) as f64 / (s1 - s0) as f64;
                v0 + t * (v1 - v0)
            }
        }
    }
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
    /// Parameters defined as empirical time series; evaluated via binary search each step
    pub(crate) series_parameters: Vec<TimeSeriesParameter>,
}
