//! Data types of a compiled model.

use commol_core::{MathExpressionContext, RateMathExpression, SeriesMode};

/// Pre-resolved storage location for a JIT-compiled rate's input variable.
///
/// Built once at engine compile time and reused on every simulation step so the
/// inner loop avoids HashMap lookups for variables whose location is already
/// known (compartments map to a direct `Vec` index; the step alias is a single
/// scalar). Parameter lookups still go through the HashMap because their
/// storage is name-keyed in `MathExpressionContext`.
#[derive(Clone)]
pub(crate) enum VarSlot {
    /// Direct index into the engine's `population` vector.
    Compartment(usize),
    /// Direct index into `MathExpressionContext`'s parameter value storage.
    ParameterIndex(usize),
    /// Look up by name in `MathExpressionContext.parameters` (covers user
    /// parameters, formula parameters, time-series, subpopulation totals,
    /// base bin sums, and partial bin-strat sums).
    Parameter(String),
    /// The step that holds the current time (`step`).
    Step,
    /// The current time (`t`). Equal to `step` in a difference model.
    Time,
    /// A mathematical constant (`pi` or `e`).
    Constant(f64),
}

#[derive(Clone)]
pub(crate) enum FastRateExpression {
    Constant(f64),
    Slot(VarSlot),
    Mul2(VarSlot, VarSlot),
    Mul3(VarSlot, VarSlot, VarSlot),
    OneMinusMul3 {
        subtract: VarSlot,
        factor: VarSlot,
        value: VarSlot,
    },
    MulAddDiv {
        first: VarSlot,
        second: VarSlot,
        add_left: VarSlot,
        add_right: VarSlot,
        denominator: VarSlot,
    },
    Program(Vec<FastRateOp>),
}

#[derive(Clone)]
pub(crate) enum FastRateOp {
    Constant(f64),
    Slot(VarSlot),
    Add,
    Sub,
    Mul,
    Div,
    Neg,
}

/// Pre-computed transition flow information for performance
#[derive(Clone)]
pub(crate) struct TransitionFlow {
    /// `None` for source-less transitions.
    pub(crate) source_index: Option<usize>,
    /// `None` for target-less transitions.
    pub(crate) target_index: Option<usize>,
    /// Further `(source, target)` pieces of a transition with several sources
    /// or targets. Each moves the same flow as the first piece.
    pub(crate) extra_pieces: Vec<(Option<usize>, Option<usize>)>,
    /// Accumulator output indices incremented by this transition flow.
    pub(crate) accumulator_indices: Vec<usize>,
    pub(crate) rate_expression: RateMathExpression,
    /// Direct evaluator for common arithmetic-only rates. This avoids JIT call
    /// overhead for simple transition expressions and falls back to
    /// `rate_expression` for unsupported formulas.
    pub(crate) fast_rate_expression: Option<FastRateExpression>,
    /// Whether the rate expression is an absolute flow rather than a per-capita rate.
    pub(crate) is_absolute_flow: bool,
    /// Pre-resolved input slots for the JIT path. `Some` when the rate is a
    /// JIT-compiled formula whose variables we could resolve at build time;
    /// `None` for `Parameter` / `Constant` rates and for `evalexpr` fallbacks.
    pub(crate) resolved_slots: Option<Vec<VarSlot>>,
}

/// Positions on the axis where an impulse transition acts.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ImpulseSchedule {
    /// The listed positions, sorted and without duplicates.
    At(Vec<f64>),
    /// `offset`, `offset + period`, `offset + 2 * period`, …
    Every { period: f64, offset: f64 },
}

/// Relative tolerance with which a position matches an impulse position.
const POSITION_TOLERANCE: f64 = 1e-9;

fn same_position(a: f64, b: f64) -> bool {
    (a - b).abs() <= POSITION_TOLERANCE * a.abs().max(b.abs()).max(1.0)
}

impl ImpulseSchedule {
    /// Whether the impulse acts at `position`.
    pub(crate) fn fires_at(&self, position: f64) -> bool {
        match self {
            Self::At(positions) => positions.iter().any(|&p| same_position(p, position)),
            Self::Every { period, offset } => {
                let k = ((position - offset) / period).round();
                k >= 0.0 && same_position(offset + k * period, position)
            }
        }
    }

    /// Positions in `[from, to]` where the impulse acts, in ascending order.
    pub(crate) fn positions_in(&self, from: f64, to: f64) -> Vec<f64> {
        match self {
            Self::At(positions) => positions
                .iter()
                .copied()
                .filter(|&p| p >= from && p <= to)
                .collect(),
            Self::Every { period, offset } => {
                let first = ((from - offset) / period).ceil().max(0.0);
                let mut positions = Vec::new();
                let mut k = first;
                loop {
                    let position = offset + k * period;
                    if position > to {
                        break;
                    }
                    positions.push(position);
                    k += 1.0;
                }
                positions
            }
        }
    }

    /// The first position that is not an integer, if any.
    pub(crate) fn first_non_integer(&self) -> Option<f64> {
        match self {
            Self::At(positions) => positions.iter().copied().find(|p| p.fract() != 0.0),
            Self::Every { period, offset } => [*offset, *period]
                .into_iter()
                .find(|value| value.fract() != 0.0),
        }
    }
}

/// A transition that moves its rate at once at the positions of a schedule.
#[derive(Clone)]
pub(crate) struct ImpulseFlow {
    /// Id of the impulse transition.
    pub(crate) transition_id: String,
    /// Pieces, rate and accumulators, compiled as for any transition.
    pub(crate) flow: TransitionFlow,
    pub(crate) schedule: ImpulseSchedule,
}

/// Pre-computed time-series parameter for O(log N) step lookup.
#[derive(Clone)]
pub(crate) struct TimeSeriesParameter {
    pub(crate) parameter_name: String,
    pub(crate) parameter_index: usize,
    /// Sorted by step for binary search.
    data: Vec<(u64, f64)>,
    mode: SeriesMode,
}

impl TimeSeriesParameter {
    pub(crate) fn new(
        parameter_name: String,
        parameter_index: usize,
        mut data: Vec<(u64, f64)>,
        mode: SeriesMode,
    ) -> Self {
        data.sort_unstable_by_key(|(step, _)| *step);
        Self {
            parameter_name,
            parameter_index,
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
    /// Index into `MathExpressionContext`'s parameter value storage, filled in
    /// after parameter slots have been reserved.
    pub(crate) parameter_index: usize,
}

/// Subpopulation mapping with only the structural data known before
/// parameter slot reservation. The builder converts this into a
/// [`SubpopulationMapping`] once the expression context indices exist.
pub(crate) struct SubpopulationLayout {
    pub(crate) contributing_compartment_indices: Vec<usize>,
    pub(crate) parameter_name: String,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct AccumulatorOutputRef {
    pub(crate) accumulator_id: String,
    pub(crate) categories: Vec<Option<String>>,
}

#[derive(Clone)]
pub(crate) struct GeneratedAccumulatorOutput {
    pub(crate) name: String,
    pub(crate) output_ref: AccumulatorOutputRef,
}

/// A model compiled into the right-hand side that every engine integrates.
///
/// It holds the expanded compartments and accumulators, the planned flows with
/// their compiled rates, and the initial state. The state of a model is one
/// vector: the compartments, in engine order, then the accumulators.
/// [`CompiledModel::derivatives`] is the right-hand side `dX/dt = Σ flows`; a
/// difference engine adds it once per step, a differential engine integrates
/// it.
#[derive(Clone)]
pub struct CompiledModel {
    pub(crate) compartments: Vec<String>,
    /// Source and applied categories of each output, parallel to `output_names`
    pub(crate) output_labels: Vec<commol_core::OutputLabels>,
    pub(crate) output_names: Vec<String>,
    /// Initial state: compartments, then accumulators (which start at zero).
    pub(crate) initial_state: Vec<f64>,
    pub(crate) expression_context: MathExpressionContext,
    /// Pre-computed transition flows for performance
    pub(crate) transition_flows: Vec<TransitionFlow>,
    /// Impulse transitions: they move their rate at once at their positions
    /// and are not part of the right-hand side.
    pub(crate) impulse_flows: Vec<ImpulseFlow>,
    /// Whether a rate or a formula parameter reads `step` or a time series,
    /// so the right-hand side jumps at integer positions.
    pub(crate) depends_on_step: bool,
    /// Pre-computed subpopulation mappings for stratifications
    pub(crate) subpopulation_mappings: Vec<SubpopulationMapping>,
    /// Parameters defined as formulas that need to be evaluated each step
    pub(crate) formula_parameters: Vec<(String, RateMathExpression)>,
    /// Parameters defined as empirical time series; evaluated via binary search each step
    pub(crate) series_parameters: Vec<TimeSeriesParameter>,
    pub(crate) n_parameter_index: usize,
    pub(crate) t_parameter_index: usize,
    /// Whether any per-step expression still needs compartment values in the
    /// expression context. JIT-resolved transition formulas read compartments
    /// directly from the state and do not need this HashMap update.
    pub(crate) requires_compartment_context: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulse_series_only_fires_on_listed_steps() {
        let series = TimeSeriesParameter::new(
            "x".to_string(),
            0,
            vec![(5, 2.0), (1, 1.0)],
            SeriesMode::Pulse,
        );

        assert_eq!(series.evaluate(0), 0.0);
        assert_eq!(series.evaluate(1), 1.0);
        assert_eq!(series.evaluate(2), 0.0);
        assert_eq!(series.evaluate(5), 2.0);
    }

    #[test]
    fn step_function_series_holds_last_value() {
        let series = TimeSeriesParameter::new(
            "x".to_string(),
            0,
            vec![(10, 3.0), (3, 1.5)],
            SeriesMode::StepFunction,
        );

        assert_eq!(series.evaluate(2), 0.0);
        assert_eq!(series.evaluate(3), 1.5);
        assert_eq!(series.evaluate(9), 1.5);
        assert_eq!(series.evaluate(10), 3.0);
        assert_eq!(series.evaluate(20), 3.0);
    }

    #[test]
    fn linear_series_interpolates_between_points() {
        let series = TimeSeriesParameter::new(
            "x".to_string(),
            0,
            vec![(10, 10.0), (0, 0.0)],
            SeriesMode::Linear,
        );

        assert_eq!(series.evaluate(0), 0.0);
        assert_eq!(series.evaluate(5), 5.0);
        assert_eq!(series.evaluate(10), 10.0);
        assert_eq!(series.evaluate(11), 0.0);
    }
}
