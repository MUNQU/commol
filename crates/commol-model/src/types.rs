//! Data types of a compiled model.

use commol_core::{MathExpression, MathExpressionContext, RateMathExpression, SeriesMode};

/// Pre-resolved storage location for a JIT-compiled rate's input variable.
///
/// Built once at engine compile time and reused on every simulation step so the
/// inner loop avoids HashMap lookups for variables whose location is already
/// known (compartments map to a direct `Vec` index; the step alias is a single
/// scalar). Parameter lookups still go through the HashMap because their
/// storage is name-keyed in `MathExpressionContext`.
#[derive(Clone)]
pub(crate) enum VarSlot {
    /// Direct index into the amounts of the compartments in the state.
    Compartment(usize),
    /// Direct index into `MathExpressionContext`'s parameter value storage.
    ParameterIndex(usize),
    /// Look up by name in `MathExpressionContext.parameters` (covers user
    /// parameters, formula parameters, series, subgroup totals,
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

/// The depth of the stack a program of a fast rate runs on; a rate whose
/// program needs more takes the JIT.
pub(crate) const FAST_STACK_DEPTH: usize = 32;

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
    /// Id of the transition the flow comes from.
    pub(crate) transition_id: String,
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
    /// Whether the rate expression is an absolute flow rather than a proportional rate.
    pub(crate) is_absolute_flow: bool,
    /// Pre-resolved input slots for the JIT path. `Some` when the rate is a
    /// JIT-compiled formula whose variables we could resolve at build time;
    /// `None` for `Parameter` / `Constant` rates.
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

/// Whether two positions on the axis are the same, within
/// [`POSITION_TOLERANCE`].
pub(crate) fn same_position(a: f64, b: f64) -> bool {
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
    /// Pieces, rate and accumulators, compiled as for any transition.
    pub(crate) flow: TransitionFlow,
    pub(crate) schedule: ImpulseSchedule,
}

/// Pre-computed series parameter for O(log N) lookup.
#[derive(Clone)]
pub(crate) struct SeriesParameter {
    pub(crate) parameter_name: String,
    pub(crate) parameter_index: usize,
    /// Sorted by position for binary search.
    data: Vec<(f64, f64)>,
    mode: SeriesMode,
}

impl SeriesParameter {
    /// Compile a series, sorted by position.
    ///
    /// # Errors
    ///
    /// Returns a message that names the parameter when a position is not
    /// finite and non-negative, two positions are the same, or a `"pulse"`
    /// series has a position that is not an integer: a pulse is a rate on
    /// one unit step.
    pub(crate) fn new(
        parameter_name: String,
        parameter_index: usize,
        mut data: Vec<(f64, f64)>,
        mode: SeriesMode,
    ) -> Result<Self, String> {
        if let Some(&(position, _)) = data.iter().find(|(p, _)| !(p.is_finite() && *p >= 0.0)) {
            return Err(format!(
                "Series parameter '{parameter_name}' has a point at {position}; a \
                 position must be finite and not negative"
            ));
        }
        if mode == SeriesMode::Pulse
            && let Some(&(position, _)) = data.iter().find(|(p, _)| p.fract() != 0.0)
        {
            return Err(format!(
                "Series parameter '{parameter_name}' is a pulse at {position}, which is \
                 not an integer step; a pulse acts on one whole step"
            ));
        }
        data.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        if let Some(pair) = data.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(format!(
                "Series parameter '{parameter_name}' has two points at {}",
                pair[0].0
            ));
        }
        Ok(Self {
            parameter_name,
            parameter_index,
            data,
            mode,
        })
    }

    /// Whether the value of the series changes between two of its points:
    /// a `"linear"` series read at `t`.
    pub(crate) fn changes_along_t(&self) -> bool {
        self.mode == SeriesMode::Linear
    }

    /// Whether the series is read at `step` rather than at `t`.
    pub(crate) fn reads_step(&self) -> bool {
        self.mode == SeriesMode::Pulse
    }

    /// The first position that is not an integer, if any.
    pub(crate) fn first_non_integer(&self) -> Option<f64> {
        self.data
            .iter()
            .map(|&(position, _)| position)
            .find(|position| position.fract() != 0.0)
    }

    /// Positions in `(from, to)` where the series jumps or bends, for a
    /// series read at `t`.
    pub(crate) fn points_in(&self, from: f64, to: f64) -> impl Iterator<Item = f64> + '_ {
        self.data
            .iter()
            .map(|&(position, _)| position)
            .filter(move |&position| position > from && position < to)
    }

    /// The value of the series at `step` (pulse) or at `t` (step function,
    /// linear).
    pub(crate) fn evaluate(&self, step: f64, t: f64) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        match self.mode {
            SeriesMode::Pulse => match self
                .data
                .binary_search_by(|(position, _)| position.total_cmp(&step))
            {
                Ok(idx) => self.data[idx].1,
                Err(_) => 0.0,
            },
            SeriesMode::StepFunction => {
                let pos = self.data.partition_point(|(position, _)| *position <= t);
                if pos == 0 { 0.0 } else { self.data[pos - 1].1 }
            }
            SeriesMode::Linear => {
                let first = self.data[0].0;
                let last = self.data[self.data.len() - 1].0;
                if t < first || t > last {
                    return 0.0;
                }
                let pos = self.data.partition_point(|(position, _)| *position <= t);
                if pos >= self.data.len() {
                    return self.data[self.data.len() - 1].1;
                }
                let (p0, v0) = self.data[pos - 1];
                let (p1, v1) = self.data[pos];
                v0 + (t - p0) / (p1 - p0) * (v1 - v0)
            }
        }
    }
}

/// Pre-computed subgroup total mapping for stratifications
#[derive(Clone)]
pub(crate) struct SubgroupTotalMapping {
    /// Compartment indices that contribute to this subgroup total
    pub(crate) contributing_compartment_indices: Vec<usize>,
    /// Variable name of this subgroup total (e.g., "total_g1")
    pub(crate) parameter_name: String,
    /// Index into `MathExpressionContext`'s parameter value storage, filled in
    /// after parameter slots have been reserved.
    pub(crate) parameter_index: usize,
}

/// Subgroup total mapping with only the structural data known before
/// parameter slot reservation. The builder converts this into a
/// [`SubgroupTotalMapping`] once the expression context indices exist.
pub(crate) struct SubgroupTotalLayout {
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

/// What one event of a flow does to the state, and the units it needs.
///
/// An event moves one unit: it takes one from each source piece, adds one to
/// each target piece and to each accumulator of the transition. The flow of
/// a deterministic engine is the number of events per unit of the axis, so
/// [`CompiledModel::derivatives`] is `Σ changes × flow`.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowChange {
    /// Id of the transition the flow comes from.
    pub(crate) transition_id: String,
    /// The compartment a proportional rate is multiplied by, or `None` for
    /// an absolute flow.
    pub(crate) proportional_to: Option<usize>,
    /// The compartments an event takes from, with how many units it takes.
    pub(crate) takes: Vec<(usize, u32)>,
    /// `(state index, change)` of one event, piece by piece (source `-1`,
    /// target `+1`), then `+1` for each accumulator; an accumulator's index
    /// follows the compartments.
    pub(crate) changes: Vec<(usize, f64)>,
}

impl FlowChange {
    /// The change of one event of `flow`, in a state of `num_compartments`
    /// compartments followed by the accumulators.
    pub(crate) fn of(flow: &TransitionFlow, num_compartments: usize) -> Self {
        let pieces = std::iter::once((flow.source_index, flow.target_index))
            .chain(flow.extra_pieces.iter().copied());
        let mut changes = Vec::new();
        let mut takes: Vec<(usize, u32)> = Vec::new();
        for (source, target) in pieces {
            if let Some(source) = source {
                changes.push((source, -1.0));
                match takes.iter_mut().find(|(index, _)| *index == source) {
                    Some((_, count)) => *count += 1,
                    None => takes.push((source, 1)),
                }
            }
            if let Some(target) = target {
                changes.push((target, 1.0));
            }
        }
        changes.extend(
            flow.accumulator_indices
                .iter()
                .map(|&index| (num_compartments + index, 1.0)),
        );
        Self {
            transition_id: flow.transition_id.clone(),
            proportional_to: if flow.is_absolute_flow {
                None
            } else {
                flow.source_index
            },
            takes,
            changes,
        }
    }

    /// Id of the transition the flow comes from.
    pub fn transition_id(&self) -> &str {
        &self.transition_id
    }

    /// The compartment a proportional rate is multiplied by, or `None` for
    /// an absolute flow.
    pub fn proportional_to(&self) -> Option<usize> {
        self.proportional_to
    }

    /// The compartments an event takes from, each with how many units it
    /// takes, in the order of the pieces.
    pub fn takes(&self) -> &[(usize, u32)] {
        &self.takes
    }

    /// `(state index, change)` of one event: `-1` for each source piece, `+1`
    /// for each target piece and each accumulator. An accumulator's index
    /// follows the compartments, as in the state.
    pub fn changes(&self) -> &[(usize, f64)] {
        &self.changes
    }
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
    /// The total the initial fractions are taken of.
    pub(crate) total: f64,
    /// Source and applied categories of each output, parallel to `output_names`
    pub(crate) output_labels: Vec<crate::labels::OutputLabels>,
    pub(crate) output_names: Vec<String>,
    /// Initial state: compartments, then accumulators (which start at zero).
    pub(crate) initial_state: Vec<f64>,
    pub(crate) expression_context: MathExpressionContext,
    /// Pre-computed transition flows for performance
    pub(crate) transition_flows: Vec<TransitionFlow>,
    /// The change of one event of each transition flow, parallel to
    /// `transition_flows`.
    pub(crate) flow_changes: Vec<FlowChange>,
    /// The change of one event of each impulse, parallel to `impulse_flows`.
    pub(crate) impulse_changes: Vec<FlowChange>,
    /// The value of each transition flow at the last evaluation, reused by
    /// [`CompiledModel::derivatives`].
    pub(crate) flow_buffer: Vec<f64>,
    /// Impulse transitions: they move their rate at once at their positions
    /// and are not part of the right-hand side.
    pub(crate) impulse_flows: Vec<ImpulseFlow>,
    /// Whether a rate or a formula parameter reads `step` or a pulse series,
    /// so the right-hand side jumps at integer positions.
    pub(crate) depends_on_step: bool,
    /// Whether a rate or a formula parameter reads `t`, or a `"linear"`
    /// series changes along it, so a rate can change between two
    /// breakpoints with the state fixed.
    pub(crate) reads_time: bool,
    /// Values that a rate or a formula parameter compares `t` with, such as
    /// `t_on` in `t >= t_on`: the right-hand side jumps where `t` reaches them.
    pub(crate) time_thresholds: Vec<MathExpression>,
    /// Pre-computed subgroup total mappings for stratifications
    pub(crate) subgroup_total_mappings: Vec<SubgroupTotalMapping>,
    /// Parameters defined as formulas that need to be evaluated each step
    pub(crate) formula_parameters: Vec<(String, RateMathExpression)>,
    /// Parameters defined as empirical series; evaluated via binary search
    /// each time the right-hand side is evaluated
    pub(crate) series_parameters: Vec<SeriesParameter>,
    pub(crate) total_parameter_index: usize,
    pub(crate) t_parameter_index: usize,
    /// Whether any per-step expression still needs compartment values in the
    /// expression context. JIT-resolved transition formulas read compartments
    /// directly from the state and do not need this HashMap update.
    pub(crate) requires_compartment_context: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(data: Vec<(f64, f64)>, mode: SeriesMode) -> SeriesParameter {
        SeriesParameter::new("x".to_string(), 0, data, mode).unwrap()
    }

    #[test]
    fn two_positions_are_the_same_within_a_relative_tolerance() {
        assert!(same_position(1e6, 1e6 + 1e-4));
        assert!(!same_position(1.0, 1.0 + 1e-8));
    }

    #[test]
    fn pulse_series_only_fires_on_listed_steps() {
        let series = series(vec![(5.0, 2.0), (1.0, 1.0)], SeriesMode::Pulse);

        assert_eq!(series.evaluate(0.0, 0.0), 0.0);
        assert_eq!(series.evaluate(1.0, 1.5), 1.0);
        assert_eq!(series.evaluate(2.0, 2.0), 0.0);
        assert_eq!(series.evaluate(5.0, 5.0), 2.0);
    }

    #[test]
    fn step_function_series_holds_last_value() {
        let series = series(vec![(10.0, 3.0), (3.0, 1.5)], SeriesMode::StepFunction);

        assert_eq!(series.evaluate(2.0, 2.0), 0.0);
        assert_eq!(series.evaluate(3.0, 3.0), 1.5);
        assert_eq!(series.evaluate(9.0, 9.0), 1.5);
        assert_eq!(series.evaluate(10.0, 10.0), 3.0);
        assert_eq!(series.evaluate(20.0, 20.0), 3.0);
    }

    #[test]
    fn step_function_series_holds_from_any_position() {
        let series = series(vec![(2.5, 1.0)], SeriesMode::StepFunction);

        assert_eq!(series.evaluate(2.0, 2.4), 0.0);
        assert_eq!(series.evaluate(2.0, 2.5), 1.0);
    }

    #[test]
    fn linear_series_interpolates_in_t() {
        let series = series(vec![(10.0, 10.0), (0.0, 0.0)], SeriesMode::Linear);

        assert_eq!(series.evaluate(0.0, 0.0), 0.0);
        assert_eq!(series.evaluate(5.0, 5.0), 5.0);
        assert_eq!(series.evaluate(2.0, 2.5), 2.5);
        assert_eq!(series.evaluate(10.0, 10.0), 10.0);
        assert_eq!(series.evaluate(11.0, 11.0), 0.0);
    }

    #[test]
    fn a_pulse_between_steps_names_its_parameter() {
        let error = SeriesParameter::new("x".to_string(), 0, vec![(1.5, 1.0)], SeriesMode::Pulse)
            .err()
            .unwrap();
        assert!(error.contains("'x' is a pulse at 1.5"), "{error}");
    }

    #[test]
    fn a_negative_or_repeated_position_names_its_parameter() {
        for data in [vec![(-1.0, 1.0)], vec![(2.0, 1.0), (2.0, 3.0)]] {
            let error = SeriesParameter::new("x".to_string(), 0, data, SeriesMode::Linear)
                .err()
                .unwrap();
            assert!(error.contains("Series parameter 'x'"), "{error}");
        }
    }
}
