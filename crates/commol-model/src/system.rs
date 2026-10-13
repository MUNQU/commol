//! The right-hand side of a compiled model, and its accessors.

use std::collections::HashMap;

use commol_core::{MathExpressionContext, OutputLabels};

use crate::rates::{Clock, evaluate_rate};
use crate::types::{CompiledModel, TransitionFlow};

// Cached parameter names to avoid string allocations
const PARAM_N: &str = "N";
const PARAM_T: &str = "t";

/// Set a reserved parameter slot during an evaluation. When `keep_cache_live`
/// is true the call routes through the name-keyed path so the cached evalexpr
/// context (used by formula parameters and slow-path rate fallbacks) stays
/// warm; otherwise the indexed path skips the cache patch.
#[inline]
fn set_step_parameter(
    ctx: &mut MathExpressionContext,
    name: &str,
    index: usize,
    value: f64,
    keep_cache_live: bool,
) -> Result<(), String> {
    if keep_cache_live {
        ctx.set_parameter_str(name, value);
        Ok(())
    } else {
        ctx.set_parameter_by_index(index, value)
    }
}

impl CompiledModel {
    /// Names of the compartments, in engine order.
    pub fn compartments(&self) -> &[String] {
        &self.compartments
    }

    /// Names of every output: the compartments, then the accumulators.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Source and applied categories of each output, parallel to
    /// [`Self::output_names`].
    pub fn output_labels(&self) -> &[OutputLabels] {
        &self.output_labels
    }

    /// Number of compartments. The accumulators follow them in the state.
    pub fn num_compartments(&self) -> usize {
        self.compartments.len()
    }

    /// Length of the state: compartments plus accumulators.
    pub fn state_len(&self) -> usize {
        self.initial_state.len()
    }

    /// Initial state: compartments, then accumulators (zero).
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    /// Change the initial population of one compartment.
    ///
    /// # Errors
    ///
    /// Returns a message when the index names no compartment or the value is
    /// negative.
    pub fn set_initial_condition(
        &mut self,
        compartment_index: usize,
        value: f64,
    ) -> Result<(), String> {
        if compartment_index >= self.compartments.len() {
            return Err(format!(
                "Invalid compartment index: {}. Model has {} compartments.",
                compartment_index,
                self.compartments.len()
            ));
        }
        if value < 0.0 {
            return Err(format!(
                "Initial condition value must be non-negative, got: {}",
                value
            ));
        }
        self.initial_state[compartment_index] = value;
        Ok(())
    }

    /// Set the value of a parameter.
    pub fn set_parameter(&mut self, parameter_id: &str, value: f64) {
        self.expression_context
            .set_parameter(parameter_id.to_string(), value);
    }

    /// Current values of the parameters, including the variables the last
    /// evaluation set (`N`, `t`, subpopulation totals).
    pub fn parameters(&self) -> &HashMap<String, f64> {
        self.expression_context.get_parameters()
    }

    /// The right-hand side of the model: the net flow into each component of
    /// the state.
    ///
    /// `state` and `out` have [`Self::state_len`] components. Every flow is
    /// computed from the compartments of `state` (the accumulators are not
    /// read), subtracted from its source, added to its target and added to
    /// each accumulator that counts it. Rates see `step` and `t`: a difference
    /// engine passes the same value for both; a differential engine passes the
    /// step that holds `t`.
    ///
    /// A difference step is `state += out`; a differential engine integrates
    /// `dX/dt = out`.
    ///
    /// # Errors
    ///
    /// Returns a message when a formula parameter or a rate cannot be
    /// evaluated, for example because a variable is missing.
    pub fn derivatives(
        &mut self,
        step: f64,
        t: f64,
        state: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        let num_compartments = self.compartments.len();
        let population = &state[..num_compartments];
        out.fill(0.0);
        let clock = self.prepare_context(step, t, population)?;

        // Stack buffer for the JIT input vector — most rate expressions
        // reference <= 16 variables.
        let mut jit_buf = [0.0_f64; 16];
        let (compartment_out, accumulator_out) = out.split_at_mut(num_compartments);
        for flow_info in &self.transition_flows {
            let flow = flow_amount(
                flow_info,
                population,
                &mut self.expression_context,
                clock,
                &mut jit_buf,
            )?;
            add_flow(flow_info, flow, compartment_out, accumulator_out);
        }

        Ok(())
    }

    /// Whether the model has impulse transitions.
    pub fn has_impulses(&self) -> bool {
        !self.impulse_flows.is_empty()
    }

    /// Whether a rate or a formula parameter reads `step` or a pulse series.
    ///
    /// When it does, the right-hand side may jump at every integer position,
    /// so a differential engine restarts its integration there.
    pub fn depends_on_step(&self) -> bool {
        self.depends_on_step
    }

    /// Positions in `[from, to]` where an impulse acts, ascending and without
    /// duplicates.
    pub fn impulse_positions(&self, from: f64, to: f64) -> Vec<f64> {
        let mut positions: Vec<f64> = self
            .impulse_flows
            .iter()
            .flat_map(|impulse| impulse.schedule.positions_in(from, to))
            .collect();
        positions.sort_by(f64::total_cmp);
        positions.dedup();
        positions
    }

    /// Positions in `(from, to)` where `t` reaches a value that a rate compares
    /// it with, ascending and without duplicates, for the current parameter
    /// values.
    ///
    /// # Errors
    ///
    /// Returns a message when such a value cannot be evaluated, for example
    /// because a parameter has no value yet.
    pub fn time_thresholds(&mut self, from: f64, to: f64) -> Result<Vec<f64>, String> {
        let mut positions = Vec::with_capacity(self.time_thresholds.len());
        for threshold in &self.time_thresholds {
            let position = threshold
                .evaluate(&mut self.expression_context)
                .map_err(|error| {
                    format!(
                        "Failed to evaluate the time threshold '{}': {error}",
                        threshold.preprocessed()
                    )
                })?;
            if position > from && position < to {
                positions.push(position);
            }
        }
        positions.sort_by(f64::total_cmp);
        positions.dedup();
        Ok(positions)
    }

    /// Positions in `(from, to)` where a series parameter read at `t` jumps or
    /// bends: the points of every `"step_function"` and `"linear"` series,
    /// ascending and without duplicates.
    pub fn series_points(&self, from: f64, to: f64) -> Vec<f64> {
        let mut positions: Vec<f64> = self
            .series_parameters
            .iter()
            .filter(|series| !series.reads_step())
            .flat_map(|series| series.points_in(from, to))
            .collect();
        positions.sort_by(f64::total_cmp);
        positions.dedup();
        positions
    }

    /// The first series point that is not an integer, with its parameter
    /// name, if any. A difference engine only steps at integers.
    pub fn first_non_integer_series_point(&self) -> Option<(String, f64)> {
        self.series_parameters.iter().find_map(|series| {
            series
                .first_non_integer()
                .map(|position| (series.parameter_name.clone(), position))
        })
    }

    /// The first impulse position that is not an integer, with its
    /// transition id, if any. A difference engine only steps at integers.
    pub fn first_non_integer_impulse(&self) -> Option<(String, f64)> {
        self.transition_ids_of_impulses()
            .zip(&self.impulse_flows)
            .find_map(|(id, impulse)| {
                impulse
                    .schedule
                    .first_non_integer()
                    .map(|position| (id.clone(), position))
            })
    }

    fn transition_ids_of_impulses(&self) -> impl Iterator<Item = &String> {
        self.impulse_flows
            .iter()
            .map(|impulse| &impulse.transition_id)
    }

    /// Apply the impulses that act at position `t` to `state`, at once.
    ///
    /// Every jump is computed from the state before any of them, as the flows
    /// of a step are. A per-capita rate moves that fraction of its source; an
    /// absolute rate moves that amount. Each accumulator that counts an
    /// impulse adds the amount it moves. Rates see `step = floor(t)` and `t`.
    ///
    /// Returns whether any impulse acted.
    ///
    /// # Errors
    ///
    /// Returns a message when a rate cannot be evaluated.
    pub fn apply_impulses(&mut self, t: f64, state: &mut [f64]) -> Result<bool, String> {
        if !self
            .impulse_flows
            .iter()
            .any(|impulse| impulse.schedule.fires_at(t))
        {
            return Ok(false);
        }
        let num_compartments = self.compartments.len();
        let before = state.to_vec();
        let population = &before[..num_compartments];
        let clock = self.prepare_context(t.floor(), t, population)?;

        let mut jit_buf = [0.0_f64; 16];
        let (compartment_out, accumulator_out) = state.split_at_mut(num_compartments);
        for impulse in &self.impulse_flows {
            if !impulse.schedule.fires_at(t) {
                continue;
            }
            let amount = flow_amount(
                &impulse.flow,
                population,
                &mut self.expression_context,
                clock,
                &mut jit_buf,
            )?;
            add_flow(&impulse.flow, amount, compartment_out, accumulator_out);
        }
        Ok(true)
    }

    /// Set every variable a rate may read for the state `population` at
    /// `step` and `t`.
    fn prepare_context(&mut self, step: f64, t: f64, population: &[f64]) -> Result<Clock, String> {
        let keep_cache_live = self.requires_compartment_context;
        let clock = Clock { step, time: t };

        self.expression_context.set_step(step);
        self.expression_context.set_time(t);

        let total_population: f64 = population.iter().sum();
        set_step_parameter(
            &mut self.expression_context,
            PARAM_N,
            self.n_parameter_index,
            total_population,
            keep_cache_live,
        )?;
        set_step_parameter(
            &mut self.expression_context,
            PARAM_T,
            self.t_parameter_index,
            t,
            keep_cache_live,
        )?;

        // Only evalexpr fallback/formula-parameter paths need compartment
        // values in the expression context. JIT-resolved transition formulas
        // and fast-path rates read compartment values directly from the state.
        if keep_cache_live {
            self.expression_context
                .set_compartments_by_index(population);
        }

        for mapping in &self.subpopulation_mappings {
            let total: f64 = mapping
                .contributing_compartment_indices
                .iter()
                .map(|&idx| population[idx])
                .sum();
            set_step_parameter(
                &mut self.expression_context,
                &mapping.parameter_name,
                mapping.parameter_index,
                total,
                keep_cache_live,
            )?;
        }

        // A pulse series is read at `step`, the others at `t` (O(log N) search).
        for ts in &self.series_parameters {
            let value = ts.evaluate(step, t);
            set_step_parameter(
                &mut self.expression_context,
                &ts.parameter_name,
                ts.parameter_index,
                value,
                keep_cache_live,
            )?;
        }

        for (param_name, rate_expr) in &self.formula_parameters {
            match rate_expr.evaluate(&mut self.expression_context) {
                Ok(value) => {
                    self.expression_context.set_parameter_str(param_name, value);
                }
                Err(error) => {
                    return Err(format!(
                        "Failed to evaluate formula parameter '{}': {}",
                        param_name, error
                    ));
                }
            }
        }

        Ok(clock)
    }
}

/// The amount a flow moves: its rate, times its source when the rate is
/// per-capita.
#[inline]
fn flow_amount(
    flow_info: &TransitionFlow,
    population: &[f64],
    ctx: &mut MathExpressionContext,
    clock: Clock,
    jit_buf: &mut [f64; 16],
) -> Result<f64, String> {
    let rate = evaluate_rate(flow_info, population, ctx, clock, jit_buf)?;
    Ok(if flow_info.is_absolute_flow {
        rate
    } else {
        flow_info.source_index.map(|i| population[i]).unwrap_or(0.0) * rate
    })
}

/// Subtract `flow` from the sources of a flow, add it to its targets and to
/// the accumulators that count it.
#[inline]
fn add_flow(
    flow_info: &TransitionFlow,
    flow: f64,
    compartment_out: &mut [f64],
    accumulator_out: &mut [f64],
) {
    if let Some(src) = flow_info.source_index {
        compartment_out[src] -= flow;
    }
    if let Some(tgt) = flow_info.target_index {
        compartment_out[tgt] += flow;
    }
    for &(src, tgt) in &flow_info.extra_pieces {
        if let Some(src) = src {
            compartment_out[src] -= flow;
        }
        if let Some(tgt) = tgt {
            compartment_out[tgt] += flow;
        }
    }
    for &accumulator_idx in &flow_info.accumulator_indices {
        accumulator_out[accumulator_idx] += flow;
    }
}
