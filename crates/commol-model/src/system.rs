//! The right-hand side of a compiled model, and its accessors.

use commol_core::{
    MathExpressionContext, SPECIAL_VAR_T, SPECIAL_VAR_TOTAL, ScaledMatrix, same_position,
};

use crate::labels::OutputLabels;

use crate::rates::{Clock, evaluate_rate};
use crate::types::{CompiledModel, FlowChange, TransitionFlow};

/// Set a reserved parameter slot during an evaluation. When `keep_cache_live`
/// is true the call goes through the name-keyed path, which the formula
/// parameters read; otherwise it writes the indexed slot only.
#[inline]
fn set_step_parameter(
    ctx: &mut MathExpressionContext,
    name: &str,
    index: usize,
    value: f64,
    keep_cache_live: bool,
) {
    if keep_cache_live {
        ctx.set_parameter_str(name, value);
    } else {
        ctx.set_parameter_by_index(index, value)
            .expect("the builder reserves the slot of every variable it sets");
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

    /// The matrices between categories, each with the current value of its
    /// scales.
    pub fn matrices(&self) -> &[ScaledMatrix] {
        &self.matrices
    }

    /// The sum of the initial amounts the model states, before any calibrated
    /// value is written.
    pub fn total(&self) -> f64 {
        self.total
    }

    /// Number of compartments. The accumulators follow them in the state.
    pub fn num_compartments(&self) -> usize {
        self.compartments.len()
    }

    /// The fraction of each category of each compartment, in engine order
    /// and in the order of the categories of its output labels: the split of
    /// a value written to compartments that hold nothing.
    pub fn fractions(&self) -> &[Vec<f64>] {
        &self.fractions
    }

    /// Each stratification whose fractions differ by the category of an
    /// earlier one, as `(stratification, by)` pairs: a category value of
    /// `by` keeps the split of the stratification on each side.
    pub fn splits_by(&self) -> &[(String, String)] {
        &self.splits_by
    }

    /// Initial state: compartments, then accumulators (zero).
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    /// Change the initial amount of one compartment.
    ///
    /// # Errors
    ///
    /// Returns a message when the index names no compartment, or the message
    /// of the rule of an initial amount
    /// ([`crate::rules::check_compartment_amount`]).
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
        crate::rules::check_compartment_amount(&self.compartments[compartment_index], value)?;
        self.initial_state[compartment_index] = value;
        let bin = &self.output_labels[compartment_index].source;
        self.unvalued_bins.retain(|unvalued| unvalued != bin);
        Ok(())
    }

    /// Check that a run has every value: each parameter has one, from the
    /// model or from [`Self::set_parameter`], and each bin whose amount the
    /// model leaves to calibrate has had a value written to one of its
    /// compartments ([`Self::set_initial_condition`]), as a calibration
    /// does. Every engine of the aggregate kind checks it before it runs.
    ///
    /// # Errors
    ///
    /// Returns a message that names every parameter and every bin without a
    /// value.
    pub fn check_values(&self) -> Result<(), String> {
        let parameters: Vec<&str> = self
            .unvalued_parameters
            .iter()
            .map(String::as_str)
            .filter(|&id| self.expression_context.get_parameter(id).is_none())
            .collect();
        let bins = &self.unvalued_bins;
        if parameters.is_empty() && bins.is_empty() {
            return Ok(());
        }
        let mut missing = Vec::new();
        if !parameters.is_empty() {
            missing.push(format!("the parameters {parameters:?}"));
        }
        if !bins.is_empty() {
            missing.push(format!("the amounts of the bins {bins:?}"));
        }
        Err(format!(
            "The model cannot run: {} have no value. Give them a value, or calibrate them \
             and write the calibrated values back",
            missing.join(" and ")
        ))
    }

    /// Set the value of a parameter: a constant, or a parameter to
    /// calibrate. When it scales a row or a column of a matrix, the entries
    /// and the spectral radius of that matrix are written again at the new
    /// value ([`ScaledMatrix::variables`]).
    ///
    /// # Errors
    ///
    /// Returns the message of [`Self::check_parameter`] when a caller cannot
    /// set the parameter. The model is unchanged after an error.
    pub fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.check_parameter(parameter_id)?;
        self.expression_context
            .set_parameter_str(parameter_id, value);
        for matrix in &mut self.matrices {
            if matrix.set_scale(parameter_id, value)
                && let Ok(values) = matrix.variables(|_| None)
            {
                for (name, value) in matrix.names().zip(values) {
                    self.expression_context.set_parameter_str(name, value);
                }
            }
        }
        Ok(())
    }

    /// Check that a caller can set a parameter ([`Self::set_parameter`]).
    ///
    /// # Errors
    ///
    /// Returns a message that names the id when it is not a parameter of the
    /// model, or when it is a formula or a series parameter: the model writes
    /// its value at each evaluation, so a value set by a caller would be
    /// lost.
    pub fn check_parameter(&self, parameter_id: &str) -> Result<(), String> {
        if self.settable_parameters.contains(parameter_id) {
            return Ok(());
        }
        Err(self.not_settable(parameter_id))
    }

    /// Why a caller cannot set `parameter_id`.
    fn not_settable(&self, parameter_id: &str) -> String {
        let source = if self
            .formula_parameters
            .iter()
            .any(|(name, _)| name == parameter_id)
        {
            "formula"
        } else if self
            .series_parameters
            .iter()
            .any(|series| series.parameter_name == parameter_id)
        {
            "series"
        } else {
            let mut settable: Vec<&str> = self
                .settable_parameters
                .iter()
                .map(String::as_str)
                .collect();
            settable.sort_unstable();
            return format!(
                "Parameter '{parameter_id}' is not in the model; the parameters that can be \
                 set are {}",
                settable.join(", ")
            );
        };
        format!("Parameter '{parameter_id}' cannot be set: its {source} sets it at each step")
    }

    /// The right-hand side of the model: the net flow into each component of
    /// the state.
    ///
    /// `state` and `out` have the length of [`Self::initial_state`]. It is
    /// `Σ changes × flow` over the flows of the model: each flow
    /// ([`Self::flow_rates`]) is subtracted from its sources, added to its
    /// targets and added to each accumulator that counts it
    /// ([`Self::flow_changes`]). Every flow is computed from the compartments
    /// of `state` (the accumulators are not read). Rates see `step` and `t`:
    /// an engine that reports only at integer steps passes the same value for
    /// both; the others pass the step that holds `t`.
    ///
    /// A step of `DifferenceEquations` is `state += out`;
    /// `DifferentialEquations` integrates `dX/dt = out`.
    ///
    /// # Errors
    ///
    /// Returns a message when a formula parameter or a rate cannot be
    /// evaluated, for example because a variable is missing, and a message
    /// that names the flow and `t` when a flow is not a number while every
    /// compartment of `state` is one. A compartment that is not a number
    /// gives flows that are not numbers without an error: `diffsol` finds
    /// the sparse pattern of the Jacobian that way.
    pub fn derivatives(
        &mut self,
        step: f64,
        t: f64,
        state: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        let mut flows = std::mem::take(&mut self.flow_buffer);
        let result = self
            .flow_rates(step, t, state, &mut flows)
            .and_then(|()| self.check_flows(t, state, &flows));
        if result.is_ok() {
            out.fill(0.0);
            for (change, &flow) in self.flow_changes.iter().zip(&flows) {
                add_changes(change, flow, out);
            }
        }
        self.flow_buffer = flows;
        result
    }

    /// Returns a message that names the first flow of `flows` that is not a
    /// number when every compartment of `state` is one: its rate at `t` is
    /// then not a number.
    fn check_flows(&self, t: f64, state: &[f64], flows: &[f64]) -> Result<(), String> {
        let Some(flow) = flows.iter().position(|flow| flow.is_nan()) else {
            return Ok(());
        };
        if state[..self.compartments.len()]
            .iter()
            .any(|amount| amount.is_nan())
        {
            return Ok(());
        }
        Err(format!(
            "The rate of flow '{}' at {t} is not a number",
            self.flow_changes[flow].transition_id
        ))
    }

    /// The value of each flow of the model: the term
    /// [`Self::derivatives`] adds for it.
    ///
    /// `out` has one component per flow, in the order of
    /// [`Self::flow_changes`]. A proportional flow is its rate times its
    /// source; an absolute flow is its rate. In a stochastic engine it is the
    /// propensity of the flow: the expected number of its events per unit of
    /// the axis.
    ///
    /// # Errors
    ///
    /// Returns a message when a formula parameter or a rate cannot be
    /// evaluated.
    pub fn flow_rates(
        &mut self,
        step: f64,
        t: f64,
        state: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        self.transition_rates(step, t, state, out)?;
        let amounts = &state[..self.compartments.len()];
        for (flow, change) in out.iter_mut().zip(&self.flow_changes) {
            if let Some(source) = change.proportional_to {
                *flow *= amounts[source];
            }
        }
        Ok(())
    }

    /// The rate of each flow of the model, as its expression gives it: per
    /// unit of its source for a proportional flow, the flow itself for an
    /// absolute one.
    ///
    /// `out` has one component per flow, in the order of
    /// [`Self::flow_changes`]. A chain binomial reads a proportional rate as
    /// the probability that one unit of the source leaves in the step.
    ///
    /// # Errors
    ///
    /// Returns a message when a formula parameter or a rate cannot be
    /// evaluated.
    pub fn transition_rates(
        &mut self,
        step: f64,
        t: f64,
        state: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        let amounts = &state[..self.compartments.len()];
        let clock = self.prepare_context(step, t, amounts)?;
        // Stack buffer for the JIT input vector — most rate expressions
        // reference <= 16 variables.
        let mut jit_buf = [0.0_f64; 16];
        for (rate, flow_info) in out.iter_mut().zip(&self.transition_flows) {
            *rate = evaluate_rate(
                flow_info,
                amounts,
                &mut self.expression_context,
                clock,
                &mut jit_buf,
            )?;
        }
        Ok(())
    }

    /// The change of the state of one event of each flow, in the order of
    /// [`Self::flow_rates`].
    pub fn flow_changes(&self) -> &[FlowChange] {
        &self.flow_changes
    }

    /// The change of the state of one event of each impulse, in the order
    /// [`Self::impulse_rates`] numbers them.
    pub fn impulse_changes(&self) -> &[FlowChange] {
        &self.impulse_changes
    }

    /// The rate of each impulse that acts at position `t`, as its expression
    /// gives it (the fraction of its source for a proportional impulse, the
    /// amount for an absolute one), written into `out` as `(impulse index,
    /// rate)`, in the order of [`Self::impulse_changes`].
    ///
    /// Every rate is computed from `state`. Rates see `step = floor(t)` and
    /// `t`, as in [`Self::apply_impulses`].
    ///
    /// # Errors
    ///
    /// Returns a message when a rate cannot be evaluated.
    pub fn impulse_rates(
        &mut self,
        t: f64,
        state: &[f64],
        out: &mut Vec<(usize, f64)>,
    ) -> Result<(), String> {
        out.clear();
        if !self
            .impulse_flows
            .iter()
            .any(|impulse| impulse.schedule.fires_at(t))
        {
            return Ok(());
        }
        let amounts = &state[..self.compartments.len()];
        let clock = self.prepare_context(t.floor(), t, amounts)?;
        let mut jit_buf = [0.0_f64; 16];
        for (index, impulse) in self.impulse_flows.iter().enumerate() {
            if impulse.schedule.fires_at(t) {
                let rate = evaluate_rate(
                    &impulse.flow,
                    amounts,
                    &mut self.expression_context,
                    clock,
                    &mut jit_buf,
                )?;
                out.push((index, rate));
            }
        }
        Ok(())
    }

    /// Whether a rate can change between two breakpoints while the state
    /// stays fixed: a rate or a formula parameter reads `t`, or a
    /// `"linear"` series changes along it.
    pub fn reads_time(&self) -> bool {
        self.reads_time
    }

    /// The positions in `(from, to]` where the right-hand side can jump,
    /// ascending and without duplicates, ending at `to`: the impulse
    /// positions, the values a rate compares `t` with, the points of a series
    /// read at `t` and, when a rate reads `step` or a pulse series, every
    /// integer.
    ///
    /// An engine in continuous time stops at each of them, so no step crosses
    /// a discontinuity.
    ///
    /// # Errors
    ///
    /// Returns a message when a value a rate compares `t` with cannot be
    /// evaluated.
    pub fn breakpoints(&mut self, from: f64, to: f64) -> Result<Vec<f64>, String> {
        let mut points: Vec<f64> = self
            .impulse_positions(from, to)
            .into_iter()
            .filter(|&position| position > from)
            .collect();
        points.extend(self.time_thresholds(from, to)?);
        points.extend(self.series_points(from, to));
        if self.depends_on_step {
            let mut n = from.floor() + 1.0;
            while n < to {
                points.push(n);
                n += 1.0;
            }
        }
        points.push(to);
        points.sort_by(f64::total_cmp);
        points.dedup_by(|later, earlier| same_position(*later, *earlier));
        points.retain(|&point| !same_position(point, from));
        Ok(points)
    }

    /// Whether the model has impulse transitions.
    pub fn has_impulses(&self) -> bool {
        !self.impulse_flows.is_empty()
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
                        threshold.formula
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

    /// Apply the impulses that act at position `t` to `state`, at once.
    ///
    /// Every jump is computed from the state before any of them, as the flows
    /// of a step are. A proportional rate moves that fraction of its source; an
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
        let amounts = &before[..num_compartments];
        let clock = self.prepare_context(t.floor(), t, amounts)?;

        let mut jit_buf = [0.0_f64; 16];
        for (impulse, change) in self.impulse_flows.iter().zip(&self.impulse_changes) {
            if !impulse.schedule.fires_at(t) {
                continue;
            }
            let amount = flow_amount(
                &impulse.flow,
                amounts,
                &mut self.expression_context,
                clock,
                &mut jit_buf,
            )?;
            add_changes(change, amount, state);
        }
        Ok(true)
    }

    /// The flows that drain compartment `compartment` when
    /// [`Self::derivatives`] reads `state` at `step` and `t`: the transition
    /// id of each and the net amount it takes from the compartment, largest
    /// first.
    ///
    /// A flow drains a compartment when it leaves it with a positive amount
    /// or enters it with a negative one.
    ///
    /// # Errors
    ///
    /// Returns a message when a formula parameter or a rate cannot be
    /// evaluated.
    pub fn outflows(
        &mut self,
        step: f64,
        t: f64,
        state: &[f64],
        compartment: usize,
    ) -> Result<Vec<(String, f64)>, String> {
        let amounts = &state[..self.compartments.len()];
        let clock = self.prepare_context(step, t, amounts)?;
        drains(
            self.transition_flows.iter(),
            amounts,
            compartment,
            &mut self.expression_context,
            clock,
        )
    }

    /// The impulses that act at `t` and drain compartment `compartment` when
    /// [`Self::apply_impulses`] reads `state`, as [`Self::outflows`] gives
    /// them.
    ///
    /// # Errors
    ///
    /// Returns a message when a rate cannot be evaluated.
    pub fn impulse_outflows(
        &mut self,
        t: f64,
        state: &[f64],
        compartment: usize,
    ) -> Result<Vec<(String, f64)>, String> {
        let amounts = &state[..self.compartments.len()];
        let clock = self.prepare_context(t.floor(), t, amounts)?;
        let firing = self
            .impulse_flows
            .iter()
            .filter(|impulse| impulse.schedule.fires_at(t))
            .map(|impulse| &impulse.flow);
        drains(
            firing,
            amounts,
            compartment,
            &mut self.expression_context,
            clock,
        )
    }

    /// Set every variable a rate may read for the compartment amounts
    /// `amounts` at `step` and `t`.
    fn prepare_context(&mut self, step: f64, t: f64, amounts: &[f64]) -> Result<Clock, String> {
        let keep_cache_live = self.requires_compartment_context;
        let clock = Clock { step, time: t };

        self.expression_context.set_step(step);
        self.expression_context.set_time(t);

        let total: f64 = amounts.iter().sum();
        set_step_parameter(
            &mut self.expression_context,
            SPECIAL_VAR_TOTAL,
            self.total_parameter_index,
            total,
            keep_cache_live,
        );
        set_step_parameter(
            &mut self.expression_context,
            SPECIAL_VAR_T,
            self.t_parameter_index,
            t,
            keep_cache_live,
        );

        // Only the formula parameters need compartment values in the
        // expression context. JIT-resolved transition formulas and fast-path
        // rates read compartment values directly from the state.
        if keep_cache_live {
            self.expression_context.set_compartments_by_index(amounts);
        }

        for mapping in &self.subgroup_total_mappings {
            let total: f64 = mapping
                .contributing_compartment_indices
                .iter()
                .map(|&idx| amounts[idx])
                .sum();
            set_step_parameter(
                &mut self.expression_context,
                &mapping.parameter_name,
                mapping.parameter_index,
                total,
                keep_cache_live,
            );
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
            );
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
/// proportional.
#[inline]
fn flow_amount(
    flow_info: &TransitionFlow,
    amounts: &[f64],
    ctx: &mut MathExpressionContext,
    clock: Clock,
    jit_buf: &mut [f64; 16],
) -> Result<f64, String> {
    let rate = evaluate_rate(flow_info, amounts, ctx, clock, jit_buf)?;
    Ok(if flow_info.is_absolute_flow {
        rate
    } else {
        flow_info.source_index.map(|i| amounts[i]).unwrap_or(0.0) * rate
    })
}

/// The flows of `flows` that drain `compartment`, with the net amount each
/// takes from it, largest first. The context is prepared for `amounts`.
fn drains<'a>(
    flows: impl Iterator<Item = &'a TransitionFlow>,
    amounts: &[f64],
    compartment: usize,
    ctx: &mut MathExpressionContext,
    clock: Clock,
) -> Result<Vec<(String, f64)>, String> {
    let mut jit_buf = [0.0_f64; 16];
    let mut drains = Vec::new();
    for flow_info in flows {
        let pieces = std::iter::once((flow_info.source_index, flow_info.target_index))
            .chain(flow_info.extra_pieces.iter().copied());
        let mut times: f64 = 0.0;
        for (source, target) in pieces {
            if source == Some(compartment) {
                times += 1.0;
            }
            if target == Some(compartment) {
                times -= 1.0;
            }
        }
        if times == 0.0 {
            continue;
        }
        let drain = times * flow_amount(flow_info, amounts, ctx, clock, &mut jit_buf)?;
        if drain > 0.0 {
            drains.push((flow_info.transition_id.clone(), drain));
        }
    }
    drains.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(drains)
}

/// Add `flow` times the change of one event to `out`: subtract it from the
/// sources of a flow, add it to its targets and to the accumulators that
/// count it.
#[inline]
fn add_changes(change: &FlowChange, flow: f64, out: &mut [f64]) {
    for &(index, sign) in &change.changes {
        out[index] += sign * flow;
    }
}
