//! Simulation methods for running the difference equations engine.

use crate::types::{DifferenceEquations, FastRateExpression, FastRateOp, TransitionFlow, VarSlot};
use commol_core::{MathExpression, MathExpressionContext, RateMathExpression};
use std::collections::HashMap;

// Cached parameter names to avoid string allocations
const PARAM_N: &str = "N";
const PARAM_T: &str = "t";

/// Set a reserved parameter slot during a simulation step. When
/// `keep_cache_live` is true the call routes through the name-keyed path so
/// the cached evalexpr context (used by formula parameters and slow-path rate
/// fallbacks) stays warm; otherwise the indexed path skips the cache patch.
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

impl DifferenceEquations {
    /// Get the current population vector.
    pub fn population(&self) -> Vec<f64> {
        self.population.clone()
    }

    /// Get the list of compartment names.
    pub fn compartments(&self) -> Vec<String> {
        self.compartments.clone()
    }

    /// Get all simulation output names: state compartments followed by accumulators.
    pub fn output_names(&self) -> Vec<String> {
        self.output_names.clone()
    }

    fn output_len(&self) -> usize {
        self.population.len() + self.accumulators.len()
    }

    fn output_row(&self) -> Vec<f64> {
        let mut row = Vec::with_capacity(self.output_len());
        row.extend_from_slice(&self.population);
        row.extend_from_slice(&self.accumulators);
        row
    }

    fn copy_output_row(&self, row: &mut Vec<f64>) {
        let output_len = self.output_len();
        if row.len() != output_len {
            row.resize(output_len, 0.0);
        }
        let population_len = self.population.len();
        row[..population_len].copy_from_slice(&self.population);
        row[population_len..].copy_from_slice(&self.accumulators);
    }

    /// Execute a single simulation step.
    ///
    /// This method:
    /// 1. Updates the expression context with current state
    /// 2. Computes flows for all transitions
    /// 3. Applies flows to update compartment populations
    /// 4. Increments the step counter
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error message if rate evaluation fails.
    pub fn step(&mut self) -> Result<(), String> {
        // Reuse compartment flows buffer instead of allocating
        self.compartment_flows.fill(0.0);

        let keep_cache_live = self.requires_compartment_context;

        // Update expression context with current population values
        self.expression_context.set_step(self.current_step);

        // Calculate and set total population N
        let total_population: f64 = self.population.iter().sum();
        set_step_parameter(
            &mut self.expression_context,
            PARAM_N,
            self.n_parameter_index,
            total_population,
            keep_cache_live,
        )?;

        // Set t as an alias for step (for convenience in formulas)
        set_step_parameter(
            &mut self.expression_context,
            PARAM_T,
            self.t_parameter_index,
            self.current_step,
            keep_cache_live,
        )?;

        // Only evalexpr fallback/formula-parameter paths need compartment
        // values in the expression context. JIT-resolved transition formulas
        // and fast-path rates read compartment values directly from
        // `population`.
        if keep_cache_live {
            self.expression_context
                .set_compartments_by_index(&self.population);
        }

        // Compute subpopulation totals using pre-computed mappings (if any)
        for mapping in &self.subpopulation_mappings {
            let total: f64 = mapping
                .contributing_compartment_indices
                .iter()
                .map(|&idx| self.population[idx])
                .sum();
            set_step_parameter(
                &mut self.expression_context,
                &mapping.parameter_name,
                mapping.parameter_index,
                total,
                keep_cache_live,
            )?;
        }

        // Evaluate time-series parameters (O(log N) binary search, no allocation)
        let step_u64 = self.current_step as u64;
        for ts in &self.series_parameters {
            let value = ts.evaluate(step_u64);
            set_step_parameter(
                &mut self.expression_context,
                &ts.parameter_name,
                ts.parameter_index,
                value,
                keep_cache_live,
            )?;
        }

        // Evaluate formula parameters and update context
        // Note: We need to clone to avoid borrow checker issues
        let formula_params = self.formula_parameters.clone();
        for (param_name, rate_expr) in &formula_params {
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

        // Use pre-computed transition flows - much faster!
        // Stack buffer for the JIT input vector — most rate expressions
        // reference <= 16 variables.
        let mut jit_buf = [0.0_f64; 16];
        for flow_info in &self.transition_flows {
            let source_population = flow_info
                .source_index
                .map(|i| self.population[i])
                .unwrap_or(0.0);

            let rate = evaluate_rate(
                flow_info,
                &self.population,
                &mut self.expression_context,
                self.current_step,
                &mut jit_buf,
            )?;

            let flow = if flow_info.is_absolute_flow {
                rate
            } else {
                source_population * rate
            };

            if let Some(src) = flow_info.source_index {
                self.compartment_flows[src] -= flow;
            }
            if let Some(tgt) = flow_info.target_index {
                self.compartment_flows[tgt] += flow;
            }
            for &accumulator_idx in &flow_info.accumulator_indices {
                self.accumulators[accumulator_idx] += flow;
            }
        }

        // Apply the calculated flows to the population vector.
        for (i, flow) in self.compartment_flows.iter().enumerate() {
            self.population[i] += flow;
        }

        // Increment step
        self.current_step += 1.0;

        Ok(())
    }

    /// Run the simulation for a specified number of steps.
    ///
    /// # Arguments
    ///
    /// * `num_steps` - Number of time steps to simulate
    ///
    /// # Returns
    ///
    /// A vector of population states, where the first element is the initial state (t=0)
    /// and subsequent elements are states at t=1, t=2, ..., t=num_steps.
    pub fn run(&mut self, num_steps: u32) -> Result<Vec<Vec<f64>>, String> {
        // Pre-allocate memory for efficiency
        let mut steps = Vec::with_capacity(num_steps as usize + 1);

        // Store initial state (t=0)
        steps.push(self.output_row());

        for _ in 0..num_steps {
            self.step()?;
            steps.push(self.output_row());
        }

        Ok(steps)
    }

    /// Optimized version that writes simulation results into a pre-allocated buffer.
    ///
    /// This method is more memory-efficient than `run()` when the caller can
    /// provide a reusable buffer.
    ///
    /// # Arguments
    ///
    /// * `num_steps` - Number of time steps to simulate
    /// * `buffer` - Pre-allocated buffer to store results
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error message if simulation fails.
    pub fn run_into_buffer(
        &mut self,
        num_steps: u32,
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        let total_steps = (num_steps + 1) as usize;
        let output_len = self.output_len();

        // Keep existing row allocations alive across repeated calls. This is
        // especially important for calibration, where the same buffer is reused
        // for hundreds or thousands of full simulations.
        if buffer.len() < total_steps {
            buffer.reserve(total_steps - buffer.len());
            while buffer.len() < total_steps {
                buffer.push(vec![0.0; output_len]);
            }
        } else if buffer.len() > total_steps {
            buffer.truncate(total_steps);
        }

        for row in buffer.iter_mut() {
            if row.len() != output_len {
                row.resize(output_len, 0.0);
            }
        }

        // Store initial state (t=0)
        self.copy_output_row(&mut buffer[0]);

        for row in buffer.iter_mut().take(total_steps).skip(1) {
            self.step()?;
            self.copy_output_row(row);
        }

        Ok(())
    }

    /// Run through the largest requested step while recording only selected
    /// steps. `time_steps` must be sorted and deduplicated by the caller.
    pub fn run_at_steps_into_buffer(
        &mut self,
        time_steps: &[u32],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        let output_len = self.output_len();

        if buffer.len() < time_steps.len() {
            buffer.reserve(time_steps.len() - buffer.len());
            while buffer.len() < time_steps.len() {
                buffer.push(vec![0.0; output_len]);
            }
        } else if buffer.len() > time_steps.len() {
            buffer.truncate(time_steps.len());
        }

        for row in buffer.iter_mut() {
            if row.len() != output_len {
                row.resize(output_len, 0.0);
            }
        }

        let mut next_record_idx = 0;
        while next_record_idx < time_steps.len() && time_steps[next_record_idx] == 0 {
            self.copy_output_row(&mut buffer[next_record_idx]);
            next_record_idx += 1;
        }

        let Some(&max_step) = time_steps.last() else {
            return Ok(());
        };

        for step in 1..=max_step {
            self.step()?;
            while next_record_idx < time_steps.len() && time_steps[next_record_idx] == step {
                self.copy_output_row(&mut buffer[next_record_idx]);
                next_record_idx += 1;
            }
        }

        Ok(())
    }
}

/// Evaluate a transition flow's rate, choosing the fastest path the rate
/// supports: the specialized fast-path interpreter, the JIT with pre-resolved
/// slots, or the evalexpr fallback.
#[inline]
fn evaluate_rate(
    flow: &TransitionFlow,
    population: &[f64],
    ctx: &mut MathExpressionContext,
    current_step: f64,
    jit_buf: &mut [f64; 16],
) -> Result<f64, String> {
    let rate_result = if let Some(fast_expr) = &flow.fast_rate_expression {
        eval_fast_rate(fast_expr, population, ctx, current_step)
    } else if let (RateMathExpression::Formula(expr), Some(slots)) =
        (&flow.rate_expression, &flow.resolved_slots)
    {
        evaluate_jit_rate(expr, slots, population, ctx, current_step, jit_buf)
    } else {
        flow.rate_expression
            .evaluate(ctx)
            .map_err(|error| error.to_string())
    };

    rate_result.map_err(|error| {
        format!(
            "Failed to evaluate rate for transition from {:?} to {:?}: {}",
            flow.source_index, flow.target_index, error
        )
    })
}

/// Resolve every variable slot the JIT-compiled rate needs, then invoke the
/// compiled function. The 16-slot stack buffer covers the vast majority of
/// rate expressions; wider expressions fall back to a one-off heap buffer.
#[inline]
fn evaluate_jit_rate(
    expr: &MathExpression,
    slots: &[VarSlot],
    population: &[f64],
    ctx: &MathExpressionContext,
    current_step: f64,
    jit_buf: &mut [f64; 16],
) -> Result<f64, String> {
    let n = slots.len();
    if n <= jit_buf.len() {
        let values = &mut jit_buf[..n];
        for (i, slot) in slots.iter().enumerate() {
            values[i] = resolve_slot(slot, population, ctx, current_step)?;
        }
        expr.call_jit_with_buffer(values)
            .map_err(|error| error.to_string())
    } else {
        let mut heap = vec![0.0; n];
        for (i, slot) in slots.iter().enumerate() {
            heap[i] = resolve_slot(slot, population, ctx, current_step)?;
        }
        expr.call_jit_with_buffer(&heap)
            .map_err(|error| error.to_string())
    }
}

/// Resolve a pre-computed `VarSlot` to its current numeric value. Compartments
/// and the step alias are O(1); parameters fall back to a single HashMap probe.
#[inline]
fn resolve_slot(
    slot: &VarSlot,
    population: &[f64],
    ctx: &MathExpressionContext,
    current_step: f64,
) -> Result<f64, String> {
    match slot {
        VarSlot::Compartment(idx) => Ok(population[*idx]),
        VarSlot::Step => Ok(current_step),
        VarSlot::ParameterIndex(idx) => ctx
            .get_parameter_by_index(*idx)
            .ok_or_else(|| format!("Parameter index '{}' not found", idx)),
        VarSlot::Parameter(name) => ctx
            .get_parameter(name)
            .ok_or_else(|| format!("Variable '{}' not found", name)),
    }
}

#[inline]
fn eval_fast_rate(
    expr: &FastRateExpression,
    population: &[f64],
    ctx: &MathExpressionContext,
    current_step: f64,
) -> Result<f64, String> {
    match expr {
        FastRateExpression::Constant(value) => return Ok(*value),
        FastRateExpression::Slot(slot) => {
            return resolve_slot(slot, population, ctx, current_step);
        }
        FastRateExpression::Mul2(a, b) => {
            return Ok(resolve_slot(a, population, ctx, current_step)?
                * resolve_slot(b, population, ctx, current_step)?);
        }
        FastRateExpression::Mul3(a, b, c) => {
            return Ok(resolve_slot(a, population, ctx, current_step)?
                * resolve_slot(b, population, ctx, current_step)?
                * resolve_slot(c, population, ctx, current_step)?);
        }
        FastRateExpression::OneMinusMul3 {
            subtract,
            factor,
            value,
        } => {
            return Ok(
                (1.0 - resolve_slot(subtract, population, ctx, current_step)?)
                    * resolve_slot(factor, population, ctx, current_step)?
                    * resolve_slot(value, population, ctx, current_step)?,
            );
        }
        FastRateExpression::MulAddDiv {
            first,
            second,
            add_left,
            add_right,
            denominator,
        } => {
            return Ok(resolve_slot(first, population, ctx, current_step)?
                * resolve_slot(second, population, ctx, current_step)?
                * (resolve_slot(add_left, population, ctx, current_step)?
                    + resolve_slot(add_right, population, ctx, current_step)?)
                / resolve_slot(denominator, population, ctx, current_step)?);
        }
        FastRateExpression::Program(_) => {}
    }

    let FastRateExpression::Program(ops) = expr else {
        unreachable!("specialized fast rate variants return above");
    };
    let mut stack = [0.0_f64; 32];
    let mut sp = 0usize;

    for op in ops {
        match op {
            FastRateOp::Constant(value) => push_stack(&mut stack, &mut sp, *value)?,
            FastRateOp::Slot(slot) => push_stack(
                &mut stack,
                &mut sp,
                resolve_slot(slot, population, ctx, current_step)?,
            )?,
            FastRateOp::Add => {
                let rhs = pop_stack(&stack, &mut sp)?;
                let lhs = pop_stack(&stack, &mut sp)?;
                push_stack(&mut stack, &mut sp, lhs + rhs)?;
            }
            FastRateOp::Sub => {
                let rhs = pop_stack(&stack, &mut sp)?;
                let lhs = pop_stack(&stack, &mut sp)?;
                push_stack(&mut stack, &mut sp, lhs - rhs)?;
            }
            FastRateOp::Mul => {
                let rhs = pop_stack(&stack, &mut sp)?;
                let lhs = pop_stack(&stack, &mut sp)?;
                push_stack(&mut stack, &mut sp, lhs * rhs)?;
            }
            FastRateOp::Div => {
                let rhs = pop_stack(&stack, &mut sp)?;
                let lhs = pop_stack(&stack, &mut sp)?;
                push_stack(&mut stack, &mut sp, lhs / rhs)?;
            }
            FastRateOp::Neg => {
                let value = pop_stack(&stack, &mut sp)?;
                push_stack(&mut stack, &mut sp, -value)?;
            }
        }
    }

    if sp == 1 {
        pop_stack(&stack, &mut sp)
    } else {
        Err("fast rate expression left an invalid stack state".to_string())
    }
}

#[inline]
fn push_stack(stack: &mut [f64; 32], sp: &mut usize, value: f64) -> Result<(), String> {
    if *sp >= stack.len() {
        return Err("fast rate expression stack overflow".to_string());
    }
    stack[*sp] = value;
    *sp += 1;
    Ok(())
}

#[inline]
fn pop_stack(stack: &[f64; 32], sp: &mut usize) -> Result<f64, String> {
    if *sp == 0 {
        return Err("fast rate expression stack underflow".to_string());
    }
    *sp -= 1;
    Ok(stack[*sp])
}

/// Implementation of the SimulationEngine trait.
impl commol_core::SimulationEngine for DifferenceEquations {
    fn run(&mut self, num_steps: u32) -> Result<Vec<Vec<f64>>, String> {
        // Delegate to existing implementation
        DifferenceEquations::run(self, num_steps)
    }

    fn step(&mut self) -> Result<(), String> {
        // Delegate to existing implementation
        DifferenceEquations::step(self)
    }

    fn compartments(&self) -> Vec<String> {
        self.compartments.clone()
    }

    fn output_names(&self) -> Vec<String> {
        DifferenceEquations::output_names(self)
    }

    fn population(&self) -> Vec<f64> {
        self.population.clone()
    }

    fn reset(&mut self) {
        // Reset population to initial state
        self.population.copy_from_slice(&self.initial_population);
        self.accumulators
            .copy_from_slice(&self.initial_accumulators);
        // Reset step counter
        self.current_step = 0.0;
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.expression_context
            .set_parameter(parameter_id.to_string(), value);
        Ok(())
    }

    fn get_parameters(&self) -> &HashMap<String, f64> {
        self.expression_context.get_parameters()
    }

    fn current_step(&self) -> f64 {
        self.current_step
    }

    fn run_into_buffer(
        &mut self,
        num_steps: u32,
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        // Delegate to optimized implementation
        DifferenceEquations::run_into_buffer(self, num_steps, buffer)
    }

    fn run_at_steps_into_buffer(
        &mut self,
        time_steps: &[u32],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        DifferenceEquations::run_at_steps_into_buffer(self, time_steps, buffer)
    }

    fn set_initial_condition(
        &mut self,
        compartment_index: usize,
        value: f64,
    ) -> Result<(), String> {
        // Validate compartment index
        if compartment_index >= self.initial_population.len() {
            return Err(format!(
                "Invalid compartment index: {}. Model has {} compartments.",
                compartment_index,
                self.initial_population.len()
            ));
        }

        // Validate value (non-negative population)
        if value < 0.0 {
            return Err(format!(
                "Initial condition value must be non-negative, got: {}",
                value
            ));
        }

        // Update initial population
        self.initial_population[compartment_index] = value;

        // Also update current population to reflect the change
        self.population[compartment_index] = value;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use commol_core::{
        Bin, BinFraction, Dynamics, InitialConditions, Model, ModelTypes, Population,
        RateMathExpression, Transition,
    };

    fn make_periodic_ab_model() -> Model {
        Model {
            name: "AB_periodic".to_string(),
            description: None,
            version: None,
            parameters: vec![],
            population: Population {
                bins: vec![
                    Bin {
                        id: "A".to_string(),
                        name: "Source".to_string(),
                    },
                    Bin {
                        id: "B".to_string(),
                        name: "Sink".to_string(),
                    },
                ],
                accumulators: vec![],
                stratifications: vec![],
                transitions: vec![],
                initial_conditions: InitialConditions {
                    population_size: 1000,
                    bin_fractions: vec![
                        BinFraction {
                            bin: "A".to_string(),
                            fraction: Some(1.0),
                        },
                        BinFraction {
                            bin: "B".to_string(),
                            fraction: Some(0.0),
                        },
                    ],
                    stratification_fractions: vec![],
                },
            },
            dynamics: Dynamics {
                typology: ModelTypes::DifferenceEquations,
                transitions: vec![Transition {
                    id: "flow".to_string(),
                    source: vec!["A".to_string()],
                    target: vec!["B".to_string()],
                    accumulators: vec![],
                    rate: Some(RateMathExpression::from_string(
                        "if(step - floor(step / 7) * 7 == 0, 0.1, 0)".to_string(),
                    )),
                    stratified_rates: None,
                    condition: None,
                    per_compartment: None,
                }],
            },
        }
    }

    fn make_missing_parameter_model() -> Model {
        Model {
            name: "AB_missing_parameter".to_string(),
            description: None,
            version: None,
            parameters: vec![],
            population: Population {
                bins: vec![
                    Bin {
                        id: "A".to_string(),
                        name: "Source".to_string(),
                    },
                    Bin {
                        id: "B".to_string(),
                        name: "Sink".to_string(),
                    },
                ],
                accumulators: vec![],
                stratifications: vec![],
                transitions: vec![],
                initial_conditions: InitialConditions {
                    population_size: 1000,
                    bin_fractions: vec![
                        BinFraction {
                            bin: "A".to_string(),
                            fraction: Some(1.0),
                        },
                        BinFraction {
                            bin: "B".to_string(),
                            fraction: Some(0.0),
                        },
                    ],
                    stratification_fractions: vec![],
                },
            },
            dynamics: Dynamics {
                typology: ModelTypes::DifferenceEquations,
                transitions: vec![Transition {
                    id: "flow".to_string(),
                    source: vec!["A".to_string()],
                    target: vec!["B".to_string()],
                    accumulators: vec![],
                    rate: Some(RateMathExpression::from_string("missing * A".to_string())),
                    stratified_rates: None,
                    condition: None,
                    per_compartment: None,
                }],
            },
        }
    }

    #[test]
    fn test_periodic_rate_fires_at_exact_steps() {
        // A→B model where rate fires at step 0 and step 7 (period=7).
        // Verifies the floor-based modulo formula compiles and evaluates
        // correctly end-to-end through the DifferenceEquations engine.
        let model = make_periodic_ab_model();
        let mut engine = DifferenceEquations::from_model(&model);
        let results = engine.run(14).unwrap();

        // results[k] = state after k step() calls; step counter k-1 was used for k>=1.
        // Compartment order: A=index 0, B=index 1.
        let b: Vec<f64> = results.iter().map(|s| s[1]).collect();
        let a: Vec<f64> = results.iter().map(|s| s[0]).collect();

        // Step 0 fires: B increases from initial zero
        assert!(b[1] > b[0], "B should increase after step 0");

        // Steps 1–6 do not fire
        for k in 2..=7 {
            assert!(
                (b[k] - b[k - 1]).abs() < 1e-10,
                "B should not change at step {}, delta = {}",
                k - 1,
                (b[k] - b[k - 1]).abs()
            );
        }

        // Step 7 fires: B increases again
        assert!(b[8] > b[7], "B should increase after step 7");

        // Steps 8–13 do not fire
        for k in 9..=14 {
            assert!(
                (b[k] - b[k - 1]).abs() < 1e-10,
                "B should not change at step {}, delta = {}",
                k - 1,
                (b[k] - b[k - 1]).abs()
            );
        }

        // Population conserved at every recorded state
        for k in 0..=14 {
            assert!(
                (a[k] + b[k] - 1000.0).abs() < 1e-8,
                "Population not conserved at step {}: A={}, B={}, total={}",
                k,
                a[k],
                b[k],
                a[k] + b[k]
            );
        }
    }

    #[test]
    fn jit_fast_path_reports_missing_parameters() {
        let model = make_missing_parameter_model();
        let mut engine = DifferenceEquations::from_model(&model);
        let error = engine
            .step()
            .expect_err("missing parameter must not be treated as zero");

        assert!(error.contains("Variable 'missing' not found"));
    }

    fn ctx_with_params(pairs: &[(&str, f64)]) -> MathExpressionContext {
        let mut ctx = MathExpressionContext::new();
        for (name, value) in pairs {
            ctx.set_parameter((*name).to_string(), *value);
        }
        ctx
    }

    fn param_slot(ctx: &MathExpressionContext, name: &str) -> VarSlot {
        VarSlot::ParameterIndex(
            ctx.parameter_index(name)
                .expect("parameter must be registered"),
        )
    }

    #[test]
    fn fast_rate_constant_returns_value() {
        let ctx = MathExpressionContext::new();
        let expr = FastRateExpression::Constant(0.42);
        let result = eval_fast_rate(&expr, &[], &ctx, 0.0).unwrap();
        assert_eq!(result, 0.42);
    }

    #[test]
    fn fast_rate_slot_resolves_compartment_and_step() {
        let ctx = MathExpressionContext::new();
        let population = vec![10.0, 20.0];

        let compartment = FastRateExpression::Slot(VarSlot::Compartment(1));
        assert_eq!(
            eval_fast_rate(&compartment, &population, &ctx, 5.0).unwrap(),
            20.0
        );

        let step = FastRateExpression::Slot(VarSlot::Step);
        assert_eq!(eval_fast_rate(&step, &population, &ctx, 5.0).unwrap(), 5.0);
    }

    #[test]
    fn fast_rate_mul2_and_mul3_compute_products() {
        let ctx = ctx_with_params(&[("k1", 2.0), ("k2", 3.0), ("k3", 4.0)]);
        let mul2 = FastRateExpression::Mul2(param_slot(&ctx, "k1"), param_slot(&ctx, "k2"));
        assert_eq!(eval_fast_rate(&mul2, &[], &ctx, 0.0).unwrap(), 6.0);

        let mul3 = FastRateExpression::Mul3(
            param_slot(&ctx, "k1"),
            param_slot(&ctx, "k2"),
            param_slot(&ctx, "k3"),
        );
        assert_eq!(eval_fast_rate(&mul3, &[], &ctx, 0.0).unwrap(), 24.0);
    }

    #[test]
    fn fast_rate_one_minus_mul3_and_mul_add_div() {
        let ctx = ctx_with_params(&[
            ("k1", 0.25),
            ("k2", 0.5),
            ("k3", 80.0),
            ("k4", 20.0),
            ("k5", 2.0),
            ("k6", 3.0),
            ("k7", 100.0),
        ]);

        let one_minus = FastRateExpression::OneMinusMul3 {
            subtract: param_slot(&ctx, "k1"),
            factor: param_slot(&ctx, "k2"),
            value: param_slot(&ctx, "k3"),
        };
        let expected = (1.0 - 0.25) * 0.5 * 80.0;
        assert!((eval_fast_rate(&one_minus, &[], &ctx, 0.0).unwrap() - expected).abs() < 1e-12);

        let mul_add_div = FastRateExpression::MulAddDiv {
            first: param_slot(&ctx, "k2"),
            second: param_slot(&ctx, "k3"),
            add_left: param_slot(&ctx, "k5"),
            add_right: param_slot(&ctx, "k6"),
            denominator: param_slot(&ctx, "k7"),
        };
        let expected = 0.5 * 80.0 * (2.0 + 3.0) / 100.0;
        assert!((eval_fast_rate(&mul_add_div, &[], &ctx, 0.0).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn fast_rate_program_evaluates_in_postfix_order() {
        let ctx = ctx_with_params(&[("k1", 5.0), ("k2", 3.0)]);
        let program = FastRateExpression::Program(vec![
            FastRateOp::Slot(param_slot(&ctx, "k1")),
            FastRateOp::Slot(param_slot(&ctx, "k2")),
            FastRateOp::Sub,
            FastRateOp::Constant(2.0),
            FastRateOp::Mul,
        ]);
        let expected = (5.0 - 3.0) * 2.0;
        assert!((eval_fast_rate(&program, &[], &ctx, 0.0).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn fast_rate_program_neg_unary_op() {
        let ctx = ctx_with_params(&[("k1", 7.0)]);
        let program = FastRateExpression::Program(vec![
            FastRateOp::Slot(param_slot(&ctx, "k1")),
            FastRateOp::Neg,
        ]);
        assert_eq!(eval_fast_rate(&program, &[], &ctx, 0.0).unwrap(), -7.0);
    }

    #[test]
    fn fast_rate_program_detects_invalid_stack_state() {
        let ctx = MathExpressionContext::new();
        let program =
            FastRateExpression::Program(vec![FastRateOp::Constant(1.0), FastRateOp::Constant(2.0)]);
        let error = eval_fast_rate(&program, &[], &ctx, 0.0).unwrap_err();
        assert!(error.contains("invalid stack state"));
    }

    #[test]
    fn fast_rate_program_underflow_is_reported() {
        let ctx = MathExpressionContext::new();
        let program = FastRateExpression::Program(vec![FastRateOp::Add]);
        let error = eval_fast_rate(&program, &[], &ctx, 0.0).unwrap_err();
        assert!(error.contains("underflow"));
    }

    #[test]
    fn fast_rate_program_overflow_is_reported() {
        let ctx = MathExpressionContext::new();
        let ops: Vec<FastRateOp> = (0..33).map(|_| FastRateOp::Constant(1.0)).collect();
        let program = FastRateExpression::Program(ops);
        let error = eval_fast_rate(&program, &[], &ctx, 0.0).unwrap_err();
        assert!(error.contains("overflow"));
    }

    fn make_three_bin_model() -> Model {
        use commol_core::{Parameter, ParameterValue};
        Model {
            name: "abc_three_bin".to_string(),
            description: None,
            version: None,
            parameters: vec![
                Parameter {
                    id: "k1".to_string(),
                    value: Some(ParameterValue::Constant(0.3)),
                    description: None,
                },
                Parameter {
                    id: "k2".to_string(),
                    value: Some(ParameterValue::Constant(0.1)),
                    description: None,
                },
            ],
            population: Population {
                bins: vec![
                    Bin {
                        id: "A".to_string(),
                        name: "Bin A".to_string(),
                    },
                    Bin {
                        id: "B".to_string(),
                        name: "Bin B".to_string(),
                    },
                    Bin {
                        id: "C".to_string(),
                        name: "Bin C".to_string(),
                    },
                ],
                accumulators: vec![],
                stratifications: vec![],
                transitions: vec![],
                initial_conditions: InitialConditions {
                    population_size: 1000,
                    bin_fractions: vec![
                        BinFraction {
                            bin: "A".to_string(),
                            fraction: Some(0.99),
                        },
                        BinFraction {
                            bin: "B".to_string(),
                            fraction: Some(0.01),
                        },
                        BinFraction {
                            bin: "C".to_string(),
                            fraction: Some(0.0),
                        },
                    ],
                    stratification_fractions: vec![],
                },
            },
            dynamics: Dynamics {
                typology: ModelTypes::DifferenceEquations,
                transitions: vec![
                    Transition {
                        id: "a_to_b".to_string(),
                        source: vec!["A".to_string()],
                        target: vec!["B".to_string()],
                        accumulators: vec![],
                        rate: Some(RateMathExpression::from_string(
                            "k1 * A * B / N".to_string(),
                        )),
                        stratified_rates: None,
                        condition: None,
                        per_compartment: None,
                    },
                    Transition {
                        id: "b_to_c".to_string(),
                        source: vec!["B".to_string()],
                        target: vec!["C".to_string()],
                        accumulators: vec![],
                        rate: Some(RateMathExpression::from_string("k2".to_string())),
                        stratified_rates: None,
                        condition: None,
                        per_compartment: None,
                    },
                ],
            },
        }
    }

    #[test]
    fn get_parameters_exposes_n_after_step_without_compartment_context() {
        use commol_core::SimulationEngine;
        let model = make_three_bin_model();
        let mut engine = DifferenceEquations::from_model(&model);

        engine.step().unwrap();

        let params = engine.get_parameters();
        let n = params
            .get("N")
            .copied()
            .expect("N must be visible in get_parameters() after a step");
        // The model conserves total population at 1000 across the first step.
        assert!((n - 1000.0).abs() < 1e-6, "expected N=1000, got {}", n);
    }

    #[test]
    fn three_bin_model_conserves_population_through_fast_path() {
        let model = make_three_bin_model();
        let mut engine = DifferenceEquations::from_model(&model);
        let results = engine.run(50).unwrap();

        for (k, state) in results.iter().enumerate() {
            let total: f64 = state.iter().sum();
            assert!(
                (total - 1000.0).abs() < 1e-6,
                "population not conserved at step {}: total = {}",
                k,
                total
            );
        }
    }
}
