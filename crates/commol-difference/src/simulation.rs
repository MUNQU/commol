//! Simulation methods for running the difference equations engine.

use crate::types::{DifferenceEquations, VarSlot};
use commol_core::RateMathExpression;
use std::collections::HashMap;

// Cached parameter names to avoid string allocations
const PARAM_N: &str = "N";
const PARAM_T: &str = "t";

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

        // Update expression context with current population values
        self.expression_context.set_step(self.current_step);

        // Calculate and set total population N (use &str to avoid allocation)
        let total_population: f64 = self.population.iter().sum();
        self.expression_context
            .set_parameter_str(PARAM_N, total_population);

        // Set t as an alias for step (for convenience in formulas)
        self.expression_context
            .set_parameter_str(PARAM_T, self.current_step);

        // Only evalexpr fallback/formula-parameter paths need compartment
        // values in the expression context. JIT-resolved transition formulas
        // read compartment values directly from `population`.
        if self.requires_compartment_context {
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
            self.expression_context
                .set_parameter_str(&mapping.parameter_name, total);
        }

        // Evaluate time-series parameters (O(log N) binary search, no allocation)
        let step_u64 = self.current_step as u64;
        for ts in &self.series_parameters {
            let value = ts.evaluate(step_u64);
            self.expression_context
                .set_parameter_str(&ts.parameter_name, value);
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

            let rate = match (&flow_info.rate_expression, &flow_info.resolved_slots) {
                // Fast path: JIT-compiled formula with pre-resolved slots.
                // Fill the buffer using direct compartment/Step indexing plus
                // one HashMap probe per parameter, then call the JIT directly.
                (RateMathExpression::Formula(expr), Some(slots)) => {
                    let n = slots.len();
                    let values: &mut [f64] = if n <= jit_buf.len() {
                        &mut jit_buf[..n]
                    } else {
                        // Rare: fall back to heap for very wide expressions.
                        let mut heap = vec![0.0; n];
                        for (i, slot) in slots.iter().enumerate() {
                            heap[i] = resolve_slot(
                                slot,
                                &self.population,
                                &self.expression_context,
                                self.current_step,
                            )
                            .map_err(|error| {
                                format!(
                                    "Failed to evaluate rate for transition from {:?} to {:?}: {}",
                                    flow_info.source_index, flow_info.target_index, error
                                )
                            })?;
                        }
                        match expr.call_jit_with_buffer(&heap) {
                            Ok(v) => {
                                let r = v;
                                let flow = if flow_info.is_absolute_flow {
                                    r
                                } else {
                                    source_population * r
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
                                continue;
                            }
                            Err(error) => {
                                return Err(format!(
                                    "Failed to evaluate rate for transition from {:?} to {:?}: {}",
                                    flow_info.source_index, flow_info.target_index, error
                                ));
                            }
                        }
                    };
                    for (i, slot) in slots.iter().enumerate() {
                        values[i] = resolve_slot(
                            slot,
                            &self.population,
                            &self.expression_context,
                            self.current_step,
                        )
                        .map_err(|error| {
                            format!(
                                "Failed to evaluate rate for transition from {:?} to {:?}: {}",
                                flow_info.source_index, flow_info.target_index, error
                            )
                        })?;
                    }
                    match expr.call_jit_with_buffer(values) {
                        Ok(v) => v,
                        Err(error) => {
                            return Err(format!(
                                "Failed to evaluate rate for transition from {:?} to {:?}: {}",
                                flow_info.source_index, flow_info.target_index, error
                            ));
                        }
                    }
                }
                // Slow path: constant, single parameter, or evalexpr fallback.
                _ => match flow_info
                    .rate_expression
                    .evaluate(&mut self.expression_context)
                {
                    Ok(rate_value) => rate_value,
                    Err(error) => {
                        return Err(format!(
                            "Failed to evaluate rate for transition from {:?} to {:?}: {}",
                            flow_info.source_index, flow_info.target_index, error
                        ));
                    }
                },
            };

            let flow = if flow_info.is_absolute_flow {
                // Absolute rate: use directly
                rate
            } else {
                // Per-capita rate: multiply by source population
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

/// Resolve a pre-computed `VarSlot` to its current numeric value. Compartments
/// and the step alias are O(1); parameters fall back to a single HashMap probe.
#[inline]
fn resolve_slot(
    slot: &VarSlot,
    population: &[f64],
    ctx: &commol_core::MathExpressionContext,
    current_step: f64,
) -> Result<f64, String> {
    match slot {
        VarSlot::Compartment(idx) => Ok(population[*idx]),
        VarSlot::Step => Ok(current_step),
        VarSlot::Parameter(name) => ctx
            .get_parameter(name)
            .ok_or_else(|| format!("Variable '{}' not found", name)),
    }
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
}
