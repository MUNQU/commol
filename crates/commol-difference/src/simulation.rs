//! Simulation methods for running the difference equations engine.

use crate::types::DifferenceEquations;
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

        // Use optimized index-based compartment update (avoids cloning strings)
        self.expression_context
            .set_compartments_by_index(&self.population);

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
        for flow_info in &self.transition_flows {
            let source_population = self.population[flow_info.source_index];

            // Evaluate the pre-parsed rate expression (use cached context)
            let rate = match flow_info
                .rate_expression
                .evaluate(&mut self.expression_context)
            {
                Ok(rate_value) => rate_value,
                Err(error) => {
                    return Err(format!(
                        "Failed to evaluate rate for transition from {} to {}: {}",
                        flow_info.source_index, flow_info.target_index, error
                    ));
                }
            };

            let flow = if flow_info.references_compartments {
                // Absolute rate: use directly
                rate
            } else {
                // Per-capita rate: multiply by source population
                source_population * rate
            };

            self.compartment_flows[flow_info.source_index] -= flow;
            self.compartment_flows[flow_info.target_index] += flow;
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
        steps.push(self.population.clone());

        for _ in 0..num_steps {
            self.step()?;
            steps.push(self.population.clone());
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

        // Ensure buffer has correct capacity
        buffer.clear();
        buffer.reserve(total_steps);

        // Store initial state (t=0)
        buffer.push(self.population.clone());

        for _ in 0..num_steps {
            self.step()?;
            buffer.push(self.population.clone());
        }

        Ok(())
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

    fn population(&self) -> Vec<f64> {
        self.population.clone()
    }

    fn reset(&mut self) {
        // Reset population to initial state
        self.population = self.initial_population.clone();
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
