//! Expression evaluation context
//!
//! This module provides the context for evaluating mathematical expressions,
//! including parameters, compartments, and special variables.

use evalexpr::{ContextWithMutableVariables, HashMapContext, Value};
use std::collections::HashMap;

use super::preprocessing::{
    SPECIAL_VAR_E, SPECIAL_VAR_N, SPECIAL_VAR_PI, SPECIAL_VAR_STEP, SPECIAL_VAR_T,
};

/// Context for evaluating mathematical expressions
#[derive(Debug, Clone)]
pub struct MathExpressionContext {
    pub(crate) parameters: HashMap<String, f64>,
    parameter_indices: HashMap<String, usize>,
    parameter_names: Vec<String>,
    parameter_values: Vec<Option<f64>>,
    pub(crate) compartments: HashMap<String, f64>,
    pub(crate) step: f64,
    /// Cached compartment names for index-based updates
    compartment_names: Vec<String>,
    /// Cached evalexpr context to avoid rebuilding
    cached_evalexpr_context: Option<HashMapContext>,
    /// Flag to indicate if cache is dirty
    cache_dirty: bool,
}

impl MathExpressionContext {
    /// Create a new expression context
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            parameter_indices: HashMap::new(),
            parameter_names: Vec::new(),
            parameter_values: Vec::new(),
            compartments: HashMap::new(),
            step: 0.0,
            compartment_names: Vec::new(),
            cached_evalexpr_context: None,
            cache_dirty: true,
        }
    }

    /// Initialize compartment names for index-based updates
    /// This should be called once during engine initialization
    pub fn init_compartments(&mut self, names: Vec<String>) {
        self.compartment_names = names.clone();
        for name in names {
            self.compartments.insert(name, 0.0);
        }
    }

    /// Set compartment values by index (much faster than by name)
    /// The values vector must match the order of compartment_names
    pub fn set_compartments_by_index(&mut self, values: &[f64]) {
        for (i, value) in values.iter().enumerate() {
            if i < self.compartment_names.len() {
                // Direct update using pre-stored reference
                if let Some(comp_value) = self.compartments.get_mut(&self.compartment_names[i]) {
                    *comp_value = *value;
                }
            }
        }

        // Update cached context directly if it exists (more efficient than rebuilding)
        if let Some(ref mut ctx) = self.cached_evalexpr_context {
            for (i, value) in values.iter().enumerate() {
                if i < self.compartment_names.len() {
                    ctx.set_value(self.compartment_names[i].clone(), Value::Float(*value))
                        .ok();
                }
            }
            // Update N (total population). Callers may pass a wider row than the
            // registered compartments (simulation outputs append accumulators after
            // the population), so only the registered compartments count towards N
            // — matching the assignment loop above.
            let total_pop: f64 = values.iter().take(self.compartment_names.len()).sum();
            ctx.set_value(SPECIAL_VAR_N.to_string(), Value::Float(total_pop))
                .ok();
        } else {
            self.cache_dirty = true;
        }
    }

    /// Set a parameter value
    pub fn set_parameter(&mut self, name: String, value: f64) {
        self.set_parameter_indexed(name, value);
        self.cache_dirty = true;
    }

    /// Set a parameter value by string reference (avoids allocation in the
    /// common case where the key already exists).
    pub fn set_parameter_str(&mut self, name: &str, value: f64) {
        if let Some(&idx) = self.parameter_indices.get(name) {
            self.parameter_values[idx] = Some(value);
            if let Some(slot) = self.parameters.get_mut(name) {
                *slot = value;
            } else {
                self.parameters.insert(name.to_string(), value);
            }
        } else {
            self.set_parameter_indexed(name.to_string(), value);
        }

        // Update cached context directly if it exists
        if let Some(ref mut ctx) = self.cached_evalexpr_context {
            ctx.set_value(name.to_string(), Value::Float(value)).ok();
        } else {
            self.cache_dirty = true;
        }
    }

    /// Pre-allocate parameter slots so subsequent `set_parameter_str` calls hit
    /// the existing-key branch and avoid allocating a fresh `String` each step.
    pub fn reserve_parameters(&mut self, names: impl IntoIterator<Item = String>) {
        for name in names {
            self.reserve_parameter_index(name);
        }
        self.cache_dirty = true;
    }

    /// Set multiple parameters
    pub fn set_parameters(&mut self, parameters: HashMap<String, f64>) {
        for (name, value) in parameters {
            self.set_parameter_indexed(name, value);
        }
        self.cache_dirty = true;
    }

    /// Set a compartment population value
    pub fn set_compartment(&mut self, name: String, value: f64) {
        self.compartments.insert(name, value);
        self.cache_dirty = true;
    }

    /// Set the current step
    pub fn set_step(&mut self, step: f64) {
        self.step = step;

        // Update cached context directly if it exists
        if let Some(ref mut ctx) = self.cached_evalexpr_context {
            ctx.set_value(SPECIAL_VAR_STEP.to_string(), Value::Float(step))
                .ok();
            ctx.set_value(SPECIAL_VAR_T.to_string(), Value::Float(step))
                .ok();
        } else {
            self.cache_dirty = true;
        }
    }

    /// Get a single parameter value by name
    pub fn get_parameter(&self, name: &str) -> Option<f64> {
        self.parameter_indices
            .get(name)
            .and_then(|&idx| self.parameter_values.get(idx).and_then(|value| *value))
    }

    /// Get a stable parameter slot for direct indexed reads in hot paths.
    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.parameter_indices.get(name).copied()
    }

    /// Get a parameter value by a previously resolved index.
    pub fn get_parameter_by_index(&self, index: usize) -> Option<f64> {
        self.parameter_values.get(index).and_then(|value| *value)
    }

    /// Set a parameter slot by a previously resolved index. Updates both the
    /// indexed value storage and the name-keyed HashMap so that
    /// `get_parameters()` and evalexpr fallback evaluation stay consistent.
    /// Skips the per-call evalexpr context patch performed by
    /// `set_parameter_str`; the next fallback evaluation will rebuild the
    /// cache from the up-to-date HashMap.
    pub fn set_parameter_by_index(&mut self, index: usize, value: f64) -> Result<(), String> {
        let Some(slot) = self.parameter_values.get_mut(index) else {
            return Err(format!("Parameter index '{}' not found", index));
        };
        *slot = Some(value);
        let name = &self.parameter_names[index];
        if let Some(map_slot) = self.parameters.get_mut(name) {
            *map_slot = value;
        } else {
            self.parameters.insert(name.clone(), value);
        }
        self.cache_dirty = true;
        Ok(())
    }

    /// Get a reference to all parameters
    pub fn get_parameters(&self) -> &HashMap<String, f64> {
        &self.parameters
    }

    fn set_parameter_indexed(&mut self, name: String, value: f64) {
        if let Some(&idx) = self.parameter_indices.get(name.as_str()) {
            self.parameter_values[idx] = Some(value);
            if let Some(slot) = self.parameters.get_mut(name.as_str()) {
                *slot = value;
            } else {
                self.parameters.insert(name, value);
            }
            return;
        }

        let idx = self.parameter_values.len();
        self.parameter_indices.insert(name.clone(), idx);
        self.parameter_names.push(name.clone());
        self.parameter_values.push(Some(value));
        self.parameters.insert(name, value);
    }

    fn reserve_parameter_index(&mut self, name: String) {
        if self.parameter_indices.contains_key(name.as_str()) {
            return;
        }

        let idx = self.parameter_values.len();
        self.parameter_indices.insert(name.clone(), idx);
        self.parameter_names.push(name);
        self.parameter_values.push(None);
    }

    /// Get or rebuild the evalexpr context (uses cache when possible)
    pub(crate) fn get_evalexpr_context(&mut self) -> &HashMapContext {
        if self.cache_dirty || self.cached_evalexpr_context.is_none() {
            let mut context = HashMapContext::new();

            // Add user-defined parameters
            for (name, value) in &self.parameters {
                context.set_value(name.clone(), Value::Float(*value)).ok();
            }

            // Add compartment values
            for (name, value) in &self.compartments {
                context.set_value(name.clone(), Value::Float(*value)).ok();
            }

            // Add special variables
            // Sum compartment values in sorted order for deterministic floating-point results.
            // HashMap iteration order is non-deterministic, and floating-point addition
            // is not associative, so summing in different orders produces different results.
            let mut sorted_values: Vec<f64> = self.compartments.values().copied().collect();
            sorted_values.sort_by(|a, b| a.total_cmp(b));
            let total_pop: f64 = sorted_values.iter().sum();
            context
                .set_value(SPECIAL_VAR_N.to_string(), Value::Float(total_pop))
                .ok();
            context
                .set_value(SPECIAL_VAR_STEP.to_string(), Value::Float(self.step))
                .ok();
            context
                .set_value(SPECIAL_VAR_T.to_string(), Value::Float(self.step))
                .ok();
            context
                .set_value(
                    SPECIAL_VAR_PI.to_string(),
                    Value::Float(std::f64::consts::PI),
                )
                .ok();
            context
                .set_value(SPECIAL_VAR_E.to_string(), Value::Float(std::f64::consts::E))
                .ok();

            self.cached_evalexpr_context = Some(context);
            self.cache_dirty = false;
        }

        self.cached_evalexpr_context.as_ref().unwrap()
    }
}

impl Default for MathExpressionContext {
    fn default() -> Self {
        Self::new()
    }
}
