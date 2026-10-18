//! Expression evaluation context
//!
//! This module provides the context for evaluating mathematical expressions,
//! including parameters, compartments, and special variables.

use std::collections::HashMap;

/// Context for evaluating mathematical expressions
#[derive(Debug, Clone, Default)]
pub struct MathExpressionContext {
    pub(crate) parameters: HashMap<String, f64>,
    parameter_indices: HashMap<String, usize>,
    parameter_names: Vec<String>,
    parameter_values: Vec<Option<f64>>,
    pub(crate) compartments: HashMap<String, f64>,
    pub(crate) step: f64,
    /// Continuous time `t`. Equal to `step` in a difference model; inside a
    /// step of a differential model `step` stays at the step that holds `t`.
    pub(crate) time: f64,
    /// Cached compartment names for index-based updates
    compartment_names: Vec<String>,
}

impl MathExpressionContext {
    /// Create a new expression context
    pub fn new() -> Self {
        Self::default()
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
    }

    /// Set a parameter value
    pub fn set_parameter(&mut self, name: String, value: f64) {
        self.set_parameter_indexed(name, value);
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
    }

    /// Pre-allocate parameter slots so subsequent `set_parameter_str` calls hit
    /// the existing-key branch and avoid allocating a fresh `String` each step.
    pub fn reserve_parameters(&mut self, names: impl IntoIterator<Item = String>) {
        for name in names {
            self.reserve_parameter_index(name);
        }
    }

    /// Set multiple parameters
    pub fn set_parameters(&mut self, parameters: HashMap<String, f64>) {
        for (name, value) in parameters {
            self.set_parameter_indexed(name, value);
        }
    }

    /// Set the amount of a compartment
    pub fn set_compartment(&mut self, name: String, value: f64) {
        self.compartments.insert(name, value);
    }

    /// Set the current step, and the time `t` to the same value.
    pub fn set_step(&mut self, step: f64) {
        self.step = step;
        self.time = step;
    }

    /// Set the time `t` and leave `step` as it is.
    ///
    /// A differential engine calls it after [`Self::set_step`], so `step` is
    /// the step that holds `t`.
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
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
    /// indexed value storage and the name-keyed HashMap, so `get_parameters()`
    /// and the compiled expressions read the same value.
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
}
