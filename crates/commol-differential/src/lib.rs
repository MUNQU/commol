//! Ordinary differential equations engine for compartment models.
//!
//! The model is compiled by `commol-model`; this crate integrates its
//! right-hand side, `dX/dt = Σ flows(t, X)`, with [`diffsol`]. Commol keeps no
//! numerical method of its own.
//!
//! - Methods: `tsit45` (explicit, the default) and `bdf` (stiff). `bdf` gets
//!   its Jacobian from a finite-difference Jacobian-vector product, dense below
//!   [`SPARSE_THRESHOLD`] state components and sparse from there.
//! - Time: on `[n, n + 1)` rates see `step = n` and the continuous `t`, so a
//!   rate that reads `step` or a pulse series is constant on the interval. A
//!   `"step_function"` or `"linear"` series is read at `t`, and the
//!   integration restarts at its points.
//! - Accumulators are state components: `dA/dt = Σ counted flows`.
//! - Impulses act at their positions: the integration stops there, the state
//!   jumps and the integration restarts from the new state.
//! - Outputs: at any sorted list of positions, from the dense output of the
//!   solver.
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use commol_core::SimulationEngine;
//! use commol_differential::DifferentialEquations;
//!
//! let mut engine = DifferentialEquations::try_from_model(&model)?;
//! let rows = engine.run_at(&[0.0, 0.5, 2.25])?;
//! ```

#![warn(missing_docs)]

mod solve;

pub use solve::SPARSE_THRESHOLD;

use std::collections::HashMap;

use commol_core::{EngineSettings, ModelValue, OdeSolver};
use commol_model::{
    CompartmentalEngine, CompiledModel, InitialConditionPlan, OutputLabels, output_sums,
};

/// Differential equations simulation engine.
///
/// It wraps a [`CompiledModel`] and integrates it in continuous time with the
/// solver settings of the model.
#[derive(Clone)]
pub struct DifferentialEquations {
    model: CompiledModel,
    settings: OdeSolver,
    /// Current state: compartments, then accumulators.
    state: Vec<f64>,
    current_time: f64,
    /// Whether the impulses at `current_time` are already in `state`.
    settled: bool,
}

impl DifferentialEquations {
    /// Compile a differential-equations model into an engine.
    ///
    /// # Errors
    ///
    /// Returns a message when the model is not a differential-equations
    /// model, when a tolerance is not positive, or when
    /// [`CompiledModel::try_from_model`] fails.
    pub fn try_from_model(model: &commol_core::Model) -> Result<Self, String> {
        let EngineSettings::DifferentialEquations(settings) = *model.engine.settings() else {
            return Err(format!(
                "The DifferentialEquations engine cannot solve a {} model",
                model.engine.engine_type().name()
            ));
        };
        Self::from_compiled(CompiledModel::try_from_model(model)?, settings)
    }

    /// Integrate a compiled model with the given solver settings, whatever
    /// its engine.
    ///
    /// # Errors
    ///
    /// Returns a message when `rtol` or `atol` is not a positive number.
    pub fn from_compiled(model: CompiledModel, settings: OdeSolver) -> Result<Self, String> {
        settings.validate()?;
        let state = model.initial_state().to_vec();
        Ok(Self {
            model,
            settings,
            state,
            current_time: 0.0,
            settled: false,
        })
    }

    /// The compiled model the engine integrates.
    pub fn model(&self) -> &CompiledModel {
        &self.model
    }

    /// The solver settings.
    pub fn settings(&self) -> &OdeSolver {
        &self.settings
    }

    /// Current population of each compartment.
    pub fn population(&self) -> Vec<f64> {
        self.state[..self.model.num_compartments()].to_vec()
    }

    /// Names of the compartments.
    pub fn compartments(&self) -> Vec<String> {
        self.model.compartments().to_vec()
    }

    /// Names of every output: compartments, then accumulators.
    pub fn output_names(&self) -> Vec<String> {
        self.model.output_names().to_vec()
    }

    /// Apply the impulses at the current time, once.
    fn settle(&mut self) -> Result<(), String> {
        if !self.settled {
            self.model
                .apply_impulses(self.current_time, &mut self.state)?;
            self.settled = true;
        }
        Ok(())
    }

    /// Integrate from the current time through the last of `positions`,
    /// writing the state at each of them into `buffer`.
    ///
    /// `positions` must be sorted, unique and not before the current time.
    /// The output at a position where an impulse acts shows the state after
    /// the jump.
    ///
    /// # Errors
    ///
    /// Returns a message when the positions are not sorted, unique, finite,
    /// non-negative and not before the current time, when a rate cannot be
    /// evaluated, or when the solver fails.
    fn integrate_to(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        commol_core::check_positions(positions, self.current_time, check_time)?;
        let output_len = self.state.len();
        buffer.resize_with(positions.len(), Vec::new);
        for row in buffer.iter_mut() {
            row.resize(output_len, 0.0);
        }

        self.settle()?;
        let first_later = positions
            .iter()
            .position(|&p| p > self.current_time)
            .unwrap_or(positions.len());
        for row in &mut buffer[..first_later] {
            row.copy_from_slice(&self.state);
        }
        let later = &positions[first_later..];
        let Some(&end) = later.last() else {
            return Ok(());
        };
        solve::integrate(
            &mut self.model,
            &self.settings,
            self.current_time,
            &mut self.state,
            later,
            &mut buffer[first_later..],
        )?;
        self.current_time = end;
        // `integrate` applied the impulses at `end`.
        self.settled = true;
        Ok(())
    }
}

/// Why a differential engine does not report at `position`, if it does not.
fn check_time(position: f64) -> Result<(), String> {
    if position.is_finite() && position >= 0.0 {
        Ok(())
    } else {
        Err(format!(
            "a differential-equations model reports at finite, non-negative positions; \
             {position} is not one"
        ))
    }
}

impl commol_core::SimulationEngine for DifferentialEquations {
    type ValuePlan = InitialConditionPlan;

    fn check_position(&self, position: f64) -> Result<(), String> {
        check_time(position)
    }

    fn step(&mut self) -> Result<(), String> {
        let end = self.current_time + 1.0;
        let mut buffer = Vec::with_capacity(1);
        self.integrate_to(&[end], &mut buffer)
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        self.integrate_to(positions, buffer)
    }

    fn output_names(&self) -> Vec<String> {
        DifferentialEquations::output_names(self)
    }

    fn reset(&mut self) {
        self.state.copy_from_slice(self.model.initial_state());
        self.current_time = 0.0;
        self.settled = false;
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.model.set_parameter(parameter_id, value);
        Ok(())
    }

    fn get_parameters(&self) -> &HashMap<String, f64> {
        self.model.parameters()
    }

    fn current_position(&self) -> f64 {
        self.current_time
    }

    fn state_outputs(&self) -> usize {
        self.model.num_compartments()
    }

    fn output_sums(&self) -> Vec<(String, Vec<usize>)> {
        output_sums(self.model.output_labels())
    }

    fn plan_values(
        &self,
        values: &[ModelValue],
        remainder: Option<&str>,
    ) -> Result<InitialConditionPlan, String> {
        InitialConditionPlan::new(self, values, remainder)
    }

    fn apply_values(&mut self, plan: &InitialConditionPlan, values: &[f64]) -> Result<f64, String> {
        plan.apply(self, values)
    }
}

impl CompartmentalEngine for DifferentialEquations {
    fn compartments(&self) -> Vec<String> {
        DifferentialEquations::compartments(self)
    }

    fn output_labels(&self) -> Vec<OutputLabels> {
        self.model.output_labels().to_vec()
    }

    fn population(&self) -> Vec<f64> {
        DifferentialEquations::population(self)
    }

    fn population_size(&self) -> f64 {
        self.model.population_size()
    }

    fn set_initial_condition(
        &mut self,
        compartment_index: usize,
        value: f64,
    ) -> Result<(), String> {
        self.model.set_initial_condition(compartment_index, value)?;
        self.state[compartment_index] = value;
        if self.current_time == 0.0 && self.settled {
            // The impulses at time 0 must act on the new initial state.
            self.state.copy_from_slice(self.model.initial_state());
            self.settled = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
