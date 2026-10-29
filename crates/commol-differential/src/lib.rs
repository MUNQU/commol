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
//! ```rust
//! use commol_core::{Model, SimulationEngine};
//! use commol_differential::DifferentialEquations;
//!
//! // A → B at the rate k1, from A = 900 and B = 100.
//! let json = r#"{"name":"m","parameters":[{"id":"k1","value":0.1}],
//!     "engine":{"type":"DifferentialEquations"},
//!     "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
//!     "accumulators":[],"stratifications":[],
//!     "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k1"}]},
//!     "structure":{"kind":"aggregate","bin_amounts":{"A":900.0,"B":100.0}}}"#;
//! let model: Model = serde_json::from_str(json).unwrap();
//! let mut engine = DifferentialEquations::try_from_model(&model).unwrap();
//! let rows = engine.run_at(&[0.0, 0.5, 2.25]).unwrap();
//! assert_eq!(rows.len(), 3);
//! assert!(rows[2][0] < rows[1][0]);
//! ```

#![warn(missing_docs)]

mod solve;

pub use solve::SPARSE_THRESHOLD;

use commol_core::{EngineSettings, EngineType, OdeSolver, SimulationError};
use commol_model::{CompartmentalEngine, CompiledModel};

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
    /// Compile a model of the `DifferentialEquations` engine.
    ///
    /// # Errors
    ///
    /// Returns a message when the model is not of this engine, when a
    /// tolerance is not positive, or when
    /// [`CompiledModel::try_from_model`] fails.
    pub fn try_from_model(model: &commol_core::Model) -> Result<Self, String> {
        let EngineSettings::DifferentialEquations(settings) = *model.engine.settings() else {
            return Err(format!(
                "The {} engine cannot run a {} model",
                EngineType::DifferentialEquations.name(),
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

    /// Current amount of each compartment, without the accumulators.
    pub fn state(&self) -> Vec<f64> {
        self.state[..self.model.num_compartments()].to_vec()
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
        commol_core::check_positions(positions, self.current_time, |position| {
            EngineType::DifferentialEquations.check_position(position)
        })?;
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

impl commol_core::SimulationEngine for DifferentialEquations {
    commol_model::delegate_to_compiled_model!();

    fn check_position(&self, position: f64) -> Result<(), String> {
        EngineType::DifferentialEquations.check_position(position)
    }

    fn engine_type(&self) -> Option<EngineType> {
        Some(EngineType::DifferentialEquations)
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        let end = self.current_time + 1.0;
        let mut buffer = Vec::with_capacity(1);
        Ok(self.integrate_to(&[end], &mut buffer)?)
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        Ok(self.integrate_to(positions, buffer)?)
    }

    /// The relative tolerance of the solver.
    fn output_accuracy(&self) -> f64 {
        self.settings.rtol
    }

    /// The absolute tolerance of the solver.
    fn output_absolute_accuracy(&self) -> f64 {
        self.settings.atol
    }

    /// The same engine with `rtol = relative` and `atol = absolute`.
    fn at_accuracy(&self, relative: f64, absolute: f64) -> Option<Self> {
        Some(Self {
            settings: OdeSolver {
                rtol: relative,
                atol: absolute,
                ..self.settings
            },
            ..self.clone()
        })
    }

    fn reset(&mut self) {
        self.state.copy_from_slice(self.model.initial_state());
        self.current_time = 0.0;
        self.settled = false;
    }

    fn current_position(&self) -> f64 {
        self.current_time
    }
}

impl CompartmentalEngine for DifferentialEquations {
    fn model(&self) -> &CompiledModel {
        &self.model
    }

    fn model_mut(&mut self) -> &mut CompiledModel {
        &mut self.model
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
