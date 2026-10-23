//! What an engine of an aggregate model gives besides the generic engine
//! interface.

use commol_core::SimulationEngine;

use crate::labels::OutputLabels;

/// An engine whose state is the amount of each compartment of a
/// aggregate model, followed by its accumulators.
///
/// Every engine of the aggregate kind implements it; the initial
/// conditions that calibration writes ([`crate::InitialConditionPlan`]) use
/// it.
pub trait CompartmentalEngine: SimulationEngine {
    /// Names of the compartments, in engine order. They are the first
    /// outputs.
    fn compartments(&self) -> Vec<String>;

    /// Source and applied categories of each output, parallel to
    /// [`SimulationEngine::output_names`].
    fn output_labels(&self) -> Vec<OutputLabels>;

    /// Initial amount of each compartment, in engine order: the state
    /// [`SimulationEngine::reset`] restores, whatever position the engine has
    /// reached.
    fn initial_state(&self) -> Vec<f64>;

    /// The sum of the initial amounts the model states, before any calibrated
    /// value is written.
    fn total(&self) -> f64;

    /// Change the initial amount of one compartment, and the current one
    /// when the engine is at its start.
    ///
    /// After this call, [`SimulationEngine::reset`] restores the new value.
    ///
    /// # Errors
    ///
    /// Returns a message when the index names no compartment or the value is
    /// negative.
    fn set_initial_condition(&mut self, compartment_index: usize, value: f64)
    -> Result<(), String>;
}
