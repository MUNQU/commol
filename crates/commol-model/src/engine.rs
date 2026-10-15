//! What an engine of a compartmental model gives besides the generic engine
//! interface.

use commol_core::SimulationEngine;

use crate::labels::OutputLabels;

/// An engine whose state is the amount of each compartment of a
/// compartmental model, followed by its accumulators.
///
/// Both engines of the compartmental kind implement it; the initial
/// conditions that calibration writes ([`crate::InitialConditionPlan`]) use
/// it.
pub trait CompartmentalEngine: SimulationEngine {
    /// Names of the compartments, in engine order. They are the first
    /// outputs.
    fn compartments(&self) -> Vec<String>;

    /// Source and applied categories of each output, parallel to
    /// [`SimulationEngine::output_names`].
    fn output_labels(&self) -> Vec<OutputLabels>;

    /// Current amount of each compartment, in engine order.
    fn state(&self) -> Vec<f64>;

    /// The total the initial fractions are taken of.
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
