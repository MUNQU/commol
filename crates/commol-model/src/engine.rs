//! What an engine of an aggregate model gives besides the generic engine
//! interface, and what the four engines of the aggregate kind share: the
//! methods they take from their compiled model, the run of an engine that
//! reports only at integer steps, and the numbers of their messages.

use commol_core::{EngineType, SimulationEngine, SimulationError, check_positions};

use crate::labels::OutputLabels;
use crate::types::CompiledModel;

/// An engine whose state is the amount of each compartment of an
/// aggregate model, followed by its accumulators.
///
/// Every engine of the aggregate kind implements it; the initial
/// conditions that calibration writes ([`crate::InitialConditionPlan`]) use
/// it. The engine gives its compiled model ([`Self::model`]), and the other
/// methods read it by default. The methods of [`SimulationEngine`] that only
/// read the compiled model come from [`crate::delegate_to_compiled_model`].
pub trait CompartmentalEngine: SimulationEngine {
    /// The compiled model the engine runs.
    fn model(&self) -> &CompiledModel;

    /// The compiled model the engine runs, for a change of a parameter
    /// ([`CompiledModel::set_parameter`]) or of the compartments that have no
    /// value ([`CompiledModel::check_values`]). The engine keeps its state apart
    /// from the model, so a change of the initial state goes through
    /// [`Self::set_initial_condition`].
    fn model_mut(&mut self) -> &mut CompiledModel;

    /// Names of the compartments, in engine order. They are the first
    /// outputs.
    fn compartments(&self) -> Vec<String> {
        self.model().compartments().to_vec()
    }

    /// Source and applied categories of each output, parallel to
    /// [`SimulationEngine::output_names`].
    fn output_labels(&self) -> Vec<OutputLabels> {
        self.model().output_labels().to_vec()
    }

    /// Initial amount of each compartment, in engine order: the state
    /// [`SimulationEngine::reset`] restores, whatever position the engine has
    /// reached.
    fn initial_state(&self) -> Vec<f64> {
        let model = self.model();
        model.initial_state()[..model.num_compartments()].to_vec()
    }

    /// The sum of the initial amounts the model states, before any calibrated
    /// value is written.
    fn total(&self) -> f64 {
        self.model().total()
    }

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

/// The methods of [`SimulationEngine`] that an engine of the aggregate kind
/// takes from its compiled model ([`CompartmentalEngine::model`]), to write
/// in its `impl SimulationEngine` block: the type `ValuePlan`
/// ([`crate::InitialConditionPlan`]), `output_names`, `state_outputs` (the
/// compartments), `output_sums` ([`crate::output_sums`]), `matrices`
/// ([`CompiledModel::matrices`]), `check_values`
/// ([`CompiledModel::check_values`]), `plan_values`, `apply_values` and
/// `set_parameter` ([`CompiledModel::set_parameter`]).
///
/// A trait cannot give a default to a method of its supertrait, so the
/// engines take these methods from this macro. `delegate_to_compiled_model!(
/// except set_parameter)` leaves out `set_parameter`, for an engine that does
/// more than change the parameter. The crate that calls it depends on
/// `commol-core`.
#[macro_export]
macro_rules! delegate_to_compiled_model {
    () => {
        $crate::delegate_to_compiled_model!(except set_parameter);

        fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
            $crate::CompartmentalEngine::model_mut(self).set_parameter(parameter_id, value)
        }
    };
    (except set_parameter) => {
        type ValuePlan = $crate::InitialConditionPlan;

        fn output_names(&self) -> Vec<String> {
            $crate::CompartmentalEngine::model(self)
                .output_names()
                .to_vec()
        }

        fn state_outputs(&self) -> usize {
            $crate::CompartmentalEngine::model(self).num_compartments()
        }

        fn output_sums(&self) -> Vec<(String, Vec<usize>)> {
            $crate::output_sums($crate::CompartmentalEngine::model(self).output_labels())
        }

        fn matrices(&self) -> Vec<::commol_core::ScaledMatrix> {
            $crate::CompartmentalEngine::model(self).matrices().to_vec()
        }

        fn check_values(&self) -> Result<(), String> {
            $crate::CompartmentalEngine::model(self).check_values()
        }

        fn plan_values(
            &self,
            values: &[::commol_core::ModelValue],
            remainder: Option<&str>,
        ) -> Result<$crate::InitialConditionPlan, String> {
            $crate::InitialConditionPlan::new(self, values, remainder)
        }

        fn apply_values(
            &mut self,
            plan: &$crate::InitialConditionPlan,
            values: &[f64],
        ) -> Result<f64, String> {
            plan.apply(self, values)
        }
    };
}

/// An engine that reports only at integer steps
/// ([`EngineType::reports_at_integer_steps`]), as [`record_steps`] runs it.
pub trait IntegerStepper {
    /// The engine, for the rule of the positions it reports at.
    const ENGINE: EngineType;

    /// The step the engine is at.
    fn current_step(&self) -> f64;

    /// Apply the impulses at the current step, once, so the state is the
    /// one the engine reports there.
    ///
    /// # Errors
    ///
    /// Returns the error of an impulse that the engine cannot apply.
    fn settle(&mut self) -> Result<(), SimulationError>;

    /// Advance one step from the settled state.
    ///
    /// # Errors
    ///
    /// Returns the error of the step.
    fn advance(&mut self) -> Result<(), SimulationError>;

    /// Write the state, compartments then accumulators, into `row`.
    fn copy_row(&self, row: &mut Vec<f64>);
}

/// Step an engine through the last of `positions`, writing its state at each
/// of them into `buffer`, one row per position.
///
/// The rows of `buffer` keep their allocations from one call to the next,
/// which a calibration that runs the same engine many times relies on.
///
/// # Errors
///
/// Returns [`SimulationError::Failed`] when a value of the model is missing
/// ([`CompiledModel::check_values`]), a position is not an integer step
/// ([`EngineType::check_position`]), or the positions are not sorted, unique
/// and not before the current step ([`check_positions`]); and the error of a
/// step or an impulse.
pub fn record_steps<E: IntegerStepper + CompartmentalEngine>(
    engine: &mut E,
    positions: &[f64],
    buffer: &mut Vec<Vec<f64>>,
) -> Result<(), SimulationError> {
    engine.model().check_values()?;
    check_positions(positions, engine.current_step(), |position| {
        E::ENGINE.check_position(position)
    })?;
    buffer.resize_with(positions.len(), Vec::new);
    engine.settle()?;
    let mut next = 0;
    loop {
        while next < positions.len() && positions[next] == engine.current_step() {
            engine.copy_row(&mut buffer[next]);
            next += 1;
        }
        if next == positions.len() {
            return Ok(());
        }
        engine.advance()?;
        engine.settle()?;
    }
}

/// `value` to six significant digits, for a message.
pub fn rounded(value: f64) -> f64 {
    format!("{value:.5e}").parse().unwrap_or(value)
}
