//! Values that calibration writes into an engine.
//!
//! Parameters are set by id. Values of the model kind, such as initial
//! conditions, are resolved and written by the engine
//! ([`SimulationEngine::plan_values`], [`SimulationEngine::apply_values`]), so
//! this crate does not know how a kind stores them. Factors and noise
//! parameters are not written: they apply to observations.

use commol_core::{ModelValue, ModelValueType, SimulationEngine, ValuePlan};

use crate::types::{CalibrationParameter, CalibrationParameterType};

/// The type of model value a calibration parameter is, or `None` for a
/// parameter, a factor or a noise parameter.
fn model_value_type(parameter_type: CalibrationParameterType) -> Option<ModelValueType> {
    match parameter_type {
        CalibrationParameterType::Parameter
        | CalibrationParameterType::Factor
        | CalibrationParameterType::Noise => None,
        CalibrationParameterType::InitialCondition => Some(ModelValueType::InitialCondition),
    }
}

/// How the values of a set of calibration parameters reach an engine.
#[derive(Clone, Debug)]
pub(crate) struct ValueWriter<E: SimulationEngine> {
    /// Index of each value of the model kind among the parameters, parallel
    /// to the plan.
    kind_indices: Vec<usize>,
    /// The values of the model kind, resolved by the engine.
    plan: E::ValuePlan,
}

impl<E: SimulationEngine> ValueWriter<E> {
    /// Check that every factor has positive bounds and the engine sets every
    /// parameter, and ask it to resolve the values of its model kind.
    ///
    /// Each parameter is set on a copy of the engine at its lower bound, so
    /// the rule of the engine ([`SimulationEngine::set_parameter`]) rejects
    /// an id it does not let a caller set before any run.
    ///
    /// # Errors
    ///
    /// Returns a message that names a factor with a bound at or below 0, or
    /// the engine's message when it does not set a parameter or does not
    /// accept a value, or an id or the remainder does not resolve.
    pub(crate) fn new(
        engine: &E,
        parameters: &[CalibrationParameter],
        remainder: Option<&str>,
    ) -> Result<Self, String> {
        let mut probe = engine.clone();
        for parameter in parameters {
            match parameter.parameter_type {
                CalibrationParameterType::Parameter => {
                    probe.set_parameter(&parameter.id, parameter.min_bound)?;
                }
                CalibrationParameterType::Factor
                    if !(parameter.min_bound > 0.0 && parameter.max_bound > 0.0) =>
                {
                    return Err(format!(
                        "Factor parameter '{}' must have positive bounds (got min={}, max={})",
                        parameter.id, parameter.min_bound, parameter.max_bound
                    ));
                }
                CalibrationParameterType::Factor
                | CalibrationParameterType::InitialCondition
                | CalibrationParameterType::Noise => {}
            }
        }
        let mut kind_indices = Vec::new();
        let mut values = Vec::new();
        for (index, parameter) in parameters.iter().enumerate() {
            if let Some(value_type) = model_value_type(parameter.parameter_type) {
                kind_indices.push(index);
                values.push(ModelValue {
                    value_type,
                    id: parameter.id.clone(),
                });
            }
        }
        let plan = engine.plan_values(&values, remainder)?;
        Ok(Self { kind_indices, plan })
    }

    /// The values of the model kind, as the engine resolved them.
    pub(crate) fn plan(&self) -> &E::ValuePlan {
        &self.plan
    }

    /// Write the values into an engine: the parameters, then the values of
    /// the model kind.
    ///
    /// # Returns
    ///
    /// How far the state is from feasible: `0.0` for a feasible state.
    ///
    /// # Errors
    ///
    /// Returns a message when the engine rejects a value.
    pub(crate) fn apply(
        &self,
        engine: &mut E,
        parameters: &[CalibrationParameter],
        values: &[f64],
    ) -> Result<f64, String> {
        for (param, value) in parameters.iter().zip(values) {
            if param.parameter_type == CalibrationParameterType::Parameter {
                engine
                    .set_parameter(&param.id, *value)
                    .map_err(|e| format!("Failed to set parameter '{}': {}", param.id, e))?;
            }
        }
        let kind_values: Vec<f64> = self.kind_indices.iter().map(|&i| values[i]).collect();
        engine.apply_values(&self.plan, &kind_values)
    }
}

/// Clamp parameter values to their defined bounds.
pub(crate) fn clamp_to_bounds(parameters: &[CalibrationParameter], values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .zip(parameters)
        .map(|(value, param)| value.clamp(param.min_bound, param.max_bound))
        .collect()
}

/// Build an engine clone with calibration values applied exactly as in
/// objective evaluation.
///
/// # Returns
///
/// The configured engine and the values it applied (clamped to their bounds).
///
/// # Errors
///
/// Returns a message when the vector length is wrong, the engine rejects a
/// value, or the values make the state infeasible.
pub(crate) fn configure_engine<E: SimulationEngine>(
    base_engine: &E,
    parameters: &[CalibrationParameter],
    writer: &ValueWriter<E>,
    values: &[f64],
) -> Result<(E, Vec<f64>), String> {
    if values.len() != parameters.len() {
        return Err(format!(
            "Expected {} parameters, got {}",
            parameters.len(),
            values.len()
        ));
    }
    let clamped = clamp_to_bounds(parameters, values);
    let mut engine = base_engine.clone();
    engine.reset();
    let excess = writer.apply(&mut engine, parameters, &clamped)?;
    if excess > 0.0 {
        return Err(writer.plan().infeasibility(excess));
    }
    Ok((engine, clamped))
}

/// Check that every factor has positive bounds and the engine sets every
/// parameter and resolves every value of its model kind and the remainder,
/// and return the plan it resolves them to.
///
/// The engine sets and resolves them as the calibration objective does, so
/// the first check that sees an id gives the engine's message, for example
/// that the model has no such parameter or that a category needs a binary
/// stratification.
///
/// # Errors
///
/// Returns a message that names a factor with a bound at or below 0, or the
/// engine's message when it does not set a parameter, a value or the
/// remainder does not resolve, or the kind does not accept a type of value.
pub fn check_model_values<E: SimulationEngine>(
    engine: &E,
    parameters: &[CalibrationParameter],
    remainder: Option<&str>,
) -> Result<E::ValuePlan, String> {
    ValueWriter::new(engine, parameters, remainder).map(|writer| writer.plan)
}

/// An engine with calibration values applied exactly as the calibration
/// objective scores them, the values it applied (clamped to their bounds),
/// and the plan of the values of its model kind.
///
/// This is the authority for writing calibrated values back to a model: the
/// engine's state is the state the objective scored.
///
/// # Errors
///
/// Returns a message when a value of the model kind or the remainder does not
/// resolve, the vector length is wrong, the engine rejects a value, or the
/// values make the state infeasible.
pub fn configured_engine<E: SimulationEngine>(
    base_engine: &E,
    parameters: &[CalibrationParameter],
    remainder: Option<&str>,
    values: &[f64],
) -> Result<(E, Vec<f64>, E::ValuePlan), String> {
    let writer = ValueWriter::new(base_engine, parameters, remainder)?;
    let (engine, applied) = configure_engine(base_engine, parameters, &writer, values)?;
    Ok((engine, applied, writer.plan))
}
