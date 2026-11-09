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

/// The values of the model kind among the parameters: the index of each
/// one among the parameters, and its type and id, in problem order.
fn model_values(parameters: &[CalibrationParameter]) -> (Vec<usize>, Vec<ModelValue>) {
    parameters
        .iter()
        .enumerate()
        .filter_map(|(index, parameter)| {
            model_value_type(parameter.parameter_type).map(|value_type| {
                (
                    index,
                    ModelValue {
                        value_type,
                        id: parameter.id.clone(),
                    },
                )
            })
        })
        .unzip()
}

/// Where the numbers that a method writes for its values come from.
#[derive(Clone, Copy, Debug)]
pub enum ValueSource<'a> {
    /// A method varies them, such as an optimizer or a sampler: any number
    /// within the bounds of each value.
    Varied,
    /// A method writes numbers it knows, such as the draws of a sample:
    /// every set of numbers it writes, each with one number per parameter
    /// in problem order.
    Known(&'a [Vec<f64>]),
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
    /// The writer of a method that writes numbers it knows: check that every
    /// factor has positive bounds and the engine sets every parameter, and
    /// ask it to resolve the values of its model kind. The engine checks
    /// each number when it writes it ([`Self::apply`]).
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
    pub(crate) fn known(
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
        let (kind_indices, values) = model_values(parameters);
        let plan = engine.plan_values(&values, remainder)?;
        Ok(Self { kind_indices, plan })
    }

    /// The writer of a method that varies the values, such as an optimizer
    /// or a sampler: [`Self::known`], then the engine checks that it takes
    /// every number such a method can write
    /// ([`SimulationEngine::check_varied_values`]), whatever the bounds.
    ///
    /// # Errors
    ///
    /// Returns the message of [`Self::known`], or the engine's message when it
    /// takes only some numbers of a value, such as an engine of whole units
    /// with no rounding and an initial condition.
    pub(crate) fn varied(
        engine: &E,
        parameters: &[CalibrationParameter],
        remainder: Option<&str>,
    ) -> Result<Self, String> {
        let writer = Self::known(engine, parameters, remainder)?;
        engine.check_varied_values(&writer.plan)?;
        Ok(writer)
    }

    /// The writer of `source` ([`Self::varied`] or [`Self::known`]), once
    /// the numbers of `source` are written to a copy of the engine and the
    /// model then has every value a run needs
    /// ([`SimulationEngine::check_values`]). A varied source writes, for each
    /// value, the point of its bounds nearest 0. A known source writes each
    /// of its sets of numbers, so the engine checks every number the method
    /// writes before any run.
    ///
    /// One copy takes every set: each set writes the same values, and
    /// [`SimulationEngine::reset`] keeps the values last written.
    ///
    /// # Errors
    ///
    /// Returns the message of the constructor of `source`, a message when a
    /// known source holds no parameter set or a set without one number per
    /// parameter, the engine's message when it rejects a number (with the
    /// index of its set for a known source), or the message of
    /// [`SimulationEngine::check_values`].
    pub(crate) fn checked(
        engine: &E,
        parameters: &[CalibrationParameter],
        remainder: Option<&str>,
        source: ValueSource<'_>,
    ) -> Result<Self, String> {
        let mut written = engine.clone();
        let mut write = |writer: &Self, numbers: &[f64]| {
            written.reset();
            writer.apply(&mut written, parameters, numbers)?;
            written.check_values()
        };
        match source {
            ValueSource::Varied => {
                let writer = Self::varied(engine, parameters, remainder)?;
                let nearest_zero: Vec<f64> = parameters
                    .iter()
                    .map(|parameter| 0.0_f64.max(parameter.min_bound).min(parameter.max_bound))
                    .collect();
                write(&writer, &nearest_zero)?;
                Ok(writer)
            }
            ValueSource::Known(sets) => {
                let writer = Self::known(engine, parameters, remainder)?;
                if sets.is_empty() {
                    return Err("There is no parameter set to write: give at least one".to_string());
                }
                for (index, numbers) in sets.iter().enumerate() {
                    if numbers.len() != parameters.len() {
                        return Err(format!(
                            "Parameter set {index} needs one number per parameter ({}), got {}",
                            parameters.len(),
                            numbers.len()
                        ));
                    }
                    write(&writer, numbers)
                        .map_err(|error| format!("Parameter set {index}: {error}"))?;
                }
                Ok(writer)
            }
        }
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
/// and return the plan it resolves them to: [`check_model_values`] without
/// its checks that the engine takes every number of a varied value and that
/// no value is missing, for a query of some of the values, such as the
/// compartments each initial condition sets. It writes no number.
///
/// # Errors
///
/// Returns a message that names a factor with a bound at or below 0, or the
/// engine's message when it does not set a parameter, a value or the
/// remainder does not resolve, or the kind does not accept a type of value.
pub fn plan_model_values<E: SimulationEngine>(
    engine: &E,
    parameters: &[CalibrationParameter],
    remainder: Option<&str>,
) -> Result<E::ValuePlan, String> {
    ValueWriter::known(engine, parameters, remainder).map(|writer| writer.plan)
}

/// Check that every factor has positive bounds, the engine sets every
/// parameter and resolves every value of its model kind and the remainder,
/// and the model has every value a run needs once they are written, and
/// return the plan it resolves them to.
///
/// The engine sets and resolves them as the calibration objective does, so
/// the first check that sees an id gives the engine's message, for example
/// that the model has no such parameter or that a category needs a binary
/// stratification. The values are written to a copy of the engine, each at
/// the point of its bounds nearest 0, and the copy checks that nothing is
/// left without a value ([`SimulationEngine::check_values`]), so a
/// calibration that leaves a parameter or an initial amount without a value
/// is refused before it starts.
///
/// # Errors
///
/// Returns a message that names a factor with a bound at or below 0, or the
/// engine's message when it does not set a parameter, a value or the
/// remainder does not resolve, the kind does not accept a type of value, the
/// engine does not take every number of a value
/// ([`SimulationEngine::check_varied_values`]), or a value the calibration
/// does not write is missing.
pub fn check_model_values<E: SimulationEngine>(
    engine: &E,
    parameters: &[CalibrationParameter],
    remainder: Option<&str>,
) -> Result<E::ValuePlan, String> {
    ValueWriter::checked(engine, parameters, remainder, ValueSource::Varied)
        .map(|writer| writer.plan)
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
    let writer = ValueWriter::known(base_engine, parameters, remainder)?;
    let (engine, applied) = configure_engine(base_engine, parameters, &writer, values)?;
    Ok((engine, applied, writer.plan))
}
