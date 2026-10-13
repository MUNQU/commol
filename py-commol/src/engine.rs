//! One engine type for the bindings that accept either engine.

use std::collections::HashMap;

use commol_core::{OutputLabels, SimulationEngine};
use commol_difference::DifferenceEquations;
use commol_differential::DifferentialEquations;
use pyo3::prelude::*;

use crate::difference::PyDifferenceEquations;
use crate::differential::PyDifferentialEquations;

/// The difference or the differential engine.
///
/// Calibration and predictions take it, so each binding exists once for both
/// engines.
#[derive(Clone)]
pub enum Engine {
    /// Discrete time.
    Difference(DifferenceEquations),
    /// Continuous time.
    Differential(DifferentialEquations),
}

/// Call `$body` with `$engine` bound to the engine inside an [`Engine`].
macro_rules! dispatch {
    ($self:expr, $engine:ident => $body:expr) => {
        match $self {
            Engine::Difference($engine) => $body,
            Engine::Differential($engine) => $body,
        }
    };
}

impl SimulationEngine for Engine {
    fn check_position(&self, position: f64) -> Result<(), String> {
        dispatch!(self, engine => engine.check_position(position))
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        dispatch!(self, engine => engine.run_at_positions_into_buffer(positions, buffer))
    }

    fn step(&mut self) -> Result<(), String> {
        dispatch!(self, engine => SimulationEngine::step(engine))
    }

    fn compartments(&self) -> Vec<String> {
        dispatch!(self, engine => SimulationEngine::compartments(engine))
    }

    fn output_labels(&self) -> Vec<OutputLabels> {
        dispatch!(self, engine => engine.output_labels())
    }

    fn output_names(&self) -> Vec<String> {
        dispatch!(self, engine => SimulationEngine::output_names(engine))
    }

    fn population(&self) -> Vec<f64> {
        dispatch!(self, engine => SimulationEngine::population(engine))
    }

    fn reset(&mut self) {
        dispatch!(self, engine => engine.reset())
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        dispatch!(self, engine => engine.set_parameter(parameter_id, value))
    }

    fn get_parameters(&self) -> &HashMap<String, f64> {
        dispatch!(self, engine => engine.get_parameters())
    }

    fn current_position(&self) -> f64 {
        dispatch!(self, engine => engine.current_position())
    }

    fn set_initial_condition(
        &mut self,
        compartment_index: usize,
        value: f64,
    ) -> Result<(), String> {
        dispatch!(self, engine => engine.set_initial_condition(compartment_index, value))
    }
}

/// A Python argument that is either engine.
#[derive(FromPyObject)]
pub enum EngineArg<'py> {
    /// A `DifferenceEquations` object.
    Difference(PyRef<'py, PyDifferenceEquations>),
    /// A `DifferentialEquations` object.
    Differential(PyRef<'py, PyDifferentialEquations>),
}

impl EngineArg<'_> {
    /// A copy of the engine the argument holds.
    pub fn engine(&self) -> Engine {
        match self {
            Self::Difference(engine) => Engine::Difference(engine.inner().clone()),
            Self::Differential(engine) => Engine::Differential(engine.inner().clone()),
        }
    }
}

/// Define a Python engine class `$py` named `$name` around the Rust engine
/// `$rust`. Both engines expose the same methods, through
/// [`SimulationEngine`].
macro_rules! engine_class {
    ($py:ident, $name:literal, $rust:ty, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(name = $name)]
        pub struct $py {
            pub(crate) inner: $rust,
        }

        #[pymethods]
        impl $py {
            /// Compile a model into the engine
            ///
            /// Args:
            ///     model: The compartment model to simulate
            ///
            /// Raises:
            ///     RuntimeError: If the engine cannot run the model's typology,
            ///         or the model cannot be compiled (for example an
            ///         initial-condition override names a compartment that is
            ///         not part of the model expansion)
            #[new]
            fn new(model: &crate::core::PyModel) -> PyResult<Self> {
                Ok(Self {
                    inner: <$rust>::try_from_model(&model.inner)
                        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?,
                })
            }

            /// Run on the default grid of a number of steps
            ///
            /// Args:
            ///     num_steps: Number of steps to simulate
            ///
            /// Returns:
            ///     The positions of the grid, from the current one (included)
            ///     one unit apart, and one output row per position
            fn run(&mut self, num_steps: u32) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
                let trajectory = commol_core::SimulationEngine::run(&mut self.inner, num_steps)
                    .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
                Ok((trajectory.positions, trajectory.rows))
            }

            /// Run through the last of the given positions on the axis
            ///
            /// Args:
            ///     positions: Sorted, unique positions to report at, not before
            ///         the current one
            ///
            /// Returns:
            ///     One output row per position
            ///
            /// Raises:
            ///     ValueError: If a position is not valid for the engine (a
            ///         difference engine reports only at integer steps), or the
            ///         positions are not sorted and unique
            ///     RuntimeError: If the simulation fails
            fn run_at(&mut self, positions: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
                let engine = &mut self.inner;
                commol_core::check_positions(
                    &positions,
                    commol_core::SimulationEngine::current_position(engine),
                    |position| commol_core::SimulationEngine::check_position(engine, position),
                )
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                commol_core::SimulationEngine::run_at(engine, &positions)
                    .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
            }

            /// Check that the engine can report at a position on the axis
            ///
            /// Args:
            ///     position: A position on the axis
            ///
            /// Raises:
            ///     ValueError: If the engine's model type does not report there
            fn check_position(&self, position: f64) -> PyResult<()> {
                commol_core::SimulationEngine::check_position(&self.inner, position)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
            }

            /// Advance one step
            fn step(&mut self) -> PyResult<()> {
                commol_core::SimulationEngine::step(&mut self.inner)
                    .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
            }

            /// Current population of each compartment
            #[getter]
            fn population(&self) -> Vec<f64> {
                commol_core::SimulationEngine::population(&self.inner)
            }

            /// Names of the compartments (e.g., ["S", "I_young", "R_old"])
            #[getter]
            fn compartments(&self) -> Vec<String> {
                commol_core::SimulationEngine::compartments(&self.inner)
            }

            /// Names of the simulation outputs, including accumulators
            #[getter]
            fn output_names(&self) -> Vec<String> {
                commol_core::SimulationEngine::output_names(&self.inner)
            }

            /// Reset the simulation to initial conditions
            fn reset(&mut self) {
                commol_core::SimulationEngine::reset(&mut self.inner);
            }

            /// Set a parameter value
            ///
            /// Args:
            ///     parameter_id: The parameter identifier
            ///     value: The new parameter value
            fn set_parameter(&mut self, parameter_id: &str, value: f64) -> PyResult<()> {
                commol_core::SimulationEngine::set_parameter(&mut self.inner, parameter_id, value)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
            }

            /// All current parameter values, by name
            fn get_parameters(&self) -> std::collections::HashMap<String, f64> {
                commol_core::SimulationEngine::get_parameters(&self.inner).clone()
            }

            /// Current position on the axis
            fn current_position(&self) -> f64 {
                commol_core::SimulationEngine::current_position(&self.inner)
            }

            /// Set the initial condition for a specific compartment
            ///
            /// Args:
            ///     compartment_index: The index of the compartment (0-based)
            ///     value: The new initial population value
            ///
            /// Raises:
            ///     ValueError: If compartment_index is invalid or value is negative
            fn set_initial_condition(
                &mut self,
                compartment_index: usize,
                value: f64,
            ) -> PyResult<()> {
                commol_core::SimulationEngine::set_initial_condition(
                    &mut self.inner,
                    compartment_index,
                    value,
                )
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(compartments={}, position={})",
                    $name,
                    commol_core::SimulationEngine::compartments(&self.inner).len(),
                    commol_core::SimulationEngine::current_position(&self.inner)
                )
            }
        }

        impl $py {
            /// The Rust engine (for internal use)
            pub(crate) fn inner(&self) -> &$rust {
                &self.inner
            }
        }
    };
}

pub(crate) use engine_class;
