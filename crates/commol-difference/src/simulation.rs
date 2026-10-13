//! The difference equations engine: one step adds the right-hand side once.

use commol_core::OutputLabels;
use commol_model::CompiledModel;
use std::collections::HashMap;

/// Difference equations simulation engine.
///
/// It wraps a [`CompiledModel`] and steps it in discrete time: each step adds
/// the flows of every transition, computed from one snapshot of the state,
/// so `X(n + 1) = X(n) + derivatives(n, X(n))`.
#[derive(Clone)]
pub struct DifferenceEquations {
    model: CompiledModel,
    /// Current state: compartments, then accumulators.
    state: Vec<f64>,
    /// Reusable buffer for the right-hand side to avoid allocations.
    rates: Vec<f64>,
    current_step: f64,
    /// Whether the impulses at `current_step` are already in `state`.
    settled: bool,
}

impl DifferenceEquations {
    /// Create a new DifferenceEquations instance from a model.
    ///
    /// # Panics
    ///
    /// Panics when [`Self::try_from_model`] returns an error.
    pub fn from_model(model: &commol_core::Model) -> Self {
        Self::try_from_model(model)
            .unwrap_or_else(|error| panic!("Failed to build population: {error}"))
    }

    /// Fallible counterpart of [`Self::from_model`].
    ///
    /// # Errors
    ///
    /// Returns a message when the model is not a difference-equations model,
    /// or when [`CompiledModel::try_from_model`] fails.
    pub fn try_from_model(model: &commol_core::Model) -> Result<Self, String> {
        if model.dynamics.typology != commol_core::ModelTypes::DifferenceEquations {
            return Err(format!(
                "The DifferenceEquations engine cannot solve a {:?} model",
                model.dynamics.typology
            ));
        }
        Self::from_compiled(CompiledModel::try_from_model(model)?)
    }

    /// Step a compiled model in discrete time, whatever its typology.
    ///
    /// # Errors
    ///
    /// Returns a message that names the impulse transition when one of its
    /// positions is not an integer, because the engine only steps at
    /// integers.
    pub fn from_compiled(model: CompiledModel) -> Result<Self, String> {
        if let Some((transition, position)) = model.first_non_integer_impulse() {
            return Err(format!(
                "Impulse transition '{transition}' acts at {position}, which is not an \
                 integer step; a difference-equations model steps only at integers"
            ));
        }
        let state = model.initial_state().to_vec();
        let rates = vec![0.0; state.len()];
        Ok(Self {
            model,
            state,
            rates,
            current_step: 0.0,
            settled: false,
        })
    }

    /// Apply the impulses at the current step, once.
    ///
    /// At step `n` the state first jumps, then the flows of the step run from
    /// the new state, so the output at `n` shows the state after the jump.
    fn settle(&mut self) -> Result<(), String> {
        if !self.settled {
            self.model
                .apply_impulses(self.current_step, &mut self.state)?;
            self.settled = true;
        }
        Ok(())
    }

    /// The compiled model the engine steps.
    pub fn model(&self) -> &CompiledModel {
        &self.model
    }

    /// Get the current population vector.
    pub fn population(&self) -> Vec<f64> {
        self.state[..self.model.num_compartments()].to_vec()
    }

    /// Get the list of compartment names.
    pub fn compartments(&self) -> Vec<String> {
        self.model.compartments().to_vec()
    }

    /// Get all simulation output names: state compartments followed by accumulators.
    pub fn output_names(&self) -> Vec<String> {
        self.model.output_names().to_vec()
    }

    fn output_len(&self) -> usize {
        self.state.len()
    }

    fn output_row(&self) -> Vec<f64> {
        self.state.clone()
    }

    fn copy_output_row(&self, row: &mut Vec<f64>) {
        let output_len = self.output_len();
        if row.len() != output_len {
            row.resize(output_len, 0.0);
        }
        row.copy_from_slice(&self.state);
    }

    /// Execute a single simulation step.
    ///
    /// The impulses of the current step act first. Then every flow is
    /// computed from the state, added to it, and the step counter advances by
    /// one.
    ///
    /// # Errors
    ///
    /// Returns an error message if rate evaluation fails.
    pub fn step(&mut self) -> Result<(), String> {
        self.settle()?;
        let step = self.current_step;
        self.model
            .derivatives(step, step, &self.state, &mut self.rates)?;
        for (value, rate) in self.state.iter_mut().zip(&self.rates) {
            *value += rate;
        }
        self.current_step += 1.0;
        self.settled = false;
        Ok(())
    }

    /// Run the simulation for a specified number of steps.
    ///
    /// # Arguments
    ///
    /// * `num_steps` - Number of time steps to simulate
    ///
    /// # Returns
    ///
    /// A vector of population states, where the first element is the initial state (t=0)
    /// and subsequent elements are states at t=1, t=2, ..., t=num_steps.
    pub fn run(&mut self, num_steps: u32) -> Result<Vec<Vec<f64>>, String> {
        // Pre-allocate memory for efficiency
        let mut steps = Vec::with_capacity(num_steps as usize + 1);

        // Store initial state (t=0)
        self.settle()?;
        steps.push(self.output_row());

        for _ in 0..num_steps {
            self.step()?;
            self.settle()?;
            steps.push(self.output_row());
        }

        Ok(steps)
    }

    /// Optimized version that writes simulation results into a pre-allocated buffer.
    ///
    /// This method is more memory-efficient than `run()` when the caller can
    /// provide a reusable buffer.
    ///
    /// # Arguments
    ///
    /// * `num_steps` - Number of time steps to simulate
    /// * `buffer` - Pre-allocated buffer to store results
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error message if simulation fails.
    pub fn run_into_buffer(
        &mut self,
        num_steps: u32,
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        let total_steps = (num_steps + 1) as usize;
        let output_len = self.output_len();

        // Keep existing row allocations alive across repeated calls. This is
        // especially important for calibration, where the same buffer is reused
        // for hundreds or thousands of full simulations.
        if buffer.len() < total_steps {
            buffer.reserve(total_steps - buffer.len());
            while buffer.len() < total_steps {
                buffer.push(vec![0.0; output_len]);
            }
        } else if buffer.len() > total_steps {
            buffer.truncate(total_steps);
        }

        for row in buffer.iter_mut() {
            if row.len() != output_len {
                row.resize(output_len, 0.0);
            }
        }

        // Store initial state (t=0)
        self.settle()?;
        self.copy_output_row(&mut buffer[0]);

        for row in buffer.iter_mut().take(total_steps).skip(1) {
            self.step()?;
            self.settle()?;
            self.copy_output_row(row);
        }

        Ok(())
    }

    /// Run through the largest requested step while recording only selected
    /// steps. `time_steps` must be sorted and deduplicated by the caller.
    pub fn run_at_steps_into_buffer(
        &mut self,
        time_steps: &[u32],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        let output_len = self.output_len();

        if buffer.len() < time_steps.len() {
            buffer.reserve(time_steps.len() - buffer.len());
            while buffer.len() < time_steps.len() {
                buffer.push(vec![0.0; output_len]);
            }
        } else if buffer.len() > time_steps.len() {
            buffer.truncate(time_steps.len());
        }

        for row in buffer.iter_mut() {
            if row.len() != output_len {
                row.resize(output_len, 0.0);
            }
        }

        let mut next_record_idx = 0;
        self.settle()?;
        while next_record_idx < time_steps.len() && time_steps[next_record_idx] == 0 {
            self.copy_output_row(&mut buffer[next_record_idx]);
            next_record_idx += 1;
        }

        let Some(&max_step) = time_steps.last() else {
            return Ok(());
        };

        for step in 1..=max_step {
            self.step()?;
            self.settle()?;
            while next_record_idx < time_steps.len() && time_steps[next_record_idx] == step {
                self.copy_output_row(&mut buffer[next_record_idx]);
                next_record_idx += 1;
            }
        }

        Ok(())
    }
}

/// Implementation of the SimulationEngine trait.
impl commol_core::SimulationEngine for DifferenceEquations {
    fn run(&mut self, num_steps: u32) -> Result<Vec<Vec<f64>>, String> {
        DifferenceEquations::run(self, num_steps)
    }

    fn step(&mut self) -> Result<(), String> {
        DifferenceEquations::step(self)
    }

    fn compartments(&self) -> Vec<String> {
        DifferenceEquations::compartments(self)
    }

    fn output_labels(&self) -> Vec<OutputLabels> {
        self.model.output_labels().to_vec()
    }

    fn output_names(&self) -> Vec<String> {
        DifferenceEquations::output_names(self)
    }

    fn population(&self) -> Vec<f64> {
        DifferenceEquations::population(self)
    }

    fn reset(&mut self) {
        self.state.copy_from_slice(self.model.initial_state());
        self.current_step = 0.0;
        self.settled = false;
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.model.set_parameter(parameter_id, value);
        Ok(())
    }

    fn get_parameters(&self) -> &HashMap<String, f64> {
        self.model.parameters()
    }

    fn current_step(&self) -> f64 {
        self.current_step
    }

    fn run_into_buffer(
        &mut self,
        num_steps: u32,
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        DifferenceEquations::run_into_buffer(self, num_steps, buffer)
    }

    fn run_at_steps_into_buffer(
        &mut self,
        time_steps: &[u32],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        DifferenceEquations::run_at_steps_into_buffer(self, time_steps, buffer)
    }

    fn set_initial_condition(
        &mut self,
        compartment_index: usize,
        value: f64,
    ) -> Result<(), String> {
        self.model.set_initial_condition(compartment_index, value)?;
        // Also update current population to reflect the change
        self.state[compartment_index] = value;
        if self.current_step == 0.0 && self.settled {
            // The impulses at step 0 must act on the new initial state.
            self.state.copy_from_slice(self.model.initial_state());
            self.settled = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use commol_core::{
        Bin, BinFraction, Dynamics, InitialConditions, Model, ModelTypes, Population,
        RateMathExpression, Transition,
    };

    /// One difference step is exactly `state + derivatives(n, state)`.
    fn assert_step_adds_derivatives(model: &Model, steps: u32) {
        let mut engine = DifferenceEquations::from_model(model);
        let mut compiled = CompiledModel::try_from_model(model).unwrap();
        for n in 0..steps {
            let before = engine.state.clone();
            let mut rates = vec![0.0; before.len()];
            compiled
                .derivatives(n as f64, n as f64, &before, &mut rates)
                .unwrap();
            engine.step().unwrap();
            let expected: Vec<f64> = before.iter().zip(&rates).map(|(x, d)| x + d).collect();
            assert_eq!(engine.state, expected, "{} at step {n}", model.name);
        }
    }

    #[test]
    fn one_step_adds_the_derivatives_exactly() {
        assert_step_adds_derivatives(&make_periodic_ab_model(), 9);
        assert_step_adds_derivatives(&make_three_bin_model(), 20);
        assert_step_adds_derivatives(&ab_model_with_rate("if(step >= 0, e * k2 * A, 0)"), 5);
        assert_step_adds_derivatives(&ab_model_with_rate("k1 * A * B / N + t * 0.001"), 5);
    }

    fn make_periodic_ab_model() -> Model {
        Model {
            name: "AB_periodic".to_string(),
            description: None,
            version: None,
            parameters: vec![],
            population: Population {
                bins: vec![
                    Bin {
                        id: "A".to_string(),
                        name: "Source".to_string(),
                    },
                    Bin {
                        id: "B".to_string(),
                        name: "Sink".to_string(),
                    },
                ],
                accumulators: vec![],
                stratifications: vec![],
                transitions: vec![],
                initial_conditions: InitialConditions {
                    population_size: 1000,
                    bin_fractions: vec![
                        BinFraction {
                            bin: "A".to_string(),
                            fraction: Some(1.0),
                        },
                        BinFraction {
                            bin: "B".to_string(),
                            fraction: Some(0.0),
                        },
                    ],
                    stratification_fractions: vec![],
                    compartment_fractions: vec![],
                },
            },
            dynamics: Dynamics {
                typology: ModelTypes::DifferenceEquations,
                transitions: vec![Transition {
                    id: "flow".to_string(),
                    source: vec!["A".to_string()],
                    target: vec!["B".to_string()],
                    accumulators: vec![],
                    rate: Some(RateMathExpression::from_string(
                        "if(step - floor(step / 7) * 7 == 0, 0.1, 0)".to_string(),
                    )),
                    stratified_rates: None,
                    per_compartment: None,
                    impulse_at: None,
                    impulse_every: None,
                }],
                solver: None,
            },
        }
    }

    fn make_missing_parameter_model() -> Model {
        Model {
            name: "AB_missing_parameter".to_string(),
            description: None,
            version: None,
            parameters: vec![],
            population: Population {
                bins: vec![
                    Bin {
                        id: "A".to_string(),
                        name: "Source".to_string(),
                    },
                    Bin {
                        id: "B".to_string(),
                        name: "Sink".to_string(),
                    },
                ],
                accumulators: vec![],
                stratifications: vec![],
                transitions: vec![],
                initial_conditions: InitialConditions {
                    population_size: 1000,
                    bin_fractions: vec![
                        BinFraction {
                            bin: "A".to_string(),
                            fraction: Some(1.0),
                        },
                        BinFraction {
                            bin: "B".to_string(),
                            fraction: Some(0.0),
                        },
                    ],
                    stratification_fractions: vec![],
                    compartment_fractions: vec![],
                },
            },
            dynamics: Dynamics {
                typology: ModelTypes::DifferenceEquations,
                transitions: vec![Transition {
                    id: "flow".to_string(),
                    source: vec!["A".to_string()],
                    target: vec!["B".to_string()],
                    accumulators: vec![],
                    rate: Some(RateMathExpression::from_string("missing * A".to_string())),
                    stratified_rates: None,
                    per_compartment: None,
                    impulse_at: None,
                    impulse_every: None,
                }],
                solver: None,
            },
        }
    }

    #[test]
    fn test_periodic_rate_fires_at_exact_steps() {
        // A→B model where rate fires at step 0 and step 7 (period=7).
        // Verifies the floor-based modulo formula compiles and evaluates
        // correctly end-to-end through the DifferenceEquations engine.
        let model = make_periodic_ab_model();
        let mut engine = DifferenceEquations::from_model(&model);
        let results = engine.run(14).unwrap();

        // results[k] = state after k step() calls; step counter k-1 was used for k>=1.
        // Compartment order: A=index 0, B=index 1.
        let b: Vec<f64> = results.iter().map(|s| s[1]).collect();
        let a: Vec<f64> = results.iter().map(|s| s[0]).collect();

        // Step 0 fires: B increases from initial zero
        assert!(b[1] > b[0], "B should increase after step 0");

        // Steps 1–6 do not fire
        for k in 2..=7 {
            assert!(
                (b[k] - b[k - 1]).abs() < 1e-10,
                "B should not change at step {}, delta = {}",
                k - 1,
                (b[k] - b[k - 1]).abs()
            );
        }

        // Step 7 fires: B increases again
        assert!(b[8] > b[7], "B should increase after step 7");

        // Steps 8–13 do not fire
        for k in 9..=14 {
            assert!(
                (b[k] - b[k - 1]).abs() < 1e-10,
                "B should not change at step {}, delta = {}",
                k - 1,
                (b[k] - b[k - 1]).abs()
            );
        }

        // Population conserved at every recorded state
        for k in 0..=14 {
            assert!(
                (a[k] + b[k] - 1000.0).abs() < 1e-8,
                "Population not conserved at step {}: A={}, B={}, total={}",
                k,
                a[k],
                b[k],
                a[k] + b[k]
            );
        }
    }

    #[test]
    fn jit_fast_path_reports_missing_parameters() {
        let model = make_missing_parameter_model();
        let mut engine = DifferenceEquations::from_model(&model);
        let error = engine
            .step()
            .expect_err("missing parameter must not be treated as zero");

        assert!(error.contains("Variable 'missing' not found"));
    }

    fn make_three_bin_model() -> Model {
        use commol_core::{Parameter, ParameterValue};
        Model {
            name: "abc_three_bin".to_string(),
            description: None,
            version: None,
            parameters: vec![
                Parameter {
                    id: "k1".to_string(),
                    value: Some(ParameterValue::Constant(0.3)),
                    description: None,
                },
                Parameter {
                    id: "k2".to_string(),
                    value: Some(ParameterValue::Constant(0.1)),
                    description: None,
                },
            ],
            population: Population {
                bins: vec![
                    Bin {
                        id: "A".to_string(),
                        name: "Bin A".to_string(),
                    },
                    Bin {
                        id: "B".to_string(),
                        name: "Bin B".to_string(),
                    },
                    Bin {
                        id: "C".to_string(),
                        name: "Bin C".to_string(),
                    },
                ],
                accumulators: vec![],
                stratifications: vec![],
                transitions: vec![],
                initial_conditions: InitialConditions {
                    population_size: 1000,
                    bin_fractions: vec![
                        BinFraction {
                            bin: "A".to_string(),
                            fraction: Some(0.99),
                        },
                        BinFraction {
                            bin: "B".to_string(),
                            fraction: Some(0.01),
                        },
                        BinFraction {
                            bin: "C".to_string(),
                            fraction: Some(0.0),
                        },
                    ],
                    stratification_fractions: vec![],
                    compartment_fractions: vec![],
                },
            },
            dynamics: Dynamics {
                typology: ModelTypes::DifferenceEquations,
                transitions: vec![
                    Transition {
                        id: "a_to_b".to_string(),
                        source: vec!["A".to_string()],
                        target: vec!["B".to_string()],
                        accumulators: vec![],
                        rate: Some(RateMathExpression::from_string(
                            "k1 * A * B / N".to_string(),
                        )),
                        stratified_rates: None,
                        per_compartment: None,
                        impulse_at: None,
                        impulse_every: None,
                    },
                    Transition {
                        id: "b_to_c".to_string(),
                        source: vec!["B".to_string()],
                        target: vec!["C".to_string()],
                        accumulators: vec![],
                        rate: Some(RateMathExpression::from_string("k2".to_string())),
                        stratified_rates: None,
                        per_compartment: None,
                        impulse_at: None,
                        impulse_every: None,
                    },
                ],
                solver: None,
            },
        }
    }

    #[test]
    fn get_parameters_exposes_n_after_step_without_compartment_context() {
        use commol_core::SimulationEngine;
        let model = make_three_bin_model();
        let mut engine = DifferenceEquations::from_model(&model);

        engine.step().unwrap();

        let params = engine.get_parameters();
        let n = params
            .get("N")
            .copied()
            .expect("N must be visible in get_parameters() after a step");
        // The model conserves total population at 1000 across the first step.
        assert!((n - 1000.0).abs() < 1e-6, "expected N=1000, got {}", n);
    }

    #[test]
    fn three_bin_model_conserves_population_through_fast_path() {
        let model = make_three_bin_model();
        let mut engine = DifferenceEquations::from_model(&model);
        let results = engine.run(50).unwrap();

        for (k, state) in results.iter().enumerate() {
            let total: f64 = state.iter().sum();
            assert!(
                (total - 1000.0).abs() < 1e-6,
                "population not conserved at step {}: total = {}",
                k,
                total
            );
        }
    }

    fn ab_model_with_rate(rate: &str) -> Model {
        let mut model = make_three_bin_model();
        model.dynamics.transitions = vec![Transition {
            id: "a_to_b".to_string(),
            source: vec!["A".to_string()],
            target: vec!["B".to_string()],
            accumulators: vec![],
            rate: Some(RateMathExpression::from_string(rate.to_string())),
            stratified_rates: None,
            per_compartment: None,
            impulse_at: None,
            impulse_every: None,
        }];
        model
    }

    #[test]
    fn pi_and_e_are_constants_in_every_rate_path() {
        // k2 = 0.1, A starts at 990: one step moves `constant * 0.1 * 990`.
        let cases = [
            ("pi * k2 * A", std::f64::consts::PI),
            ("e * k2 * A", std::f64::consts::E),
            ("k2 * pi * A + 0 * B", std::f64::consts::PI),
            ("if(step >= 0, e * k2 * A, 0)", std::f64::consts::E),
        ];
        for (rate, constant) in cases {
            let model = ab_model_with_rate(rate);
            let mut engine = DifferenceEquations::from_model(&model);
            let results = engine
                .run(1)
                .unwrap_or_else(|error| panic!("{rate}: {error}"));
            let expected = 990.0 - constant * 0.1 * 990.0;
            assert!(
                (results[1][0] - expected).abs() < 1e-9,
                "{rate}: expected A = {expected}, got {}",
                results[1][0]
            );
        }
    }

    #[test]
    fn pi_alone_is_a_per_capita_constant_rate() {
        let model = ab_model_with_rate("pi / 10");
        let mut engine = DifferenceEquations::from_model(&model);
        let results = engine.run(1).unwrap();
        let expected = 990.0 - std::f64::consts::PI / 10.0 * 990.0;
        assert!((results[1][0] - expected).abs() < 1e-9);
    }
}
