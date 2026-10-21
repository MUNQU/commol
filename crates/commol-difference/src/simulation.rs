//! The difference equations engine: one step adds the right-hand side once.

use commol_core::{ModelValue, SimulationError};
use commol_model::{
    CompartmentalEngine, CompiledModel, InitialConditionPlan, OutputLabels, output_sums,
};

/// How far below zero, relative to the total at the start of a step, an
/// amount may fall: the rounding of flows that empty a compartment exactly.
const AMOUNT_TOLERANCE: f64 = 1e-12;

/// The lowest amount a compartment may hold after a step that starts from
/// `amounts`.
fn lowest_amount(amounts: &[f64]) -> f64 {
    -AMOUNT_TOLERANCE * amounts.iter().sum::<f64>()
}

/// `value` to six significant digits, for a message.
fn rounded(value: f64) -> f64 {
    format!("{value:.5e}").parse().unwrap_or(value)
}

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
    /// Reusable buffer for the right-hand side to avoid allocations. Before
    /// the impulses of a step act, it holds the state they start from.
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
            .unwrap_or_else(|error| panic!("Failed to build the engine: {error}"))
    }

    /// Fallible counterpart of [`Self::from_model`].
    ///
    /// # Errors
    ///
    /// Returns a message when the model is not a difference-equations model,
    /// or when [`CompiledModel::try_from_model`] fails: among its checks
    /// ([`commol_model::check_model`]), an impulse or a series point that is
    /// not at an integer step.
    pub fn try_from_model(model: &commol_core::Model) -> Result<Self, String> {
        let engine_type = model.engine.engine_type();
        if engine_type != commol_core::EngineType::DifferenceEquations {
            return Err(format!(
                "The DifferenceEquations engine cannot solve a {} model",
                engine_type.name()
            ));
        }
        let model = CompiledModel::try_from_model(model)?;
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
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Infeasible`] when the impulses leave a
    /// compartment below zero, and [`SimulationError::Failed`] when a rate
    /// cannot be evaluated.
    fn settle(&mut self) -> Result<(), SimulationError> {
        if self.settled {
            return Ok(());
        }
        if self.model.has_impulses() {
            self.rates.copy_from_slice(&self.state);
            if self
                .model
                .apply_impulses(self.current_step, &mut self.state)?
            {
                self.check_impulses()?;
            }
        }
        self.settled = true;
        Ok(())
    }

    /// Check the state the impulses of the current step wrote, from the
    /// state in `rates` they started from.
    fn check_impulses(&mut self) -> Result<(), SimulationError> {
        let num_compartments = self.model.num_compartments();
        let before = &self.rates[..num_compartments];
        let lowest = lowest_amount(before);
        let Some(compartment) = (0..num_compartments).find(|&i| self.state[i] < lowest) else {
            return Ok(());
        };
        let step = self.current_step;
        let drains = self
            .model
            .impulse_outflows(step, &self.rates, compartment)?;
        Err(self.below_zero(
            format!("The impulses at step {step}"),
            compartment,
            self.rates[compartment],
            self.state[compartment],
            &drains,
        ))
    }

    /// The error for a compartment that `cause` leaves at `after`, from
    /// `before`, with the flows that drain it.
    fn below_zero(
        &self,
        cause: String,
        compartment: usize,
        before: f64,
        after: f64,
        drains: &[(String, f64)],
    ) -> SimulationError {
        let drains: Vec<String> = drains
            .iter()
            .map(|(transition, amount)| format!("'{transition}' {}", rounded(*amount)))
            .collect();
        SimulationError::Infeasible {
            position: self.current_step,
            message: format!(
                "{cause} leave compartment '{}' at {after}, below zero: it held {before}, \
                 and the flows out of it move {}. A difference step cannot move more \
                 than a compartment holds",
                self.model.compartments()[compartment],
                drains.join(", "),
                after = rounded(after),
                before = rounded(before),
            ),
        }
    }

    /// Current amount of each compartment, without the accumulators.
    pub fn state(&self) -> Vec<f64> {
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

    /// Copy the state into a row that `record_steps` sized to the outputs.
    fn copy_output_row(&self, row: &mut [f64]) {
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
    /// Returns [`SimulationError::Infeasible`] when the impulses or the flows
    /// leave a compartment below zero (by more than the rounding of a flow
    /// that empties it), and the state stays as it was before them. Returns
    /// [`SimulationError::Failed`] when a rate cannot be evaluated.
    pub fn step(&mut self) -> Result<(), SimulationError> {
        self.settle()?;
        let step = self.current_step;
        self.model
            .derivatives(step, step, &self.state, &mut self.rates)?;
        let num_compartments = self.model.num_compartments();
        let lowest = lowest_amount(&self.state[..num_compartments]);
        if let Some(compartment) =
            (0..num_compartments).find(|&i| self.state[i] + self.rates[i] < lowest)
        {
            let drains = self.model.outflows(step, step, &self.state, compartment)?;
            return Err(self.below_zero(
                format!("The flows of step {step}"),
                compartment,
                self.state[compartment],
                self.state[compartment] + self.rates[compartment],
                &drains,
            ));
        }
        for (value, rate) in self.state.iter_mut().zip(&self.rates) {
            *value += rate;
        }
        self.current_step += 1.0;
        self.settled = false;
        Ok(())
    }

    /// Step through the largest of `steps`, recording the outputs at each of
    /// them into `buffer`.
    ///
    /// `steps` are sorted, unique and not before the current step; the
    /// caller checks them ([`commol_core::check_positions`]).
    fn record_steps(
        &mut self,
        steps: &[u32],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        let output_len = self.output_len();

        // Keep existing row allocations alive across repeated calls. This is
        // especially important for calibration, where the same buffer is reused
        // for hundreds or thousands of full simulations.
        buffer.resize_with(steps.len(), Vec::new);
        for row in buffer.iter_mut() {
            if row.len() != output_len {
                row.resize(output_len, 0.0);
            }
        }

        self.settle()?;
        let mut next = 0;
        loop {
            while next < steps.len() && f64::from(steps[next]) == self.current_step {
                self.copy_output_row(&mut buffer[next]);
                next += 1;
            }
            if next == steps.len() {
                return Ok(());
            }
            self.step()?;
            self.settle()?;
        }
    }
}

/// Why a difference engine does not report at `position`, if it does not.
fn check_step(position: f64) -> Result<(), String> {
    if position >= 0.0 && position.fract() == 0.0 && position <= f64::from(u32::MAX) {
        Ok(())
    } else {
        Err(format!(
            "a difference-equations model reports only at integer steps; {position} is not one"
        ))
    }
}

/// Implementation of the SimulationEngine trait.
impl commol_core::SimulationEngine for DifferenceEquations {
    type ValuePlan = InitialConditionPlan;

    fn check_position(&self, position: f64) -> Result<(), String> {
        check_step(position)
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        commol_core::check_positions(positions, self.current_step, check_step)?;
        // `check_step` accepts only integers that fit in a `u32`.
        let steps: Vec<u32> = positions.iter().map(|&position| position as u32).collect();
        self.record_steps(&steps, buffer)
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        DifferenceEquations::step(self)
    }

    fn output_names(&self) -> Vec<String> {
        DifferenceEquations::output_names(self)
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

    fn current_position(&self) -> f64 {
        self.current_step
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

impl CompartmentalEngine for DifferenceEquations {
    fn compartments(&self) -> Vec<String> {
        DifferenceEquations::compartments(self)
    }

    fn output_labels(&self) -> Vec<OutputLabels> {
        self.model.output_labels().to_vec()
    }

    fn initial_state(&self) -> Vec<f64> {
        self.model.initial_state()[..self.model.num_compartments()].to_vec()
    }

    fn total(&self) -> f64 {
        self.model.total()
    }

    fn set_initial_condition(
        &mut self,
        compartment_index: usize,
        value: f64,
    ) -> Result<(), String> {
        self.model.set_initial_condition(compartment_index, value)?;
        // Also update the current state to reflect the change
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
        Aggregate, Bin, EngineType, InitialConditions, Model, ModelEngine, ModelStructure, Process,
        SimulationEngine, Transition,
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
        assert_step_adds_derivatives(&ab_model_with_rate("k1 * A * B / total + t * 0.001"), 5);
    }

    fn make_periodic_ab_model() -> Model {
        Model {
            name: "AB_periodic".to_string(),
            description: None,
            version: None,
            parameters: vec![],
            process: Process {
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
                transitions: vec![Transition {
                    id: "flow".to_string(),
                    source: vec!["A".to_string()],
                    target: vec!["B".to_string()],
                    accumulators: vec![],
                    rate: Some("if(step - floor(step / 7) * 7 == 0, 0.1, 0)".to_string()),
                    stratified_rates: None,
                    per_compartment: None,
                    impulse_at: None,
                    impulse_every: None,
                }],
            },
            structure: ModelStructure::Aggregate(Aggregate {
                variant: None,
                initial_conditions: InitialConditions {
                    bin_amounts: [
                        ("A".to_string(), Some(1000.0)),
                        ("B".to_string(), Some(0.0)),
                    ]
                    .into_iter()
                    .collect(),
                    stratification_fractions: Default::default(),
                    compartment_amounts: Default::default(),
                },
                neighbors: None,
            }),
            engine: ModelEngine::new(EngineType::DifferenceEquations),
        }
    }

    fn make_missing_parameter_model() -> Model {
        Model {
            name: "AB_missing_parameter".to_string(),
            description: None,
            version: None,
            parameters: vec![],
            process: Process {
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
                transitions: vec![Transition {
                    id: "flow".to_string(),
                    source: vec!["A".to_string()],
                    target: vec!["B".to_string()],
                    accumulators: vec![],
                    rate: Some("missing * A".to_string()),
                    stratified_rates: None,
                    per_compartment: None,
                    impulse_at: None,
                    impulse_every: None,
                }],
            },
            structure: ModelStructure::Aggregate(Aggregate {
                variant: None,
                initial_conditions: InitialConditions {
                    bin_amounts: [
                        ("A".to_string(), Some(1000.0)),
                        ("B".to_string(), Some(0.0)),
                    ]
                    .into_iter()
                    .collect(),
                    stratification_fractions: Default::default(),
                    compartment_amounts: Default::default(),
                },
                neighbors: None,
            }),
            engine: ModelEngine::new(EngineType::DifferenceEquations),
        }
    }

    #[test]
    fn test_periodic_rate_fires_at_exact_steps() {
        // A→B model where rate fires at step 0 and step 7 (period=7).
        // Verifies the floor-based modulo formula compiles and evaluates
        // correctly end-to-end through the DifferenceEquations engine.
        let model = make_periodic_ab_model();
        let mut engine = DifferenceEquations::from_model(&model);
        let results = engine.run(14).unwrap().rows;

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
                // coverage: the failure of a test, the message of its assert
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
                // coverage: the failure of a test, the message of its assert
                k - 1,
                (b[k] - b[k - 1]).abs()
            );
        }

        // Total kept at every recorded state
        for k in 0..=14 {
            assert!(
                (a[k] + b[k] - 1000.0).abs() < 1e-8,
                "Total not kept at step {}: A={}, B={}, total={}",
                k,
                // coverage: the failure of a test, the message of its assert
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
            .expect_err("missing parameter must not be treated as zero")
            .to_string();

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
            process: Process {
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
                transitions: vec![
                    Transition {
                        id: "a_to_b".to_string(),
                        source: vec!["A".to_string()],
                        target: vec!["B".to_string()],
                        accumulators: vec![],
                        rate: Some("k1 * A * B / total".to_string()),
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
                        rate: Some("k2".to_string()),
                        stratified_rates: None,
                        per_compartment: None,
                        impulse_at: None,
                        impulse_every: None,
                    },
                ],
            },
            structure: ModelStructure::Aggregate(Aggregate {
                variant: None,
                initial_conditions: InitialConditions {
                    bin_amounts: [
                        ("A".to_string(), Some(990.0)),
                        ("B".to_string(), Some(10.0)),
                        ("C".to_string(), Some(0.0)),
                    ]
                    .into_iter()
                    .collect(),
                    stratification_fractions: Default::default(),
                    compartment_amounts: Default::default(),
                },
                neighbors: None,
            }),
            engine: ModelEngine::new(EngineType::DifferenceEquations),
        }
    }

    #[test]
    fn three_bin_model_keeps_its_total_through_fast_path() {
        let model = make_three_bin_model();
        let mut engine = DifferenceEquations::from_model(&model);
        let results = engine.run(50).unwrap().rows;

        for (k, state) in results.iter().enumerate() {
            let total: f64 = state.iter().sum();
            assert!(
                (total - 1000.0).abs() < 1e-6,
                "total not kept at step {}: total = {}",
                k,
                total
            );
        }
    }

    fn ab_model_with_rate(rate: &str) -> Model {
        let mut model = make_three_bin_model();
        model.process.transitions = vec![Transition {
            id: "a_to_b".to_string(),
            source: vec!["A".to_string()],
            target: vec!["B".to_string()],
            accumulators: vec![],
            rate: Some(rate.to_string()),
            stratified_rates: None,
            per_compartment: None,
            impulse_at: None,
            impulse_every: None,
        }];
        model
    }

    /// A transition of the three-bin model, an impulse at `impulse_at` if
    /// there are positions.
    fn transition(
        id: &str,
        source: &str,
        target: &str,
        rate: &str,
        impulse_at: &[f64],
    ) -> Transition {
        Transition {
            id: id.to_string(),
            source: vec![source.to_string()],
            target: vec![target.to_string()],
            accumulators: vec![],
            rate: Some(rate.to_string()),
            stratified_rates: None,
            per_compartment: None,
            impulse_at: (!impulse_at.is_empty()).then(|| impulse_at.to_vec()),
            impulse_every: None,
        }
    }

    /// The three-bin model (`A` = 990, `B` = 10, `C` = 0) with `transitions`.
    fn three_bins_with(transitions: Vec<Transition>) -> Model {
        let mut model = make_three_bin_model();
        model.process.transitions = transitions;
        model
    }

    /// The position and the message of an infeasible run, or a panic.
    fn infeasible(error: SimulationError) -> (f64, String) {
        match error {
            SimulationError::Infeasible { position, message } => (position, message),
            // coverage: the failure of a test, a run that is not infeasible
            SimulationError::Failed(message) => panic!("expected an infeasible run: {message}"),
        }
    }

    #[test]
    fn flows_that_move_more_than_a_compartment_holds_stop_the_first_step() {
        let mut engine = DifferenceEquations::from_model(&three_bins_with(vec![
            transition("a_to_c", "A", "C", "0.5", &[]),
            transition("a_to_b", "A", "B", "0.7", &[]),
        ]));

        let (position, message) = infeasible(engine.run(3).unwrap_err());

        assert_eq!(position, 0.0);
        assert_eq!(
            message,
            "The flows of step 0 leave compartment 'A' at -198, below zero: \
             it held 990, and the flows out of it move 'a_to_b' 693, 'a_to_c' 495. A \
             difference step cannot move more than a compartment holds"
        );
        // The state stays as it was before the step.
        assert_eq!(engine.state(), vec![990.0, 10.0, 0.0]);
        assert_eq!(engine.current_step, 0.0);
    }

    #[test]
    fn a_rate_that_turns_negative_stops_the_step_where_an_amount_drops_below_zero() {
        // From step 7 the flow runs backwards and takes 2 × 990 from `B`.
        let mut engine = DifferenceEquations::from_model(&three_bins_with(vec![transition(
            "a_to_b",
            "A",
            "B",
            "if(step >= 7, -2, 0)",
            &[],
        )]));

        let (position, message) = infeasible(engine.run(20).unwrap_err());

        assert_eq!(position, 7.0);
        assert!(
            message.starts_with("The flows of step 7 leave compartment 'B' at -1970")
                && message.contains("it held 10, and the flows out of it move 'a_to_b' 1980."),
            "{message}"
        );
    }

    #[test]
    fn proportional_rates_that_sum_to_one_empty_a_compartment() {
        let mut engine = DifferenceEquations::from_model(&three_bins_with(vec![
            transition("a_to_b", "A", "B", "0.1", &[]),
            transition("a_to_c", "A", "C", "0.2", &[]),
            transition("a_to_b_again", "A", "B", "0.7", &[]),
        ]));

        let rows = engine.run(3).unwrap().rows;

        assert!(rows[1][0].abs() < 1e-12 * 1000.0, "{}", rows[1][0]);
        assert!(((rows[3][1] + rows[3][2]) - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn an_impulse_that_moves_more_than_its_source_holds_stops_its_step() {
        let mut engine = DifferenceEquations::from_model(&three_bins_with(vec![transition(
            "dose",
            "B",
            "C",
            "1.5",
            &[3.0],
        )]));

        let (position, message) = infeasible(engine.run(5).unwrap_err());

        assert_eq!(position, 3.0);
        assert_eq!(
            message,
            "The impulses at step 3 leave compartment 'B' at -5, below zero: it held 10, \
             and the flows out of it move 'dose' 15. A difference step cannot move more \
             than a compartment holds"
        );
    }

    #[test]
    fn an_accumulator_may_count_a_negative_net_flow() {
        let mut model = three_bins_with(vec![Transition {
            accumulators: vec!["moved".to_string()],
            ..transition("a_to_b", "A", "B", "-0.001", &[])
        }]);
        model.process.accumulators = vec![commol_core::Accumulator {
            id: "moved".to_string(),
            name: "Moved".to_string(),
        }];
        let mut engine = DifferenceEquations::from_model(&model);

        let rows = engine.run(2).unwrap().rows;

        assert!(rows[2][3] < 0.0, "{:?}", rows[2]);
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
                .unwrap_or_else(|error| panic!("{rate}: {error}"))
                .rows;
            let expected = 990.0 - constant * 0.1 * 990.0;
            assert!(
                (results[1][0] - expected).abs() < 1e-9,
                "{rate}: expected A = {expected}, got {}",
                // coverage: the failure of a test, the message of its assert
                results[1][0]
            );
        }
    }

    #[test]
    fn pi_alone_is_a_proportional_constant_rate() {
        let model = ab_model_with_rate("pi / 10");
        let mut engine = DifferenceEquations::from_model(&model);
        let results = engine.run(1).unwrap().rows;
        let expected = 990.0 - std::f64::consts::PI / 10.0 * 990.0;
        assert!((results[1][0] - expected).abs() < 1e-9);
    }

    #[test]
    fn a_run_reports_at_its_grid() {
        let mut engine = DifferenceEquations::from_model(&make_three_bin_model());
        let trajectory = engine.run(3).unwrap();
        assert_eq!(trajectory.positions, vec![0.0, 1.0, 2.0, 3.0]);

        let mut fresh = DifferenceEquations::from_model(&make_three_bin_model());
        let rows = fresh.run_at(&[1.0, 3.0]).unwrap();
        assert_eq!(
            rows,
            vec![trajectory.rows[1].clone(), trajectory.rows[3].clone()]
        );
    }

    #[test]
    fn a_run_continues_from_the_current_step() {
        let mut whole = DifferenceEquations::from_model(&make_three_bin_model());
        let expected = whole.run(5).unwrap();

        let mut engine = DifferenceEquations::from_model(&make_three_bin_model());
        engine.run(2).unwrap();
        let rest = engine.run(3).unwrap();
        assert_eq!(rest.positions, vec![2.0, 3.0, 4.0, 5.0]);
        assert_eq!(rest.rows, expected.rows[2..].to_vec());
    }

    #[test]
    fn integer_positions_are_accepted_and_others_rejected() {
        let engine = DifferenceEquations::from_model(&make_three_bin_model());
        assert_eq!(engine.check_position(0.0), Ok(()));
        assert_eq!(engine.check_position(12.0), Ok(()));
        for position in [2.5, -1.0, f64::NAN, f64::INFINITY] {
            let error = engine.check_position(position).unwrap_err();
            assert!(
                error.contains("only at integer steps") && error.contains(&position.to_string()),
                "{position}: {error}"
            );
        }
    }

    #[test]
    fn unsorted_or_past_positions_are_rejected() {
        let mut engine = DifferenceEquations::from_model(&make_three_bin_model());
        let error = engine.run_at(&[3.0, 1.0]).unwrap_err().to_string();
        assert!(error.contains("sorted and unique"), "{error}");
        let error = engine.run_at(&[2.5]).unwrap_err().to_string();
        assert!(error.contains("2.5 is not one"), "{error}");

        engine.run(4).unwrap();
        let error = engine.run_at(&[2.0]).unwrap_err().to_string();
        assert!(error.contains("before the current position 4"), "{error}");
    }

    #[test]
    fn the_engine_writes_initial_conditions_that_reset_keeps() {
        use commol_core::{ModelValue, ModelValueType};
        let mut engine = DifferenceEquations::from_model(&make_three_bin_model());
        let plan = engine
            .plan_values(
                &[ModelValue {
                    value_type: ModelValueType::InitialCondition,
                    id: "B".to_string(),
                }],
                Some("A"),
            )
            .unwrap();

        assert_eq!(engine.apply_values(&plan, &[250.0]).unwrap(), 0.0);
        engine.step().unwrap();
        engine.reset();

        assert_eq!(engine.state(), vec![750.0, 250.0, 0.0]);
        assert_eq!(engine.total(), 1000.0);
    }

    #[test]
    fn the_state_is_the_compartments_and_a_bin_total_is_a_sum() {
        let engine = DifferenceEquations::from_model(&make_three_bin_model());

        assert_eq!(engine.state_outputs(), 3);
        // Without stratifications no name sums several outputs.
        assert!(engine.output_sums().is_empty());
    }
}
