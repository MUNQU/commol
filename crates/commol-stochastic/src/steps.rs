//! The discrete-time stochastic engine: the chain binomial.

use commol_core::{EngineSettings, EngineType, ModelValue, SimulationError};
use commol_model::{
    CompartmentalEngine, CompiledModel, InitialConditionPlan, OutputLabels, output_sums,
};
use commol_random::sample;

use crate::units::{Units, rounded};

/// The discrete-time stochastic engine of an aggregate model: the chain
/// binomial.
///
/// Each step draws how many whole units each flow moves, from the state at
/// the start of the step. The units of a compartment leave by its
/// proportional flows in one multinomial draw, each rate being the
/// probability that one unit leaves by that flow in the step, so a unit
/// never leaves twice and an amount never falls below zero. An absolute flow
/// moves a Poisson number of units of mean its rate, capped by what its
/// sources still hold, in the order of the model; an inflow is not capped.
/// The mean of one step, while no cap acts, is the step of the difference
/// engine.
#[derive(Clone)]
pub struct StochasticSteps {
    units: Units,
    current_step: f64,
    /// Whether the impulses at `current_step` are already in the state.
    settled: bool,
    /// The source compartment of each group of proportional flows, and the
    /// flows of the group, in the order of the model.
    groups: Vec<(usize, Vec<usize>)>,
    /// The absolute flows, in the order of the model.
    absolute: Vec<usize>,
    /// Reused buffers: the rate and the number of events of each flow, what
    /// each compartment still holds, and the probabilities and counts of a
    /// multinomial draw.
    rates: Vec<f64>,
    events: Vec<f64>,
    holdings: Vec<f64>,
    probabilities: Vec<f64>,
    counts: Vec<u64>,
}

impl StochasticSteps {
    /// Compile a model of the `StochasticSteps` engine.
    ///
    /// # Errors
    ///
    /// Returns a message when the model is not of this engine, when
    /// [`CompiledModel::try_from_model`] fails (a bin whose amount is not whole, or
    /// amounts that are not whole and no rounding), or when an impulse or a
    /// series point is not at an integer step.
    pub fn try_from_model(model: &commol_core::Model) -> Result<Self, String> {
        let EngineSettings::StochasticSteps(settings) = *model.engine.settings() else {
            return Err(format!(
                "The StochasticSteps engine cannot run a {} model",
                model.engine.engine_type().name()
            ));
        };
        let compiled = CompiledModel::try_from_model(model)?;
        if let Some((transition, position)) = compiled.first_non_integer_impulse() {
            return Err(format!(
                "Impulse transition '{transition}' acts at {position}, which is not an \
                 integer step; the StochasticSteps engine steps only at integers"
            ));
        }
        if let Some((parameter, position)) = compiled.first_non_integer_series_point() {
            return Err(format!(
                "Series parameter '{parameter}' has a point at {position}, which is not \
                 an integer step; the StochasticSteps engine steps only at integers"
            ));
        }
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut absolute = Vec::new();
        for (index, change) in compiled.flow_changes().iter().enumerate() {
            match change.proportional_to() {
                Some(source) => match groups.iter_mut().find(|(s, _)| *s == source) {
                    Some((_, flows)) => flows.push(index),
                    None => groups.push((source, vec![index])),
                },
                None => absolute.push(index),
            }
        }
        let flows = compiled.flow_changes().len();
        let num_compartments = compiled.num_compartments();
        Ok(Self {
            units: Units::new(compiled, EngineType::StochasticSteps, settings.rounding),
            current_step: 0.0,
            settled: false,
            groups,
            absolute,
            rates: vec![0.0; flows],
            events: vec![0.0; flows],
            holdings: vec![0.0; num_compartments],
            probabilities: Vec::new(),
            counts: Vec::new(),
        })
    }

    /// Current amount of each compartment, without the accumulators.
    pub fn state(&self) -> Vec<f64> {
        self.units.state[..self.units.model.num_compartments()].to_vec()
    }

    /// Apply the impulses at the current step, once.
    fn settle(&mut self) -> Result<(), SimulationError> {
        self.units.begin();
        if !self.settled {
            self.units.apply_impulses(self.current_step)?;
            self.settled = true;
        }
        Ok(())
    }

    /// Execute one step: the impulses of the step, then the draws of every
    /// flow from the state after them.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Infeasible`] that names the compartment,
    /// the step and the flows when a rate is negative, when a proportional
    /// rate is above 1, or when the proportional rates out of one compartment
    /// sum to more than 1; and when an impulse is not valid. The state stays
    /// as it was before the flows. Returns [`SimulationError::Failed`] when a
    /// rate cannot be evaluated.
    pub fn step(&mut self) -> Result<(), SimulationError> {
        self.settle()?;
        let step = self.current_step;
        let units = &mut self.units;
        units
            .model
            .transition_rates(step, step, &units.state, &mut self.rates)?;
        self.check_rates(step)?;

        let units = &mut self.units;
        let num_compartments = units.model.num_compartments();
        self.holdings
            .copy_from_slice(&units.state[..num_compartments]);
        for (source, flows) in &self.groups {
            self.probabilities.clear();
            self.probabilities
                .extend(flows.iter().map(|&flow| self.rates[flow]));
            self.counts.resize(flows.len(), 0);
            let trials = self.holdings[*source] as u64;
            sample::multinomial(
                &mut units.generator,
                trials,
                &self.probabilities,
                &mut self.counts,
            );
            for (&flow, &count) in flows.iter().zip(&self.counts) {
                self.events[flow] = count as f64;
                self.holdings[*source] -= count as f64;
            }
        }
        for &flow in &self.absolute {
            let mut count = sample::poisson(&mut units.generator, self.rates[flow]) as f64;
            let change = &units.model.flow_changes()[flow];
            for &(compartment, taken) in change.takes() {
                count = count.min((self.holdings[compartment] / f64::from(taken)).floor());
            }
            for &(compartment, taken) in change.takes() {
                self.holdings[compartment] -= count * f64::from(taken);
            }
            self.events[flow] = count;
        }
        for (change, &count) in units.model.flow_changes().iter().zip(&self.events) {
            for &(index, sign) in change.changes() {
                units.state[index] += sign * count;
            }
        }
        self.current_step += 1.0;
        self.settled = false;
        Ok(())
    }

    /// Check that every rate of the step is a rate the draws can take.
    fn check_rates(&self, step: f64) -> Result<(), SimulationError> {
        let model = &self.units.model;
        let changes = model.flow_changes();
        let name = |index: usize| model.compartments()[index].as_str();
        let infeasible = |message: String| SimulationError::Infeasible {
            position: step,
            message,
        };
        if let Some(flow) = self.rates.iter().position(|rate| rate.is_nan()) {
            return Err(SimulationError::Failed(format!(
                "The rate of flow '{}' at step {step} is not a number",
                changes[flow].transition_id()
            )));
        }
        if let Some((flow, rate)) = self.rates.iter().enumerate().find(|(_, rate)| **rate < 0.0) {
            let change = &changes[flow];
            let compartment = match change.takes().first() {
                Some(&(source, _)) => format!("out of compartment '{}'", name(source)),
                None => {
                    let target = change
                        .changes()
                        .iter()
                        .find(|(index, sign)| *sign > 0.0 && *index < model.num_compartments())
                        .map_or("nowhere", |&(index, _)| name(index));
                    format!("into compartment '{target}'")
                }
            };
            return Err(infeasible(format!(
                "At step {step}, flow '{}' {compartment} has the rate {}. The \
                 StochasticSteps engine draws whole units with it, so a rate cannot be \
                 negative",
                change.transition_id(),
                rounded(*rate),
            )));
        }
        for (source, flows) in &self.groups {
            let sum: f64 = flows.iter().map(|&flow| self.rates[flow]).sum();
            if sum <= 1.0 {
                continue;
            }
            let listed: Vec<String> = flows
                .iter()
                .map(|&flow| {
                    format!(
                        "'{}' {}",
                        changes[flow].transition_id(),
                        rounded(self.rates[flow])
                    )
                })
                .collect();
            let what = if flows.len() == 1 {
                format!("flow {} is above 1", listed[0])
            } else {
                format!(
                    "flows {} sum to {}, above 1",
                    listed.join(", "),
                    rounded(sum)
                )
            };
            return Err(infeasible(format!(
                "At step {step}, the proportional rates out of compartment '{}': {what}. \
                 The StochasticSteps engine reads a proportional rate as the probability \
                 that one unit leaves in a step, so the rates out of one compartment must \
                 sum to at most 1",
                name(*source),
            )));
        }
        Ok(())
    }

    /// Step through the largest of `steps`, recording the outputs at each of
    /// them into `buffer`.
    fn record_steps(
        &mut self,
        steps: &[u32],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        buffer.resize_with(steps.len(), Vec::new);
        self.settle()?;
        let mut next = 0;
        loop {
            while next < steps.len() && f64::from(steps[next]) == self.current_step {
                self.units.copy_row(&mut buffer[next]);
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

/// Why the engine does not report at `position`, if it does not.
fn check_step(position: f64) -> Result<(), String> {
    if position >= 0.0 && position.fract() == 0.0 && position <= f64::from(u32::MAX) {
        Ok(())
    } else {
        Err(format!(
            "a StochasticSteps model reports only at integer steps; {position} is not one"
        ))
    }
}

impl commol_core::SimulationEngine for StochasticSteps {
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
        StochasticSteps::step(self)
    }

    fn output_names(&self) -> Vec<String> {
        self.units.model.output_names().to_vec()
    }

    fn state_outputs(&self) -> usize {
        self.units.model.num_compartments()
    }

    fn output_sums(&self) -> Vec<(String, Vec<usize>)> {
        output_sums(self.units.model.output_labels())
    }

    fn reset(&mut self) {
        self.units.reset();
        self.current_step = 0.0;
        self.settled = false;
    }

    fn set_seed(&mut self, seed: u64) {
        self.units.set_seed(seed);
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.units.model.set_parameter(parameter_id, value);
        Ok(())
    }

    fn current_position(&self) -> f64 {
        self.current_step
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

impl CompartmentalEngine for StochasticSteps {
    fn compartments(&self) -> Vec<String> {
        self.units.model.compartments().to_vec()
    }

    fn output_labels(&self) -> Vec<OutputLabels> {
        self.units.model.output_labels().to_vec()
    }

    fn initial_state(&self) -> Vec<f64> {
        self.units.model.initial_state()[..self.units.model.num_compartments()].to_vec()
    }

    fn total(&self) -> f64 {
        self.units.model.total()
    }

    fn set_initial_condition(
        &mut self,
        compartment_index: usize,
        value: f64,
    ) -> Result<(), String> {
        self.units.set_initial_condition(compartment_index, value)?;
        if self.current_step == 0.0 && self.settled {
            // The run starts again from the new initial state.
            commol_core::SimulationEngine::reset(self);
        }
        Ok(())
    }
}
