//! The continuous-time stochastic engine: events one at a time (Gillespie's
//! direct method), or many in a leap (tau-leaping).

use commol_core::{
    EngineSettings, EngineType, EventMethod, ModelValue, SimulationError, StochasticEventsSettings,
};
use commol_model::{
    CompartmentalEngine, CompiledModel, InitialConditionPlan, OutputLabels, output_sums,
};
use commol_random::sample;

use crate::units::{Units, rounded};

/// The order of the reactions Cao, Gillespie and Petzold (2006) bound a leap
/// with. A rate of commol is a formula, whose order is unknown, so the engine
/// takes the order of a reaction of two units, which gives shorter leaps
/// than a first order.
const LEAP_ORDER: f64 = 2.0;

/// An adaptive leap shorter than this many mean waits between events is not
/// worth its Poisson draws: the engine draws one exact event instead (Cao,
/// Gillespie and Petzold 2006).
const EXACT_BELOW_EVENTS: f64 = 10.0;

/// How many times a leap that leaves an amount below zero is halved before
/// the engine draws one exact event instead.
const LEAP_HALVINGS: usize = 30;

/// The deepest recursion of the adaptive Simpson rule.
const SIMPSON_DEPTH: u32 = 40;

/// Relative tolerance with which two positions on the axis are the same.
const POSITION_TOLERANCE: f64 = 1e-9;

fn same_position(a: f64, b: f64) -> bool {
    (a - b).abs() <= POSITION_TOLERANCE * a.abs().max(b.abs()).max(1.0)
}

/// What happens next, from the last change of the state.
enum Next {
    /// An event at this position.
    Event(f64),
    /// The next breakpoint comes first; the integrated propensity up to it.
    Break(f64),
    /// Nothing before the end of the run.
    Beyond,
}

/// A leap of tau-leaping drawn but not yet reached: it ends at `at` and adds
/// `counts` events of each flow.
#[derive(Clone)]
struct Leap {
    at: f64,
    counts: Vec<f64>,
}

/// The continuous-time stochastic engine of an aggregate model.
///
/// Whole units move one event at a time: an event of a flow takes one unit
/// from each source and adds one to each target and accumulator. The
/// propensity of a flow, its expected events per unit of the axis, is its
/// value in the differential engine: the rate times the source for a
/// proportional rate, the rate for an absolute one. A flow whose sources
/// hold fewer units than an event takes has no events. The expected change
/// of the state is the right-hand side of the differential engine.
///
/// - Direct method (Gillespie 1977): the time to the next event is the
///   position where the total propensity, integrated from the last change of
///   the state, reaches an exponential draw; the event is drawn in proportion
///   to the propensities there. When no rate reads `t` the total is constant
///   between breakpoints and the time is a quotient. When a rate reads `t`
///   the engine integrates it (Anderson 2007), within the tolerance of its
///   settings.
/// - Tau-leaping: a Poisson number of events of each flow in a leap, of the
///   fixed length `tau` or chosen as Cao, Gillespie and Petzold (2006) do.
///
/// The engine stops at every breakpoint of the model
/// ([`CompiledModel::breakpoints`]), where the impulses act; no wait and no
/// leap crosses one. A run does not depend on the positions it reports at:
/// the output at a position is the state after the last event before it.
#[derive(Clone)]
pub struct StochasticEvents {
    units: Units,
    settings: StochasticEventsSettings,
    /// Where the engine reports: the end of the last run.
    position: f64,
    /// The last change of the state: an event, a leap or a breakpoint. Waits
    /// are measured from it.
    clock: f64,
    /// The integrated propensity left before the next event, an exponential
    /// draw; `None` until the engine draws it.
    remaining: Option<f64>,
    /// Whether the impulses at the clock are in the state.
    settled: bool,
    /// A leap drawn but not reached.
    leap: Option<Leap>,
    /// Whether every leap from the clock left an amount below zero, so the
    /// engine takes exact events until the state changes.
    no_leap: bool,
    /// The propensity of each flow at the last evaluation.
    propensities: Vec<f64>,
}

impl StochasticEvents {
    /// Compile a model of the `StochasticEvents` engine.
    ///
    /// # Errors
    ///
    /// Returns a message when the model is not of this engine, or when
    /// [`CompiledModel::try_from_model`] fails (a bin whose amount is not whole, or
    /// amounts that are not whole and no rounding).
    pub fn try_from_model(model: &commol_core::Model) -> Result<Self, String> {
        let EngineSettings::StochasticEvents(settings) = *model.engine.settings() else {
            return Err(format!(
                "The StochasticEvents engine cannot run a {} model",
                model.engine.engine_type().name()
            ));
        };
        settings.validate()?;
        let compiled = CompiledModel::try_from_model(model)?;
        let flows = compiled.flow_changes().len();
        Ok(Self {
            units: Units::new(compiled, EngineType::StochasticEvents, settings.rounding),
            settings,
            position: 0.0,
            clock: 0.0,
            remaining: None,
            settled: false,
            leap: None,
            no_leap: false,
            propensities: vec![0.0; flows],
        })
    }

    /// The compiled model the engine runs.
    pub fn model(&self) -> &CompiledModel {
        &self.units.model
    }

    /// Current amount of each compartment, without the accumulators.
    pub fn state(&self) -> Vec<f64> {
        self.units.state[..self.units.model.num_compartments()].to_vec()
    }

    /// Run through the last of `positions`, writing the state at each of them
    /// into `buffer`.
    fn run_to(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        commol_core::check_positions(positions, self.position, check_time)?;
        buffer.resize_with(positions.len(), Vec::new);
        self.units.begin();
        if !self.settled {
            self.units.apply_impulses(self.clock)?;
            self.settled = true;
        }
        let Some(&end) = positions.last() else {
            return Ok(());
        };
        // The points past `end` do not matter, and every point up to `end`
        // is a breakpoint of the model, not the end of the search.
        let breakpoints: Vec<f64> = self
            .units
            .model
            .breakpoints(self.clock, end + 1.0)?
            .into_iter()
            .filter(|&point| point <= end)
            .collect();
        let mut next_break = 0;
        let mut next = 0;
        loop {
            let limit = breakpoints
                .get(next_break)
                .copied()
                .unwrap_or(f64::INFINITY);
            let (stop, change) = self.next_change(limit, end)?;
            while next < positions.len() && positions[next] < stop && positions[next] <= end {
                self.units.copy_row(&mut buffer[next]);
                next += 1;
            }
            if stop > end {
                // The loop above copied every position up to `end`.
                self.position = end;
                return Ok(());
            }
            match change {
                Change::Event => {
                    let flow = self.pick(stop, limit)?;
                    self.fire(flow, 1.0);
                }
                Change::Leap => {
                    let leap = self.leap.take().expect("a leap to reach");
                    for (flow, &count) in leap.counts.iter().enumerate() {
                        self.fire(flow, count);
                    }
                }
                Change::Break(integral) => {
                    if let Some(remaining) = &mut self.remaining {
                        *remaining = (*remaining - integral).max(0.0);
                    }
                }
            }
            self.clock = stop;
            self.no_leap = false;
            if next_break < breakpoints.len() && same_position(stop, limit) {
                self.clock = limit;
                self.units.apply_impulses(limit)?;
                next_break += 1;
            }
            while next < positions.len() && positions[next] <= self.clock {
                self.units.copy_row(&mut buffer[next]);
                next += 1;
            }
        }
    }

    /// The next change of the state before `limit`, the next breakpoint: its
    /// position and what it is, or a position past `end` when nothing changes
    /// up to `end`.
    fn next_change(&mut self, limit: f64, end: f64) -> Result<(f64, Change), SimulationError> {
        if self.settings.method == EventMethod::TauLeaping {
            if let Some(leap) = &self.leap {
                return Ok((leap.at, Change::Leap));
            }
            if !self.no_leap
                && let Some(found) = self.draw_leap(limit)?
            {
                return Ok(found);
            }
        }
        Ok(match self.next_event(limit, end)? {
            Next::Event(at) => (at, Change::Event),
            Next::Break(integral) => (limit, Change::Break(integral)),
            Next::Beyond => (f64::INFINITY, Change::Break(0.0)),
        })
    }

    /// Add `count` events of `flow` to the state. After an event the next
    /// wait is a new draw.
    fn fire(&mut self, flow: usize, count: f64) {
        if count == 0.0 {
            return;
        }
        let change = &self.units.model.flow_changes()[flow];
        for &(index, sign) in change.changes() {
            self.units.state[index] += sign * count;
        }
        self.remaining = None;
    }

    /// The propensity of every flow at `step` and `t`, into
    /// `self.propensities`, and their total.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Infeasible`] that names the flow when a
    /// propensity is negative, and [`SimulationError::Failed`] when a rate
    /// cannot be evaluated or is not a number.
    fn propensities(&mut self, step: f64, t: f64) -> Result<f64, SimulationError> {
        let units = &mut self.units;
        units
            .model
            .flow_rates(step, t, &units.state, &mut self.propensities)?;
        let mut total = 0.0;
        for (propensity, change) in self.propensities.iter_mut().zip(units.model.flow_changes()) {
            if propensity.is_nan() {
                return Err(SimulationError::Failed(format!(
                    "The rate of flow '{}' at {t} is not a number",
                    change.transition_id()
                )));
            }
            if *propensity < 0.0 {
                return Err(SimulationError::Infeasible {
                    position: t,
                    message: format!(
                        "Flow '{}' has the propensity {} at {t}. The StochasticEvents \
                         engine draws its events with it, so it cannot be negative",
                        change.transition_id(),
                        rounded(*propensity),
                    ),
                });
            }
            if change.takes().iter().any(|&(compartment, units_taken)| {
                units.state[compartment] < f64::from(units_taken)
            }) {
                *propensity = 0.0;
            }
            total += *propensity;
        }
        Ok(total)
    }

    /// The total propensity at `t`, on the piece that ends at `limit`: rates
    /// see the step of the clock, and `t` below `limit`, so a value that
    /// jumps at `limit` acts only from there.
    fn total_at(&mut self, t: f64, limit: f64) -> Result<f64, SimulationError> {
        let t = if limit.is_finite() {
            t.min(limit.next_down())
        } else {
            t
        };
        self.propensities(self.clock.floor(), t)
    }

    /// The next event of the direct method, from the clock: where the total
    /// propensity integrated from the clock reaches the remaining draw.
    fn next_event(&mut self, limit: f64, end: f64) -> Result<Next, SimulationError> {
        let remaining = match self.remaining {
            Some(remaining) => remaining,
            None => {
                let draw = sample::exponential(&mut self.units.generator, 1.0);
                self.remaining = Some(draw);
                draw
            }
        };
        let start = self.clock;
        let first = self.total_at(start, limit)?;
        if !self.units.model.reads_time() {
            if first <= 0.0 {
                return Ok(if limit.is_finite() {
                    Next::Break(0.0)
                } else {
                    Next::Beyond
                });
            }
            let at = start + remaining / first;
            return Ok(if at < limit {
                Next::Event(at)
            } else {
                Next::Break(first * (limit - start))
            });
        }
        self.next_event_in_time(remaining, first, limit, end)
    }

    /// [`Self::next_event`] when a rate reads `t`: the total is integrated
    /// piece by piece from the clock, the pieces growing from the mean wait
    /// at the clock, until it reaches `remaining`; then the position is found
    /// by a Newton search kept inside its bracket. The pieces do not depend on
    /// `end`, so neither does the event; past `end` the search stops.
    fn next_event_in_time(
        &mut self,
        remaining: f64,
        first: f64,
        limit: f64,
        end: f64,
    ) -> Result<Next, SimulationError> {
        let tolerance = self.settings.tolerance;
        let start = self.clock;
        let mut width = if first > 0.0 {
            (remaining / first).min(1.0)
        } else {
            1.0
        };
        let mut low = start;
        let mut below = 0.0;
        let high = loop {
            if low >= limit {
                return Ok(Next::Break(below));
            }
            if low > end {
                return Ok(Next::Beyond);
            }
            let high = (low + width).min(limit);
            let piece = self.integral(low, high, limit, tolerance * remaining)?;
            if below + piece >= remaining {
                break high;
            }
            below += piece;
            low = high;
            width *= 2.0;
        };
        // `below + ∫ from low to at` reaches `remaining` in `[low, high]`.
        let (mut left, mut right) = (low, high);
        let mut at = {
            let rate = self.total_at(low, limit)?;
            if rate > 0.0 {
                (low + (remaining - below) / rate).clamp(low, high)
            } else {
                0.5 * (low + high)
            }
        };
        for _ in 0..100 {
            let gap = below + self.integral(low, at, limit, tolerance * remaining)? - remaining;
            if gap.abs() <= tolerance * remaining {
                break;
            }
            if gap > 0.0 {
                right = at;
            } else {
                left = at;
            }
            if right - left <= tolerance * at.abs().max(1.0) {
                break;
            }
            let rate = self.total_at(at, limit)?;
            let newton = if rate > 0.0 {
                at - gap / rate
            } else {
                f64::NAN
            };
            at = if newton > left && newton < right {
                newton
            } else {
                0.5 * (left + right)
            };
        }
        Ok(if at < limit {
            Next::Event(at)
        } else {
            Next::Break(below + self.integral(low, limit, limit, tolerance * remaining)?)
        })
    }

    /// The total propensity integrated from `from` to `to`, with the state
    /// fixed, by the adaptive Simpson rule within `tolerance`.
    fn integral(
        &mut self,
        from: f64,
        to: f64,
        limit: f64,
        tolerance: f64,
    ) -> Result<f64, SimulationError> {
        if to <= from {
            return Ok(0.0);
        }
        let middle = 0.5 * (from + to);
        let (a, m, b) = (
            self.total_at(from, limit)?,
            self.total_at(middle, limit)?,
            self.total_at(to, limit)?,
        );
        let whole = (to - from) / 6.0 * (a + 4.0 * m + b);
        self.simpson(
            from,
            to,
            a,
            m,
            b,
            whole,
            tolerance.max(f64::MIN_POSITIVE),
            limit,
            SIMPSON_DEPTH,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn simpson(
        &mut self,
        from: f64,
        to: f64,
        a: f64,
        m: f64,
        b: f64,
        whole: f64,
        tolerance: f64,
        limit: f64,
        depth: u32,
    ) -> Result<f64, SimulationError> {
        let middle = 0.5 * (from + to);
        let left_middle = 0.5 * (from + middle);
        let right_middle = 0.5 * (middle + to);
        let lm = self.total_at(left_middle, limit)?;
        let rm = self.total_at(right_middle, limit)?;
        let left = (middle - from) / 6.0 * (a + 4.0 * lm + m);
        let right = (to - middle) / 6.0 * (m + 4.0 * rm + b);
        let error = left + right - whole;
        if depth == 0 || error.abs() <= 15.0 * tolerance {
            return Ok(left + right + error / 15.0);
        }
        Ok(self.simpson(
            from,
            middle,
            a,
            lm,
            m,
            left,
            0.5 * tolerance,
            limit,
            depth - 1,
        )? + self.simpson(
            middle,
            to,
            m,
            rm,
            b,
            right,
            0.5 * tolerance,
            limit,
            depth - 1,
        )?)
    }

    /// The flow of an event at `at`, drawn in proportion to the propensities
    /// there.
    fn pick(&mut self, at: f64, limit: f64) -> Result<usize, SimulationError> {
        let total = self.total_at(at, limit)?;
        let point = self.units.generator.uniform() * total;
        let mut sum = 0.0;
        let mut last = 0;
        // The last flow of positive propensity takes a point that the
        // rounding of `uniform() * total` puts at the total.
        for (flow, &propensity) in self.propensities.iter().enumerate() {
            if propensity > 0.0 {
                sum += propensity;
                last = flow;
                if point < sum {
                    break;
                }
            }
        }
        Ok(last)
    }

    /// Draw a leap from the clock, or `None` when the engine takes one exact
    /// event instead: the leap would hold too few events, or every leap
    /// leaves an amount below zero.
    fn draw_leap(&mut self, limit: f64) -> Result<Option<(f64, Change)>, SimulationError> {
        let start = self.clock;
        let total = self.total_at(start, limit)?;
        if total <= 0.0 {
            return Ok(None);
        }
        let mut tau = match self.settings.tau {
            Some(tau) => tau,
            None => {
                let tau = self.adaptive_tau();
                if tau * total < EXACT_BELOW_EVENTS {
                    return Ok(None);
                }
                tau
            }
        };
        let num_compartments = self.units.model.num_compartments();
        let mut counts = vec![0.0; self.propensities.len()];
        for _ in 0..LEAP_HALVINGS {
            let reaches_limit = start + tau >= limit;
            let length = if reaches_limit { limit - start } else { tau };
            let mut after = self.units.state[..num_compartments].to_vec();
            for (flow, &propensity) in self.propensities.iter().enumerate() {
                counts[flow] =
                    sample::poisson(&mut self.units.generator, propensity * length) as f64;
                for &(index, sign) in self.units.model.flow_changes()[flow].changes() {
                    if index < num_compartments {
                        after[index] += sign * counts[flow];
                    }
                }
            }
            if after.iter().all(|&amount| amount >= 0.0) {
                let at = if reaches_limit { limit } else { start + tau };
                self.leap = Some(Leap { at, counts });
                self.remaining = None;
                return Ok(Some((at, Change::Leap)));
            }
            tau = 0.5 * length;
        }
        self.no_leap = true;
        Ok(None)
    }

    /// The leap of Cao, Gillespie and Petzold (2006): the longest in which the
    /// mean and the standard deviation of the change of each compartment stay
    /// within `epsilon` of its amount, over the order [`LEAP_ORDER`].
    fn adaptive_tau(&self) -> f64 {
        let num_compartments = self.units.model.num_compartments();
        let mut mean = vec![0.0; num_compartments];
        let mut variance = vec![0.0; num_compartments];
        for (change, &propensity) in self
            .units
            .model
            .flow_changes()
            .iter()
            .zip(&self.propensities)
        {
            if propensity <= 0.0 {
                continue;
            }
            let mut net: Vec<(usize, f64)> = Vec::new();
            for &(index, sign) in change.changes() {
                if index >= num_compartments {
                    continue;
                }
                match net.iter_mut().find(|(i, _)| *i == index) {
                    Some((_, value)) => *value += sign,
                    None => net.push((index, sign)),
                }
            }
            for (index, value) in net {
                mean[index] += value * propensity;
                variance[index] += value * value * propensity;
            }
        }
        let mut tau = f64::INFINITY;
        for index in 0..num_compartments {
            let bound = (self.settings.epsilon * self.units.state[index] / LEAP_ORDER).max(1.0);
            if mean[index] != 0.0 {
                tau = tau.min(bound / mean[index].abs());
            }
            if variance[index] > 0.0 {
                tau = tau.min(bound * bound / variance[index]);
            }
        }
        tau
    }
}

/// The change at the end of a wait.
enum Change {
    /// One event, of a flow drawn when it happens.
    Event,
    /// The leap drawn.
    Leap,
    /// A breakpoint, with the propensity integrated up to it.
    Break(f64),
}

/// Why the engine does not report at `position`, if it does not.
fn check_time(position: f64) -> Result<(), String> {
    if position.is_finite() && position >= 0.0 {
        Ok(())
    } else {
        Err(format!(
            "a StochasticEvents model reports at finite, non-negative positions; \
             {position} is not one"
        ))
    }
}

impl commol_core::SimulationEngine for StochasticEvents {
    type ValuePlan = InitialConditionPlan;

    fn check_position(&self, position: f64) -> Result<(), String> {
        check_time(position)
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        self.run_to(positions, buffer)
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        let mut buffer = Vec::with_capacity(1);
        self.run_to(&[self.position + 1.0], &mut buffer)
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
        self.position = 0.0;
        self.clock = 0.0;
        self.remaining = None;
        self.settled = false;
        self.leap = None;
        self.no_leap = false;
    }

    fn set_seed(&mut self, seed: u64) {
        self.units.set_seed(seed);
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.units.model.set_parameter(parameter_id, value);
        Ok(())
    }

    fn current_position(&self) -> f64 {
        self.position
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

impl CompartmentalEngine for StochasticEvents {
    fn compartments(&self) -> Vec<String> {
        self.units.model.compartments().to_vec()
    }

    fn output_labels(&self) -> Vec<OutputLabels> {
        self.units.model.output_labels().to_vec()
    }

    fn state(&self) -> Vec<f64> {
        StochasticEvents::state(self)
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
        if self.position == 0.0 && self.settled {
            // The run starts again from the new initial state.
            commol_core::SimulationEngine::reset(self);
        }
        Ok(())
    }
}
