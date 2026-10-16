//! Stochastic engines of aggregate compartment models.
//!
//! The model is compiled by `commol-model`, as for the deterministic engines;
//! these engines move whole units of it at random. Each flow of the model is
//! one kind of event: an event takes one unit from each source of the flow,
//! and adds one to each target and to each accumulator that counts it
//! ([`commol_model::CompiledModel::flow_changes`]).
//!
//! | Rate | [`StochasticEvents`] (continuous time) | [`StochasticSteps`] (discrete time) |
//! |---|---|---|
//! | Proportional, `k` | Each unit of the source leaves at hazard `k`: propensity `k × source` | Each unit leaves with probability `k` in a step: a binomial draw, one multinomial per source |
//! | Absolute, `r` | Events at rate `r` | A Poisson draw of mean `r`, capped by what the sources still hold |
//! | No source, `r` | Events at rate `r` | A Poisson draw of mean `r` |
//!
//! The expected change of the state is the right-hand side of the
//! differential engine in continuous time, and the step of the difference
//! engine in discrete time while no cap acts, so the deterministic engines
//! are the mean-field limits of these.
//!
//! Both engines draw every random number from one `commol_random::Generator`
//! seeded by `SimulationEngine::set_seed`, and start from whole units: a
//! total that is not whole is an error, and initial amounts that are not
//! whole need a [`commol_core::Rounding`] in the settings of the engine.

#![warn(missing_docs)]

mod events;
mod steps;
mod units;

pub use events::StochasticEvents;
pub use steps::StochasticSteps;
