//! Reproducible random numbers for the stochastic engines and the inference
//! methods of commol.
//!
//! What the stochastic engines of every model kind and the inference methods
//! share, and no model:
//!
//! - [`Generator`]: one `Xoshiro256PlusPlus` generator, seeded from a `u64`
//!   by SplitMix64, whose sequence is the same on every platform and version;
//!   it implements the generator trait of `rand_core`, so the particle swarm
//!   of a calibration draws with it;
//! - [`replicate_seed`]: the seed of replicate `index` of a run seeded with
//!   `seed`;
//! - [`Stream`]: the one registry of the seed streams of every method that
//!   draws from a seed, the starts of a calibration and the inference
//!   methods, each with its own number, method and purpose;
//! - [`seed_or_drawn`]: the one place a seed is drawn, from the entropy of the
//!   operating system, for a caller that sets none;
//! - the samplers of [`sample`] (exponential, Poisson, binomial,
//!   multinomial, normal, gamma, an index below `n`) and the uniforms of
//!   [`Generator`], written here so that their values do not change with a
//!   dependency; the inference methods draw with nothing else;
//! - [`SumTree`]: the rates of a set of events, to pick the next one in
//!   `O(log n)`. The stochastic engine of the test model kind uses it;
//!   `StochasticEvents` picks an event by a scan of its propensities;
//! - [`latin_hypercube`]: a Latin hypercube on the unit cube, which the
//!   starts of a calibration and the designs of the inference methods map to
//!   their parameters.
//!
//! A run is reproducible from its seed on one platform and one commol
//! version. The rates an engine draws with come from the system's `libm`
//! through the JIT, so on another platform a run is equal in distribution and
//! can differ draw by draw.

#![warn(missing_docs)]

mod drawn;
mod generator;
mod hypercube;
pub mod sample;
mod streams;
mod sum_tree;

pub use drawn::seed_or_drawn;
pub use generator::{Generator, replicate_seed, splitmix64};
pub use hypercube::latin_hypercube;
pub use streams::Stream;
pub use sum_tree::SumTree;
