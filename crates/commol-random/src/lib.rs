//! Reproducible random numbers for the stochastic engines of commol.
//!
//! What every stochastic engine of every model kind shares, and no model:
//!
//! - [`Generator`]: one `Xoshiro256PlusPlus` generator, seeded from a `u64`
//!   by SplitMix64, whose sequence is the same on every platform and version;
//! - [`replicate_seed`]: the seed of replicate `index` of a run seeded with
//!   `seed`, and [`stream_seed`]: the seed of one purpose of a method;
//! - the samplers of [`sample`] (exponential, Poisson, binomial,
//!   multinomial), written here so that their values do not change with a
//!   dependency;
//! - [`SumTree`]: the rates of a set of events, to pick the next one in
//!   `O(log n)`;
//! - [`latin_hypercube`]: a Latin hypercube on the unit cube, which the
//!   starts of a calibration and the designs of the inference methods map to
//!   their parameters.
//!
//! A run is reproducible from its seed on one platform and one commol
//! version. The rates an engine draws with come from the system's `libm`
//! through the JIT, so on another platform a run is equal in distribution and
//! can differ draw by draw.

#![warn(missing_docs)]

mod generator;
mod hypercube;
pub mod sample;
mod sum_tree;

pub use generator::{Generator, replicate_seed, splitmix64, stream_seed};
pub use hypercube::latin_hypercube;
pub use sum_tree::SumTree;
