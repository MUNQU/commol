//! Selection methods for cluster representatives.
//!
//! This module provides various algorithms for selecting diverse representatives
//! from a pool of candidates within a cluster.

mod crowding;
mod latin_hypercube;
pub mod maximin;

pub use crowding::select_by_crowding_distance;
pub use latin_hypercube::select_by_latin_hypercube;
pub use maximin::select_by_maximin_distance;
