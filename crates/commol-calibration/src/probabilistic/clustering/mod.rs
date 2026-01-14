//! Cluster-based representative selection.
//!
//! This module provides functionality for selecting diverse representatives
//! from clustered calibration evaluations.

mod distribution;
mod methods;
mod representatives;

pub use representatives::{select_cluster_representatives, ClusterRepresentativeConfig};
