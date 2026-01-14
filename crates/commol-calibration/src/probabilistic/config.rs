//! Configuration for ensemble selection algorithms.
//!
//! This module provides the `EnsembleSelectionConfig` struct which contains
//! all configurable parameters for NSGA-II ensemble selection and cluster
//! representative selection algorithms.

/// Configuration for ensemble selection algorithms.
///
/// This struct contains all configurable parameters for the NSGA-II ensemble
/// selection and cluster representative selection algorithms. All values have
/// sensible defaults matching the previous hardcoded behavior.
#[derive(Debug, Clone)]
pub struct EnsembleSelectionConfig {
    /// Safety margin factor for CI width bounds normalization (default: 0.1).
    /// Used to avoid edge cases in CI width estimation.
    pub ci_margin_factor: f64,

    /// Sample sizes to try when estimating CI width bounds (default: [10, 20, 50, 100]).
    pub ci_sample_sizes: Vec<usize>,

    /// NSGA-II crossover probability (default: 0.9).
    pub nsga_crossover_probability: f64,

    /// Minimum k for k-nearest neighbors in density estimation (default: 5).
    pub k_neighbors_min: usize,

    /// Maximum k for k-nearest neighbors in density estimation (default: 10).
    pub k_neighbors_max: usize,

    /// Exponential weight for sparsity bonus in maximin selection (default: 2.0).
    pub sparsity_weight: f64,

    /// Weight for stratum fit vs quality in latin_hypercube selection (default: 10.0).
    pub stratum_fit_weight: f64,
}

impl Default for EnsembleSelectionConfig {
    fn default() -> Self {
        Self {
            ci_margin_factor: 0.1,
            ci_sample_sizes: vec![10, 20, 50, 100],
            nsga_crossover_probability: 0.9,
            k_neighbors_min: 5,
            k_neighbors_max: 10,
            sparsity_weight: 2.0,
            stratum_fit_weight: 10.0,
        }
    }
}

impl EnsembleSelectionConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set CI margin factor.
    pub fn with_ci_margin_factor(mut self, factor: f64) -> Self {
        self.ci_margin_factor = factor;
        self
    }

    /// Set CI sample sizes for bounds estimation.
    pub fn with_ci_sample_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.ci_sample_sizes = sizes;
        self
    }

    /// Set NSGA-II crossover probability.
    pub fn with_nsga_crossover_probability(mut self, probability: f64) -> Self {
        self.nsga_crossover_probability = probability;
        self
    }

    /// Set k-neighbors bounds for density estimation.
    pub fn with_k_neighbors_bounds(mut self, min: usize, max: usize) -> Self {
        self.k_neighbors_min = min;
        self.k_neighbors_max = max;
        self
    }

    /// Set sparsity weight for maximin selection.
    pub fn with_sparsity_weight(mut self, weight: f64) -> Self {
        self.sparsity_weight = weight;
        self
    }

    /// Set stratum fit weight for latin hypercube selection.
    pub fn with_stratum_fit_weight(mut self, weight: f64) -> Self {
        self.stratum_fit_weight = weight;
        self
    }
}
