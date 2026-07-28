//! Configuration for representative selection.

/// Configuration for cluster representative selection algorithms.
#[derive(Debug, Clone)]
pub struct RepresentativeSelectionConfig {
    /// Minimum k for k-nearest neighbors in density estimation (default: 5).
    pub k_neighbors_min: usize,

    /// Maximum k for k-nearest neighbors in density estimation (default: 10).
    pub k_neighbors_max: usize,

    /// Exponential weight for sparsity bonus in maximin selection (default: 2.0).
    pub sparsity_weight: f64,

    /// Weight for stratum fit vs quality in latin_hypercube selection (default: 10.0).
    pub stratum_fit_weight: f64,
}

impl Default for RepresentativeSelectionConfig {
    fn default() -> Self {
        Self {
            k_neighbors_min: 5,
            k_neighbors_max: 10,
            sparsity_weight: 2.0,
            stratum_fit_weight: 10.0,
        }
    }
}

impl RepresentativeSelectionConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
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
