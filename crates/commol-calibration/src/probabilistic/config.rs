//! Configuration for ensemble selection and representative selection.

/// Prediction points used for the confidence-interval width objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CIWidthScope {
    /// Use only observed time/compartment points.
    ObservedPoints,
    /// Use all compartments at time steps that have observations.
    ObservedStepsAllCompartments,
    /// Use the full generated trajectory.
    FullTrajectory,
}

/// Algorithm used for ensemble subset selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleAlgorithm {
    /// Greedy subset selection with local search.
    GreedyLocalSearch,
    /// NSGA-II multi-objective optimization.
    Nsga2,
}

/// Configuration for ensemble selection and representative selection algorithms.
#[derive(Debug, Clone)]
pub struct EnsembleSelectionConfig {
    /// Safety margin factor for CI width bounds normalization (default: 0.1).
    /// Used to avoid edge cases in CI width estimation.
    pub ci_margin_factor: f64,

    /// Sample sizes to try when estimating CI width bounds (default: [10, 20, 50, 100]).
    pub ci_sample_sizes: Vec<usize>,

    /// Crossover probability for algorithms that use crossover (default: 0.9).
    pub crossover_probability: f64,

    /// Prediction scope for the CI width objective.
    pub ci_width_scope: CIWidthScope,

    /// Ensemble subset-selection algorithm.
    pub ensemble_algorithm: EnsembleAlgorithm,

    /// Minimum metric points before objective percentile evaluation uses Rayon.
    pub parallel_objective_threshold: usize,

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
            crossover_probability: 0.9,
            ci_width_scope: CIWidthScope::FullTrajectory,
            ensemble_algorithm: EnsembleAlgorithm::Nsga2,
            parallel_objective_threshold: 512,
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

    /// Set crossover probability.
    pub fn with_crossover_probability(mut self, probability: f64) -> Self {
        self.crossover_probability = probability;
        self
    }

    /// Set the prediction scope for the CI width objective.
    pub fn with_ci_width_scope(mut self, scope: CIWidthScope) -> Self {
        self.ci_width_scope = scope;
        self
    }

    /// Set the ensemble subset-selection algorithm.
    pub fn with_ensemble_algorithm(mut self, algorithm: EnsembleAlgorithm) -> Self {
        self.ensemble_algorithm = algorithm;
        self
    }

    /// Set the metric-point threshold for parallel objective evaluation.
    pub fn with_parallel_objective_threshold(mut self, threshold: usize) -> Self {
        self.parallel_objective_threshold = threshold;
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
