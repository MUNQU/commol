//! Type definitions for calibration

use serde::{Deserialize, Serialize};

use crate::statistics;

/// Type of value being calibrated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationParameterType {
    /// Model parameter (e.g., k1, k2)
    Parameter,
    /// A value of the state the model starts from. The engine of the model
    /// kind resolves its id; a compartmental engine takes a bin, a
    /// compartment or a category.
    InitialCondition,
    /// Factor that multiplies the model output of an observation before
    /// comparison
    Factor,
    /// A parameter of an observation model, such as the standard deviation
    /// of a normal model. It never reaches the engine.
    Noise,
}

/// Represents an observed data point to fit against
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedDataPoint {
    /// Position of the observation on the axis. The engine decides which
    /// positions its model type accepts.
    pub at: f64,

    /// Name of the observed series: the output itself, or the name of an
    /// aggregate of several outputs. Messages and the per-series
    /// normalization use it.
    pub label: String,

    /// Outputs summed before comparison; at least one.
    pub outputs: Vec<String>,

    /// Observed value of the summed outputs
    pub value: f64,

    /// Weight for this observation (default 1.0)
    /// Higher weights give more importance to this data point in the loss function
    pub weight: f64,

    /// Optional factor parameter ID that multiplies the model output before
    /// comparison. Useful when observed data is in different units or there
    /// is an unknown proportionality constant.
    pub factor_id: Option<String>,

    /// Optional window length for cumulative-output observations.
    /// When set, the predicted value is `output(at) - output(at - window)`.
    pub window: Option<f64>,

    /// The trials of a binomial observation model that takes them from each
    /// observation; `None` for every other observation.
    #[serde(default)]
    pub trials: Option<u64>,
}

impl ObservedDataPoint {
    /// Create an observation of one output with default weight of 1.0
    pub fn new(at: f64, output: String, value: f64) -> Self {
        Self {
            at,
            label: output.clone(),
            outputs: vec![output],
            value,
            weight: 1.0,
            factor_id: None,
            window: None,
            trials: None,
        }
    }

    /// Create an observation of the sum of several outputs, named `label`
    pub fn of_sum(at: f64, label: String, outputs: Vec<String>, value: f64) -> Self {
        Self {
            label,
            outputs,
            ..Self::new(at, String::new(), value)
        }
    }

    /// Create an observation of one output with a factor parameter
    pub fn with_factor(at: f64, output: String, value: f64, factor_id: String) -> Self {
        Self {
            factor_id: Some(factor_id),
            ..Self::new(at, output, value)
        }
    }

    /// Create a windowed observation of a cumulative output.
    pub fn with_window(at: f64, output: String, value: f64, window: f64) -> Self {
        Self {
            window: Some(window),
            ..Self::new(at, output, value)
        }
    }
}

/// Parameter to be calibrated with its bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationParameter {
    /// Parameter identifier (parameter ID for parameters, bin ID for initial conditions)
    pub id: String,

    /// Type of value being calibrated
    pub parameter_type: CalibrationParameterType,

    /// Minimum allowed value
    pub min_bound: f64,

    /// Maximum allowed value
    pub max_bound: f64,

    /// Optional initial guess (if None, will use midpoint of bounds)
    pub initial_guess: Option<f64>,
}

impl CalibrationParameter {
    /// Create a new calibration parameter (defaults to Parameter type)
    pub fn new(id: String, min_bound: f64, max_bound: f64) -> Self {
        Self {
            id,
            parameter_type: CalibrationParameterType::Parameter,
            min_bound,
            max_bound,
            initial_guess: None,
        }
    }

    /// Create a new calibration parameter with initial guess
    pub fn with_initial_guess(
        id: String,
        min_bound: f64,
        max_bound: f64,
        initial_guess: f64,
    ) -> Self {
        Self {
            id,
            parameter_type: CalibrationParameterType::Parameter,
            min_bound,
            max_bound,
            initial_guess: Some(initial_guess),
        }
    }

    /// Create a new calibration parameter with explicit type
    pub fn with_type(
        id: String,
        parameter_type: CalibrationParameterType,
        min_bound: f64,
        max_bound: f64,
    ) -> Self {
        Self {
            id,
            parameter_type,
            min_bound,
            max_bound,
            initial_guess: None,
        }
    }

    /// Create a new calibration parameter with type and initial guess
    pub fn with_type_and_guess(
        id: String,
        parameter_type: CalibrationParameterType,
        min_bound: f64,
        max_bound: f64,
        initial_guess: f64,
    ) -> Self {
        Self {
            id,
            parameter_type,
            min_bound,
            max_bound,
            initial_guess: Some(initial_guess),
        }
    }

    /// Get the initial value, or midpoint of bounds if not specified
    pub fn initial_value(&self) -> f64 {
        self.initial_guess
            .unwrap_or_else(|| (self.min_bound + self.max_bound) / 2.0)
    }
}

/// Configuration for loss function calculation.
///
/// Every loss uses the observation weight `w` the same way. With a residual
/// `r = (observed - predicted) × normalization`:
///
/// | Loss | Value |
/// |---|---|
/// | `SumSquaredError` | `Σ w·r²` |
/// | `RootMeanSquaredError` | `√(Σ w·r² / Σ w)` |
/// | `MeanAbsoluteError` | `Σ w·|r| / Σ w` |
///
/// With every weight at 1 they are the usual unweighted losses.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum LossConfig {
    /// Weighted sum of squared errors: `Σ w·r²`.
    #[default]
    SumSquaredError,

    /// Weighted root mean squared error: `√(Σ w·r² / Σ w)`.
    RootMeanSquaredError,

    /// Weighted mean absolute error: `Σ w·|r| / Σ w`.
    MeanAbsoluteError,
}

impl LossConfig {
    /// Contribution of one residual with weight `weight` to the running sum.
    #[inline]
    pub fn term(self, residual: f64, weight: f64) -> f64 {
        match self {
            LossConfig::SumSquaredError | LossConfig::RootMeanSquaredError => {
                weight * residual * residual
            }
            LossConfig::MeanAbsoluteError => weight * residual.abs(),
        }
    }

    /// Loss from the running sum of [`Self::term`] and the sum of the weights.
    ///
    /// Returns 0 for a mean loss whose weights sum to 0.
    #[inline]
    pub fn finish(self, sum: f64, weight_sum: f64) -> f64 {
        match self {
            LossConfig::SumSquaredError => sum,
            LossConfig::RootMeanSquaredError if weight_sum > 0.0 => (sum / weight_sum).sqrt(),
            LossConfig::MeanAbsoluteError if weight_sum > 0.0 => sum / weight_sum,
            _ => 0.0,
        }
    }

    /// Loss of `(residual, weight)` pairs.
    pub fn aggregate(self, residuals: impl IntoIterator<Item = (f64, f64)>) -> f64 {
        let (sum, weight_sum) =
            residuals
                .into_iter()
                .fold((0.0, 0.0), |(sum, weight_sum), (residual, weight)| {
                    (sum + self.term(residual, weight), weight_sum + weight)
                });
        self.finish(sum, weight_sum)
    }
}

impl LossConfig {
    /// Loss of observations against the replicates of a stochastic engine.
    ///
    /// Each item is `(observed, predictions, weight, normalization)`, with one
    /// prediction per replicate. With [`ReplicateSummary::Crps`] the loss is
    /// the weighted mean of the continuous ranked probability scores,
    /// `Σ w·crps / Σ w`, whatever the configured loss; with any other summary
    /// it is this loss of the residuals `(observed - summary) × normalization`.
    /// The predictions are reordered.
    pub fn replicate_loss<'a>(
        self,
        summary: ReplicateSummary,
        observations: impl IntoIterator<Item = (f64, &'a mut [f64], f64, f64)>,
    ) -> f64 {
        match summary {
            ReplicateSummary::Crps => {
                let (sum, weight_sum) = observations.into_iter().fold(
                    (0.0, 0.0),
                    |(sum, weight_sum), (observed, predictions, weight, normalization)| {
                        let score = crps(predictions, observed) * normalization;
                        (sum + weight * score, weight_sum + weight)
                    },
                );
                if weight_sum > 0.0 {
                    sum / weight_sum
                } else {
                    0.0
                }
            }
            summary => self.aggregate(observations.into_iter().map(
                |(observed, predictions, weight, normalization)| {
                    (
                        (observed - summary.summarize(predictions)) * normalization,
                        weight,
                    )
                },
            )),
        }
    }
}

/// What the observations are compared with when each evaluation runs several
/// replicates of a stochastic engine. The user chooses it; there is no
/// default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReplicateSummary {
    /// The mean of the replicates.
    Mean,
    /// The median of the replicates: [`Self::Quantile`] at 0.5.
    Median,
    /// A quantile of the replicates at a level in `[0, 1]`: the one quantile
    /// of commol ([`statistics::quantiles`] with equal weights, NumPy's
    /// `method="hazen"`).
    Quantile(f64),
    /// The empirical distribution of the replicates, scored by the
    /// continuous ranked probability score ([`crps`]).
    Crps,
}

impl ReplicateSummary {
    /// The summary of a name: `"mean"`, `"median"`, `"crps"`, or
    /// `"quantile"` with its `level`.
    ///
    /// # Errors
    ///
    /// Returns a message that lists the names when `name` is none of them, a
    /// quantile has no level, or another summary has one.
    pub fn from_name(name: &str, level: Option<f64>) -> Result<Self, String> {
        match (name, level) {
            ("mean", None) => Ok(Self::Mean),
            ("median", None) => Ok(Self::Median),
            ("crps", None) => Ok(Self::Crps),
            ("quantile", Some(level)) => Ok(Self::Quantile(level)),
            _ => Err(format!(
                "Unknown replicate summary '{name}' with quantile {level:?}; the summaries are \
                 'mean', 'median', 'crps' and 'quantile' with a level"
            )),
        }
    }

    /// The summary of a set of replicate values. It is NaN when a value is
    /// NaN.
    ///
    /// # Panics
    ///
    /// Panics on [`Self::Crps`], which scores a distribution instead of
    /// summarizing it, and when `values` is empty.
    pub fn summarize(self, values: &[f64]) -> f64 {
        assert!(!values.is_empty(), "a summary of no replicates");
        match self {
            Self::Mean => values.iter().sum::<f64>() / values.len() as f64,
            Self::Median => quantile(values, 0.5),
            Self::Quantile(level) => quantile(values, level),
            Self::Crps => panic!("the CRPS scores a distribution; use `crps`"),
        }
    }
}

/// The quantile `level` of equally weighted `values`, NaN when a value is
/// NaN.
fn quantile(values: &[f64], level: f64) -> f64 {
    let weights = vec![1.0; values.len()];
    statistics::quantiles(values, &weights, &[level]).map_or(f64::NAN, |q| q[0])
}

/// The continuous ranked probability score of the empirical distribution of
/// `predictions` at `observed`: `E|X - y| - ½ E|X - X'|`, with `X` and `X'`
/// drawn from the predictions. A proper scoring rule; with one prediction it
/// is the absolute error. The predictions are sorted.
pub fn crps(predictions: &mut [f64], observed: f64) -> f64 {
    predictions.sort_by(f64::total_cmp);
    let weight = 1.0 / predictions.len().max(1) as f64;
    sorted_crps(predictions.iter().map(|&value| (value, weight)), observed)
}

/// The continuous ranked probability score of a weighted sample at
/// `observed`: `Σ wᵢ|xᵢ - y| - ½ Σᵢⱼ wᵢwⱼ|xᵢ - xⱼ|`. The weights sum to 1;
/// with equal weights it is [`crps`].
///
/// # Panics
///
/// Panics when `values` and `weights` differ in length.
pub fn weighted_crps(values: &[f64], weights: &[f64], observed: f64) -> f64 {
    assert_eq!(values.len(), weights.len(), "one weight per value");
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    sorted_crps(order.into_iter().map(|i| (values[i], weights[i])), observed)
}

/// The CRPS of `(value, weight)` pairs sorted by value, whose weights sum
/// to 1. For sorted values, `Σᵢⱼ wᵢwⱼ|xᵢ - xⱼ| = 2 Σᵢ wᵢxᵢ (Cᵢ₋₁ - (1 - Cᵢ))`
/// with `Cᵢ` the cumulative weight.
fn sorted_crps(pairs: impl Iterator<Item = (f64, f64)>, observed: f64) -> f64 {
    let mut to_observed = 0.0;
    let mut pairs_sum = 0.0;
    let mut below = 0.0;
    let mut total = 0.0;
    let collected: Vec<(f64, f64)> = pairs.collect();
    for &(_, weight) in &collected {
        total += weight;
    }
    for (value, weight) in collected {
        to_observed += weight * (value - observed).abs();
        let above = total - below - weight;
        pairs_sum += 2.0 * weight * value * (below - above);
        below += weight;
    }
    to_observed - 0.5 * pairs_sum
}

/// The replicates of each evaluation of a stochastic engine: common random
/// numbers, so the loss is a deterministic function of the parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Replicates {
    /// The seed of each replicate, the same at every evaluation.
    pub seeds: Vec<u64>,
    /// What the observations are compared with.
    pub summary: ReplicateSummary,
}

impl Replicates {
    /// `count` replicates seeded from `seed`
    /// ([`commol_random::replicate_seed`]).
    ///
    /// # Errors
    ///
    /// Returns a message when `count` is 0, or the message of
    /// [`statistics::check_levels`] when a quantile level is outside
    /// `[0, 1]`.
    pub fn new(count: usize, seed: u64, summary: ReplicateSummary) -> Result<Self, String> {
        if count == 0 {
            return Err("A stochastic engine needs at least one replicate".to_string());
        }
        if let ReplicateSummary::Quantile(level) = summary {
            statistics::check_levels(&[level])?;
        }
        Ok(Self {
            seeds: (0..count as u64)
                .map(|index| commol_random::replicate_seed(seed, index))
                .collect(),
            summary,
        })
    }
}

/// A constraint on calibration parameters and/or compartment values defined as a mathematical expression
///
/// Constraints are mathematical expressions that must evaluate to >= 0 for the
/// constraint to be satisfied. When the expression evaluates to < 0, the constraint
/// is violated and a penalty is applied.
///
/// # Constraint Types
///
/// **Parameter-only constraints** (`at: None`):
/// - Evaluated once before simulation starts
/// - Can only reference calibration parameter IDs
/// - Use for relationships between parameters
///
/// **Constraints at positions** (`at: Some(vec)`):
/// - Evaluated at the given positions during simulation
/// - Can reference both parameter IDs AND compartment names
/// - Use for constraining compartment values or dynamic relationships
///
/// # Examples
///
/// ## Parameter-only constraints:
/// ```rust,ignore
/// // k1/k2 must be <= 5
/// CalibrationConstraint::new(
///     "ratio_bound".to_string(),
///     "5.0 - k1/k2".to_string(),
/// )
///
/// // k1 must be >= k2
/// CalibrationConstraint::new(
///     "ordering".to_string(),
///     "k1 - k2".to_string(),
/// )
///
/// // Sum of parameters <= 1.0
/// CalibrationConstraint::new(
///     "sum_bound".to_string(),
///     "1.0 - (param1 + param2 + param3)".to_string(),
/// )
/// ```
///
/// ## Constraints at positions (output values):
/// ```rust,ignore
/// // B must stay <= 500 at specific positions
/// CalibrationConstraint::new(
///     "max_b".to_string(),
///     "500.0 - B".to_string(),
/// )
/// .with_positions(vec![10.0, 20.0, 30.0, 40.0])
///
/// // The total must stay <= 1000
/// CalibrationConstraint::new(
///     "total_bound".to_string(),
///     "1000.0 - (A + B + C)".to_string(),
/// )
/// .with_positions(vec![0.0, 5.0, 10.0, 15.0, 20.0])
///
/// // Ratio of compartments with parameter reference
/// CalibrationConstraint::new(
///     "ratio_bound".to_string(),
///     "threshold - B/A".to_string(),
/// )
/// .with_positions(vec![10.0, 20.0, 30.0])
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConstraint {
    /// Unique identifier for this constraint
    pub id: String,

    /// Mathematical expression that must evaluate >= 0 for constraint satisfaction
    ///
    /// The expression can reference calibration parameters by their IDs.
    /// When `at` is specified, the expression can also reference output
    /// values (A, B, C, etc.) at those positions.
    ///
    /// Examples:
    /// - "5.0 - k1/k2" → k1/k2 must be <= 5
    /// - "k1 - k2" → k1 >= k2
    /// - "1.0 - (param1 + param2)" → param1 + param2 <= 1.0
    /// - "500.0 - B" (with `at`) → B <= 500 at those positions
    pub expression: String,

    /// Penalty weight multiplier (default 1.0)
    ///
    /// Higher values make this constraint more important relative to others.
    /// The violation of this constraint is `weight × (−value)` when its
    /// expression evaluates to `value < 0`.
    pub weight: f64,

    /// Optional positions on the axis at which to evaluate this constraint
    ///
    /// - If None: Constraint is evaluated once before simulation using parameter values only
    /// - If Some(vec): Constraint is evaluated at each position during simulation,
    ///   and can reference both parameters and output values
    pub at: Option<Vec<f64>>,
}

impl CalibrationConstraint {
    /// Create a new calibration constraint
    pub fn new(id: String, expression: String) -> Self {
        Self {
            id,
            expression,
            weight: 1.0,
            at: None,
        }
    }

    /// Set the weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Set the positions at which the constraint is evaluated
    pub fn with_positions(mut self, at: Vec<f64>) -> Self {
        self.at = Some(at);
        self
    }
}

/// Result from a calibration run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Best parameter values found
    pub best_parameters: Vec<f64>,

    /// Parameter names (in same order as best_parameters)
    pub parameter_names: Vec<String>,

    /// Final loss value achieved
    pub final_loss: f64,

    /// Number of iterations performed
    pub iterations: usize,

    /// Whether the optimization converged
    pub converged: bool,

    /// Termination reason
    pub termination_reason: String,
}

/// A single evaluation of a feasible point during an optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationEvaluation {
    /// Parameter values the objective scored, in problem order
    pub parameters: Vec<f64>,

    /// Loss value for this parameter set
    pub loss: f64,
}

/// Result of an optimization with every evaluation of a feasible point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResultWithHistory {
    /// The result of the optimization
    pub result: CalibrationResult,

    /// Every evaluation of a feasible point, in the order they were made
    pub evaluations: Vec<CalibrationEvaluation>,
}

#[cfg(test)]
mod loss_tests {
    use super::LossConfig;

    // Residuals 3 and -1 with weights 2 and 0.5.
    const WEIGHTED: [(f64, f64); 2] = [(3.0, 2.0), (-1.0, 0.5)];
    const UNWEIGHTED: [(f64, f64); 2] = [(3.0, 1.0), (-1.0, 1.0)];

    #[test]
    fn sse_multiplies_each_squared_residual_by_its_weight() {
        let loss = LossConfig::SumSquaredError.aggregate(WEIGHTED);
        assert!((loss - (2.0 * 9.0 + 0.5 * 1.0)).abs() < 1e-12);
    }

    #[test]
    fn rmse_is_the_weighted_root_mean_square() {
        let loss = LossConfig::RootMeanSquaredError.aggregate(WEIGHTED);
        assert!((loss - (18.5_f64 / 2.5).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn mae_is_the_weighted_mean_absolute_residual() {
        let loss = LossConfig::MeanAbsoluteError.aggregate(WEIGHTED);
        assert!((loss - (2.0 * 3.0 + 0.5 * 1.0) / 2.5).abs() < 1e-12);
    }

    #[test]
    fn unit_weights_give_the_unweighted_losses() {
        assert!((LossConfig::SumSquaredError.aggregate(UNWEIGHTED) - 10.0).abs() < 1e-12);
        assert!(
            (LossConfig::RootMeanSquaredError.aggregate(UNWEIGHTED) - 5.0_f64.sqrt()).abs() < 1e-12
        );
        assert!((LossConfig::MeanAbsoluteError.aggregate(UNWEIGHTED) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn mean_losses_of_zero_total_weight_are_zero() {
        assert_eq!(
            LossConfig::RootMeanSquaredError.aggregate([(3.0, 0.0)]),
            0.0
        );
        assert_eq!(LossConfig::MeanAbsoluteError.aggregate([]), 0.0);
    }
}

#[cfg(test)]
mod replicate_tests {
    use super::{LossConfig, ReplicateSummary, Replicates, crps, weighted_crps};

    #[test]
    fn the_crps_of_one_replicate_is_the_absolute_error() {
        assert_eq!(crps(&mut [7.0], 4.5), 2.5);
        assert_eq!(crps(&mut [-1.0], 2.0), 3.0);
    }

    #[test]
    fn the_crps_of_a_known_ensemble_is_its_closed_form() {
        let values: [f64; 4] = [1.0, 4.0, 2.0, 8.0];
        let observed: f64 = 3.0;
        let n = values.len() as f64;
        let to_observed: f64 = values.iter().map(|x| (x - observed).abs()).sum::<f64>() / n;
        let between: f64 = values
            .iter()
            .flat_map(|x| values.iter().map(move |y| (x - y).abs()))
            .sum::<f64>()
            / (n * n);
        let expected = to_observed - 0.5 * between;

        let mut predictions = values;
        assert!((crps(&mut predictions, observed) - expected).abs() < 1e-12);
        assert!((expected - 0.8125).abs() < 1e-12);
    }

    #[test]
    fn the_weighted_crps_of_equal_weights_is_the_crps() {
        let values = [1.0, 4.0, 2.0, 8.0];
        assert!((weighted_crps(&values, &[0.25; 4], 3.0) - 0.8125).abs() < 1e-12);
        // A member of twice the weight counts as two equal members.
        let doubled = weighted_crps(&[1.0, 4.0], &[2.0 / 3.0, 1.0 / 3.0], 3.0);
        assert!((doubled - crps(&mut [1.0, 1.0, 4.0], 3.0)).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "the CRPS scores a distribution")]
    fn the_crps_is_not_a_summary() {
        ReplicateSummary::Crps.summarize(&[1.0, 2.0]);
    }

    #[test]
    fn summaries_reduce_the_replicates() {
        let values = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(ReplicateSummary::Mean.summarize(&values), 3.0);
        assert_eq!(ReplicateSummary::Median.summarize(&values), 3.0);
        assert_eq!(ReplicateSummary::Quantile(0.25).summarize(&values), 1.75);
        assert_eq!(ReplicateSummary::Quantile(0.1).summarize(&values), 1.0);
        assert_eq!(ReplicateSummary::Quantile(1.0).summarize(&values), 5.0);
    }

    #[test]
    fn a_replicate_that_is_not_a_number_gives_no_summary() {
        let values = [1.0, f64::NAN, 3.0];
        for summary in [
            ReplicateSummary::Mean,
            ReplicateSummary::Median,
            ReplicateSummary::Quantile(0.9),
        ] {
            assert!(summary.summarize(&values).is_nan(), "{summary:?}");
        }
    }

    #[test]
    fn a_replicate_loss_compares_the_summary_with_each_observation() {
        let mut first = [1.0, 3.0];
        let mut second = [10.0, 14.0];
        let loss = LossConfig::SumSquaredError.replicate_loss(
            ReplicateSummary::Mean,
            [
                (4.0, first.as_mut_slice(), 2.0, 3.0),
                (12.0, second.as_mut_slice(), 1.0, 1.0),
            ],
        );
        // Means 2 and 12: residuals 2 × 3 and 0, weighted 2 and 1.
        assert_eq!(loss, 72.0);
    }

    #[test]
    fn the_crps_loss_is_the_weighted_mean_score() {
        let mut first = [7.0];
        let mut second = [1.0, 4.0, 2.0, 8.0];
        let loss = LossConfig::SumSquaredError.replicate_loss(
            ReplicateSummary::Crps,
            [
                (4.5, first.as_mut_slice(), 1.0, 1.0),
                (3.0, second.as_mut_slice(), 3.0, 2.0),
            ],
        );
        assert!((loss - (2.5 + 3.0 * 2.0 * 0.8125) / 4.0).abs() < 1e-12);
        let mut only = [2.0];
        let unweighted = LossConfig::SumSquaredError.replicate_loss(
            ReplicateSummary::Crps,
            [(1.0, only.as_mut_slice(), 0.0, 1.0)],
        );
        assert_eq!(unweighted, 0.0);
    }

    #[test]
    fn a_summary_is_named_and_only_a_quantile_has_a_level() {
        assert_eq!(
            ReplicateSummary::from_name("quantile", Some(0.9)),
            Ok(ReplicateSummary::Quantile(0.9))
        );
        assert_eq!(
            ReplicateSummary::from_name("crps", None),
            Ok(ReplicateSummary::Crps)
        );
        for (name, level) in [("mean", Some(0.5)), ("quantile", None), ("mode", None)] {
            assert!(
                ReplicateSummary::from_name(name, level)
                    .unwrap_err()
                    .starts_with(&format!("Unknown replicate summary '{name}'")),
                "{name}"
            );
        }
    }

    #[test]
    fn replicates_take_fixed_seeds_and_reject_bad_input() {
        let replicates = Replicates::new(3, 9, ReplicateSummary::Median).unwrap();
        assert_eq!(replicates.seeds.len(), 3);
        assert_eq!(
            replicates.seeds,
            Replicates::new(3, 9, ReplicateSummary::Mean).unwrap().seeds
        );
        assert_eq!(replicates.seeds[1], commol_random::replicate_seed(9, 1));
        assert!(Replicates::new(0, 9, ReplicateSummary::Mean).is_err());
        assert!(Replicates::new(2, 9, ReplicateSummary::Quantile(1.5)).is_err());
    }
}
