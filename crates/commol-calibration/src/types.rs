//! Type definitions for calibration

use serde::{Deserialize, Serialize};

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

    /// Create an observation of one output with a custom weight
    pub fn with_weight(at: f64, output: String, value: f64, weight: f64) -> Self {
        Self {
            weight,
            ..Self::new(at, output, value)
        }
    }

    /// Create an observation of one output with a factor parameter
    pub fn with_factor(at: f64, output: String, value: f64, factor_id: String) -> Self {
        Self {
            factor_id: Some(factor_id),
            ..Self::new(at, output, value)
        }
    }

    /// Create an observation of one output with both weight and factor
    pub fn with_weight_and_factor(
        at: f64,
        output: String,
        value: f64,
        weight: f64,
        factor_id: String,
    ) -> Self {
        Self {
            weight,
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

    /// Check if a value is within the parameter bounds
    pub fn is_within_bounds(&self, value: f64) -> bool {
        value >= self.min_bound && value <= self.max_bound
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
    /// The median of the replicates.
    Median,
    /// A quantile of the replicates, in `[0, 1]`, linearly interpolated
    /// between order statistics.
    Quantile(f64),
    /// The empirical distribution of the replicates, scored by the
    /// continuous ranked probability score ([`crps`]).
    Crps,
}

impl ReplicateSummary {
    /// The summary of a set of replicate values, which are reordered.
    ///
    /// # Panics
    ///
    /// Panics on [`Self::Crps`], which scores a distribution instead of
    /// summarizing it, and when `values` is empty.
    pub fn summarize(self, values: &mut [f64]) -> f64 {
        assert!(!values.is_empty(), "a summary of no replicates");
        match self {
            Self::Mean => values.iter().sum::<f64>() / values.len() as f64,
            Self::Median => quantile(values, 0.5),
            Self::Quantile(level) => quantile(values, level),
            Self::Crps => panic!("the CRPS scores a distribution; use `crps`"),
        }
    }
}

/// The quantile `level` of `values`, with linear interpolation between order
/// statistics (the default of NumPy). The values are sorted.
fn quantile(values: &mut [f64], level: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let position = (values.len() - 1) as f64 * level.clamp(0.0, 1.0);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    values[lower] + (position - lower as f64) * (values[upper] - values[lower])
}

/// The continuous ranked probability score of the empirical distribution of
/// `predictions` at `observed`: `E|X - y| - ½ E|X - X'|`, with `X` and `X'`
/// drawn from the predictions. A proper scoring rule; with one prediction it
/// is the absolute error. The predictions are sorted.
pub fn crps(predictions: &mut [f64], observed: f64) -> f64 {
    let n = predictions.len();
    if n == 0 {
        return 0.0;
    }
    predictions.sort_by(f64::total_cmp);
    let count = n as f64;
    let to_observed = predictions
        .iter()
        .map(|value| (value - observed).abs())
        .sum::<f64>()
        / count;
    // Σ_ij |x_i - x_j| = 2 Σ_i (2i - n + 1) x_(i) for the sorted values.
    let pairs: f64 = predictions
        .iter()
        .enumerate()
        .map(|(index, value)| (2.0 * index as f64 - count + 1.0) * value)
        .sum::<f64>()
        * 2.0;
    to_observed - 0.5 * pairs / (count * count)
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
    /// Returns a message when `count` is 0 or a quantile level is outside
    /// `[0, 1]`.
    pub fn new(count: usize, seed: u64, summary: ReplicateSummary) -> Result<Self, String> {
        if count == 0 {
            return Err("A stochastic calibration needs at least one replicate".to_string());
        }
        if let ReplicateSummary::Quantile(level) = summary {
            if !(0.0..=1.0).contains(&level) {
                return Err(format!("A quantile level is between 0 and 1, got {level}"));
            }
        }
        Ok(Self {
            seeds: (0..count as u64)
                .map(|index| commol_random::replicate_seed(seed, index))
                .collect(),
            summary,
        })
    }
}

impl std::fmt::Display for LossConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LossConfig::SumSquaredError => write!(f, "Sum Squared Error"),
            LossConfig::RootMeanSquaredError => write!(f, "Root Mean Squared Error"),
            LossConfig::MeanAbsoluteError => write!(f, "Mean Absolute Error"),
        }
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

    /// Human-readable description (optional, for diagnostics)
    pub description: Option<String>,

    /// Penalty weight multiplier (default 1.0)
    ///
    /// Higher values make this constraint more important relative to others.
    /// The penalty for violating this constraint is: weight * violation^2
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
            description: None,
            weight: 1.0,
            at: None,
        }
    }

    /// Set the description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
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

    /// Check if this is a parameter-only constraint
    pub fn is_parameter_only(&self) -> bool {
        self.at.is_none()
    }

    /// Check if this constraint is evaluated at positions of a run
    pub fn has_positions(&self) -> bool {
        self.at.is_some()
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

impl CalibrationResult {
    /// Get parameters as a HashMap for easy lookup
    pub fn parameters_map(&self) -> std::collections::HashMap<String, f64> {
        self.parameter_names
            .iter()
            .zip(self.best_parameters.iter())
            .map(|(name, value)| (name.clone(), *value))
            .collect()
    }
}

/// A single evaluation during calibration optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationEvaluation {
    /// Parameter values for this evaluation
    pub parameters: Vec<f64>,

    /// Loss value for this parameter set
    pub loss: f64,

    /// Model predictions at each position for all compartments
    /// `predictions[position_idx][compartment_idx]`
    pub predictions: Vec<Vec<f64>>,
}

/// Result from a calibration run with history of all evaluations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResultWithHistory {
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

    /// All unique evaluations performed during optimization (deduplicated)
    pub evaluations: Vec<CalibrationEvaluation>,
}

impl CalibrationResultWithHistory {
    /// Convert to a standard CalibrationResult (without history)
    pub fn to_standard_result(&self) -> CalibrationResult {
        CalibrationResult {
            best_parameters: self.best_parameters.clone(),
            parameter_names: self.parameter_names.clone(),
            final_loss: self.final_loss,
            iterations: self.iterations,
            converged: self.converged,
            termination_reason: self.termination_reason.clone(),
        }
    }

    /// Get parameters as a HashMap for easy lookup
    pub fn parameters_map(&self) -> std::collections::HashMap<String, f64> {
        self.parameter_names
            .iter()
            .zip(self.best_parameters.iter())
            .map(|(name, value)| (name.clone(), *value))
            .collect()
    }
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
    use super::{crps, LossConfig, ReplicateSummary, Replicates};

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
    fn summaries_reduce_the_replicates() {
        let values = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(ReplicateSummary::Mean.summarize(&mut values.clone()), 3.0);
        assert_eq!(ReplicateSummary::Median.summarize(&mut values.clone()), 3.0);
        assert_eq!(
            ReplicateSummary::Quantile(0.25).summarize(&mut values.clone()),
            2.0
        );
        assert_eq!(
            ReplicateSummary::Quantile(0.1).summarize(&mut values.clone()),
            1.4
        );
        assert_eq!(
            ReplicateSummary::Quantile(1.0).summarize(&mut values.clone()),
            5.0
        );
    }

    #[test]
    fn a_replicate_loss_compares_the_summary_with_each_observation() {
        let mut first = [1.0, 3.0];
        let mut second = [10.0, 14.0];
        let loss = LossConfig::SumSquaredError.replicate_loss(
            ReplicateSummary::Mean,
            [
                (4.0, first.as_mut_slice(), 2.0, 1.0),
                (12.0, second.as_mut_slice(), 1.0, 1.0),
            ],
        );
        // Means 2 and 12: residuals 2 and 0, weighted 2 and 1.
        assert_eq!(loss, 8.0);
    }

    #[test]
    fn the_crps_loss_is_the_weighted_mean_score() {
        let mut first = [7.0];
        let mut second = [1.0, 4.0, 2.0, 8.0];
        let loss = LossConfig::SumSquaredError.replicate_loss(
            ReplicateSummary::Crps,
            [
                (4.5, first.as_mut_slice(), 1.0, 1.0),
                (3.0, second.as_mut_slice(), 3.0, 1.0),
            ],
        );
        assert!((loss - (2.5 + 3.0 * 0.8125) / 4.0).abs() < 1e-12);
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
