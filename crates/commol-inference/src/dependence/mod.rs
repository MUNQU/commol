//! Whether the values of each series of a problem are dependent, and how
//! strongly: the autocorrelation of each series, a portmanteau test robust
//! to heteroscedasticity, the integrated autocorrelation time `τ` with its
//! band under independence, and for the series together their correlations
//! at lag 0 and the factor by which the dependence multiplies the
//! information.
//!
//! | Module | Holds |
//! |---|---|
//! | `grid` | The series on the grid of their smallest spacing, and the positions they share |
//! | `sets` | The statistics of one set of values, and their mean over several sets |
//! | `law` | The tests and bands of a set against a reference |
//! | `of_values` | [`series_dependence`]: any values, one per observation, against a wild band |
//! | `of_residuals` | [`residual_dependence`]: the residuals of a predictive against a reference |
//!
//! The statistics of a series read its values on its grid
//! ([`commol_statistics::dependence`]): a series with two observations at
//! one position, or with a position off its grid, has none, with the
//! reason; an observation of weight 0, which a replicate of a resampler
//! can hold, is left out.

pub(crate) mod grid;
pub(crate) mod law;
pub(crate) mod of_residuals;
pub(crate) mod of_values;
pub(crate) mod sets;

pub use of_residuals::{ResidualKind, ResidualSettings, residual_dependence};
pub use of_values::series_dependence;

use crate::ranges::MAX_LAG;

/// The largest lag of each series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxLag {
    /// `min(10, n / 5)` lags, at least 1, for a series of `n` values.
    Auto,
    /// This many lags for every series.
    Lags(usize),
}

impl MaxLag {
    /// The lags of a series of `values` values.
    pub fn of(self, values: usize) -> usize {
        match self {
            Self::Auto => (values / 5).clamp(1, 10),
            Self::Lags(lags) => lags,
        }
    }

    /// Check the number of lags with its rule ([`MAX_LAG`]).
    ///
    /// # Errors
    ///
    /// Returns the message of the rule.
    pub fn check(self) -> Result<(), String> {
        match self {
            Self::Auto => Ok(()),
            Self::Lags(lags) => MAX_LAG.check(lags),
        }
    }
}

/// The spread of the p-values of the draws of a sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawPValues {
    /// The weighted median.
    pub median: f64,
    /// The weighted share at or below `1 − confidence_level`.
    pub share_below: f64,
}

/// The dependence of one series.
#[derive(Debug, Clone, PartialEq)]
pub struct SeriesStatistics {
    /// The label of the series.
    pub label: String,
    /// The values the series reads: its observations of weight above 0.
    pub values: usize,
    /// Why the series has no statistics.
    pub reason: Option<String>,
    /// The largest lag.
    pub max_lag: usize,
    /// The autocorrelation at lags `1 … max_lag`; None at a lag with no
    /// pair or of a constant series.
    pub correlations: Vec<Option<f64>>,
    /// The pairs of values each lag reads.
    pub pairs: Vec<usize>,
    /// The robust portmanteau `Q̃`.
    pub portmanteau: Option<f64>,
    /// The lags in its sum.
    pub degrees_of_freedom: usize,
    /// The p-value of `Q̃` under the reference, with one draw.
    pub p_value: Option<f64>,
    /// The spread of the p-values of `Q̃` of the draws, with many.
    pub draw_p_values: Option<DrawPValues>,
    /// The integrated autocorrelation time.
    pub tau: Option<f64>,
    /// Why `τ` is undefined.
    pub tau_reason: Option<String>,
    /// The central interval of `τ` under independence at the confidence
    /// level, with one draw; an undefined `τ` is `−∞`.
    pub tau_band: Option<(f64, f64)>,
    /// The largest lag in the sum of `τ`.
    pub cut_lag: Option<usize>,
}

/// The correlation at lag 0 of two series.
#[derive(Debug, Clone, PartialEq)]
pub struct SeriesCorrelation {
    /// The label of the first series.
    pub first: String,
    /// The label of the second series.
    pub second: String,
    /// The correlation over their shared positions.
    pub correlation: Option<f64>,
}

/// The dependence of the series together.
#[derive(Debug, Clone, PartialEq)]
pub struct ProblemDependence {
    /// `Var(Σz) / ΣVar(z)` over the standardized series `z`.
    pub variance_factor: Option<f64>,
    /// `τ` of `Σz`.
    pub tau: Option<f64>,
    /// Why `τ` of `Σz` is undefined.
    pub tau_reason: Option<String>,
    /// `LRV(Σz) / ΣVar(z)`, their product: the factor by which the
    /// dependence within and across series multiplies the information.
    pub information_factor: Option<f64>,
    /// The p-value of the information factor under the reference, with one
    /// draw.
    pub p_value: Option<f64>,
    /// The spread of the p-values of the draws, with many.
    pub draw_p_values: Option<DrawPValues>,
}

/// A dependence analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct SeriesDependence {
    /// Each series, in order of first appearance of its label.
    pub series: Vec<SeriesStatistics>,
    /// Each pair of series with statistics.
    pub correlations: Vec<SeriesCorrelation>,
    /// The series together.
    pub problem: ProblemDependence,
    /// The draws of the sample; 1 for values.
    pub draws: usize,
    /// The index and reason of each replicate of a reference that failed.
    pub failed: Vec<(usize, String)>,
    /// The refits kept although their optimizer did not converge.
    pub not_converged: usize,
}
