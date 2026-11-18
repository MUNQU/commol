//! Python bindings of the dependence analyses of `commol-inference`: of any
//! values, one per observation, and of the residuals of a predictive of a
//! calibration problem.

use commol_calibration::observed_data;
use commol_inference::dependence::{
    self, DrawPValues, MaxLag, ResidualKind, ResidualSettings, SeriesDependence,
};
use commol_inference::ranges::{MAX_LAG, RANDOMIZATIONS, REPLICATES};
use commol_inference::settings::dependence::Reference;
use commol_inference::settings::resampling::NoiseSettings;
use pyo3::prelude::*;

use crate::inference::{PyCalibrationProblem, PyParameterSample, method_error, value_error};
use crate::parallelism::{parallel, parallel_logged};
use crate::records::StepLog;

/// Read `max_lag`, a whole number or `"auto"`, with the rule of its setting.
///
/// # Errors
///
/// A `ValueError` for a text other than `"auto"` or a number outside the
/// rule, and a `TypeError` for another type.
fn max_lag(value: &Bound<'_, PyAny>) -> PyResult<MaxLag> {
    if let Ok(text) = value.extract::<String>() {
        return if text == "auto" {
            Ok(MaxLag::Auto)
        } else {
            Err(value_error(format!(
                "max_lag is a whole number of at least 1 or \"auto\", got '{text}'"
            )))
        };
    }
    let lags: i64 = value.extract()?;
    crate::counts::count(&MAX_LAG, lags).map(MaxLag::Lags)
}

/// The spread of the p-values of the draws, as `(median, share_below)`.
fn spread(values: Option<DrawPValues>) -> Option<(f64, f64)> {
    values.map(|v| (v.median, v.share_below))
}

/// The dependence of one series
#[pyclass(name = "SeriesStatistics", frozen, get_all)]
pub struct PySeriesStatistics {
    /// The label of the series
    label: String,
    /// The values it reads
    values: usize,
    /// Why it has no statistics
    reason: Option<String>,
    /// The largest lag
    max_lag: usize,
    /// The autocorrelation at each lag
    correlations: Vec<Option<f64>>,
    /// The pairs each lag reads
    pairs: Vec<usize>,
    /// The robust portmanteau
    portmanteau: Option<f64>,
    /// The lags in its sum
    degrees_of_freedom: usize,
    /// Its p-value, with one draw
    p_value: Option<f64>,
    /// The median and the share below the level of the p-values of the
    /// draws, with many
    draw_p_values: Option<(f64, f64)>,
    /// The integrated autocorrelation time
    tau: Option<f64>,
    /// Why it is undefined
    tau_reason: Option<String>,
    /// Its band under independence
    tau_band: Option<(f64, f64)>,
    /// The largest lag in its sum
    cut_lag: Option<usize>,
}

/// A dependence analysis
#[pyclass(name = "SeriesDependence", frozen, get_all)]
pub struct PySeriesDependence {
    /// Each series
    series: Vec<Py<PySeriesStatistics>>,
    /// `(first, second, correlation)` of each pair of series
    correlations: Vec<(String, String, Option<f64>)>,
    /// `Var(Σz) / ΣVar(z)`
    variance_factor: Option<f64>,
    /// `τ` of `Σz`
    tau: Option<f64>,
    /// Why it is undefined
    tau_reason: Option<String>,
    /// `LRV(Σz) / ΣVar(z)`
    information_factor: Option<f64>,
    /// Its p-value, with one draw
    p_value: Option<f64>,
    /// The spread of the p-values of the draws, with many
    draw_p_values: Option<(f64, f64)>,
    /// The draws of the sample
    draws: usize,
    /// Each replicate of the reference that failed
    failed: Vec<String>,
    /// The refits kept although their optimizer did not converge
    not_converged: usize,
}

/// The analysis as Python reads it.
fn python(py: Python<'_>, analysis: SeriesDependence) -> PyResult<PySeriesDependence> {
    let series = analysis
        .series
        .into_iter()
        .map(|s| {
            Py::new(
                py,
                PySeriesStatistics {
                    label: s.label,
                    values: s.values,
                    reason: s.reason,
                    max_lag: s.max_lag,
                    correlations: s.correlations,
                    pairs: s.pairs,
                    portmanteau: s.portmanteau,
                    degrees_of_freedom: s.degrees_of_freedom,
                    p_value: s.p_value,
                    draw_p_values: spread(s.draw_p_values),
                    tau: s.tau,
                    tau_reason: s.tau_reason,
                    tau_band: s.tau_band,
                    cut_lag: s.cut_lag,
                },
            )
        })
        .collect::<PyResult<_>>()?;
    let problem = analysis.problem;
    Ok(PySeriesDependence {
        series,
        correlations: analysis
            .correlations
            .into_iter()
            .map(|c| (c.first, c.second, c.correlation))
            .collect(),
        variance_factor: problem.variance_factor,
        tau: problem.tau,
        tau_reason: problem.tau_reason,
        information_factor: problem.information_factor,
        p_value: problem.p_value,
        draw_p_values: spread(problem.draw_p_values),
        draws: analysis.draws,
        failed: commol_inference::resampling::FitUnit::Replicate.entries(&analysis.failed),
        not_converged: analysis.not_converged,
    })
}

/// The dependence of `values`, one per observation, with a wild band of
/// `replicates` sets whose signs come from `seed`
///
/// Args:
///     observations: The JSON of the observations
///     values: One value per observation
///     max_lag: The largest lag, or "auto"
///     replicates: The wild sets of the band
///     confidence_level: The level of the band
///     seed: The seed of the signs
///
/// Raises:
///     ValueError: If a setting is outside its range, or the values are not
///         one finite value per observation
#[pyfunction]
fn series_dependence(
    py: Python<'_>,
    observations: &str,
    values: Vec<f64>,
    max_lag: &Bound<'_, PyAny>,
    replicates: i64,
    confidence_level: f64,
    seed: i128,
) -> PyResult<PySeriesDependence> {
    let observations = observed_data(&crate::json::read::<Vec<_>>(observations)?);
    let max_lag = self::max_lag(max_lag)?;
    let replicates: usize = crate::counts::count(&REPLICATES, replicates)?;
    let seed: u64 = crate::counts::count(&commol_core::SEED, seed)?;
    let analysis = parallel(py, || {
        dependence::series_dependence(
            &observations,
            &values,
            max_lag,
            replicates,
            confidence_level,
            seed,
        )
    })?
    .map_err(method_error)?;
    python(py, analysis)
}

/// The dependence of the residuals of `sample` at `observations`, against
/// `reference`; every random number comes from the seed of the problem
///
/// Args:
///     problem: The calibration problem
///     sample: The draws
///     observations: The JSON of the observations
///     noise: The JSON of `DistributionNoise` (quantile residuals) or
///         `NoNoise` (additive residuals)
///     max_lag: The largest lag, or "auto"
///     randomizations: The randomizations of the quantile residuals
///     reference: The JSON of `AtParameters` or `Refitted`
///     confidence_level: The level of the band of `τ`
///     logger: The logger of a record of each refit that finishes
///
/// Raises:
///     TypeError: If the noise is `ResidualNoise`
///     ValueError: If a setting is outside its range, the sample's
///         parameters are not the problem's, the residuals do not fit the
///         problem or the reference does not fit the sample or the
///         observations
///     InfeasibleStateError: If a draw leaves the values its model allows
///     RuntimeError: If every refit fails
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn residual_dependence(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    sample: &PyParameterSample,
    observations: &str,
    noise: &str,
    max_lag: &Bound<'_, PyAny>,
    randomizations: i64,
    reference: &str,
    confidence_level: f64,
    logger: Py<PyAny>,
) -> PyResult<PySeriesDependence> {
    let observations = observed_data(&crate::json::read::<Vec<_>>(observations)?);
    let kind = match crate::json::read::<NoiseSettings>(noise)? {
        NoiseSettings::Distribution => ResidualKind::Quantile,
        NoiseSettings::None => ResidualKind::Additive,
        NoiseSettings::Residual(_) => {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "A dependence analysis reads the residuals of the predictive itself: \
                 DistributionNoise() for the quantile residuals of observation models, or \
                 NoNoise() for the additive residuals of a loss; ResidualNoise(...) adds the \
                 residuals of a fit as noise",
            ));
        }
    };
    let settings = ResidualSettings {
        max_lag: self::max_lag(max_lag)?,
        randomizations: crate::counts::count(&RANDOMIZATIONS, randomizations)?,
        reference: crate::json::read::<Reference>(reference)?,
        confidence_level,
    };
    let log = StepLog::new(logger);
    let analysis = parallel_logged(py, &log, |steps| {
        dependence::residual_dependence(
            &problem.inner,
            problem.optimizer(),
            &sample.inner,
            &observations,
            kind,
            &settings,
            problem.seed,
            steps,
        )
    })?
    .map_err(method_error)?;
    python(py, analysis)
}

/// Register the dependence analyses with the inference module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySeriesStatistics>()?;
    m.add_class::<PySeriesDependence>()?;
    m.add_function(wrap_pyfunction!(series_dependence, m)?)?;
    m.add_function(wrap_pyfunction!(residual_dependence, m)?)?;
    Ok(())
}
