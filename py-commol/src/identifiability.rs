//! Python bindings of the identifiability analyses of `commol-inference`.
//!
//! Each function takes the problem built once by a `Calibrator`, the JSON
//! of its method ([`commol_inference::settings::identifiability`]) and the
//! handle of the result the method reads, and returns the analysis as plain
//! values; Python builds the report.

use crate::parallelism::parallel;
use commol_calibration::FitRecord;
use commol_core::SimulationEngine;
use commol_inference::identifiability::{
    self, CollinearityVerdict, Reference, SampleVerdict, StructuralVerdict, Subset,
};
use commol_inference::sensitivity::{Design, FiniteDifferences};
use commol_inference::settings::identifiability::{
    CollinearitySettings, RankDesign, Reference as ReferenceSettings,
    SampleIdentifiabilitySettings, StartsIdentifiabilitySettings, StructuralRankSettings,
};
use pyo3::prelude::*;

use crate::inference::{
    PyCalibrationProblem, PyParameterSample, method_error, python_list, value_error,
};

/// The methods that accept a stochastic engine, for the message of one that
/// does not.
const PRACTICAL_METHODS: &str = "SampleIdentifiability and StartsIdentifiability";

/// The finite differences of `problem` on `design`, for the method `method`,
/// which differentiates the model.
///
/// # Errors
///
/// A `ValueError` when the engine is stochastic or the design does not fit
/// the problem.
fn source(
    problem: &PyCalibrationProblem,
    design: Design,
    method: &str,
) -> PyResult<FiniteDifferences<crate::engine::Engine>> {
    if problem.inner.forward().base_engine().is_stochastic() {
        return Err(value_error(format!(
            "{method} differentiates the model, which needs a deterministic engine: the \
             runs of a stochastic engine change in jumps with its parameters. \
             {PRACTICAL_METHODS} accept a stochastic engine."
        )));
    }
    FiniteDifferences::new(&problem.inner, design).map_err(method_error)
}

/// A sample whose parameters are those of `problem`, in order.
///
/// # Errors
///
/// A `ValueError` that names both lists when they differ.
fn problem_sample<'a>(
    problem: &PyCalibrationProblem,
    sample: &'a PyParameterSample,
) -> PyResult<&'a commol_inference::ParameterSample> {
    let names = problem.names();
    if sample.inner.names() != names.as_slice() {
        return Err(value_error(format!(
            "The sample's parameters {} are not the problem's {}.",
            python_list(sample.inner.names()),
            python_list(&names)
        )));
    }
    Ok(&sample.inner)
}

/// The names of `names` that `values` has no value of.
fn missing(names: &[String], values: &commol_core::json::Pairs<f64>) -> Vec<String> {
    names
        .iter()
        .filter(|name| values.get(name).is_none())
        .cloned()
        .collect()
}

/// The values of `values` in the order of `names`, which it has all of.
fn ordered(names: &[String], values: &commol_core::json::Pairs<f64>) -> Vec<f64> {
    names
        .iter()
        .map(|name| *values.get(name).expect("checked by missing"))
        .collect()
}

/// The rank of the scaled sensitivities at one point
#[pyclass(name = "PointRank", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyPointRank {
    /// The parameter values of the point
    values: Vec<f64>,
    /// The singular values, largest first
    singular_values: Vec<f64>,
    /// The threshold a singular value must pass
    threshold: f64,
    /// How many singular values pass it
    rank: usize,
}

/// The structural rank of a problem
#[pyclass(name = "StructuralAnalysis", frozen, get_all)]
pub struct PyStructuralAnalysis {
    /// Each feasible point
    points: Vec<PyPointRank>,
    /// Why each infeasible point was skipped
    skipped: Vec<String>,
    /// The rank at most points
    generic_rank: usize,
    /// The projection of each parameter on the null space
    projections: Vec<f64>,
    /// "identifiable", "in_combination" or "non_identifiable", per parameter
    verdicts: Vec<String>,
    /// Each identifiable combination as `(parameter index, exponent)` pairs
    combinations: Vec<Vec<(usize, f64)>>,
}

/// The structural rank at the points of the settings, rows of a Latin
/// hypercube over the bounds drawn from the seed of the problem
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `StructuralRank`
///
/// Raises:
///     ValueError: If a setting is not valid, the engine is stochastic, or
///         every point is infeasible
#[pyfunction]
fn structural_rank(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<PyStructuralAnalysis> {
    let settings: StructuralRankSettings = crate::json::read(settings)?;
    let design = match settings.design {
        RankDesign::Observations => Design::Observations,
        RankDesign::Dense => Design::Dense,
    };
    let source = source(problem, design, "StructuralRank")?;
    let analysis = parallel(py, || {
        identifiability::structural_rank(&source, settings.points, problem.seed)
    })?
    .map_err(method_error)?;
    Ok(PyStructuralAnalysis {
        points: analysis
            .points
            .into_iter()
            .map(|point| PyPointRank {
                values: point.values,
                singular_values: point.singular_values,
                threshold: point.threshold,
                rank: point.rank,
            })
            .collect(),
        skipped: analysis.skipped,
        generic_rank: analysis.generic_rank,
        projections: analysis.projections,
        verdicts: analysis
            .verdicts
            .iter()
            .map(|verdict| {
                match verdict {
                    StructuralVerdict::Identifiable => "identifiable",
                    StructuralVerdict::InCombination => "in_combination",
                    StructuralVerdict::NonIdentifiable => "non_identifiable",
                }
                .to_string()
            })
            .collect(),
        combinations: analysis
            .combinations
            .into_iter()
            .map(|combination| combination.exponents)
            .collect(),
    })
}

/// The verdict of one parameter of a collinearity analysis and its causes
#[pyclass(name = "CollinearityVerdict", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyCollinearityVerdict {
    /// "determined" or "poorly_determined"
    verdict: String,
    /// No observation is sensitive to it
    insensitive: bool,
    /// The other parameters, by index, whose pair with it has an index at
    /// or above the threshold
    collinear_with: Vec<usize>,
}

impl From<CollinearityVerdict> for PyCollinearityVerdict {
    fn from(verdict: CollinearityVerdict) -> Self {
        Self {
            verdict: verdict.verdict.name().to_string(),
            insensitive: verdict.insensitive,
            collinear_with: verdict.collinear_with,
        }
    }
}

/// Brun's collinearity analysis of a problem
#[pyclass(name = "CollinearityAnalysis", frozen, get_all)]
pub struct PyCollinearityAnalysis {
    /// The weighted median of the sensitivity measure of each parameter
    sensitivity: Vec<f64>,
    /// The weighted median of the index of each pair
    pair_gamma: Vec<Vec<f64>>,
    /// The subset of lowest index of each size, as
    /// `(parameters, gamma, share below the threshold)`
    best_by_size: Vec<(Vec<usize>, f64, f64)>,
    /// The largest subsets whose index is below the threshold, as above
    admissible: Vec<(Vec<usize>, f64, f64)>,
    /// The verdict of each parameter
    verdicts: Vec<PyCollinearityVerdict>,
    /// How many points were used
    points: usize,
    /// Why each infeasible point was skipped
    skipped: Vec<String>,
}

fn subsets(subsets: Vec<Subset>) -> Vec<(Vec<usize>, f64, f64)> {
    subsets
        .into_iter()
        .map(|s| (s.parameters, s.gamma, s.share_below))
        .collect()
}

/// Where a collinearity analysis reads the model: the JSON of a
/// `CalibrationResult`, or a sample.
#[derive(FromPyObject)]
enum At<'py> {
    /// The draws of a sample, with their weights.
    Sample(PyRef<'py, PyParameterSample>),
    /// The best values of a fit.
    Fit(String),
}

/// Brun's indices at the fit or the draws of `at`
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `Collinearity`
///     at: The JSON of a `CalibrationResult`, or a sample
///
/// Raises:
///     ValueError: If a setting is not valid, the engine is stochastic, the
///         fit lacks a parameter or the sample's parameters are not the
///         problem's, or every point is infeasible
#[pyfunction]
fn collinearity(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
    at: At<'_>,
) -> PyResult<PyCollinearityAnalysis> {
    let settings: CollinearitySettings = crate::json::read(settings)?;
    let source = source(problem, Design::Observations, "Collinearity")?;
    let names = problem.names();
    let (points, weights) = match &at {
        At::Fit(fit) => {
            let fit: FitRecord = crate::json::read(fit)?;
            let missing = missing(&names, &fit.best_parameters);
            if !missing.is_empty() {
                return Err(value_error(format!(
                    "The result has no value of {}.",
                    python_list(&missing)
                )));
            }
            (vec![ordered(&names, &fit.best_parameters)], vec![1.0])
        }
        At::Sample(sample) => {
            let sample = problem_sample(problem, sample)?;
            (sample.values().to_vec(), sample.weights().to_vec())
        }
    };
    let analysis = parallel(py, || {
        identifiability::collinearity(
            &source,
            &points,
            &weights,
            settings.max_subset_size,
            settings.threshold,
        )
    })?
    .map_err(method_error)?;
    Ok(PyCollinearityAnalysis {
        sensitivity: analysis.sensitivity,
        pair_gamma: analysis.pair_gamma,
        best_by_size: subsets(analysis.best_by_size),
        admissible: subsets(analysis.admissible),
        verdicts: analysis.verdicts.into_iter().map(Into::into).collect(),
        points: analysis.points,
        skipped: analysis.skipped,
    })
}

/// The verdict of one parameter of a sample and its causes
#[pyclass(name = "SampleVerdict", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PySampleVerdict {
    /// "determined" or "poorly_determined"
    verdict: String,
    /// The other parameters, by index, correlated with it above the
    /// threshold, with the correlation
    correlated_with: Vec<(usize, f64)>,
    /// The number of modes of its marginal
    modes: usize,
}

impl From<SampleVerdict> for PySampleVerdict {
    fn from(verdict: SampleVerdict) -> Self {
        Self {
            verdict: verdict.verdict.name().to_string(),
            correlated_with: verdict.correlated_with,
            modes: verdict.modes,
        }
    }
}

/// The practical identifiability of a sample
#[pyclass(name = "SampleAnalysis", frozen, get_all)]
pub struct PySampleAnalysis {
    /// The contraction of each parameter against its reference
    contraction: Vec<f64>,
    /// The weighted correlation matrix
    correlation: Vec<Vec<f64>>,
    /// The pairs `(j, k, r)` whose absolute correlation is above the
    /// threshold
    correlated_pairs: Vec<(usize, usize, f64)>,
    /// The modes of each marginal
    modes: Vec<Vec<f64>>,
    /// The verdict of each parameter
    verdicts: Vec<PySampleVerdict>,
}

/// The contraction, correlations and modes of `sample`
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `SampleIdentifiability`
///     sample: The sample
///
/// Raises:
///     ValueError: If a setting is not valid, the sample's parameters are
///         not the problem's, or a prior is missing or does not fit its
///         bounds
#[pyfunction]
fn sample_identifiability(
    problem: &PyCalibrationProblem,
    settings: &str,
    sample: &PyParameterSample,
) -> PyResult<PySampleAnalysis> {
    let settings: SampleIdentifiabilitySettings = crate::json::read(settings)?;
    let sample = problem_sample(problem, sample)?;
    let priors = match (settings.reference, &settings.priors) {
        (ReferenceSettings::Prior, Some(priors)) => Some(
            priors
                .priors(sample.names(), sample.bounds())
                .map_err(value_error)?,
        ),
        _ => None,
    };
    let reference = match &priors {
        Some(priors) => Reference::Prior(priors),
        None => Reference::Bounds,
    };
    let analysis =
        identifiability::sample_identifiability(sample, reference, settings.correlation_threshold)
            .map_err(method_error)?;
    Ok(PySampleAnalysis {
        contraction: analysis.contraction,
        correlation: analysis.correlation,
        correlated_pairs: analysis.correlated_pairs,
        modes: analysis.modes,
        verdicts: analysis.verdicts.into_iter().map(Into::into).collect(),
    })
}

/// The comparison of the starts of a calibration
#[pyclass(name = "StartsAnalysis", frozen, get_all)]
pub struct PyStartsAnalysis {
    /// The converged starts within the loss tolerance of the best
    near_best: Vec<usize>,
    /// The largest difference of each parameter between them, as a share of
    /// its bound width
    spread: Vec<f64>,
    /// Whether that spread is above the parameter tolerance
    flagged: Vec<bool>,
}

/// Compare the ends of the starts of a fit that converged
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `StartsIdentifiability`
///     result: The JSON of the `CalibrationResult` whose starts are compared
///
/// Raises:
///     ValueError: If a setting is not valid, a start lacks a parameter, or
///         fewer than two starts converged
#[pyfunction]
fn starts_identifiability(
    problem: &PyCalibrationProblem,
    settings: &str,
    result: &str,
) -> PyResult<PyStartsAnalysis> {
    let settings: StartsIdentifiabilitySettings = crate::json::read(settings)?;
    let result: FitRecord = crate::json::read(result)?;
    let names = problem.names();
    let mut ends = Vec::with_capacity(result.starts.len());
    for start in &result.starts {
        let missing = missing(&names, &start.end);
        if !missing.is_empty() {
            return Err(value_error(format!(
                "A start has no value of {}.",
                python_list(&missing)
            )));
        }
        ends.push(ordered(&names, &start.end));
    }
    let losses: Vec<f64> = result.starts.iter().map(|start| start.loss).collect();
    let converged: Vec<bool> = result.starts.iter().map(|start| start.converged).collect();
    let analysis = identifiability::starts_identifiability(
        &ends,
        &losses,
        &converged,
        &problem.inner.parameter_bounds(),
        settings.parameter_tolerance,
        settings.loss_tolerance,
    )
    .map_err(method_error)?;
    Ok(PyStartsAnalysis {
        near_best: analysis.near_best,
        spread: analysis.spread,
        flagged: analysis.flagged,
    })
}

/// Register the identifiability analyses with the inference module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPointRank>()?;
    m.add_class::<PyStructuralAnalysis>()?;
    m.add_class::<PyCollinearityVerdict>()?;
    m.add_class::<PyCollinearityAnalysis>()?;
    m.add_class::<PySampleVerdict>()?;
    m.add_class::<PySampleAnalysis>()?;
    m.add_class::<PyStartsAnalysis>()?;
    m.add_function(wrap_pyfunction!(structural_rank, m)?)?;
    m.add_function(wrap_pyfunction!(collinearity, m)?)?;
    m.add_function(wrap_pyfunction!(sample_identifiability, m)?)?;
    m.add_function(wrap_pyfunction!(starts_identifiability, m)?)?;
    Ok(())
}
