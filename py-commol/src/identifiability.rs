//! Python bindings of the identifiability analyses of `commol-inference`.
//!
//! Each function takes the problem built once by a `Calibrator`, or a
//! sample, and returns the analysis as plain values; Python builds the
//! report.

use crate::parallelism::parallel;
use commol_inference::identifiability::{
    self, CollinearityVerdict, Reference, SampleVerdict, StructuralVerdict, Subset,
};
use commol_inference::sensitivity::{Design, FiniteDifferences};
use pyo3::prelude::*;

use crate::inference::{PyCalibrationProblem, PyParameterSample, method_error};

fn source(
    problem: &PyCalibrationProblem,
    dense: bool,
) -> PyResult<FiniteDifferences<crate::engine::Engine>> {
    let design = if dense {
        Design::Dense
    } else {
        Design::Observations
    };
    FiniteDifferences::new(&problem.inner, design).map_err(method_error)
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

/// The structural rank at `points` rows of a Latin hypercube over the bounds
///
/// Args:
///     problem: The calibration problem
///     points: How many points, at least 1
///     dense: The dense design instead of the observations
///     seed: The seed of the points
///
/// Raises:
///     ValueError: If the engine is stochastic, `points` is 0, or every
///         point is infeasible
#[pyfunction]
fn structural_rank(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    #[pyo3(from_py_with = crate::counts::required::points)] points: usize,
    dense: bool,
    seed: u64,
) -> PyResult<PyStructuralAnalysis> {
    let source = source(problem, dense)?;
    let analysis = parallel(py, || {
        identifiability::structural_rank(&source, points, seed)
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

/// Brun's indices at `points` with `weights`
///
/// Args:
///     problem: The calibration problem
///     points: The parameter values of each point
///     weights: The weight of each point
///     max_subset_size: The largest subset, from 2 to the number of
///         parameters
///     threshold: The index below which a subset can be fitted together
///
/// Raises:
///     ValueError: If the engine is stochastic, a setting is not valid
///         (a negative max_subset_size included), or every point is
///         infeasible
#[pyfunction]
fn collinearity(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    points: Vec<Vec<f64>>,
    weights: Vec<f64>,
    #[pyo3(from_py_with = crate::counts::required::max_subset_size)] max_subset_size: usize,
    threshold: f64,
) -> PyResult<PyCollinearityAnalysis> {
    let source = source(problem, false)?;
    let analysis = parallel(py, || {
        identifiability::collinearity(&source, &points, &weights, max_subset_size, threshold)
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
///     sample: The sample
///     correlation_threshold: Strictly between 0 and 1
///     priors: The prior of each parameter, the reference of the
///         contraction; None compares with the uniform on the bounds
///
/// Raises:
///     ValueError: If `correlation_threshold` is not strictly between 0 and
///         1, or a prior is missing or does not fit its bounds
#[pyfunction]
#[pyo3(signature = (sample, correlation_threshold, priors=None))]
fn sample_identifiability(
    sample: &PyParameterSample,
    correlation_threshold: f64,
    priors: Option<Vec<crate::smc::PriorArgs>>,
) -> PyResult<PySampleAnalysis> {
    let priors = priors
        .map(|given| crate::smc::priors(sample.inner.names(), sample.inner.bounds(), Some(given)))
        .transpose()?;
    let reference = match &priors {
        Some(priors) => Reference::Prior(priors),
        None => Reference::Bounds,
    };
    let analysis =
        identifiability::sample_identifiability(&sample.inner, reference, correlation_threshold)
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

/// Compare the ends of the starts that converged
///
/// Raises:
///     ValueError: If the inputs differ in length, a tolerance is negative,
///         or fewer than two starts converged
#[pyfunction]
fn starts_identifiability(
    ends: Vec<Vec<f64>>,
    losses: Vec<f64>,
    converged: Vec<bool>,
    bounds: Vec<(f64, f64)>,
    parameter_tolerance: f64,
    loss_tolerance: f64,
) -> PyResult<PyStartsAnalysis> {
    let analysis = identifiability::starts_identifiability(
        &ends,
        &losses,
        &converged,
        &bounds,
        parameter_tolerance,
        loss_tolerance,
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
