//! Python bindings of the profile likelihood of `commol-inference`.
//!
//! [`profile_likelihood`] and [`prediction_profile`] take the problem built
//! once by a `Calibrator` and the JSON of the method, fit the problem,
//! trace every profile in Rust and return them as plain values; Python
//! builds the result.

use crate::parallelism::parallel;
use commol_inference::profiles::{
    Profile, ProfileEnd, ProfileTracer, ProfileVerdict, ProfiledQuantity, StructuralReading,
    Structure, Undetermined, beale, bootstrap_thresholds, chi_square,
};
use commol_inference::resampling::FitUnit;
use commol_inference::settings::profiles::{
    PredictionProfileSettings, ProfileLikelihoodSettings, Threshold, ThresholdKind,
};
use pyo3::prelude::*;

use crate::calibration::PyStarts;
use crate::inference::{
    PyCalibrationProblem, ResamplerSpec, best_values, method_error, python_list, value_error,
};

/// Where a side leaves the box: the value, and the index of the free
/// parameter on its bound.
type BoxExit = (f64, usize);

/// One profile, as the runner returns it
#[pyclass(name = "Profile", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyProfile {
    /// The value of the quantity at the fit
    center: f64,
    /// The threshold of the statistic
    threshold: f64,
    /// The value of each point, increasing
    values: Vec<f64>,
    /// The statistic of each point
    statistics: Vec<f64>,
    /// Every parameter of each point, in problem order
    parameters: Vec<Vec<f64>>,
    /// The multiplier of each point of a prediction profile
    multipliers: Vec<Option<f64>>,
    /// Whether the refit of each point converged
    converged: Vec<bool>,
    /// Why the side below and the side above end
    ends: (String, String),
    /// Where the statistic crosses the threshold below and above
    interval: (Option<f64>, Option<f64>),
    /// Evaluations of the objective
    evaluations: usize,
    /// The rise of each kept replicate, with a bootstrap threshold
    rises: Option<Vec<f64>>,
    /// "identifiable", "practically_non_identifiable" or
    /// "structurally_non_identifiable"; None when the threshold is below the
    /// tolerance
    verdict: Option<String>,
    /// The lowest statistic
    lowest: f64,
    /// Where the side below and the side above leave the box: the value and
    /// the index of the free parameter on its bound, or None
    box_exits: (Option<BoxExit>, Option<BoxExit>),
    /// What the resets of a profile of a parameter say, or None
    structure: Option<PyStructure>,
    /// For each side, the free parameter whose bound makes the crossing, or
    /// None
    box_crossings: (Option<usize>, Option<usize>),
    /// The error of a refit at the fit, for a profile of a parameter
    refit_error: Option<f64>,
    /// A refit at the fit lands farther from it than the tolerance
    inaccurate_refit: bool,
    /// A refit is below the fit by more than the tolerance
    below_fit: bool,
}

/// What the resets of a flat profile say, as the runner returns it
#[pyclass(name = "ProfileStructure", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyStructure {
    /// "in_combination", "non_identifiable", or None
    verdict: Option<String>,
    /// Why there is no verdict: "below_tolerance", "not_flat", "below_fit",
    /// "stochastic", "on_bound" or "undecided"; None with a verdict
    undetermined: Option<String>,
    /// The free parameter on its bound at the fit, for "on_bound"
    on_bound: Option<usize>,
    /// The largest rise of a feasible reset
    largest_rise: Option<f64>,
    /// The free parameters that compensate, by index
    compensating: Vec<usize>,
    /// The largest rise of resetting the others together, when above the
    /// tolerance
    joint_rise: Option<f64>,
}

impl From<Structure> for PyStructure {
    fn from(structure: Structure) -> Self {
        let (verdict, undetermined, on_bound) = match structure.verdict {
            Ok(StructuralReading::InCombination) => (Some("in_combination"), None, None),
            Ok(StructuralReading::NonIdentifiable) => (Some("non_identifiable"), None, None),
            Err(why) => {
                let parameter = match why {
                    Undetermined::OnBound(index) => Some(index),
                    _ => None,
                };
                (None, Some(why.name()), parameter)
            }
        };
        Self {
            verdict: verdict.map(str::to_string),
            undetermined: undetermined.map(str::to_string),
            on_bound,
            largest_rise: structure.largest_rise,
            compensating: structure.compensating,
            joint_rise: structure.joint_rise,
        }
    }
}

fn end_name(end: ProfileEnd) -> String {
    match end {
        ProfileEnd::Crossed => "crossed",
        ProfileEnd::Bound => "bound",
        ProfileEnd::MaxSteps => "max_steps",
        ProfileEnd::Unreachable => "unreachable",
    }
    .to_string()
}

impl PyProfile {
    fn new(
        profile: Profile,
        rises: Option<Vec<f64>>,
        verdict: Option<ProfileVerdict>,
        structure: Option<Structure>,
        tolerance: f64,
    ) -> Self {
        let [lower, upper] = profile.interval();
        Self {
            box_crossings: (profile.box_crossings[0], profile.box_crossings[1]),
            refit_error: profile.refit_error,
            inaccurate_refit: profile.inaccurate_refit(tolerance),
            below_fit: profile.below_fit(tolerance),
            verdict: verdict.map(|verdict| verdict.name().to_string()),
            lowest: profile.lowest(),
            structure: structure.map(PyStructure::from),
            box_exits: {
                let [below, above] = profile.box_exits();
                (below, above)
            },
            center: profile.center,
            threshold: profile.threshold,
            values: profile.points.iter().map(|p| p.value).collect(),
            statistics: profile.points.iter().map(|p| p.statistic).collect(),
            multipliers: profile.points.iter().map(|p| p.multiplier).collect(),
            converged: profile.points.iter().map(|p| p.converged).collect(),
            parameters: profile.points.into_iter().map(|p| p.parameters).collect(),
            ends: (end_name(profile.ends[0]), end_name(profile.ends[1])),
            interval: (lower, upper),
            evaluations: profile.evaluations,
            rises,
        }
    }
}

/// The profiles of a problem, as the runner returns them
#[pyclass(name = "ProfileRun", frozen, get_all)]
pub struct PyProfileRun {
    /// One profile per quantity: the parameters, then the predictions
    profiles: Vec<PyProfile>,
    /// The objective at the fit
    fit_loss: f64,
    /// The numerical tolerance of a profile value, max(τ, ν)
    tolerance: f64,
    /// The tolerance of the optimizer, τ
    optimizer_tolerance: f64,
    /// The noise floor of the objective at the fit, ν
    noise_floor: f64,
    /// The floor of a comparison of two evaluations, or None
    evaluation_floor: Option<f64>,
    /// The replicates kept by a bootstrap threshold
    replicates: Option<Vec<usize>>,
    /// Each replicate of a bootstrap threshold that failed:
    /// "Replicate <index> failed: <reason>"
    failed: Vec<String>,
    /// Replicates of a bootstrap threshold kept although their optimizer
    /// did not converge
    not_converged: usize,
}

/// Fit the problem with `starts` starts and trace the profile of each of
/// `quantities` from the fit, with `threshold`.
fn trace(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    starts: usize,
    quantities: Vec<ProfiledQuantity>,
    threshold: &Threshold,
    max_steps: usize,
) -> PyResult<(PyStarts, PyProfileRun)> {
    let fitted = problem.fit(py, starts)?;
    let fit = best_values(&fitted);
    let problem_ref = &problem.inner;
    let confidence_level = threshold.confidence_level;
    let tracer = ProfileTracer::new(problem_ref, problem.optimizer(), &fit, problem.seed)
        .map_err(method_error)?;
    let mut rises = None;
    let mut replicates = None;
    let mut failed = Vec::new();
    let mut not_converged = 0;
    let thresholds = match (threshold.kind, &threshold.bootstrap) {
        (ThresholdKind::ChiSquare, _) => {
            vec![chi_square(problem_ref, confidence_level).map_err(method_error)?; quantities.len()]
        }
        (ThresholdKind::Beale, _) => vec![
            beale(problem_ref, tracer.fit_loss(), confidence_level)
                .map_err(method_error)?;
            quantities.len()
        ],
        (ThresholdKind::BootstrapCalibrated, bootstrap) => {
            let bootstrap = bootstrap
                .as_ref()
                .expect("a calibrated threshold has its bootstrap (Threshold::check)");
            let centers = quantities
                .iter()
                .map(|quantity| tracer.center(quantity))
                .collect::<Result<Vec<f64>, _>>()
                .map_err(method_error)?;
            let runner = problem.runner(&fit, bootstrap.starts())?;
            let (resampler, _) = problem.resampler(&ResamplerSpec::of(bootstrap), runner.fit())?;
            let count = bootstrap.replicates();
            let run = parallel(py, || {
                bootstrap_thresholds(
                    &runner,
                    resampler.as_ref(),
                    count,
                    &quantities,
                    &centers,
                    confidence_level,
                )
            })?
            .map_err(method_error)?;
            rises = Some(run.rises);
            replicates = Some(run.replicates);
            failed = FitUnit::Replicate.entries(&run.failed);
            not_converged = run.not_converged;
            run.thresholds
        }
    };
    let traced = parallel(py, || tracer.trace_all(&quantities, &thresholds, max_steps))?;
    let mut profiles = Vec::with_capacity(traced.len());
    for (index, (profile, quantity)) in traced.into_iter().zip(&quantities).enumerate() {
        let profile = profile.map_err(method_error)?;
        let rises = rises
            .as_ref()
            .map(|rises: &Vec<Vec<f64>>| rises[index].clone());
        // A profile of a parameter that does not cross on both sides gets its
        // resets.
        let structure = match quantity {
            ProfiledQuantity::Parameter(parameter)
                if !matches!(
                    profile.verdict(tracer.tolerance()),
                    None | Some(ProfileVerdict::Identifiable)
                ) =>
            {
                Some(parallel(py, || {
                    tracer.read_structure(*parameter, &profile)
                })?)
            }
            _ => None,
        };
        let verdict = tracer.verdict_with(&profile, structure.as_ref());
        profiles.push(PyProfile::new(
            profile,
            rises,
            verdict,
            structure,
            tracer.tolerance(),
        ));
    }
    let run = PyProfileRun {
        profiles,
        fit_loss: tracer.fit_loss(),
        tolerance: tracer.tolerance(),
        optimizer_tolerance: tracer.optimizer_tolerance(),
        noise_floor: tracer.noise_floor(),
        evaluation_floor: tracer.evaluation_floor(),
        replicates,
        failed,
        not_converged,
    };
    Ok((PyStarts { inner: fitted }, run))
}

/// Fit the problem and trace the profile of each parameter from the fit
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `ProfileLikelihood`
///
/// Returns:
///     The starts of the fit, and the profiles in the order of the
///     parameters of the settings
///
/// Raises:
///     ValueError: If a setting is not valid, a name is not a parameter of
///         the problem, the threshold does not fit the problem, or the fit
///         is not feasible
///     RuntimeError: If the fit or a refit fails, or every replicate of the
///         bootstrap fails
#[pyfunction]
fn profile_likelihood(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(PyStarts, PyProfileRun)> {
    let settings: ProfileLikelihoodSettings = crate::json::read(settings)?;
    let names = problem.names();
    let quantities = settings
        .parameters
        .iter()
        .map(|name| {
            names
                .iter()
                .position(|parameter| parameter == name)
                .map(ProfiledQuantity::Parameter)
                .ok_or_else(|| {
                    value_error(format!(
                        "'{name}' is not a parameter of the problem; its parameters are {}.",
                        python_list(&names)
                    ))
                })
        })
        .collect::<PyResult<Vec<_>>>()?;
    trace(
        py,
        problem,
        settings.starts,
        quantities,
        &settings.threshold,
        settings.max_steps,
    )
}

/// Fit the problem and trace the profile of each prediction from the fit
///
/// Args:
///     problem: The calibration problem
///     settings: The JSON of a `PredictionProfile`
///
/// Returns:
///     The starts of the fit, and the profiles in the order of the
///     predictions of the settings
///
/// Raises:
///     ValueError: If a setting is not valid, a prediction does not fit the
///         model, the threshold does not fit the problem, or the fit is not
///         feasible
///     RuntimeError: If the fit or a refit fails, or every replicate of the
///         bootstrap fails
#[pyfunction]
fn prediction_profile(
    py: Python<'_>,
    problem: &PyCalibrationProblem,
    settings: &str,
) -> PyResult<(PyStarts, PyProfileRun)> {
    let settings: PredictionProfileSettings = crate::json::read(settings)?;
    let quantities = settings
        .specs
        .iter()
        .map(|spec| ProfiledQuantity::Prediction(spec.point()))
        .collect();
    trace(
        py,
        problem,
        settings.starts,
        quantities,
        &settings.threshold,
        settings.max_steps,
    )
}

/// Register the profile bindings with the inference module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProfile>()?;
    m.add_class::<PyStructure>()?;
    m.add_class::<PyProfileRun>()?;
    m.add_function(wrap_pyfunction!(profile_likelihood, m)?)?;
    m.add_function(wrap_pyfunction!(prediction_profile, m)?)?;
    Ok(())
}
