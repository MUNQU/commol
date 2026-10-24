//! Python bindings of the sensitivity analyses of `commol-inference`.
//!
//! [`PyForwardAnalysis`] wraps a [`ForwardAnalysis`], built once from a
//! `SensitivityAnalysis`: each method is one call into Rust, which builds
//! the design, runs it and estimates. The quantities cross as one JSON
//! list ([`crate::quantity`]); the results cross as plain values, and
//! Python builds the result classes.

use crate::parallelism::parallel;
use commol_inference::design::InputSpace;
use commol_inference::global_sensitivity::{ChaosExpansion, ForwardAnalysis, Truncation};
use commol_inference::{NoNoise, Predictive, observation_draws};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::calibration::{PyCalibrationParameter, PyObservedDataPoint};
use crate::engine::{Engine, PyEngine};
use crate::inference::{PyObservationDraws, PyParameterSample, PyPredictive, method_error};
use crate::quantity::read_quantities;
use crate::smc::{PriorArgs, priors};

/// The draws of a method and their log weights, `(values, log_weights)`.
type Draws = (Vec<Vec<f64>>, Vec<f64>);

/// The draws, the quantities and weight of each run, and the seed of each
/// draw of a propagation.
type PropagationValues = (Draws, Vec<Vec<f64>>, Vec<f64>, Vec<u64>);

/// `μ`, `μ*`, `σ` and the number of points of a Morris screening.
type MorrisValues = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, usize);

/// The draws, the coefficients, the p-values and the effective size of a
/// PRCC.
type PrccValues = (Draws, Vec<Vec<f64>>, Vec<Vec<f64>>, f64);

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// The inputs, from their parameters and their distributions or sample.
fn input_space(
    parameters: &[PyCalibrationParameter],
    distributions: Option<Vec<PriorArgs>>,
    sample: Option<&PyParameterSample>,
) -> PyResult<InputSpace> {
    let names: Vec<String> = parameters.iter().map(|p| p.inner.id.clone()).collect();
    let bounds: Vec<(f64, f64)> = parameters
        .iter()
        .map(|p| (p.inner.min_bound, p.inner.max_bound))
        .collect();
    match sample {
        Some(sample) => Ok(InputSpace::Sample(sample.inner.clone())),
        None => Ok(InputSpace::Independent(priors(
            &names,
            &bounds,
            distributions,
        )?)),
    }
}

/// A forward model with no observations, the inputs it varies and its
/// seed, built once
#[pyclass(name = "ForwardAnalysis", frozen)]
pub struct PyForwardAnalysis {
    inner: ForwardAnalysis<Engine>,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyForwardAnalysis {
    /// Build a forward analysis
    ///
    /// Args:
    ///     engine: The simulation engine built from the model
    ///     parameters: The inputs, in order
    ///     distributions: The distribution of each input as
    ///         `(input, kind, parameters)`, for independent inputs; None is
    ///         uniform on the bounds
    ///     sample: The draws of the inputs, for inputs from a sample; it
    ///         takes the place of `distributions`
    ///     initial_condition_remainder: Compartment or bin that takes what
    ///         the initial conditions leave, if any
    ///     replicates: Replicates at each point, for a stochastic engine; a
    ///         deterministic engine reads none
    ///     seed: Seed of every random number of the analysis; None draws one,
    ///         which `seed` reports, and a stochastic engine needs it set
    ///
    /// Raises:
    ///     ValueError: If the inputs are not valid for the engine, replicates
    ///         is negative, or a stochastic engine has no replicates or no
    ///         seed
    #[new]
    #[pyo3(signature = (engine, parameters, distributions=None, sample=None, initial_condition_remainder=None, replicates=None, seed=None))]
    fn new(
        engine: PyRef<'_, PyEngine>,
        parameters: Vec<PyCalibrationParameter>,
        distributions: Option<Vec<PriorArgs>>,
        sample: Option<PyRef<'_, PyParameterSample>>,
        initial_condition_remainder: Option<String>,
        replicates: Option<i64>,
        seed: Option<i128>,
    ) -> PyResult<Self> {
        let seed: Option<u64> = crate::counts::optional_count(&commol_core::SEED, seed)?;
        let replicates: Option<usize> =
            crate::counts::optional_count(&commol_calibration::ranges::REPLICATES, replicates)?;
        let inputs = input_space(&parameters, distributions, sample.as_deref())?;
        let inner = ForwardAnalysis::new(
            engine.engine(),
            parameters.into_iter().map(|p| p.inner).collect(),
            initial_condition_remainder.as_deref(),
            replicates,
            inputs,
            seed,
        )
        .map_err(method_error)?;
        Ok(Self { inner })
    }

    /// The seed of every random number of the analysis
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    /// Monte Carlo propagation: the draws, the quantities and weight of each
    /// run, and the seed of each draw
    ///
    /// Raises:
    ///     ValueError: If the draws do not fit the inputs, or a quantity is
    ///         not a quantity or does not fit the model
    ///     InfeasibleStateError: If a draw leaves the values its model allows
    fn propagate(
        &self,
        py: Python<'_>,
        draws: Option<i64>,
        quantities: &str,
    ) -> PyResult<PropagationValues> {
        let draws: Option<usize> =
            crate::counts::optional_count(&commol_inference::ranges::DRAWS, draws)?;
        let quantities = read_quantities(quantities)?;
        let result =
            parallel(py, || self.inner.propagate(draws, &quantities))?.map_err(method_error)?;
        let draws = (result.sample.values().to_vec(), result.sample.log_weights());
        Ok((draws, result.values, result.weights, result.draw_seeds))
    }

    /// The PRCC of each input for each quantity: the draws, the coefficients
    /// and p-values `[quantity][input]`, and Kish's effective size
    ///
    /// Raises:
    ///     ValueError: If the draws do not fit the inputs, a quantity does not
    ///         fit the model or is NaN at a draw, or the ranks of the inputs
    ///         are collinear
    ///     InfeasibleStateError: If a point leaves the values its model allows
    fn prcc(&self, py: Python<'_>, draws: Option<i64>, quantities: &str) -> PyResult<PrccValues> {
        let draws: Option<usize> =
            crate::counts::optional_count(&commol_inference::ranges::DRAWS, draws)?;
        let quantities = read_quantities(quantities)?;
        let (sample, result) =
            parallel(py, || self.inner.prcc(draws, &quantities))?.map_err(method_error)?;
        Ok((
            (sample.values().to_vec(), sample.log_weights()),
            result.coefficients,
            result.p_values,
            result.effective_size,
        ))
    }

    /// Morris screening: `μ`, `μ*` and `σ` of each input for each quantity,
    /// `[quantity][input]`, and the number of points run
    ///
    /// Raises:
    ///     ValueError: If the inputs come from a sample, the levels are not
    ///         even and at least 2, or a quantity does not fit the model or
    ///         is NaN at a point
    ///     InfeasibleStateError: If a point leaves the values its model allows
    fn morris(
        &self,
        py: Python<'_>,
        trajectories: i64,
        levels: i64,
        quantities: &str,
    ) -> PyResult<MorrisValues> {
        let trajectories: usize =
            crate::counts::count(&commol_inference::ranges::TRAJECTORIES, trajectories)?;
        let levels: usize = crate::counts::count(&commol_inference::ranges::LEVELS, levels)?;
        let quantities = read_quantities(quantities)?;
        let (result, runs) = parallel(py, || self.inner.morris(trajectories, levels, &quantities))?
            .map_err(method_error)?;
        Ok((result.mu, result.mu_star, result.sigma, runs))
    }

    /// Sobol indices on the design of Saltelli, with bootstrap intervals
    ///
    /// Raises:
    ///     ValueError: If the inputs come from a sample, `draws` or
    ///         `replicates` is below 2, the level is not strictly between 0
    ///         and 1, or a quantity does not fit the model or is NaN at a point
    ///     InfeasibleStateError: If a point leaves the values its model allows
    fn sobol(
        &self,
        py: Python<'_>,
        draws: i64,
        replicates: i64,
        confidence_level: f64,
        quantities: &str,
    ) -> PyResult<SobolValues> {
        let draws: usize = crate::counts::count(&commol_inference::ranges::TWO_DRAWS, draws)?;
        let replicates: usize =
            crate::counts::count(&commol_inference::ranges::TWO_REPLICATES, replicates)?;
        let quantities = read_quantities(quantities)?;
        let (result, runs) = parallel(py, || {
            self.inner
                .sobol(draws, replicates, confidence_level, &quantities)
        })?
        .map_err(method_error)?;
        let [first_lower, first_upper] = result.first_interval;
        let [total_lower, total_upper] = result.total_interval;
        Ok(SobolValues {
            first: result.first,
            total: result.total,
            first_lower,
            first_upper,
            total_lower,
            total_upper,
            runs,
        })
    }

    /// A polynomial chaos expansion of `degree` on a Latin hypercube of
    /// `draws` points, and its Sobol indices; the basis keeps the terms whose
    /// degrees have a q-norm of `q_norm` at most `degree` and hold at most
    /// `max_interaction` inputs (`None`: no limit)
    ///
    /// Raises:
    ///     ValueError: If the truncation is not valid, the inputs come from a
    ///         sample, there are not more points than terms, or a quantity
    ///         does not fit the model or is NaN at a point
    ///     InfeasibleStateError: If a point leaves the values its model allows
    fn chaos(
        &self,
        py: Python<'_>,
        degree: i64,
        q_norm: f64,
        max_interaction: Option<i64>,
        draws: i64,
        quantities: &str,
    ) -> PyResult<PyChaosRun> {
        let max_interaction: Option<usize> = crate::counts::optional_count(
            &commol_inference::ranges::MAX_INTERACTION,
            max_interaction,
        )?;
        let degree: usize = crate::counts::count(&commol_inference::ranges::DEGREE, degree)?;
        let draws: usize = crate::counts::count(&commol_inference::ranges::TWO_DRAWS, draws)?;
        let truncation = Truncation::new(q_norm, max_interaction).map_err(method_error)?;
        let quantities = read_quantities(quantities)?;
        let (expansion, runs) = parallel(py, || {
            self.inner.chaos(degree, truncation, draws, &quantities)
        })?
        .map_err(method_error)?;
        Ok(PyChaosRun {
            mean: expansion.mean(),
            variance: expansion.variance(),
            first: expansion.first(),
            total: expansion.total(),
            indices: expansion.indices,
            coefficients: expansion.coefficients,
            loo_error: expansion.loo_error,
            runs,
        })
    }

    /// The run of every draw of `sample` at `positions`, in parallel, each
    /// replicate of a stochastic engine seeded from the seed of the analysis
    ///
    /// Raises:
    ///     ValueError: Before any draw runs, if the sample's inputs are not
    ///         the analysis', or the positions are not sorted and unique or
    ///         one is a position the engine rejects
    ///     InfeasibleStateError: If a draw leaves the values its model allows
    fn predictive(
        &self,
        py: Python<'_>,
        sample: &PyParameterSample,
        positions: Vec<f64>,
    ) -> PyResult<PyPredictive> {
        parallel(py, || {
            Predictive::new(
                self.inner.forward(),
                &sample.inner,
                &positions,
                self.inner.seed(),
            )
        })?
        .map(|inner| PyPredictive { inner })
        .map_err(method_error)
    }

    /// Every draw's value of `observations`, read through the observation
    /// operator, with no observation noise: residual noise needs the
    /// residuals of a fit, and an analysis has no data
    ///
    /// Raises:
    ///     ValueError: If the sample's inputs are not the analysis', or an
    ///         observation names an unknown output
    ///     InfeasibleStateError: If a draw leaves the values its model allows
    fn observation_draws(
        &self,
        py: Python<'_>,
        sample: &PyParameterSample,
        observations: Vec<PyObservedDataPoint>,
    ) -> PyResult<PyObservationDraws> {
        let observations: Vec<_> = observations.into_iter().map(|o| o.inner).collect();
        parallel(py, || {
            observation_draws(
                self.inner.forward(),
                &sample.inner,
                &observations,
                &NoNoise,
                self.inner.seed(),
            )
        })?
        .map(|inner| PyObservationDraws { inner })
        .map_err(method_error)
    }
}

/// The indices of a Sobol analysis, their intervals and the number of
/// points run
#[pyclass(name = "SobolRun", frozen, get_all)]
pub struct SobolValues {
    /// The first-order indices, `[quantity][input]`
    first: Vec<Vec<f64>>,
    /// The total indices, `[quantity][input]`
    total: Vec<Vec<f64>>,
    /// The lower ends of the intervals of the first-order indices
    first_lower: Vec<Vec<f64>>,
    /// The upper ends of the intervals of the first-order indices
    first_upper: Vec<Vec<f64>>,
    /// The lower ends of the intervals of the total indices
    total_lower: Vec<Vec<f64>>,
    /// The upper ends of the intervals of the total indices
    total_upper: Vec<Vec<f64>>,
    /// The number of points run
    runs: usize,
}

/// A polynomial chaos expansion of each quantity and its Sobol indices
#[pyclass(name = "ChaosRun", frozen, get_all)]
pub struct PyChaosRun {
    /// The degree of each input in each term, `[term][input]`
    indices: Vec<Vec<usize>>,
    /// The coefficient of each term for each quantity, `[quantity][term]`
    coefficients: Vec<Vec<f64>>,
    /// The relative leave-one-out error of each quantity
    loo_error: Vec<f64>,
    /// The mean of each quantity
    mean: Vec<f64>,
    /// The variance of each quantity
    variance: Vec<f64>,
    /// The first-order indices, `[quantity][input]`
    first: Vec<Vec<f64>>,
    /// The total indices, `[quantity][input]`
    total: Vec<Vec<f64>>,
    /// The number of points run
    runs: usize,
}

/// The value of each quantity of a polynomial chaos expansion at each point
///
/// The terms come from the degree, the truncation and the number of inputs,
/// so only the coefficients cross.
///
/// Args:
///     parameters: The inputs, in order, with their bounds
///     distributions: The distribution of each input, `(input, kind,
///         parameters)`
///     degree: The degree of the expansion
///     q_norm: The q of the q-norm of the basis, in (0, 1]
///     max_interaction: The largest number of inputs of a term, or None
///     coefficients: The coefficient of each term for each quantity
///     points: The values of the inputs at each point, `[point][input]`
///
/// Returns:
///     The value of each quantity at each point, `[point][quantity]`
///
/// Raises:
///     ValueError: If a distribution does not fit its bounds, the truncation
///         is not valid, a row of coefficients has not one value per term,
///         or a point has not one value per input or a value outside the
///         bounds of its input
#[pyfunction]
fn chaos_evaluate(
    parameters: Vec<PyCalibrationParameter>,
    distributions: Vec<PriorArgs>,
    degree: i64,
    q_norm: f64,
    max_interaction: Option<i64>,
    coefficients: Vec<Vec<f64>>,
    points: Vec<Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    let degree: usize = crate::counts::count(&commol_inference::ranges::DEGREE, degree)?;
    let max_interaction: Option<usize> =
        crate::counts::optional_count(&commol_inference::ranges::MAX_INTERACTION, max_interaction)?;
    let inputs = input_space(&parameters, Some(distributions), None)?;
    let truncation = Truncation::new(q_norm, max_interaction).map_err(method_error)?;
    let expansion = ChaosExpansion::new(parameters.len(), degree, truncation, coefficients)
        .map_err(method_error)?;
    let unit = inputs
        .to_unit("PolynomialChaos", &points)
        .map_err(value_error)?;
    Ok(unit.iter().map(|u| expansion.evaluate(u)).collect())
}

/// Register the sensitivity bindings in the inference module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyForwardAnalysis>()?;
    m.add_class::<SobolValues>()?;
    m.add_class::<PyChaosRun>()?;
    m.add_function(wrap_pyfunction!(chaos_evaluate, m)?)?;
    Ok(())
}
