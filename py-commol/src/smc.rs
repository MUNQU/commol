//! Python bindings of the distributions, the priors and the sequential Monte
//! Carlo methods of `commol-inference`.
//!
//! A prior crosses as `(parameter, kind, parameters)`; `None` in place of
//! the list is `Priors.uniform_on_bounds()`. Each runner takes the problem
//! built once by a `Calibrator` and returns the run as plain values; Python
//! builds the result.

use commol_inference::{Distribution, Family, Priors};
use commol_random::{Generator, replicate_seed};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A prior as Python passes it: the parameter, the kind and its parameters.
pub(crate) type PriorArgs = (String, String, Vec<f64>);

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// The priors of `names` with `bounds`; `None` is uniform on the bounds.
pub(crate) fn priors(
    names: &[String],
    bounds: &[(f64, f64)],
    given: Option<Vec<PriorArgs>>,
) -> PyResult<Priors> {
    match given {
        None => Priors::uniform_on_bounds(names, bounds),
        Some(given) => {
            let families = given
                .into_iter()
                .map(|(name, kind, parameters)| {
                    Family::from_name(&kind, &parameters).map(|family| (name, family))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(value_error)?;
            Priors::new(names, bounds, &families)
        }
    }
    .map_err(value_error)
}

/// A distribution truncated to the bounds of a parameter
///
/// Args:
///     kind: "uniform", "log_uniform", "beta", "normal", "log_normal" or
///         "gamma"
///     parameters: The parameters of the kind, in order
///     lower: The lower bound
///     upper: The upper bound
///
/// Raises:
///     ValueError: If a parameter is not valid or the bounds do not fit the
///         kind
#[pyclass(name = "Distribution", frozen)]
pub struct PyDistribution {
    inner: Distribution,
}

#[pymethods]
impl PyDistribution {
    #[new]
    fn new(kind: &str, parameters: Vec<f64>, lower: f64, upper: f64) -> PyResult<Self> {
        let family = Family::from_name(kind, &parameters).map_err(value_error)?;
        Distribution::new(family, lower, upper)
            .map(|inner| Self { inner })
            .map_err(value_error)
    }

    /// The log density at `x`: -inf outside the bounds
    fn log_density(&self, x: f64) -> f64 {
        self.inner.log_density(x)
    }

    /// The distribution function at `x`
    fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }

    /// The quantile at `p`, in [0, 1]
    ///
    /// Raises:
    ///     ValueError: If `p` is not in [0, 1]
    fn quantile(&self, p: f64) -> PyResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(value_error(format!(
                "A quantile level is between 0 and 1, got {p}"
            )));
        }
        Ok(self.inner.quantile(p))
    }

    /// The mean
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// The variance
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// `count` draws from the generator seeded with `seed`
    fn sample(&self, count: usize, seed: u64) -> Vec<f64> {
        let mut generator = Generator::from_seed(replicate_seed(seed, 0));
        (0..count)
            .map(|_| self.inner.sample(&mut generator))
            .collect()
    }
}

/// Register the classes and functions of this module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistribution>()?;
    Ok(())
}
