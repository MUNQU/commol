//! The check of the range rules of an object of the Python API, which the
//! Python model of the object calls when it is built.

use commol_core::{Ranges, Rule, Setting, check_object};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList, PyTuple};

/// The rules of every object: the replicates of a simulation, the axis
/// patterns, a calibration and the methods.
const TABLES: &[&[Ranges]] = &[
    commol_core::range::RANGES,
    commol_calibration::ranges::RANGES,
    commol_inference::ranges::RANGES,
];

/// The setting `value` with its own type: an int (a bool too) is a whole
/// number, a float is a number (NaN and infinity included), a list or a
/// tuple is a list, and anything else, such as a string or None, is not a
/// number.
fn setting(value: &Bound<'_, PyAny>) -> Setting {
    if value.is_instance_of::<PyInt>() {
        value
            .extract::<i128>()
            .map_or_else(|_| Setting::Number(nearest(value)), Setting::Whole)
    } else if let Ok(number) = value.cast::<PyFloat>() {
        Setting::Number(number.value())
    } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        Setting::List(
            value
                .try_iter()
                .expect("a list or a tuple is iterable")
                .flatten()
                .map(|item| setting(&item))
                .collect(),
        )
    } else {
        Setting::Other
    }
}

/// The number nearest to the int `value`, which is beyond `i128`: infinity
/// beyond `f64`, with the sign of `value`.
fn nearest(value: &Bound<'_, PyAny>) -> f64 {
    value.extract::<f64>().unwrap_or_else(|_| {
        if value.gt(0).unwrap_or(false) {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    })
}

/// Check the settings of an object with the range rule of each
///
/// Args:
///     object: The class of the object, such as "Sobol"
///     settings: Each setting with a range rule, by its name, as the object
///         holds it
///
/// Raises:
///     ValueError: With the message of the rule of the first setting
///         outside its range, or when no rules name the object
#[pyfunction]
fn check_ranges(object: &str, settings: &Bound<'_, PyDict>) -> PyResult<()> {
    let settings: Vec<(String, Setting)> = settings
        .iter()
        .map(|(name, value)| (name.to_string(), setting(&value)))
        .collect();
    check_object(TABLES, object, &settings).map_err(PyValueError::new_err)
}

/// A value outside the range of a setting.
#[derive(IntoPyObject)]
enum Outside {
    Whole(i128),
    Number(f64),
}

/// Every setting that has a range rule: its object, its name, a value
/// outside its range and the message of the rule for that value
#[pyfunction]
fn range_rules() -> Vec<(&'static str, &'static str, Outside, String)> {
    TABLES
        .iter()
        .flat_map(|table| table.iter())
        .flat_map(|ranges| {
            ranges.rules.iter().map(|rule| {
                let (outside, error) = match rule {
                    Rule::Count(rule) => {
                        (Outside::Whole(rule.outside()), rule.check(rule.outside()))
                    }
                    Rule::Real(rule) => {
                        (Outside::Number(rule.outside()), rule.check(rule.outside()))
                    }
                };
                let message = error.expect_err("a value outside the range breaks the rule");
                (ranges.object, rule.name(), outside, message)
            })
        })
        .collect()
}

/// Add the check of the range rules to `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(range_rules, m)?)?;
    Ok(())
}
