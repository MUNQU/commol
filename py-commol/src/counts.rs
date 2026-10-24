//! The counts the bindings take from Python, read with the rule of their
//! setting in the engine.
//!
//! A binding takes a count as `i64`, and a seed, whose range ends at
//! 2^64 - 1, as `i128`, so PyO3 reads only the type of Python: a value that
//! is not a whole number raises its `TypeError`, and one beyond the integer
//! type its `OverflowError`. [`count`] then reads the number with the rule
//! of the setting ([`commol_core::Count`]), which gives one message for
//! every value outside the range, a negative one included.

use commol_core::Count;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Read `value` with `rule` as the type of the setting.
///
/// # Errors
///
/// `ValueError` with the message of the rule when `value` is outside its
/// range.
pub(crate) fn count<T: TryFrom<i128>>(rule: &Count, value: impl Into<i128>) -> PyResult<T> {
    rule.read(value).map_err(PyValueError::new_err)
}

/// Read `value` with `rule`, or None when it is None.
///
/// # Errors
///
/// As [`count`].
pub(crate) fn optional_count<T: TryFrom<i128>>(
    rule: &Count,
    value: Option<impl Into<i128>>,
) -> PyResult<Option<T>> {
    value.map(|value| count(rule, value)).transpose()
}
