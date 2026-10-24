//! The counts the bindings take from Python: one reading, with one message
//! for a number that is not a count.
//!
//! A binding reads each count with `#[pyo3(from_py_with = ...)]` and the
//! reader of its name, from [`required`] or, for a count it may omit, from
//! [`optional`], so the message names the setting.

use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

/// An unsigned integer type that a binding takes as a count.
pub(crate) trait Count: Sized {
    /// The largest count of the type.
    const MAX: u64;

    /// The count `value`, or the error of PyO3.
    fn read(value: &Bound<'_, PyAny>) -> PyResult<Self>;
}

macro_rules! count_types {
    ($($kind:ty),* $(,)?) => {$(
        impl Count for $kind {
            const MAX: u64 = <$kind>::MAX as u64;

            fn read(value: &Bound<'_, PyAny>) -> PyResult<Self> {
                value.extract()
            }
        }
    )*};
}

count_types!(u32, u64, usize);

/// Read the count `name` from `value`.
///
/// A negative number, or one above the largest count of the type, raises
/// `ValueError` with a message that names the setting, in place of the
/// `OverflowError` of PyO3; a value that is not a whole number raises the
/// `TypeError` of PyO3. The method then checks the rule of its setting, such
/// as "at least 1".
pub(crate) fn count<T: Count>(name: &str, value: &Bound<'_, PyAny>) -> PyResult<T> {
    match T::read(value) {
        Err(error) if error.is_instance_of::<PyOverflowError>(value.py()) => {
            let range = if value.lt(0)? {
                "from 0".to_string()
            } else {
                format!("from 0 to {}", T::MAX)
            };
            Err(PyValueError::new_err(format!(
                "{name} is a whole number {range}, got {value}"
            )))
        }
        read => read,
    }
}

/// The count `name` of `value`, or None when `value` is None.
pub(crate) fn optional_count<T: Count>(
    name: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<T>> {
    if value.is_none() {
        Ok(None)
    } else {
        count(name, value).map(Some)
    }
}

/// `improvement_stop` of a particle swarm, `(window, tol_abs, tol_rel)`,
/// with the count `window`; None when it is None.
pub(crate) fn improvement_stop(value: &Bound<'_, PyAny>) -> PyResult<Option<(u64, f64, f64)>> {
    if value.is_none() {
        return Ok(None);
    }
    let (window, tol_abs, tol_rel): (Bound<'_, PyAny>, f64, f64) = value.extract()?;
    Ok(Some((count("window", &window)?, tol_abs, tol_rel)))
}

/// `residual_noise` of observation draws, `(fit, block_length,
/// residual_scale)`, with the count `block_length`; None when it is None.
pub(crate) fn residual_noise(
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<(Vec<f64>, usize, String)>> {
    if value.is_none() {
        return Ok(None);
    }
    let (fit, block_length, scale): (Vec<f64>, Bound<'_, PyAny>, String) = value.extract()?;
    Ok(Some((fit, count("block_length", &block_length)?, scale)))
}

macro_rules! readers {
    ($read:ident -> $out:ident: $($name:ident),* $(,)?) => {$(
        #[doc = concat!("The count `", stringify!($name), "`, as [`", stringify!($read), "`](super::", stringify!($read), ") reads it.")]
        pub(crate) fn $name<T: Count>(value: &Bound<'_, PyAny>) -> PyResult<$out<T>> {
            super::$read(stringify!($name), value)
        }
    )*};
}

/// The readers of the counts a binding requires, one per name.
pub(crate) mod required {
    use super::Count;
    use pyo3::prelude::*;

    type Same<T> = T;

    readers!(count -> Same:
        count,
        degree,
        draws,
        header_interval,
        levels,
        max_iterations,
        max_steps,
        max_subset_size,
        num_steps,
        particles,
        points,
        replicates,
        starts,
        trajectories,
    );
}

/// The readers of the counts a binding may omit, one per name.
pub(crate) mod optional {
    use super::Count;
    use pyo3::prelude::*;

    readers!(optional_count -> Option:
        block_length,
        count,
        draws,
        max_interaction,
        replicates,
        trials,
    );
}
