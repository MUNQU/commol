//! The threads of the parallel methods: the setting that
//! `commol.set_threads` and `commol.threads` change, and the one helper,
//! [`parallel`], through which every binding runs its work, with its
//! sibling [`parallel_logged`] for a job that logs its steps.
//!
//! The setting is a `contextvars.ContextVar` that holds a [`PyParallelism`],
//! so each Python thread, and each block of `commol.threads`, has its own.
//! Without a value it is [`commol_core::Parallelism::default`].

use commol_core::{Levels, Parallelism};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

use crate::records::{ChannelSteps, StepLog};

/// The context variable of the setting, built at its first use.
static SETTING: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// The context variable that holds the setting of this context.
fn setting(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    SETTING
        .get_or_try_init(py, || {
            let kwargs = PyDict::new(py);
            kwargs.set_item("default", py.None())?;
            let variable = py
                .import("contextvars")?
                .getattr("ContextVar")?
                .call(("commol_parallelism",), Some(&kwargs))?;
            Ok::<_, PyErr>(variable.unbind())
        })
        .map(|variable| variable.bind(py))
}

/// A number of threads and the levels that run in parallel
///
/// Args:
///     threads: The number of threads, at least 1
///     levels: The JSON of "all" or "outer"
///
/// Raises:
///     ValueError: If `threads` is below 1 or `levels` is neither name
#[pyclass(name = "Parallelism", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyParallelism {
    inner: Parallelism,
}

// coverage: code that a PyO3 attribute generates: the wrappers of the methods without arguments
// count 0 while the methods run
#[pymethods]
impl PyParallelism {
    #[new]
    fn new(threads: i64, levels: &str) -> PyResult<Self> {
        let levels: Levels = crate::json::read(levels)?;
        Parallelism::new(threads, levels)
            .map(|inner| Self { inner })
            .map_err(PyValueError::new_err)
    }

    /// The number of threads
    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }

    /// The levels that run in parallel: "all" or "outer"
    #[getter]
    fn levels(&self) -> &'static str {
        self.inner.levels().name()
    }
}

/// The setting of this context, or the default.
fn current_setting(py: Python<'_>) -> PyResult<Parallelism> {
    let value = setting(py)?.call_method0("get")?;
    if value.is_none() {
        return Ok(Parallelism::default());
    }
    Ok(value.extract::<PyParallelism>()?.inner)
}

/// The setting of this context: one thread per core and every level when
/// none is set
#[pyfunction]
fn current(py: Python<'_>) -> PyResult<PyParallelism> {
    current_setting(py).map(|inner| PyParallelism { inner })
}

/// The context variable that holds the setting
#[pyfunction(name = "setting")]
fn setting_variable(py: Python<'_>) -> PyResult<Py<PyAny>> {
    setting(py).map(|variable| variable.clone().unbind())
}

/// Run `job` with the GIL released, in the thread pool of the setting of
/// this context, and return its result. Every `par_iter` of `job`, nested
/// levels included, runs in that pool.
///
/// # Errors
///
/// A `TypeError` when the context variable holds something else than a
/// `Parallelism`, and a `RuntimeError` when the system cannot start the
/// threads of the pool.
pub(crate) fn parallel<T: Send>(py: Python<'_>, job: impl FnOnce() -> T + Send) -> PyResult<T> {
    let parallelism = current_setting(py)?;
    py.detach(|| parallelism.install(job))
        .map_err(PyRuntimeError::new_err)
}

/// [`parallel`], with the steps of `log` lent to `job`: the job runs on a
/// thread of its own, which waits for the pool, while this thread writes
/// the record of each step the job sends ([`crate::records`]). The GIL is
/// held only for each record.
///
/// # Errors
///
/// As [`parallel`]; the error of the logger when it cannot say which
/// levels it takes; and an error that is no `Exception` (a
/// `KeyboardInterrupt`) of a record, raised once the job has ended.
pub(crate) fn parallel_logged<T: Send>(
    py: Python<'_>,
    log: &StepLog,
    job: impl FnOnce(&ChannelSteps) -> T + Send,
) -> PyResult<T> {
    let parallelism = current_setting(py)?;
    let (steps, drain) = log.channel(py)?;
    let (result, interrupt) = py.detach(|| {
        std::thread::scope(|scope| {
            let worker = scope.spawn(move || parallelism.install(|| job(&steps)));
            let interrupt = drain.drain();
            let result = worker
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            (result, interrupt)
        })
    });
    match interrupt {
        Some(error) => Err(error),
        None => result.map_err(PyRuntimeError::new_err),
    }
}

/// Register the setting of the threads.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyParallelism>()?;
    m.add_function(wrap_pyfunction!(current, m)?)?;
    m.add_function(wrap_pyfunction!(setting_variable, m)?)?;
    Ok(())
}
