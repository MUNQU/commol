//! The threads of the parallel methods and the levels that run in parallel.
//!
//! A [`Parallelism`] names a number of threads and the [`Levels`] that run
//! in parallel. [`Parallelism::install`] runs a job in the thread pool of
//! that choice, built once and kept for the next job with the same choice.
//! Inside the job, every `par_iter` of commol and of the argmin fork uses
//! that pool, nested levels included, so the job uses at most `threads`
//! threads. The global pool of `rayon` is never used, and the environment
//! variable `RAYON_NUM_THREADS` does not apply.
//!
//! A level that nests in another one, such as the particles of a swarm in
//! the starts of a calibration, reads [`inner_levels_in_parallel`]: each
//! thread of a pool holds the levels of its pool.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Which levels run in parallel when a method nests them, such as the
/// starts of a calibration (the outer level) and the particles of the swarm
/// of each start (an inner level).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Levels {
    /// Every level runs in parallel in the one pool; work stealing balances
    /// them.
    #[default]
    All,
    /// Only the outer level runs in parallel; the particles of a swarm run
    /// one after the other.
    Outer,
}

impl Levels {
    /// The levels of a name: `"all"` or `"outer"`.
    ///
    /// # Errors
    ///
    /// Returns a message that lists the names when `name` is none of them.
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "all" => Ok(Self::All),
            "outer" => Ok(Self::Outer),
            _ => Err(format!(
                "Unknown levels '{name}'; the levels are 'all' and 'outer'"
            )),
        }
    }

    /// The name of the levels: `"all"` or `"outer"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Outer => "outer",
        }
    }
}

thread_local! {
    /// The levels of the pool of this thread; [`Levels::All`] on a thread of
    /// no commol pool.
    static LEVELS: Cell<Levels> = const { Cell::new(Levels::All) };
}

/// Whether a level nested in another one runs in parallel on this thread:
/// true in a pool of [`Levels::All`] and on a thread of no commol pool.
pub fn inner_levels_in_parallel() -> bool {
    LEVELS.get() == Levels::All
}

/// A number of threads and the levels that run in parallel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Parallelism {
    threads: usize,
    levels: Levels,
}

impl Default for Parallelism {
    /// One thread per core of the machine, every level in parallel.
    fn default() -> Self {
        Self {
            threads: std::thread::available_parallelism().map_or(1, usize::from),
            levels: Levels::All,
        }
    }
}

impl Parallelism {
    /// `threads` threads, with `levels` in parallel.
    ///
    /// # Errors
    ///
    /// Returns a message that names `threads` when it is below 1.
    pub fn new(threads: i64, levels: Levels) -> Result<Self, String> {
        match usize::try_from(threads) {
            Ok(threads) if threads >= 1 => Ok(Self { threads, levels }),
            _ => Err(format!(
                "The number of threads must be at least 1, got {threads}"
            )),
        }
    }

    /// The number of threads.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The levels that run in parallel.
    pub fn levels(&self) -> Levels {
        self.levels
    }

    /// Run `job` in the thread pool of this choice, and return its result.
    ///
    /// # Errors
    ///
    /// Returns a message when the system cannot start the threads of the
    /// pool.
    pub fn install<R: Send>(&self, job: impl FnOnce() -> R + Send) -> Result<R, String> {
        Ok(self.pool()?.install(job))
    }

    /// The pool of this choice, built at its first use.
    fn pool(&self) -> Result<Arc<ThreadPool>, String> {
        static POOLS: OnceLock<Mutex<HashMap<Parallelism, Arc<ThreadPool>>>> = OnceLock::new();
        let mut pools = POOLS
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(pool) = pools.get(self) {
            return Ok(Arc::clone(pool));
        }
        let levels = self.levels;
        let pool = ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .thread_name(|index| format!("commol-{index}"))
            .start_handler(move |_| LEVELS.set(levels))
            .build()
            .map_err(|error| format!("Cannot start {} threads: {error}", self.threads))?;
        let pool = Arc::new(pool);
        pools.insert(*self, Arc::clone(&pool));
        Ok(pool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::collections::HashSet;

    #[test]
    fn a_number_of_threads_below_one_is_rejected_with_its_value() {
        assert_eq!(
            Parallelism::new(0, Levels::All),
            Err("The number of threads must be at least 1, got 0".to_string())
        );
        assert_eq!(
            Parallelism::new(-3, Levels::Outer),
            Err("The number of threads must be at least 1, got -3".to_string())
        );
        let parallelism = Parallelism::new(3, Levels::Outer).unwrap();
        assert_eq!(
            (parallelism.threads(), parallelism.levels()),
            (3, Levels::Outer)
        );
    }

    #[test]
    fn the_levels_are_read_from_their_name() {
        for levels in [Levels::All, Levels::Outer] {
            assert_eq!(Levels::from_name(levels.name()), Ok(levels));
        }
        assert_eq!(
            Levels::from_name("inner"),
            Err("Unknown levels 'inner'; the levels are 'all' and 'outer'".to_string())
        );
    }

    #[test]
    fn the_default_is_one_thread_per_core_and_every_level() {
        let parallelism = Parallelism::default();
        assert_eq!(
            parallelism.threads(),
            std::thread::available_parallelism().unwrap().get()
        );
        assert_eq!(parallelism.levels(), Levels::All);
    }

    /// The threads that run the items of a nested parallel loop.
    fn threads_of_nested_loops() -> HashSet<String> {
        (0..16)
            .into_par_iter()
            .flat_map(|_| {
                (0..16)
                    .into_par_iter()
                    .map(|_| {
                        std::thread::sleep(std::time::Duration::from_micros(200));
                        std::thread::current().name().unwrap_or("").to_string()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn a_job_and_its_nested_loops_run_on_the_threads_of_its_pool() {
        let parallelism = Parallelism::new(2, Levels::All).unwrap();
        let names = parallelism.install(threads_of_nested_loops).unwrap();
        assert!(!names.is_empty() && names.len() <= 2, "{names:?}");
        assert!(
            names.iter().all(|name| name.starts_with("commol-")),
            "{names:?}"
        );
        let one = Parallelism::new(1, Levels::All).unwrap();
        assert_eq!(one.install(threads_of_nested_loops).unwrap().len(), 1);
    }

    #[test]
    fn each_thread_of_a_pool_holds_the_levels_of_its_pool() {
        assert!(inner_levels_in_parallel());
        for levels in [Levels::All, Levels::Outer] {
            let parallelism = Parallelism::new(3, levels).unwrap();
            let seen: HashSet<bool> = parallelism
                .install(|| {
                    (0..64)
                        .into_par_iter()
                        .map(|_| inner_levels_in_parallel())
                        .collect()
                })
                .unwrap();
            assert_eq!(seen, HashSet::from([levels == Levels::All]));
        }
    }
}
