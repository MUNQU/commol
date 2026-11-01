//! The log records of the steps of a long method.
//!
//! A binding of a method that runs for long gives its job a
//! [`ChannelSteps`] ([`crate::parallelism::parallel_logged`]). The loops of
//! the job send their events to it from the threads of the pool; it turns
//! each event into an owned [`Step`] and sends it into a channel. The thread
//! that called the method drains the channel ([`Drain`]): it counts the
//! steps of each stage and writes one record per step to the logger of the
//! method's module, with the GIL, so a record has the thread and the
//! `contextvars` context of the caller, and the counts rise by one.
//!
//! The outermost loop of a method logs at `INFO`; the starts of a replicate
//! and the generations of the runs of a calibrated `η` log at `DEBUG`. A
//! level that the logger does not take sends nothing. A loop of model runs
//! logs at each 1 % of its points, not at each point.
//!
//! The drain also reads the signals of Python while it waits: a Ctrl-C, or
//! a `KeyboardInterrupt` that a record raises, sets the stop that the
//! [`ChannelSteps`] give the loops ([`Steps::stopped`]), so the job starts
//! no step and ends with the steps already running.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::{Duration, Instant};

use commol_calibration::{CalibrationResult, StartEvent};
use commol_core::Steps;
use commol_inference::profiles::{ProfileEnd, SideEvent};
use commol_inference::resampling::{FitUnit, RunnerEvent, RunnerStage};
use commol_inference::runs::RunEvent;
use commol_inference::smc::SmcEvent;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

/// The level `DEBUG` of Python's `logging`.
const DEBUG: u8 = 10;
/// The level `INFO` of Python's `logging`.
const INFO: u8 = 20;

/// The longest wait of the drain between two reads of the signals.
const SIGNALS: Duration = Duration::from_millis(100);

/// Where the records of a method go: the logger of its module, the names of
/// the quantities its profiles trace, in their order, and what it fits once
/// per perturbation.
pub(crate) struct StepLog {
    logger: Py<PyAny>,
    quantities: Vec<String>,
    unit: FitUnit,
}

impl StepLog {
    /// The records of a method that logs to `logger`, whose fits are
    /// replicates.
    pub(crate) fn new(logger: Py<PyAny>) -> Self {
        Self {
            logger,
            quantities: Vec::new(),
            unit: FitUnit::Replicate,
        }
    }

    /// These records, with the names of the quantities of the profiles.
    pub(crate) fn with_quantities(self, quantities: Vec<String>) -> Self {
        Self { quantities, ..self }
    }

    /// These records, of a method that fits a `unit` once per
    /// perturbation: a block left out, an origin.
    pub(crate) fn with_unit(self, unit: FitUnit) -> Self {
        Self { unit, ..self }
    }

    /// The steps that a job sends to, and the drain of their records.
    ///
    /// # Errors
    ///
    /// The error of the logger when it cannot say which levels it takes.
    pub(crate) fn channel(&self, py: Python<'_>) -> PyResult<(ChannelSteps, Drain)> {
        let logger = self.logger.bind(py);
        let takes = |level: u8| -> PyResult<bool> {
            logger.call_method1("isEnabledFor", (level,))?.is_truthy()
        };
        let (sender, receiver) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let steps = ChannelSteps {
            sender,
            info: takes(INFO)?,
            debug: takes(DEBUG)?,
            stop: Arc::clone(&stop),
        };
        let drain = Drain {
            receiver,
            logger: self.logger.clone_ref(py),
            counts: Counts::new(self.quantities.clone(), self.unit),
            stop,
        };
        Ok((steps, drain))
    }
}

/// A loop whose steps are counted together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Stage {
    /// A loop of fits.
    Fits(FitStage),
    /// The sides of the profiles.
    Sides,
    /// A loop of model runs.
    Runs,
    /// The generations of a run of the sampler: of the method's own run, or
    /// of the run of a replicate of a calibrated `η`.
    Generations(Option<usize>),
    /// The runs of the replicates of a calibrated `η`.
    Samples,
    /// The rounds that raise the largest `η` of a calibrated `η`.
    Growths,
}

impl Stage {
    /// The level of the records of this stage: `DEBUG` for the starts of a
    /// replicate and the generations of the run of a replicate, `INFO` for
    /// the outermost loops.
    fn level(self) -> u8 {
        match self {
            Self::Fits(FitStage::Starts(Some(_))) | Self::Generations(Some(_)) => DEBUG,
            _ => INFO,
        }
    }
}

/// A stage whose steps count up to a total that its beginning gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Counted {
    /// A loop of fits.
    Fits(FitStage),
    /// The sides of the profiles.
    Sides,
    /// A loop of model runs.
    Runs,
    /// The runs of the replicates of a calibrated `η`.
    Samples,
}

impl Counted {
    fn stage(self) -> Stage {
        match self {
            Self::Fits(stage) => Stage::Fits(stage),
            Self::Sides => Stage::Sides,
            Self::Runs => Stage::Runs,
            Self::Samples => Stage::Samples,
        }
    }
}

/// A loop of fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FitStage {
    /// The starts of a fit: of the method's own fit, or of a replicate.
    Starts(Option<usize>),
    /// The fits of the replicates.
    Replicates,
    /// The refits of the replicates with each quantity held.
    Refits,
}

/// How a fit ended.
#[derive(Debug, Clone, Copy)]
enum FitEnd {
    /// It failed.
    Failed,
    /// A refit, which gives no loss of its own, ended.
    Refitted,
    /// It ended at `loss`, converged or not.
    Ended { loss: f64, converged: bool },
}

/// What a step tells, owned, so it crosses to the calling thread.
#[derive(Debug, Clone, Copy)]
enum Step {
    /// A stage begins with `total` steps.
    Began { stage: Counted, total: usize },
    /// Fit `index` of `stage` ended: a start, a replicate or a refit.
    Fit {
        stage: FitStage,
        index: usize,
        end: FitEnd,
    },
    /// Side `side` of quantity `quantity` ended, with its number of points
    /// and why, or it failed.
    Side {
        quantity: usize,
        side: usize,
        end: Option<(usize, ProfileEnd)>,
    },
    /// A point of a loop of model runs ran.
    Ran,
    /// Generation `index` of the run of `replicate`, or of the method's own
    /// run, reached `level` on the way to `last`.
    Generation {
        replicate: Option<usize>,
        index: usize,
        level: f64,
        last: f64,
        effective_size: f64,
        acceptance_rate: f64,
    },
    /// The run of replicate `index` of a calibrated `η` ended, or failed.
    Sampled { index: usize, failed: bool },
    /// Round `round` raised the largest `η` to `eta`.
    Growth { round: usize, eta: f64 },
}

impl Step {
    fn stage(&self) -> Stage {
        match self {
            Self::Began { stage, .. } => stage.stage(),
            Self::Fit { stage, .. } => Stage::Fits(*stage),
            Self::Side { .. } => Stage::Sides,
            Self::Ran => Stage::Runs,
            Self::Generation { replicate, .. } => Stage::Generations(*replicate),
            Self::Sampled { .. } => Stage::Samples,
            Self::Growth { .. } => Stage::Growths,
        }
    }
}

/// How a fit of `result` ended.
fn fit_end(result: Result<&CalibrationResult, ()>) -> FitEnd {
    match result {
        Ok(result) => FitEnd::Ended {
            loss: result.final_loss,
            converged: result.converged,
        },
        Err(()) => FitEnd::Failed,
    }
}

/// The step of an event of the starts of `replicate`, or of the method's
/// own fit.
fn start_step(replicate: Option<usize>, event: StartEvent<'_>) -> Step {
    let stage = FitStage::Starts(replicate);
    match event {
        StartEvent::Began { total } => Step::Began {
            stage: Counted::Fits(stage),
            total,
        },
        StartEvent::Finished { index, outcome } => Step::Fit {
            stage,
            index,
            end: fit_end(outcome.result.as_ref().map_err(|_| ())),
        },
    }
}

/// The receiver of the events of a job: it sends the steps of the levels
/// that the logger takes into the channel of a [`Drain`], and tells the
/// loops of the job to stop once the drain sets its stop.
pub(crate) struct ChannelSteps {
    sender: Sender<Step>,
    info: bool,
    debug: bool,
    stop: Arc<AtomicBool>,
}

impl ChannelSteps {
    /// Send `step` when the logger takes its level, and a stage only when
    /// it has a step. A drain that is gone takes nothing more.
    fn keep(&self, step: Step) {
        let taken = match step.stage().level() {
            DEBUG => self.debug,
            _ => self.info,
        };
        if taken && !matches!(step, Step::Began { total: 0, .. }) {
            let _ = self.sender.send(step);
        }
    }

    /// Whether the drain has set the stop.
    fn stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
}

impl Steps<StartEvent<'_>> for ChannelSteps {
    fn send(&self, event: StartEvent<'_>) {
        self.keep(start_step(None, event));
    }

    fn stopped(&self) -> bool {
        self.stop()
    }
}

impl Steps<RunnerEvent<'_>> for ChannelSteps {
    fn send(&self, event: RunnerEvent<'_>) {
        self.keep(match event {
            RunnerEvent::Began { stage, total } => Step::Began {
                stage: Counted::Fits(match stage {
                    RunnerStage::Fits => FitStage::Replicates,
                    RunnerStage::Refits => FitStage::Refits,
                }),
                total,
            },
            RunnerEvent::Fitted { index, outcome } => Step::Fit {
                stage: FitStage::Replicates,
                index,
                end: fit_end(outcome.as_ref().map_err(|_| ())),
            },
            RunnerEvent::Start { index, event } => start_step(Some(index), event),
            RunnerEvent::Refitted { index, outcome } => Step::Fit {
                stage: FitStage::Refits,
                index,
                end: match outcome {
                    Ok(_) => FitEnd::Refitted,
                    Err(_) => FitEnd::Failed,
                },
            },
        });
    }

    fn stopped(&self) -> bool {
        self.stop()
    }
}

impl Steps<SideEvent<'_>> for ChannelSteps {
    fn send(&self, event: SideEvent<'_>) {
        self.keep(match event {
            SideEvent::Began { total } => Step::Began {
                stage: Counted::Sides,
                total,
            },
            SideEvent::Finished {
                quantity,
                side,
                outcome,
            } => Step::Side {
                quantity,
                side,
                end: outcome.ok().map(|end| (end.points, end.end)),
            },
        });
    }

    fn stopped(&self) -> bool {
        self.stop()
    }
}

impl Steps<RunEvent> for ChannelSteps {
    fn send(&self, event: RunEvent) {
        self.keep(match event {
            RunEvent::Began { total } => Step::Began {
                stage: Counted::Runs,
                total,
            },
            RunEvent::Ran { .. } => Step::Ran,
        });
    }

    fn stopped(&self) -> bool {
        self.stop()
    }
}

impl Steps<SmcEvent<'_>> for ChannelSteps {
    fn send(&self, event: SmcEvent<'_>) {
        self.keep(match event {
            SmcEvent::Generation {
                replicate,
                index,
                step,
                last,
            } => Step::Generation {
                replicate,
                index,
                level: step.level,
                last,
                effective_size: step.effective_size,
                acceptance_rate: step.acceptance_rate,
            },
            SmcEvent::Replicates { total } => Step::Began {
                stage: Counted::Samples,
                total,
            },
            SmcEvent::Replicate { index, outcome } => Step::Sampled {
                index,
                failed: outcome.is_err(),
            },
            SmcEvent::Growth { round, eta } => Step::Growth { round, eta },
        });
    }

    fn stopped(&self) -> bool {
        self.stop()
    }
}

/// The calling thread's end of the channel: it counts the steps, writes
/// their records and sets the stop of the job.
pub(crate) struct Drain {
    receiver: Receiver<Step>,
    logger: Py<PyAny>,
    counts: Counts,
    stop: Arc<AtomicBool>,
}

impl Drain {
    /// Write the record of each step until the job drops its steps, with
    /// the GIL for each record only, and read the signals of Python at
    /// least every [`SIGNALS`].
    ///
    /// A record whose handler raises an `Exception` is dropped, so a
    /// logging failure never costs the result of the method. The first
    /// error of the signals, or the first other error of a record, such as
    /// a `KeyboardInterrupt`, sets the stop, ends the records and is
    /// returned once the job has ended.
    pub(crate) fn drain(mut self) -> Option<PyErr> {
        let mut interrupt = None;
        let mut read = Instant::now();
        loop {
            let step = match self.receiver.recv_timeout(SIGNALS) {
                Ok(step) => Some(step),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => break,
            };
            if interrupt.is_some() {
                continue;
            }
            if let Some(message) = step.and_then(|step| {
                let message = self.counts.message(step)?;
                Some((step.stage().level(), message))
            }) {
                interrupt = self.log(message);
            }
            if interrupt.is_none() && read.elapsed() >= SIGNALS {
                read = Instant::now();
                interrupt = Python::attach(|py| py.check_signals()).err();
            }
            if interrupt.is_some() {
                self.stop.store(true, Ordering::Relaxed);
            }
        }
        interrupt
    }

    /// Write the record `message` at its level, and give the error of a
    /// handler that is no `Exception`.
    fn log(&self, (level, message): (u8, String)) -> Option<PyErr> {
        Python::attach(
            |py| match self.logger.bind(py).call_method1("log", (level, message)) {
                Err(error) if !error.is_instance_of::<PyException>(py) => Some(error),
                _ => None,
            },
        )
    }
}

/// The counts of the steps of each stage, the names of the quantities of
/// the profiles, and what the method fits once per perturbation.
struct Counts {
    quantities: Vec<String>,
    unit: FitUnit,
    /// The finished steps and the total of each stage.
    stages: HashMap<Stage, (usize, usize)>,
}

impl Counts {
    fn new(quantities: Vec<String>, unit: FitUnit) -> Self {
        Self {
            quantities,
            unit,
            stages: HashMap::new(),
        }
    }

    /// Count `step` and give its record, or `None` for a point of a loop of
    /// model runs that does not reach the next 1 % of its points.
    fn message(&mut self, step: Step) -> Option<String> {
        let unit = self.unit;
        let (done, total) = self.stages.entry(step.stage()).or_insert((0, 0));
        let mut count = || {
            *done += 1;
            format!("({done}/{total})")
        };
        Some(match step {
            Step::Began {
                stage,
                total: steps,
            } => {
                *done = 0;
                *total = steps;
                began(stage, steps, unit)
            }
            Step::Fit { stage, index, end } => finished(stage, index, end, unit, &count()),
            Step::Side {
                quantity,
                side,
                end,
            } => {
                let name = &self.quantities[quantity];
                let side = if side == 0 { "lower" } else { "upper" };
                let count = count();
                match end {
                    Some((points, end)) => format!(
                        "Profile of {name}, {side} side: {} after {points} points {count}",
                        crate::profiles::end_name(end)
                    ),
                    None => format!("Profile of {name}, {side} side failed {count}"),
                }
            }
            Step::Ran => {
                *done += 1;
                let percent = *done * 100 / *total;
                if percent == (*done - 1) * 100 / *total {
                    return None;
                }
                format!("Ran {done} of {total} points ({percent}%)")
            }
            Step::Generation {
                replicate,
                index,
                level,
                last,
                effective_size,
                acceptance_rate,
            } => {
                let generation = format!(
                    "generation {index}: level {level:.6e} toward {last:.6e}, effective size \
                     {effective_size:.1}, acceptance {acceptance_rate:.3}"
                );
                match replicate {
                    Some(replicate) => format!("Replicate {replicate}: {generation}"),
                    None => capital(&generation),
                }
            }
            Step::Sampled { index, failed } => {
                let end = if failed { "failed" } else { "sampled" };
                format!("Replicate {index} {end} {}", count())
            }
            Step::Growth { round, eta } => format!("Round {round} raised eta to {eta:.6e}"),
        })
    }
}

/// `text` with its first letter in upper case.
fn capital(text: &str) -> String {
    text[..1].to_uppercase() + &text[1..]
}

/// `count` things named `name`, in the plural when they are not one.
fn plural(count: usize, name: &str) -> String {
    if count == 1 {
        format!("1 {name}")
    } else {
        format!("{count} {name}s")
    }
}

/// The record of a stage of `total` steps that begins, in a method whose
/// fits are each a `unit`.
fn began(stage: Counted, total: usize, unit: FitUnit) -> String {
    let unit = unit.name();
    match stage {
        Counted::Fits(FitStage::Starts(None)) => format!("Fitting {}", plural(total, "start")),
        Counted::Fits(FitStage::Starts(Some(index))) => format!(
            "{} {index}: fitting {}",
            capital(unit),
            plural(total, "start")
        ),
        Counted::Fits(FitStage::Replicates) => format!("Fitting {}", plural(total, unit)),
        Counted::Fits(FitStage::Refits) => {
            format!("Refitting {} with each quantity held", plural(total, unit))
        }
        Counted::Sides => format!("Tracing {}", plural(total, "profile side")),
        Counted::Runs => format!("Running the model at {}", plural(total, "point")),
        Counted::Samples => format!("Sampling {}", plural(total, "replicate")),
    }
}

/// The record of fit `index` of `stage`, a `unit`, which ended with `end`.
fn finished(stage: FitStage, index: usize, end: FitEnd, unit: FitUnit, count: &str) -> String {
    let unit = capital(unit.name());
    let name = match stage {
        FitStage::Starts(None) => format!("Start {index}"),
        FitStage::Starts(Some(fit)) => format!("{unit} {fit}: start {index}"),
        FitStage::Replicates | FitStage::Refits => format!("{unit} {index}"),
    };
    match end {
        FitEnd::Failed => format!("{name} failed {count}"),
        FitEnd::Refitted => format!("{name} refitted {count}"),
        FitEnd::Ended { loss, converged } => {
            let convergence = if converged {
                "converged"
            } else {
                "did not converge"
            };
            format!("{name} finished {count}: loss {loss:.6e}, {convergence}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn began(stage: Counted, total: usize) -> Step {
        Step::Began { stage, total }
    }

    fn fit(stage: FitStage, index: usize, end: FitEnd) -> Step {
        Step::Fit { stage, index, end }
    }

    #[test]
    fn a_stage_of_no_step_has_no_record() {
        let (sender, receiver) = channel();
        let steps = ChannelSteps {
            sender,
            info: true,
            debug: true,
            stop: Arc::default(),
        };
        steps.send(RunnerEvent::Began {
            stage: RunnerStage::Refits,
            total: 0,
        });
        drop(steps);
        assert_eq!(receiver.into_iter().count(), 0);
    }

    #[test]
    fn the_steps_of_a_stage_count_up_to_its_total() {
        let mut counts = Counts::new(Vec::new(), FitUnit::Replicate);
        let ended = FitEnd::Ended {
            loss: 2.0,
            converged: false,
        };
        let messages: Vec<_> = [
            began(Counted::Fits(FitStage::Replicates), 2),
            fit(FitStage::Replicates, 1, FitEnd::Failed),
            fit(FitStage::Replicates, 0, ended),
        ]
        .into_iter()
        .filter_map(|step| counts.message(step))
        .collect();
        assert_eq!(
            messages,
            [
                "Fitting 2 replicates",
                "Replicate 1 failed (1/2)",
                "Replicate 0 finished (2/2): loss 2.000000e0, did not converge",
            ]
        );
    }

    #[test]
    fn a_refit_that_fails_is_logged_as_failed() {
        let (sender, receiver) = channel();
        let steps = ChannelSteps {
            sender,
            info: true,
            debug: false,
            stop: Arc::default(),
        };
        steps.send(RunnerEvent::Began {
            stage: RunnerStage::Refits,
            total: 1,
        });
        steps.send(RunnerEvent::Refitted {
            index: 2,
            outcome: &Err("no fit".to_string()),
        });
        drop(steps);
        let mut counts = Counts::new(Vec::new(), FitUnit::Replicate);
        let messages: Vec<_> = receiver
            .into_iter()
            .filter_map(|step| counts.message(step))
            .collect();
        assert_eq!(
            messages,
            [
                "Refitting 1 replicate with each quantity held",
                "Replicate 2 failed (1/1)",
            ]
        );
    }

    #[test]
    fn the_starts_of_each_replicate_count_apart_and_log_at_debug() {
        let mut counts = Counts::new(Vec::new(), FitUnit::Replicate);
        let ended = FitEnd::Ended {
            loss: 1.0,
            converged: true,
        };
        let steps = [
            began(Counted::Fits(FitStage::Starts(Some(0))), 1),
            began(Counted::Fits(FitStage::Starts(Some(1))), 1),
            fit(FitStage::Starts(Some(1)), 0, ended),
            fit(FitStage::Starts(Some(0)), 0, ended),
        ];
        assert!(steps.iter().all(|step| step.stage().level() == DEBUG));
        let messages: Vec<_> = steps
            .into_iter()
            .filter_map(|step| counts.message(step))
            .collect();
        assert_eq!(
            messages,
            [
                "Replicate 0: fitting 1 start",
                "Replicate 1: fitting 1 start",
                "Replicate 1: start 0 finished (1/1): loss 1.000000e0, converged",
                "Replicate 0: start 0 finished (1/1): loss 1.000000e0, converged",
            ]
        );
    }

    #[test]
    fn the_points_of_a_loop_of_runs_log_at_each_percent() {
        let mut counts = Counts::new(Vec::new(), FitUnit::Replicate);
        let mut messages = vec![counts.message(began(Counted::Runs, 250)).unwrap()];
        messages.extend((0..250).filter_map(|_| counts.message(Step::Ran)));
        assert_eq!(messages.len(), 101);
        assert_eq!(messages[0], "Running the model at 250 points");
        assert_eq!(messages[1], "Ran 3 of 250 points (1%)");
        assert_eq!(messages[100], "Ran 250 of 250 points (100%)");

        let mut counts = Counts::new(Vec::new(), FitUnit::Replicate);
        counts.message(began(Counted::Runs, 2));
        let few: Vec<_> = (0..2).filter_map(|_| counts.message(Step::Ran)).collect();
        assert_eq!(few, ["Ran 1 of 2 points (50%)", "Ran 2 of 2 points (100%)"]);
    }

    #[test]
    fn a_generation_gives_its_level_and_the_run_of_a_replicate_logs_at_debug() {
        let mut counts = Counts::new(Vec::new(), FitUnit::Replicate);
        let generation = |replicate| Step::Generation {
            replicate,
            index: 3,
            level: 0.25,
            last: 1.0,
            effective_size: 812.0,
            acceptance_rate: 0.31,
        };
        assert_eq!(generation(None).stage().level(), INFO);
        assert_eq!(generation(Some(2)).stage().level(), DEBUG);
        let messages: Vec<_> = [
            generation(None),
            generation(Some(2)),
            began(Counted::Samples, 2),
            Step::Sampled {
                index: 1,
                failed: false,
            },
            Step::Sampled {
                index: 0,
                failed: true,
            },
            Step::Growth { round: 1, eta: 8.0 },
        ]
        .into_iter()
        .filter_map(|step| counts.message(step))
        .collect();
        assert_eq!(
            messages,
            [
                "Generation 3: level 2.500000e-1 toward 1.000000e0, effective size 812.0, \
                 acceptance 0.310",
                "Replicate 2: generation 3: level 2.500000e-1 toward 1.000000e0, effective \
                 size 812.0, acceptance 0.310",
                "Sampling 2 replicates",
                "Replicate 1 sampled (1/2)",
                "Replicate 0 failed (2/2)",
                "Round 1 raised eta to 8.000000e0",
            ]
        );
    }

    #[test]
    fn the_fits_of_a_method_are_named_by_its_unit() {
        let mut counts = Counts::new(Vec::new(), FitUnit::Block);
        let ended = FitEnd::Ended {
            loss: 1.0,
            converged: true,
        };
        let messages: Vec<_> = [
            began(Counted::Fits(FitStage::Replicates), 1),
            began(Counted::Fits(FitStage::Starts(Some(0))), 1),
            fit(FitStage::Starts(Some(0)), 0, ended),
            fit(FitStage::Replicates, 0, ended),
        ]
        .into_iter()
        .filter_map(|step| counts.message(step))
        .collect();
        assert_eq!(
            messages,
            [
                "Fitting 1 block",
                "Block 0: fitting 1 start",
                "Block 0: start 0 finished (1/1): loss 1.000000e0, converged",
                "Block 0 finished (1/1): loss 1.000000e0, converged",
            ]
        );
    }
}
