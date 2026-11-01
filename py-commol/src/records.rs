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
//! log at `DEBUG`. A level that the logger does not take sends nothing.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};

use commol_calibration::{CalibrationResult, StartEvent};
use commol_core::Steps;
use commol_inference::profiles::{ProfileEnd, SideEvent};
use commol_inference::resampling::{RunnerEvent, RunnerStage};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

/// The level `DEBUG` of Python's `logging`.
const DEBUG: u8 = 10;
/// The level `INFO` of Python's `logging`.
const INFO: u8 = 20;

/// Where the records of a method go: the logger of its module, and the
/// names of the quantities its profiles trace, in their order.
pub(crate) struct StepLog {
    logger: Py<PyAny>,
    quantities: Vec<String>,
}

impl StepLog {
    /// The records of a method that logs to `logger`.
    pub(crate) fn new(logger: Py<PyAny>) -> Self {
        Self {
            logger,
            quantities: Vec::new(),
        }
    }

    /// These records, with the names of the quantities of the profiles.
    pub(crate) fn with_quantities(self, quantities: Vec<String>) -> Self {
        Self { quantities, ..self }
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
        let steps = ChannelSteps {
            sender,
            info: takes(INFO)?,
            debug: takes(DEBUG)?,
        };
        let drain = Drain {
            receiver,
            logger: self.logger.clone_ref(py),
            counts: Counts::new(self.quantities.clone()),
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
}

impl Stage {
    /// The level of the records of this stage: `DEBUG` for the starts of a
    /// replicate, `INFO` for the outermost loops.
    fn level(self) -> u8 {
        match self {
            Self::Fits(FitStage::Starts(Some(_))) => DEBUG,
            _ => INFO,
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
    Began { stage: Stage, total: usize },
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
}

impl Step {
    fn stage(&self) -> Stage {
        match self {
            Self::Began { stage, .. } => *stage,
            Self::Fit { stage, .. } => Stage::Fits(*stage),
            Self::Side { .. } => Stage::Sides,
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
            stage: Stage::Fits(stage),
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
/// that the logger takes into the channel of a [`Drain`].
pub(crate) struct ChannelSteps {
    sender: Sender<Step>,
    info: bool,
    debug: bool,
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
}

impl Steps<StartEvent<'_>> for ChannelSteps {
    fn send(&self, event: StartEvent<'_>) {
        self.keep(start_step(None, event));
    }
}

impl Steps<RunnerEvent<'_>> for ChannelSteps {
    fn send(&self, event: RunnerEvent<'_>) {
        self.keep(match event {
            RunnerEvent::Began { stage, total } => Step::Began {
                stage: Stage::Fits(match stage {
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
}

impl Steps<SideEvent<'_>> for ChannelSteps {
    fn send(&self, event: SideEvent<'_>) {
        self.keep(match event {
            SideEvent::Began { total } => Step::Began {
                stage: Stage::Sides,
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
}

/// The calling thread's end of the channel: it counts the steps and writes
/// their records.
pub(crate) struct Drain {
    receiver: Receiver<Step>,
    logger: Py<PyAny>,
    counts: Counts,
}

impl Drain {
    /// Write the record of each step until the job drops its steps, with
    /// the GIL for each record only.
    ///
    /// A record whose handler raises an `Exception` is dropped, so a
    /// logging failure never costs the result of the method. The first
    /// other error, such as a `KeyboardInterrupt`, ends the records and is
    /// returned once the job has ended.
    pub(crate) fn drain(mut self) -> Option<PyErr> {
        let mut interrupt = None;
        while let Ok(step) = self.receiver.recv() {
            if interrupt.is_some() {
                continue;
            }
            let message = self.counts.message(step);
            Python::attach(|py| {
                let logged = self
                    .logger
                    .bind(py)
                    .call_method1("log", (step.stage().level(), message));
                if let Err(error) = logged
                    && !error.is_instance_of::<PyException>(py)
                {
                    interrupt = Some(error);
                }
            });
        }
        interrupt
    }
}

/// The counts of the steps of each stage, and the names of the quantities
/// of the profiles.
struct Counts {
    quantities: Vec<String>,
    /// The finished steps and the total of each stage.
    stages: HashMap<Stage, (usize, usize)>,
}

impl Counts {
    fn new(quantities: Vec<String>) -> Self {
        Self {
            quantities,
            stages: HashMap::new(),
        }
    }

    /// Count `step` and give its record.
    fn message(&mut self, step: Step) -> String {
        let (done, total) = self.stages.entry(step.stage()).or_insert((0, 0));
        let mut count = || {
            *done += 1;
            format!("({done}/{total})")
        };
        match step {
            Step::Began {
                stage,
                total: steps,
            } => {
                *done = 0;
                *total = steps;
                began(stage, steps)
            }
            Step::Fit { stage, index, end } => finished(stage, index, end, &count()),
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
        }
    }
}

/// `count` things named `name`, in the plural when they are not one.
fn plural(count: usize, name: &str) -> String {
    if count == 1 {
        format!("1 {name}")
    } else {
        format!("{count} {name}s")
    }
}

/// The record of a stage of `total` steps that begins.
fn began(stage: Stage, total: usize) -> String {
    match stage {
        Stage::Fits(FitStage::Starts(None)) => format!("Fitting {}", plural(total, "start")),
        Stage::Fits(FitStage::Starts(Some(replicate))) => {
            format!("Replicate {replicate}: fitting {}", plural(total, "start"))
        }
        Stage::Fits(FitStage::Replicates) => format!("Fitting {}", plural(total, "replicate")),
        Stage::Fits(FitStage::Refits) => format!(
            "Refitting {} with each quantity held",
            plural(total, "replicate")
        ),
        Stage::Sides => format!("Tracing {}", plural(total, "profile side")),
    }
}

/// The record of fit `index` of `stage`, which ended with `end`.
fn finished(stage: FitStage, index: usize, end: FitEnd, count: &str) -> String {
    let name = match stage {
        FitStage::Starts(None) => format!("Start {index}"),
        FitStage::Starts(Some(replicate)) => format!("Replicate {replicate}: start {index}"),
        FitStage::Replicates | FitStage::Refits => format!("Replicate {index}"),
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

    fn began(stage: Stage, total: usize) -> Step {
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
        let mut counts = Counts::new(Vec::new());
        let ended = FitEnd::Ended {
            loss: 2.0,
            converged: false,
        };
        let messages: Vec<_> = [
            began(Stage::Fits(FitStage::Replicates), 2),
            fit(FitStage::Replicates, 1, FitEnd::Failed),
            fit(FitStage::Replicates, 0, ended),
        ]
        .into_iter()
        .map(|step| counts.message(step))
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
        let mut counts = Counts::new(Vec::new());
        let messages: Vec<_> = receiver
            .into_iter()
            .map(|step| counts.message(step))
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
        let mut counts = Counts::new(Vec::new());
        let ended = FitEnd::Ended {
            loss: 1.0,
            converged: true,
        };
        let steps = [
            began(Stage::Fits(FitStage::Starts(Some(0))), 1),
            began(Stage::Fits(FitStage::Starts(Some(1))), 1),
            fit(FitStage::Starts(Some(1)), 0, ended),
            fit(FitStage::Starts(Some(0)), 0, ended),
        ];
        assert!(steps.iter().all(|step| step.stage().level() == DEBUG));
        let messages: Vec<_> = steps.into_iter().map(|step| counts.message(step)).collect();
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
}
