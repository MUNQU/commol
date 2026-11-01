# Progress

[Back to Design decisions](../design-decisions.md)

How a method that runs for long tells where it is: one log record per step it
finishes; and how a Ctrl-C stops it.

## A long method logs each step from the calling thread

A loop of a long method sends an event to a receiver when a stage begins and
when a step of it finishes: the port `Steps<E>` of
`crates/commol-core/src/progress.rs`, with one `send`. Each crate owns the events
of its loops: `StartEvent` of the starts (`optimize_starts_with_observer`),
`RunnerEvent` of the fits and refits of a `ReplicateRunner` (`with_steps`),
`SideEvent` of the sides of `ProfileTracer::trace_all`. An event borrows the
outcome of its step, so a loop holds its receiver as
`&(dyn for<'e> Steps<Event<'e>> + '_)`, and `NoSteps` drops every event.

The receiver of the extension is `ChannelSteps` (`py-commol/src/records.rs`).
A binding gives its job one through `parallel_logged`
(`py-commol/src/parallelism.rs`), with the logger of the module of its runner:
the job runs on a thread of its own, which waits for the pool, and the workers
send owned steps into a channel. The calling thread drains the channel: it
counts the steps of each stage, and of the starts of each replicate, and
writes one record per step with the GIL. So a record has the thread and the
`contextvars` context of the caller, the count of a stage rises by one, and a
worker never waits for the GIL. The outermost loop of a method logs at `INFO`,
the starts of a replicate at `DEBUG`; a level the logger does not take sends
nothing, and a stage of no step logs no record. A handler that raises an
`Exception` loses its record and never the result; a `KeyboardInterrupt` is
raised once the job has ended.

An identifiability method logs its start and its end, with its duration, on
`commol.identifiability` (`_logger` of `IdentifiabilityMethod`, read by
`run_method`). A build logs one `INFO` record, with the outputs read from the
engine; each element of the builder logs at `DEBUG`.

| Alternative | Why not |
|---|---|
| The crate `log` with `pyo3-log`, or `tracing` with a layer | The core would own the names, levels and nesting of the Python loggers; `pyo3-log` keeps the levels it read, so a later `setLevel` is not seen; a global logger also forwards the `log` records of cranelift and diffsol; it is state of the process, and a Rust test needs a global subscriber. |
| A worker takes the GIL and logs | Each record gets a new `Dummy-N` thread and an empty `contextvars` context, two counts can come out of order, and a worker waits for the GIL. |
| A `began` and a `finished` with a depth | One type cannot serve a replicate and its starts, and the start loops of replicates run at once, so a depth or a bare `began(total)` cannot say which loop a step is of. |
| A Python callback argument of each method | Every public signature changes for a concern of the application. |
| A record per point of a profile | A side takes up to `max_steps` refits per quantity: thousands of records for a prediction profile. |

## The step of each method is its outermost loop

| Method | Step | Event |
|---|---|---|
| A bootstrap, a block jackknife, jackknife+, EnbPI, rolling conformal | One fit of the `ReplicateRunner`: a replicate, a block left out, an origin; the binding names it (`StepLog::with_unit`, `FitUnit`) | `RunnerEvent` |
| A Gibbs or Bayesian posterior, ABC-SMC | One generation of `SmcSampler::advance`, with its level and the last level | `SmcEvent::Generation` |
| A calibrated `η` by bootstrap coverage | One replicate run, then one round that raises `η`; the generations of a replicate log at `DEBUG` | `SmcEvent` |
| ABC rejection, a Laplace mixture, a sensitivity method, a predictive | One model run: a draw of the prior, a draw of the importance sampler, a point of `BatchRunner`, a set of `run_sets` | `RunEvent` |

A model run takes milliseconds, so the drain writes a record of a loop of
model runs at each 1 % of its points only, at most 100 per stage. The number
of generations of SMC is not known before the run, so a generation gives its
level and the last level, not a count over a total. Split and adaptive
conformal, Laplace, the sandwich and the delta band have no long loop, and
log no step.

| Alternative | Why not |
|---|---|
| One record per model run | A Sobol design of 12 000 points writes 12 000 lines and takes the GIL per point. |
| A record every N seconds | Not deterministic, so no test can check the records. |

## Ctrl-C stops a long method

`Steps::stopped` tells a loop to stop; its default never stops. A loop that
sends events reads it before each step, the starts of a replicate too, and a
step that begins once it is set does not run: it fails with `STOPPED`, the
error of its method. The method ends soon, and the binding drops its result
for the `KeyboardInterrupt`.

The step is the unit of the stop, so a step that has begun runs to its end:
the moves of a generation of SMC, the walk of a side of a profile, and the
draws of the prior that start a run of Gibbs, Bayesian or ABC SMC, or a
replicate of a calibrated eta, all end before the method sees the stop. Each
draw of ABC rejection is a step of its own, which reads the stop.

`ChannelSteps` reads a flag that it shares with the drain. The drain waits
for a step at most 100 ms (`recv_timeout`), and reads the signals of Python
(`check_signals`, with the GIL) at least that often, also when the logger
takes no record. An error of the signals, or an error of a record that is no
`Exception`, sets the flag; the drain then writes no record until the job
ends, and returns the error. `check_signals` acts only on the main thread,
the one thread to which Python gives a `KeyboardInterrupt`.

| Alternative | Why not |
|---|---|
| A `Stop` argument of each loop | It changes the signature of every loop; `stopped` comes with the receiver the loops already hold. |
| Kill the threads of the pool | Rust cannot kill a thread of rayon safely, and a fit stopped in the middle leaves no result to drop cleanly. |
| A stop only between methods | One bootstrap is the long run. |
| A typed error `Stopped` of each method | It changes the error of each crate for a value Python never sees. |

The cost is one atomic load per step, and a wake of the calling thread every
100 ms; that thread waits for the job and holds no thread of the pool.

**Guards:** `crates/commol-core/src/progress.rs` `tests`;
`crates/commol-calibration/tests/test_starts.rs::no_start_runs_once_the_steps_stop`;
`crates/commol-inference/tests/test_profiles.rs::no_replicate_and_no_refit_runs_once_the_steps_stop`,
`::no_side_runs_once_the_steps_stop`,
`::no_start_of_a_replicate_runs_once_the_steps_stop`;
`crates/commol-inference/tests/test_smc.rs::a_run_sends_each_generation_with_its_level`,
`::the_runs_of_a_calibrated_eta_send_their_replicates_and_generations`,
`::no_generation_runs_once_the_steps_stop`,
`::a_calibrated_eta_stops_inside_a_round`,
`::a_draw_of_abc_rejection_that_fails_is_sent_as_run`,
`::no_draw_of_abc_rejection_runs_once_the_steps_stop`;
`crates/commol-inference/tests/test_predictive.rs::a_predictive_sends_each_draw_and_runs_none_once_the_steps_stop`;
`crates/commol-inference/tests/test_local.rs::a_mixture_sends_each_draw_and_runs_none_once_the_steps_stop`,
`::a_draw_of_an_importance_sample_that_fails_is_sent_as_run`;
`crates/commol-inference/tests/test_global_sensitivity.rs` (the batch runner);
`crates/commol-calibration/tests/test_starts.rs::the_starts_send_their_beginning_then_each_start_that_finishes`;
`crates/commol-inference/tests/test_profiles.rs::a_bootstrap_threshold_sends_its_fits_then_its_refits`;
`py-commol/src/records.rs` `tests`;
`tests/calibration/test_progress.py::test_each_finished_start_is_logged`;
`tests/uncertainty/test_progress.py` (`test_every_long_method_logs_its_steps`,
`test_an_interrupt_stops_the_steps`,
`test_an_interrupt_stops_a_method_that_logs_nothing`);
`tests/suite/test_logging.py::test_a_build_logs_one_record`.
