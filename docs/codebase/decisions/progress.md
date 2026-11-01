# Progress

[Back to Design decisions](../design-decisions.md)

How a method that runs for long tells where it is: one log record per step it
finishes.

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

**Guards:** `crates/commol-core/src/progress.rs` `tests`;
`crates/commol-calibration/tests/test_starts.rs::the_starts_send_their_beginning_then_each_start_that_finishes`;
`crates/commol-inference/tests/test_profiles.rs::a_bootstrap_threshold_sends_its_fits_then_its_refits`;
`py-commol/src/records.rs` `tests`;
`tests/calibration/test_progress.py::test_each_finished_start_is_logged`;
`tests/uncertainty/test_progress.py`;
`tests/suite/test_logging.py::test_a_build_logs_one_record`.
