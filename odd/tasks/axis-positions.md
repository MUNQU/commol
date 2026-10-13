# Refactor: one kind of position on the axis, carried by the results

Goal: every public name that takes or gives a position on the axis takes or
gives a `float`, the engine alone decides which positions its model type
accepts, and results carry their positions. Reports, windows, plots and
ensembles read a value at a position, not at a list index.
Motivation: audit T0 of `odd/tasks/axis-scale-refactor.md` (themes A, B and
C). Part of the API was written for the difference engine only, where a list
index is a step. A differential model, or any later type, does not fit it.
Branch: `refactor/axis-positions` from `develop`, after PR #102 is merged.
First of four ODDs; the order of all pending work is in
`odd/execution-order.md`.
Engram mirror: topic `odd/axis-positions/tasks` (project `commol`).
Runners: the Rust tests, clippy and rustdoc (from the root), `maturin develop
--release`, then `pytest`, `pytest -m docs`, `ruff check .`, `ty check commol`
and `vulture` (from `py-commol/`), `mkdocs build --strict` (from the root).

## Current state (verified 2026-10-13 on `develop` at `34d75eb` with PR #102)

The findings, with their places, are in `odd/tasks/axis-scale-refactor.md`,
section "Audit T0":

- A1–A5: a position has several names and types. The engine trait has
  `run(num_steps: u32)` and `run_at_steps_into_buffer(&[u32])` beside
  `run_at_positions_into_buffer(&[f64])`. `Simulation.run`, the windows, the
  reports, `TimePattern.window`/`linear_ramp` and time-series parameters take
  `int`. `step` names both a position and a count; `t` is documented as an
  alias of `step`.
- B1–B3: results carry no positions. Reports, windows, plots and the ensemble
  statistics read a list index as the position;
  `statistics_calculator.py` picks between an index and an engine query with
  `isinstance(step, int)`.
- C1–C3: Python repeats the rules of the engine on which positions a model
  accepts (`context/calibration.py:56`, `context/dynamics.py:282`,
  `api/simulation.py:203`).

Already on positions: the calibration crate (`observed_time_steps: Vec<f64>`,
`run_at_positions_into_buffer`), `Simulation.run_at`, `ObservedDataPoint` and
`CalibrationConstraint` (`int | float`), impulses (`float`), `observed_report`
(`ceil(step / period) − 1`).

## Specs

Every spec follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — The engine owns positions

- Fix: the engine trait names a position one way, as `f64` (D2). Each engine
  checks the positions where it receives them and names the object that holds
  the first one it rejects: the observation or constraint when it builds a
  calibration problem, the impulse transition or time-series parameter when
  it builds a model. A difference engine rejects a non-integer one. Python
  passes positions on and shows the engine's error: C1–C3 are removed. Today
  only the Python check names the observation (`context/calibration.py:56`);
  `integer_steps` and `calibration_problem.rs:460` do not (L6).
- Check: the Rust tests of each engine on accepted and rejected positions,
  each asserting the name of the object in the message; the Python tests that
  today expect the Python messages expect the engine's.

### S2 — Results carry their positions

- Fix: `Simulation.run` and `run_at` return the positions with the values,
  and `output_format` goes (D1); the ensemble trajectories of the probabilistic calibration too
  (`generate_predictions_parallel` and the calibrated variant take positions,
  A2).
- Check: a difference run has positions `0, 1, …, n`; a differential `run_at`
  has the requested positions; existing reads of `results["A"]` still work.

### S3 — Windows, reports, plots and ensembles read by position

- Fix: `window_end_steps`, `windowed_totals`, `Simulation.windowed_totals`,
  `sample`, `series_report`, `accumulators_report`, `simulation_report`, the
  plotter (x axis and windowed line) and `statistics_calculator` read a value
  at a position from S2 results and accept non-integer windows for a model
  type that reports there. The window grid of `TimeScale` (`window_start`,
  `window_end`, `window_index`) moves to `api/windows.py`; `window_index` uses
  the rule of `observed_report`, which calls it. `isinstance(step, int)` and
  `_as_step` go.
- Check: a difference model gives the same reports, windows and plots as
  today; a differential model windows and reports at non-integer positions
  and matches `run_at`; `observed_report` and `window_index` agree.

### S4 — Time-series parameters at any position

- Fix: time-series points take `float` positions (D4). In a differential
  model `"linear"` interpolates in `t` and `"step_function"` holds from any
  position; the engine restarts at the points, as it does at the values of
  PR #102.
- Check: a linear series in a differential model matches the integral of its
  interpolant; a difference model gives the same results as today.

### S5 — `TimePattern.window` and `linear_ramp` at any position

- Fix: both read `t` and take `float` bounds. A difference model has
  `t = step` (`DifferenceEquations::step` passes `step` as both), so it gives
  the same results; a differential model gets a window that opens at any
  position and a continuous ramp.
- Check: the existing pattern tests unchanged; in a differential model a
  window `[2.5, 4)` moves `1 − e^(−1.5 a)` and a ramp matches its integral.

### S6 — Names and documentation

- Fix: the names of positions, lengths and counts follow D3. The docs and
  docstrings stop calling `t` an alias of `step` (A5). Names with more than
  one meaning are renamed too (L7): `ObservedDataPoint.compartment` /
  `compartments` → `output` / `outputs`, with `label` for the name of an
  aggregate; `TimePattern.window(start, end, amount)` → `interval`, since
  `window` becomes a length (D3); `source_ids` / `source_id` of
  `Simulation.total_series`, `windowed_totals`, `group_outputs` and
  `outputs_for` → `ids` / `id`, since `source` is also the source of a
  transition; `NelderMeadConfig.initial_step` → `initial_simplex_size`
  (unreleased); `scale_id` / `CalibrationParameterType.SCALE` → `factor_id` /
  `FACTOR`, since `scale` is also the scale of a prior and of a mutation.
  Names that say "time" for the axis (L8): `TimePattern` → `AxisPattern`
  (module `api/patterns.py`, pages `docs/guide/axis-patterns.md` and
  `docs/api/axis-patterns.md`); "time series" → "series"
  (`TimeSeriesValue` → `SeriesValue`, "series parameter" in the docs; Rust
  `TimeSeriesParameter` and `ParameterValue::TimeSeries` too, at no format
  cost since the enum is `untagged`). The expression variable `t` stays: the
  usual letter for the independent variable of an ODE, and `x` would clash
  with compartment names. Guides, API pages,
  `design-decisions.md` and the changelog follow the definition of done.
- Check: `pytest -m docs`, `mkdocs build --strict`; a search for the old names
  finds none outside the changelog; a test over `commol.__all__` checks that
  every Pydantic class sets `extra="forbid"`, and `ObservedDataPoint(…,
  window_steps=7)` raises an error that names `window_steps`.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S2 | How do results carry their positions? | A `dict` subclass from output name to values, as today, with a `positions` attribute and a `rows()` method (every output at each position, in `simulation_outputs` order). `output_format` goes: `run` and `run_at` return one type. `results["A"]`, `json.dumps`, `pandas.DataFrame(results)` and `isinstance(results, dict)` keep working. | A reserved `"positions"` key: it can clash with an output name and every loop over outputs must skip it. A `(positions, results)` tuple: breaks every caller for no gain over an attribute. Keep `list_of_lists` as a `list` subclass with `.positions`: two result shapes for one thing, and the `@overload` on `Literal` stays. Keep it as a plain list: its rows carry no positions (B1). Cost: `output_format="list_of_lists"` breaks; no module of `commol` uses it (two tests and one guide example do), and the changelog maps it to `run(...).rows()`. | approved (L2) |
| D2 | S1 | What does the engine trait keep? | One way to run: at positions (`f64`). `run(n)` stays as a convenience that runs at the engine's default grid, built in Rust, and returns that grid. `run_into_buffer(u32)` and `run_at_steps_into_buffer(&[u32])` leave the trait; the difference engine keeps its integer loop as its own method and calls it from `run_at_positions_into_buffer` after it checks the positions. | Keep both ways: two names for one thing, and the default `run_at_positions_into_buffer` keeps the integer rule in the trait (A1 of the audit), which a new model type inherits. Remove `run(n)`: every user must build a grid for the most common call. Performance (measured, L3): positions cost 1–2 % over `run(n)` when Python passes all `n + 1` of them, none when Rust builds the grid; the loop is the same. | approved (L3) |
| D3 | S6 | Names of positions, lengths and counts | A position is `at` (as in `impulse_at`, `TimePattern.pulse(at=…)`); a length is `window` or `period`; a count of steps keeps `num_steps`. So `ObservedDataPoint.at` and `window`, `CalibrationConstraint.at`, `Simulation.run_at(at)`, `windowed_totals(at=…)`. The expression variable `step` does not change. No error per old name (L10): every public Pydantic class sets `extra="forbid"`, so an old or misspelt name raises an error that names it; the changelog lists the mapping. | Keep `step`/`window_steps`/`time_steps`: one word for a position and a count, the trap of A5. `position`/`window_length`: clear but longer, and unlike the names that exist (`impulse_at`). | approved (L4, L10) |
| D4 | S4 | Positions of a time-series parameter | `float` positions (`f64` in Rust). `"linear"` interpolates in `t`; `"step_function"` holds from any position and the engine restarts at its points; `"pulse"` keeps integer positions (a pulse is a rate on one unit interval, `ode-engine` D12). The engine rejects a position its model type does not accept (D2). | Keep `int`: a differential model cannot hold a series between integers, and `"linear"` stays a staircase, a wrong result (L5). Allow any position for `"pulse"` too: in a differential model a pulse at a point moves nothing, silently. Cost: `"linear"` series in a differential model change results; the ODE engine is not released, and a difference model gives the same results. | approved (L5) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1 | Engine trait on positions (D2); each engine checks its positions; remove C1–C3; a benchmark test of `run(n)` against `run_at` on the full grid | direct | |
| T2 | S2 | Results with positions (D1), in `Simulation` and the ensemble predictions | direct | |
| T3 | S3 | Windows, reports, plots and ensemble statistics by position; window grid into `windows.py` | direct | |
| T4 | S4 | Time-series parameters at `float` positions (D4); restarts at their points | direct | |
| T5 | S5 | `window` and `linear_ramp` on `t` with `float` bounds | direct | |
| T6 | S6 | Names (D3, L7, L8); `extra="forbid"` on every public Pydantic class, with a test over `commol.__all__`; docs, API pages, design decisions, changelog | direct | |
| T7 | all | Final audit: every runner | direct | |

## Delivery

- Strategy: one PR to `develop` from `refactor/axis-positions`, squash-merged
  (`refactor(api): …`).
- Order: T1 → T2 → T3 → T4 → T5 → T6 → T7.
- Version impact: breaking (names of D3, results of D1, trait of D2). It can
  ship in one minor version with the two other ODDs if all three merge before
  the release. Changelog "Changed" and "Removed".

## Evidence

(Filled during implementation.)

## Log

- L1 (user, 2026-10-13): split the audit T0 of
  `odd/tasks/axis-scale-refactor.md` into three ODDs; this one takes themes
  A, B and C, and the window grid and S5/S6 of that ODD.
- L2 (user, 2026-10-13): approved D1 with `rows()` in place of
  `output_format="list_of_lists"`.
- L3 (user, 2026-10-13): asked whether D2 keeps today's performance. Measured
  on the release extension, SIR stratified 10 × 5 (150 outputs), difference
  model, median of 40 runs: `run(n)` 0.247–0.250 ms (n = 100) and
  5.12–5.19 ms (n = 2000); `run_at` on the full grid 0.249–0.253 and
  5.18–5.27 ms; `run_at` every 50 positions 0.121–0.122 and 2.43–2.48 ms.
  Approved D2.
- L4 (user, 2026-10-13): approved D3. Uses to rename (code, tests, docs):
  `step=` 5/46/31, `window_steps` 91/15/24, `time_steps` 66/14/9,
  `at_steps` 7/0/2.
- L5 (user, 2026-10-13): approved D4. Evidence: a linear series from 0 at
  `t = 0` to 1 at `t = 10` in a differential model gives A(10) = 11.109, the
  staircase `1000·e^−4.5`, not the ramp `1000·e^−5` = 6.738.
- L6 (user, 2026-10-13): while approving `axis-scale-refactor` D2, made a
  condition of S1: the engine's error names the object that holds a rejected
  position, as the Python check it replaces does.
- L7 (user, 2026-10-13): approved five renames of names with more than one
  meaning, found by introspecting `commol.__all__` (S6). Kept: `population`
  (a key of the saved model; NSGA-II's is scoped by its class), `conditions`
  (initial conditions is a fixed term; as a selector it means one thing),
  `mode`/`method`/`strategy`/`type` (scoped), `point`, `pulse` against
  `impulse` (documented), the plotter's `series` (`code-findings` D5).
- L8 (user, 2026-10-13): asked why class names say "Time" when the axis is
  any independent variable. Approved `AxisPattern` and "series" in S6; the
  general prose ("over time") goes to `axis-scale-refactor` S3. `TimePattern`
  is the most used of the renamed names (173 uses in code, tests and docs).
- L9 (user, 2026-10-13): the `population` name kept in L7 is settled by
  `model-kind-interface` D1 (its L6): `Model.population` goes, and
  `population_size` keeps its one meaning, a head count.
- L10 (user, 2026-10-13): D3 reviewed under the alpha rule. No hand-written
  error per old name. Found: 16 of 22 public Pydantic classes accept unknown
  fields silently (`ObservedDataPoint(…, window=7)` drops `window` today);
  only the six probabilistic configs set `extra="forbid"`. Approved
  `extra="forbid"` on every public Pydantic class, which also catches typos.
