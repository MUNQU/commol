# Refactor: units and rates for any model type

Goal: replace `TimeScale` with pieces that fit any model axis and any model
type. Unit conversion works for any independent variable (`Model.step_unit`);
a rate conversion asks the model type what a rate means; the guides and
docstrings say what a rate means per type instead of stating the difference
rule as the general one.
Motivation: after `odd/ode-engine.md`, the axis of a model is whatever
independent variable it differentiates over (time is one case), and a
`DifferentialEquations` model needs continuous rates. `TimeScale` assumes a
time axis and discrete-time rates. Audit T0 below found that more of the API
assumes one model type; that work is split into three ODDs (L5).
Branch: `refactor/axis-scale` from `develop`, after
`odd/axis-positions.md`, `odd/model-type-registry.md` and
`odd/model-kind-interface.md` are merged.
Last of four ODDs; the order of all pending work is in
`odd/README.md`.
Engram mirror: topic `odd/axis-scale-refactor/tasks` (project `commol`).
Runners: `pytest` and `pytest -m docs` (from `py-commol/`), `ruff check .`,
`ty check commol`, `mkdocs build --strict` (from the root).

## Current state (verified 2026-10-13 on `develop` at `3c13102`)

`TimeScale` (`py-commol/commol/api/time_scale.py`), public since v0.2.0
(`1db87df`, tag `v0.2.0`), holds `step_seconds` and mixes three jobs:

| Job | Members | Problem |
|---|---|---|
| Unit conversion | `steps_from_seconds/minutes/hours/days/weeks`, `steps_per_hour/day/week`, `from_step_unit` | Time only. Hand-written second constants duplicate `pint`, which the unit check already uses (`utils/equations.py`, `ureg`). Not connected to `Model.step_unit`. |
| Rate conversion | `rate_from_mean_duration`, `rate_from_probability` | Discrete-time formulas: `1 − exp(−h/D)` and `1 − (1 − p)^(1/n)`. A differential model needs `1/D` and `−ln(1 − p)/n`; with the per-step value it moves too little. Today only a docstring and a guide warning say so (`docs/guide/core-concepts.md`, "Steps and physical time"). |
| Window grid | `window_start`, `window_end`, `window_index` | Step arithmetic; it never reads `step_seconds`. `window_index` uses `max(step − 1, 0) // period`, wrong for a non-integer step (`observed_report` in `api/reporting.py` uses `ceil(step / period) − 1` since the ODE PR). |

Users: `py-commol/tests/simulation/test_time_scale.py`,
`docs/guide/core-concepts.md`, `docs/guide/simulations.md`,
`docs/api/time-scale.md`, `docs/codebase/design-decisions.md` ("Physical time
is not in the model"). No other module of `commol` calls `TimeScale`.

Where the audit findings go:

| Themes | ODD |
|---|---|
| A, B, C (positions, results, rules Python repeats); the window grid of `TimeScale`; `TimePattern.window` and `linear_ramp` | `odd/axis-positions.md` |
| D (model types); what a rate means per type, as the engine's answer | `odd/model-type-registry.md` |
| E (rate conversions and wording), F (units), `TimeScale` | this ODD |
| E3 (pulses) | none: `ode-engine` D12 stays |

## Audit T0 (2026-10-13, `develop` at `34d75eb` with PR #102)

Question: which code and docs assume one model type (most often the
difference engine), so that another type (differential today; stochastic,
delay or spatial later) does not fit the public API? Read: the engine trait,
both engines, the calibration crate, the bindings and stubs, every module of
`commol/api` and `commol/context`, and the guides. Findings by theme:

### A. A position on the axis has several names and types

| ID | Where | Finding |
|---|---|---|
| A1 | `commol-core/src/simulation_engine.rs` | The engine trait has `run(num_steps: u32)`, `run_into_buffer(u32)`, `run_at_steps_into_buffer(&[u32])` and `run_at_positions_into_buffer(&[f64])`: two ways to name a position. `step()` advances "one time unit"; `current_step()` returns a position. |
| A2 | `_commol_rs.pyi`, `commol-calibration/src/probabilistic/calibration.rs:184` | `engine.run(num_steps: int)`; `generate_predictions_parallel` and `generate_calibrated_predictions_parallel` take `time_steps: int` and run `0..n`. |
| A3 | `api/simulation.py`, `api/windows.py`, `api/reporting.py`, `api/plotter.py`, `api/time_patterns.py` | `Simulation.run(num_steps: int)`, `windowed_totals(window_steps: int, at_steps: Iterable[int])`, `window_end_steps`, `sample(steps: Iterable[int])`, the three reports, `TimePattern.window`/`linear_ramp` (`int` bounds on `step`). |
| A4 | `context/parameter.py:8`, `commol-model/src/types.rs` (`evaluate(step: u64)`) | A time-series parameter holds `(int, float)` pairs and is read at `floor(t)`: in a differential model the `"linear"` mode is a staircase and no point can sit between integers. |
| A5 | names | `step` names a position (`ObservedDataPoint.step`, `run_at(steps)`; Rust `time_step`) and a count (`num_steps`, `window_steps`). `t` is documented as an alias of `step` (`math_expression/preprocessing.rs:12`, `math_expression/mod.rs:17`, `ModelBuilder.add_transition` docstring), which is false in a differential model. |

### B. Results do not carry their positions

| ID | Where | Finding |
|---|---|---|
| B1 | `Simulation.run` | `dict_of_lists` has no positions; every later step reads list index = position. |
| B2 | `api/reporting.py`, `api/windows.py`, `api/plotter.py:628` (`x = range(len(series))`) | Reports, windows and plots index a series by step. The plotter's windowed line raises for a non-integer window (`plotter.py:315`). |
| B3 | `api/probabilistic_calibrator.py:694`, `api/probabilistic/statistics_calculator.py:32,147` | Trajectories cover `0..floor(max step)`; a non-integer observation is fetched from the engine apart. `isinstance(step, int)` picks between a list index and an engine query, and `_as_step` turns `24.0` into `24` "so it indexes a trajectory": the number type carries the meaning. |

### C. Python repeats rules the engine owns (AGENTS.md golden rule)

| ID | Where | Duplicates |
|---|---|---|
| C1 | `context/calibration.py:56` | Non-integer observation, window or constraint step in a difference model; the engine has `integer_steps` (`simulation_engine.rs`). |
| C2 | `context/dynamics.py:282` | Non-integer impulse position in a difference model; the engine has `first_non_integer_impulse` (`commol-difference`). |
| C3 | `api/simulation.py:203` | `run_at` positions sorted, unique, non-negative; the engines check their positions too. |

### D. The set of model types is closed and spread out

| ID | Where | Finding |
|---|---|---|
| D-1 | `constants.ModelTypes`, `commol-core/src/types/dynamics.rs` | Two typologies, in Python and Rust. |
| D-2 | `py-commol/src/engine.rs` (`Engine`, `EngineArg`, `dispatch!`), `context/_model/engine_state.py:106`, `context/_model/equations.py:318` | Each new type edits an enum, a macro, an `if/else` that picks the engine class, and an `if/else` that writes the left-hand side of `print_equations`. |
| D-3 | `Dynamics.solver: OdeSolver`, `ModelBuilder(solver=...)`, `dynamics.py:282` | Settings of one engine live in the generic model and builder; the builder takes them before the typology is chosen at `build(typology)`. |

### E. The meaning of a rate per type is not in one place

| ID | Where | Finding |
|---|---|---|
| E1 | `api/time_scale.py` | Rate conversions are per-step only (S2). |
| E2 | `docs/guide/units-and-equations.md:66`, `core-concepts.md:6,57`, `quickstart.md:55`, `context/model.py:774`, `context/_model/units.py:62,182`, `ModelBuilder.add_transition` docstring ("evaluated ONCE per time step"), `commol-core/src/types/dynamics.rs:8` | "Every flow applies once per step" is written as the general rule; it is the rule of the difference engine. |
| E3 | `TimePattern.pulse`, `pulses`, `periodic` | A pulse is a rate for one unit interval in a differential model (`ode-engine` D12). |

### F. Units

| ID | Where | Finding |
|---|---|---|
| F1 | `context/_model/units.py:14` (`TIME_DIMENSIONS`) | Without a declared `step_unit`, the step unit is inferred only from units of time; a length axis must be declared. |

### What a model-type-neutral API needs (from A–F)

1. One kind of position: a `float` on the axis everywhere (inputs and
   results), with the engine alone deciding which positions it accepts and
   saying so in its error (C1–C3, A1–A5).
2. Results that carry their positions, so reports, windows, plots and
   ensembles read a value at a position, not at a list index (B1–B3).
3. One registry of model types: each type names its engine, its settings and
   its equation form, so a new type adds an entry, not edits in five places
   (D-1–D-3).
4. The meaning of a rate asked from the type, not assumed (E1–E3, S2).

## Specs

Every spec follows the [documentation definition of done](../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — Unit conversion for any axis

- Fix: a converter built from a step unit of any single dimension (the rule of
  `check_step_unit`, `context/_model/units.py`), on `pint`. It turns a value
  in a compatible unit into steps (`3 day` with a step of `h` → 72; `2 km` with
  a step of `m` → 2000) and raises, naming both units, when the dimension
  differs. The model's `step_unit` can build it (D1). One method returns a
  position (D2); a name that needs a whole number of steps checks it.
- Check: tests for a time axis and a length axis; an incompatible unit raises
  and names both; `4.1 hour` with a step of `minute` gives exactly `246`.

### S2 — Rate conversions that ask the model type

- Fix: the conversion from a mean duration and from a probability over a
  period takes the model type (or the model) and asks it what a rate means
  (`model-type-registry` S1): the per-step probability for
  `DifferenceEquations`, the continuous rate for `DifferentialEquations` (D3).
- Fix (formula): the per-step rate of a difference model stays
  `1 − e^(−h/D)`, which keeps the survival of the continuous model at every
  step; its mean residence is `1/r` (10.508 steps for `D = 10`), not `D`. The
  alternative `r = h/D` keeps the mean and loses the survival (`0.590` against
  `0.607` at step 5) (L7).
- Check: a difference model and a differential model built with the converted
  rates both move `p` of a bin with one outflow over the period (the bin keeps
  `(1 − r)^n` and `e^(−r n)` of its start, both `1 − p`); from a mean duration
  `D`, the difference model keeps `e^(−n/D)` at each step `n` and the
  differential model has mean residence `D`.

### S3 — The meaning of a rate, per model type, in the docs

- Fix: every place of E2 says what a rate means for each model type, or links
  to the one page that does, instead of "every flow applies once per step".
  General prose says "along the axis", not "over time"; examples may keep
  time, the most common axis (`axis-positions` L8).
- Check: a search for "once per step" finds it only where the difference
  engine is meant; `mkdocs build --strict`.

### S4 — Time is a unit like any other

- Fix (D6): the step unit is the declared `step_unit` or, when there is none,
  `step`, the default increment of any axis (its own dimension `[step]`, not
  time). `TIME_DIMENSIONS` and the inference in `UnitChecker._step_unit` go,
  and with them the messages that say "unit of time".
- Fix: the hand-written list of time aliases (`context/_model/helpers.py:289`)
  goes: a unit means what `pint` defines, and an unknown unit becomes its own
  dimension, for time as for any quantity. The list did not work as meant:
  `pint` reads `decade` as dimensionless and `quarter` as a mass, so
  `1/quarter` was a mass without error (L10).
- Fix: in the check, `t` (a position on the axis) has the step unit, and a
  number compared with it is read in that unit, as the engine does; `step`
  (a count of increments) stays dimensionless.
- Fix: `rate_from_mean_duration` becomes `rate_from_mean_span(span, unit)`;
  "duration" is a word of time.
- Check: a model on a length axis and one on a time axis go through the same
  path; a model with `1/day` rates and no `step_unit` fails the check with a
  message that says to declare it; with `step_unit="day"`, the check rejects
  a rate in `1/quarter` or `1/mo` and names the unit; with `step_unit="day"`,
  a proportional rate `k * t` passes when `k` is in `1/day**2`.

### S5 — Migration and documentation

- Fix: `TimeScale` goes or stays as a thin alias (D4). The window members map
  to the window grid of `axis-positions` S3. `docs/guide/core-concepts.md`
  ("Steps and physical time"), `docs/api/time-scale.md`, `docs/api/index.md`,
  `design-decisions.md`, changelog ("Changed" or "Removed", D4), and the
  warning added by the ODE PR is removed once S2 fixes the cause.
- Check: `pytest -m docs` (every new public name has an API entry),
  `mkdocs build --strict`.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1 | Where does the converter come from? | A class built from a unit, plus `Model.axis_units()` (name in D5) that builds it from the step unit of the unit check (`UnitChecker._step_unit`): the declared `step_unit`, else the one inferred from the parameter units, else `step`. It never raises by itself: `step` is a `pint` unit of dimension `[step]`, so a value in a physical unit raises the dimension error of S1, which adds "declare `step_unit` in `ModelBuilder`" when the step unit is `step`. | Only from the model: a user without a built model (a script that prepares parameters) cannot convert. Only standalone: the model's `step_unit` and the converter can disagree, and nothing checks it. Raise when the unit is `step`: it is a valid unit, and the dimension error already says what is wrong. | approved (L2) |
| D2 | S1 | What does a conversion return? | One method, `steps(value, unit) -> float`, that snaps a result within `1e-9` of an integer to it (`4.1 hour` → `246.0` minutes, not `245.99999999999997`). Whether a position must be whole is the engine's check (`axis-positions` D2); pulses keep integers (`axis-positions` D4). | Two methods, `steps -> int` and `position -> float`: the user picks a method by the type the receiving name wants, a difference-engine detail (L4). One method with a `whole=` flag: the same choice as a flag. Cost: a non-whole position fails where the engine receives it, and the error does not name the source unit; it names the value and, by the condition of `axis-positions` S1 (L6), the object that holds it. The snap means it fails only on a real mistake (`36 hour` with a step of `day`). | approved (L6) |
| D3 | S2 | How does a rate conversion know what a rate means? | It takes the model type, `rate_from_mean_duration(duration, unit, typology)`, and asks the engine's list of types (`model-type-registry` S1); the model-bound converter of D1 passes its model's type. | Two function pairs (`per_step_rate_…`, `continuous_rate_…`): the user can still pick the wrong one, and a new type needs a new pair. Formulas per typology in Python: the engine is the source of truth, and a new type edits Python too. A global default typology: hidden state. | approved (L7) |
| D4 | S5 | What happens to `TimeScale`? | Remove it in 0.3.0, with a changelog "Removed" entry that maps every member: `steps_from_*`, `steps_per_*`, `from_step_unit` → the converter of S1 and its model method; `rate_from_mean_duration`, `rate_from_probability` → the conversions of S2; `window_start`, `window_end`, `window_index` → `api/windows.py` (`axis-positions` S3). It is in one release only (`v0.2.0`, 2026-10-12; `1db87df` is only in that tag). | A deprecated alias for one minor version: less breakage, but its rate methods keep the trap the refactor removes (per-step rates for a differential model), so the alias would warn on every rate call or give wrong values. Keep it unchanged beside the new pieces: two ways to do one thing, one wrong for differential models. Cost: breaks code written against 0.2.0; 0.3.0 breaks for `axis-positions` D1 and D3 too. | approved (L8) |
| D5 | S1 | Name of the converter | `AxisUnits`, built from a step unit, and `Model.axis_units()`. | `AxisScale`: "scale" already has three meanings in commol (the `SCALE` calibration parameter and `scale_id`, the `scale` of a prior, the `scale` of a mutation) and matplotlib's axis scale (linear or log) in the plotter. `StepUnit`: next to the string field `Model.step_unit`, two things with almost one name. `StepScale`: the "scale" clash. `UnitConverter`: too general; it does not say it converts to the model's axis. | approved (L9) |
| D6 | S4 | How is the step unit found? | No inference: the declared `step_unit`, else `step`. The inference was approved in `code-findings` D6 because there was no way to declare the step unit; `step_unit` exists now (ODE PR). | Keep the inference to units of time: no break, but time keeps a privilege the axis does not have, and code that guesses stays. Infer from any dimension the rates divide by: a `1/m` parameter can be a density, so the check would guess wrong without error. Require `step_unit` always: breaks models without units, which need none. Cost: a model with `1/…` rates and no `step_unit` fails the opt-in check until it declares one (the quickstart and two test files). | approved (L10) |

## Tasks

| ID | Specs | Task | Route | Status |
|---|---|---|---|---|
| T0 | all | Audit of every place that assumes one model type (section "Audit T0") | direct | done (2026-10-13) |
| T1 | S1 | Converter on `pint` from a step unit (D1, D2, D5); tests on a time and a length axis | direct | done |
| T2 | S2 | Rate conversions that ask the model type (D3); closed-form tests in both engines | direct | done |
| T3 | S3 | The meaning of a rate per type in guides and docstrings (E2) | direct | done |
| T4 | S4 | Time as a unit like any other: no inference (D6), no time aliases, `t` with the step unit, `rate_from_mean_span` | direct | done |
| T5 | S5 | Remove or alias `TimeScale` (D4); documentation, API pages, changelog; remove the ODE-PR warning | direct | done |
| T6 | all | Final audit: every runner | direct | done |

## Delivery

- Strategy: one PR to `develop` from `refactor/axis-scale`, squash-merged
  (`refactor(api): …`).
- Order: T1 → T2 → T3 → T4 → T5 → T6.
- Version impact: breaking for users of `TimeScale` (D4); changelog
  "Added" (converter, rate conversions per model type) and "Removed" or
  "Changed". It can ship in one minor version with the two other ODDs.

## Evidence

Runners on `refactor/axis-scale` (2026-10-15), before commit; delivered by PR #112:

| Runner | Result |
|---|---|
| Rust tests (six crates) | all pass; +2 in `commol-core` (`a_rate_from_a_mean_span_keeps_the_exponential_survival`, `a_rate_from_a_probability_moves_it_over_the_period`) |
| `cargo test -p commol --features test-kind` | pass |
| `cargo fmt --check`, `clippy -D warnings`, `cargo doc` with `-D warnings` | clean |
| `pytest` | 1005 passed, 3 xfailed; `test_axis_units.py` (27) replaces `test_time_scale.py` |
| `pytest -m docs` | 62 passed |
| `ruff check`, `ruff format --check`, `ty check commol`, `vulture` | clean |
| `mkdocs build --strict` | built, exit 0 |

Checks of the specs:

| Spec | Evidence |
|---|---|
| S1 | `TestSteps`: `3 day` → `72.0` hours, `2 km` → `2000.0` m, `4.1 hour` → `246.0` minutes; another dimension names both units; on `step` the error says to declare `step_unit`. |
| S2 | `TestRatesMoveTheBinInBothEngines`: with the rate from `p = 0.3` over a week, A keeps `700` at day 7 in both engines; with the rate from a mean of 10 days, A keeps `1000 e^(−n/10)` at every day `n` in both; the continuous rate is `1/10`. The per-step mean residence is `10.508` (`TestRates`). |
| S3 | "once per step" is only in text about the difference engine (`quickstart.md`, `mental-model.md`, `design-decisions.md`, `commol-model` docs). "What a rate means" in `core-concepts.md` is the one page; the units guide, `simulations.md`, `quickstart.md` and the `add_transition` docstring link to it. |
| S4 | `TestStepUnit`, `TestAxisUnit`, `TestPositionOnTheAxis`: `1/day` without `step_unit` fails and says to declare it; with `step_unit="day"`, `1/week`, `1/quarter` and `1/mo` fail and are named; `k * t * A` passes with `k` in `1/day**2`; a length and a time axis go through one path. |
| S5 | `TimeScale` and its page are gone; `AxisUnits` has an API page (`test_public_name_has_an_api_entry`); the ODE-PR warning is gone; the changelog has "Added", "Changed" and "Removed" with the mapping of every member. |

## Log

- L1 (user, 2026-10-13): during the ODE PR, asked whether `TimeScale` still
  makes sense. Agreed: add a warning in the ODE PR (rate methods are
  discrete-time) and refactor in this separate ODD.
- L2 (user, 2026-10-13): approved D1, with the step unit taken from the unit
  check (declared, inferred, else `step`) and no special error for `step`.
- L3 (user, 2026-10-13): while reviewing D2, noted that a set of methods was
  designed for the difference engine only and must be adapted to the
  differential one. Checking it found two faults of the ODE engine (`step`
  stayed 0 in rate expressions; the solver stepped over a window on `t`),
  fixed first in PR #102. Scope widened; the inventory became audit T0.
- L4 (user, 2026-10-13): choosing a conversion method by the type of the
  receiving name is odd for a user; D2 reworked to one method.
- L5 (user, 2026-10-13): after asking whether the ODD makes the API fit any
  model type, approved a full audit (T0, above), then split the work into
  three ODDs: `axis-positions` (A, B, C, window grid, patterns),
  `model-type-registry` (D, meaning of a rate per type) and this one (E, F,
  `TimeScale`). S3, S5 and S6 of the earlier scope moved to `axis-positions`;
  D3 now asks the model type.
- L6 (user, 2026-10-13): approved D2 on condition that the engine's error
  names the object that holds a rejected position (added to `axis-positions`
  S1).
- L7 (user, 2026-10-13): approved D3. The check of S2 was wrong for the
  difference model: `1 − e^(−1/D)` keeps the survival, not the mean
  (`1/r = 10.508` for `D = 10`); the formula stays and the check now says so.
- L8 (user, 2026-10-13): approved D4 (remove `TimeScale` in 0.3.0).
- L9 (user, 2026-10-13): approved D5 (`AxisUnits`, `Model.axis_units()`)
  and asked to check the public API for other words with several meanings
  that should be renamed.
- L10 (user, 2026-10-13): asked whether units could go entirely. They are
  optional already and the engine never reads them; they serve the opt-in
  check, `print_equations` and `AxisUnits`. Approved D6 without inference,
  then asked that time be treated like any other unit, and that `step` is an
  increment of any unit, not of time. Approved S4: no time aliases, `t` with
  the step unit, `step` a dimensionless count, `rate_from_mean_span`.
- L11 (user, 2026-10-14): asked that D2 follow `neutral-state-terms` D5,
  where "amount" means only the value of a compartment. The argument of
  `steps` is `value`, the name `pint` gives the number of a quantity
  (`Quantity(value, units)`) and the name `AxisPattern` takes. S1 and D1 say
  "value" too. The decision is otherwise unchanged.
- L12 (implementation, 2026-10-15): choices made while implementing,
  within the approved decisions.
  - D3: the engine is a field of `AxisUnits` (`AxisUnits(step_unit,
    engine)`), not an argument of each rate method. `Model.axis_units()`
    passes the model's engine; a converter without one converts values and
    raises on a rate, naming the fix. One source of the engine per converter.
  - D3: the formulas are in Rust, `RateMeaning::rate_from_mean_span` and
    `rate_from_probability` (`commol-core/src/types/engine.rs`), bound on
    `EngineType`. Python holds no formula per meaning, so a new engine with an
    existing meaning changes no Python, and a new meaning changes only Rust.
  - D5: `AxisUnits` is in `context/axis_units.py`, not `api/`, because
    `Model.axis_units()` builds it and `context/` does not import `api/`.
  - S4: the rule "a number compared with `t` is read in its unit" applies to
    every quantity and to sums and differences too (pint needs one unit for
    each), and the check now reads `if(c, a, b)` (both branches share a unit)
    and `min`/`max`. A number a function computes stays dimensionless. Before,
    any rate with `if` failed the check with a syntax error.
  - S4: a step unit or a unit that `pint` does not define is a unit of its
    own dimension (`define_units`, shared by the check and `AxisUnits`).
  - The patterns of `AxisPattern` hold no units, so a rate made of one alone
    fails the check of a model with bin units, before and after.
