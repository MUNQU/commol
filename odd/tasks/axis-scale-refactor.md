# Refactor: `TimeScale` into axis units, rate conversions and window grids

Goal: replace `TimeScale` with pieces that fit any model axis and both
engines. Unit conversion works for any independent variable (`Model.step_unit`),
rate conversions give the right rate for the typology, and window grids live
with the other window code and accept non-integer steps.
Motivation: after `odd/archive/tasks/ode-engine.md`, the axis of a model is whatever
independent variable it differentiates over (time is one case), and a
`DifferentialEquations` model needs continuous rates. `TimeScale` assumes a
time axis and discrete-time rates.
Branch: `refactor/axis-scale` from `develop` (the ODE engine, PR #101, is
merged as `3c13102`).
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

## Specs

Every spec follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — Unit conversion for any axis

- Fix: a converter built from a step unit of any single dimension (the rule of
  `check_step_unit`, `context/_model/units.py`), on `pint`. It turns an amount
  in a compatible unit into steps (`3 day` with a step of `h` → 72; `2 km` with
  a step of `m` → 2000) and raises, naming both units, when the dimension
  differs. A whole number of steps is required where the result is a step
  count (D2). The model's `step_unit` can build it (D1).
- Check: tests for a time axis and a length axis; an incompatible unit raises
  and names both; a non-whole result raises as `steps_from_seconds` does today.

### S2 — Rate conversions that know the typology

- Fix: the conversion from a mean duration and from a probability over a
  period takes the typology (or the model) and returns the per-step
  probability for `DifferenceEquations` and the continuous rate for
  `DifferentialEquations` (D3).
- Check: a difference model and a differential model built with the converted
  rates both move `p` of a bin with one outflow over the period (the bin keeps
  `(1 − r)^n` and `e^(−r n)` of its start, both `1 − p`); the mean residence time of a one-bin outflow
  matches `D` in both engines.

### S3 — Window grids with the other window code

- Fix: `window_start`, `window_end` and `window_index` move to `api/windows.py`
  next to `window_end_steps` and `windowed_totals`, and accept non-integer
  steps and periods for a differential model. `window_index` uses the rule of
  `observed_report` (`ceil(step / period) − 1`, step 0 in window 0), and
  `observed_report` calls it, so the rule lives once.
- Check: the existing `TestWindowGrid` cases; a non-integer step lands in the
  window that holds it; `observed_report` and `window_index` agree.

### S4 — Migration and documentation

- Fix: `TimeScale` goes or stays as a thin alias (D4). `docs/guide/core-concepts.md`
  ("Steps and physical time"), `docs/api/time-scale.md`, `docs/api/index.md`,
  `design-decisions.md`, changelog ("Changed" or "Removed", D4), and the
  warning added by the ODE PR is removed once S2 fixes the cause.
- Check: `pytest -m docs` (every new public name has an API entry),
  `mkdocs build --strict`.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1 | Where does the converter come from? | A class built from a unit, plus `Model.axis_scale()` (name in D5) that builds it from `step_unit` and raises when the model has none. | Only from the model: a user without a built model (a script that prepares parameters) cannot convert. Only standalone: the model's `step_unit` and the converter can disagree, and nothing checks it. | open |
| D2 | S1 | What does a conversion return? | Two methods: `steps(amount, unit) -> int` (raises when not whole, as today) and `position(amount, unit) -> float` (any value, for `run_at`, observations and impulses of a differential model). | One method returning `float`: callers that need a count (`TimePattern.periodic`, windows of a difference model) must round and lose today's early error. One method with a `whole=` flag: one return type per flag, which `ty` cannot express without overloads. | open |
| D3 | S2 | How does a rate conversion know the engine? | It takes the typology: `rate_from_mean_duration(duration, unit, typology)`; the model-bound converter of D1 passes its model's typology. | Two function pairs (`per_step_rate_…`, `continuous_rate_…`): the user can still pick the wrong one. Read the typology from a global default: hidden state. | open |
| D4 | S4 | What happens to `TimeScale`? | Remove it in the next minor version, with a changelog "Removed" entry that maps every member to its replacement. Commol is 0.x and `TimeScale` has existed one release. | A deprecated alias for one minor version: less breakage, but its rate methods keep the discrete-time trap the refactor removes, so the alias would warn on every rate call. Keep it unchanged beside the new pieces: two ways to do one thing. | open |
| D5 | S1 | Name of the converter | `AxisScale` | `StepScale`: says what it converts to, but not that it is about the model's axis. `UnitScale`: too general; every pint quantity has a unit. | open |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1 | Converter on `pint` from a step unit (D1, D2, D5); tests on a time and a length axis | direct | |
| T2 | S2 | Rate conversions by typology (D3); closed-form tests in both engines | direct | |
| T3 | S3 | Window grid into `windows.py` with non-integer steps; `observed_report` uses `window_index` | direct | |
| T4 | S4 | Remove or alias `TimeScale` (D4); documentation, API pages, changelog; remove the ODE-PR warning | direct | |
| T5 | all | Final audit: every runner | direct | |

## Delivery

- Strategy: one PR to `develop` from `refactor/axis-scale`, squash-merged
  (`refactor(api): …`).
- Order: T1 → T2 → T3 → T4 → T5.
- Version impact: breaking for users of `TimeScale` (D4); changelog
  "Added" (converter, typology-aware rates) and "Removed" or "Changed".

## Evidence

(Filled during implementation.)

## Log

- L1 (user, 2026-10-13): during the ODE PR, asked whether `TimeScale` still
  makes sense. Agreed: add a warning in the ODE PR (rate methods are
  discrete-time) and refactor in this separate ODD.
