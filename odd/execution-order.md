# Execution order of the pending work

The order in which the open PR and the ODDs in `odd/tasks/` run, and why.
Each row is one branch and one PR to `develop`, unless it says otherwise.
Written 2026-10-13 on `develop` at `34d75eb`. Update it when an ODD is added,
merged or archived.

## Order

| # | Work | Tasks | Depends on | Breaking |
|---|---|---|---|---|
| 0 | PR #102, `fix/ode-step-in-rates`: the ODE engine restarts where a rate reads `step` or compares `t` with a fixed value | merged as `4c1fcb2` (2026-10-13) | — | No |
| 1 | `odd/tasks/reference-model-tests.md` | T1 → T5 | #102 | No |
| 2 | `odd/tasks/axis-positions.md` | T1 → T7 | 1 | Yes |
| 3 | `odd/tasks/model-type-registry.md` | T1 → T4 | 2 | Yes |
| 4 | `odd/tasks/model-kind-interface.md` | T0 → T6 | 3 | Yes |
| 5 | `odd/tasks/axis-scale-refactor.md` | T1 → T6 (T0 done) | 4 | Yes |
| 6 | Release 0.3.0 (`docs/development/release.md`) | — | 2–5 | — |
| 7 | `odd/tasks/library-benchmark.md` | T1 → T5 | 6 | No |

## Why this order

- **#102 first.** It fixes wrong results of the ODE engine (`step` stayed 0 in
  rate expressions; the solver stepped over a window on `t`). Every later
  test of a differential model needs it.
- **Reference model tests before the refactors.** Steps 2–5 change the public
  API four times. Tests against closed-form and published results guard the
  behavior through all of them, and they can find engine faults, as the work
  that led to #102 did, before the refactors build on them. Cost: each
  refactor migrates these tests too; most of them go through `tests/support/`,
  so the renames land in one place.
- **`axis-positions` before the registry.** It sets one kind of position and
  results with positions, which every later ODD uses, and it holds most of the
  renames (S6), so the later ODDs are written against the final names.
- **Registry, then model-kind interface.** The registry makes the list of
  engines, each naming its model kind, and `Model.engine` (its T2); the
  model-kind interface then makes `Model.structure` (its T0) and checks the
  kind against that list. `model-kind-interface` D1 is split this way so no
  field is made twice (`model-type-registry` L8, `model-kind-interface` L7).
- **`axis-scale-refactor` last of the refactors.** `AxisUnits` reads
  `step_unit` and the engine of the new `Model`, and its rate conversions ask
  the registry what a rate means.
- **One release for every break.** Steps 2–5 all break the API; commol is
  alpha, so the break is no cost, and one 0.3.0 release with one changelog
  mapping is easier to read than four.
- **Benchmark last.** It compares commol with other libraries on the API and
  performance users will get, so it runs on 0.3.0.

## Not in parallel

Steps 1–5 change the same tests and modules (`tests/support/`,
`api/reporting.py`, `context/`), so they run one after the other. Each starts
from `develop` after the previous PR merges.
