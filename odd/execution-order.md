# Execution order of the pending work

The order in which the open PR and the ODDs in `odd/tasks/` run, and why.
Each row is one branch and one PR to `develop`, unless it says otherwise.
Written 2026-10-13 on `develop` at `34d75eb`. Update it when an ODD is added,
merged or archived.

## Order

| # | Work | Tasks | Depends on | Breaking |
|---|---|---|---|---|
| 0 | PR #102, `fix/ode-step-in-rates`: the ODE engine restarts where a rate reads `step` or compares `t` with a fixed value | merged as `4c1fcb2` (2026-10-13) | — | No |
| 1 | `odd/archive/tasks/reference-model-tests.md` | merged as `eda0bbc` (PR #103, 2026-10-13) | #102 | No |
| 2 | `odd/archive/tasks/axis-positions.md` | merged by PR #104 (squash of `729894c`, 2026-10-13) | 1 | Yes |
| 3 | `odd/archive/tasks/model-type-registry.md` | merged by PR #105 (squash of `131a5ab`, 2026-10-14) | 2 | Yes |
| 4 | `odd/archive/tasks/model-kind-interface.md` | merged by PR #106 (squash `77aa9fd`, 2026-10-14) | 3 | Yes |
| 5 | `odd/archive/tasks/neutral-state-terms.md` | delivered by PR #108 | 4 | Yes |
| 6 | `odd/archive/tasks/domain-agnostic-audit.md` | delivered by PR #109 | 5 | Yes |
| 7 | `odd/tasks/axis-scale-refactor.md` | T1 → T6 (T0 done) | 6 | Yes |
| 8 | `odd/tasks/difference-negative-amounts.md` | T1 → T5 | 7 | Yes |
| 9 | Release 0.3.0 (`docs/development/release.md`) | — | 2–8 | — |
| 10 | `odd/tasks/library-benchmark.md` | T1 → T5 | 9 | No |
| 11 | `odd/tasks/model-kind-foundation.md` | T0 → T8 | 9 | Yes |
| 12 | `odd/tasks/stochastic-compartmental.md` | T1 → T8 | 11 | No |
| 13 | `odd/tasks/markov-chains.md` | T1 → T8 | 11 (S1–S4), 12 (S5) | No |
| 14 | `odd/tasks/integral-outputs.md` | T1 → T7 | 12, 13 | No |
| 15 | `odd/tasks/network-models.md` | T1 → T8 | 11, 12, 14 | No |
| 16 | `odd/tasks/agent-based-models.md` | T1 → T8 | 15 | No |

## Why this order

- **#102 first.** It fixes wrong results of the ODE engine (`step` stayed 0 in
  rate expressions; the solver stepped over a window on `t`). Every later
  test of a differential model needs it.
- **Reference model tests before the refactors.** Steps 2–7 change the public
  API six times. Tests against closed-form and published results guard the
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
- **`neutral-state-terms` before `axis-scale-refactor`.** It renames
  "population" to state, amount and total, so `axis-scale-refactor` is
  written against the final names and its tests migrate once; that ODD also
  adds `steps(amount, unit)`, which must not give "amount" a third meaning
  (`neutral-state-terms` D5).
- **`domain-agnostic-audit` before `axis-scale-refactor`.** It renames
  "per-capita" to "proportional" and the test models to abstract ones, so
  `axis-scale-refactor`, which writes "per-capita" (S4) and rewrites tests of
  `units/`, is written against the final names and its tests migrate once.
- **`axis-scale-refactor` last of the refactors.** `AxisUnits` reads
  `step_unit` and the engine of the new `Model`, and its rate conversions ask
  the registry what a rate means.
- **Negative amounts after the refactors, before the release.** Seven
  difference runs of the suite go negative today, four of them in published
  examples. The fix is a bug fix that breaks such models, so it ships in
  0.3.0 with the other breaks; it touches `tests/support/` and the guide,
  which steps 2–7 change, so it runs after them. The stochastic and Markov
  ODDs reuse its error.
- **One release for every break.** Steps 2–8 all break the API; commol is
  alpha, so the break is no cost, and one 0.3.0 release with one changelog
  mapping is easier to read than four.
- **Benchmark last.** It compares commol with other libraries on the API and
  performance users will get, so it runs on 0.3.0.
- **Foundation before any new kind or engine.** `model-kind-foundation` gives
  the description (`describe`, which replaces `print_equations`), seeds,
  replicates and calibration with common random numbers. Every later ODD
  builds on them; it changes no package code the benchmark measures, so the
  two can run in parallel.
- **Stochastic engine first of the new work.** It reuses the whole
  compartmental description, so it costs least, and it gives the event
  engines and the test oracles that Markov chain trajectories, networks and
  agents need.
- **Markov chains next.** Most of a chain fits the aggregate kind already;
  its builder, matrix and analysis (S1–S4) need only the foundation and may go
  before the stochastic engine if the user wants them earlier.
- **Integral outputs right after Markov chains.** Rewards are the main use of
  Markov chains, so they follow the chains with no gap (Markov D4).
  They come before networks and agents, so those kinds add integrals of their
  counts with the declaration already in place.
- **Networks before agents.** Both deliver one structure, `Agents`
  (`agent-based-models.md` D6). The network ODD delivers it first, with a
  graph and no attributes, and the variant `Network`: the first real second
  kind, which tests the generic layers with less to build. The agent ODD then
  adds attributes, `Complete`, `Groups` and Python rules on the same
  structure and engines.

## Not in parallel

Steps 1–8 change the same tests and modules (`tests/support/`,
`api/reporting.py`, `context/`), so they run one after the other. Each starts
from `develop` after the previous PR merges. Steps 11–16 also run one after
the other, in the order of the table: each builds on the interfaces the one
before adds.
