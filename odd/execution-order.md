# Execution order of the pending work

The order in which the open PR and the ODDs in `odd/tasks/` run, and why.
Each row is one branch and one PR to `develop`, unless it says otherwise.
Written 2026-10-13 on `develop` at `34d75eb`; reordered 2026-10-15 at
`20a61a6` for the uncertainty and identifiability program. Update it when an
ODD is added, merged or archived.

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
| 7 | `odd/archive/tasks/axis-scale-refactor.md` | delivered by PR #112 | 6 | Yes |
| 8 | `odd/archive/tasks/difference-negative-amounts.md` | delivered by PR #113 | 7 | Yes |
| 9 | Release 0.3.0 (`docs/development/release.md`) | released 2026-10-15 (PR #114, tag `v0.3.0`, PyPI) | 2–8 | — |
| 10 | `odd/archive/tasks/model-kind-foundation.md` | delivered on `refactor/model-kind-foundation` (2026-10-16) | 9 | Yes |
| 11 | `odd/tasks/stochastic-compartmental.md` | T1 → T8 | 10 | No |
| 12 | `odd/tasks/inference-foundation.md` | T0 → T9 | 10, 11 | Yes |
| 13 | `odd/tasks/identifiability.md` | T1 → T7 | 12 | No |
| 14 | `odd/tasks/resampling-uncertainty.md` | T1 → T8 | 12 | No |
| 15 | `odd/tasks/conformal-prediction.md` | T1 → T7 | 14 | No |
| 16 | `odd/tasks/smc-inference.md` | T1 → T6 | 11, 13, 14 | No |
| 17 | `odd/tasks/observation-models.md` | T1 → T8 | 14, 16 | No |
| 18 | `odd/tasks/profile-likelihood.md` | T1 → T6 | 13, 14, 17 | No |
| 19 | `odd/tasks/local-approximations.md` | T1 → T6 | 13, 14, 16, 17 | No |
| 20 | `odd/tasks/mcmc-samplers.md` | T1 → T7 | 16, 17, 19 | No |
| 21 | `odd/tasks/gradient-inference.md` | T1 → T8 | 13, 19, 20 | No |
| 22 | `odd/tasks/state-space-inference.md` | T1 → T8 | 11, 17, 20 | No |
| 23 | `odd/tasks/sensitivity-analysis.md` | T1 → T8 | 12, 16 | No |
| 24 | `odd/tasks/model-comparison.md` | T1 → T6 | 16, 17, 19, 20 | No |
| 25 | `odd/tasks/emulation-discrepancy.md` | T1 → T6 | 13, 17, 20, 23 | No |
| 26 | Release 0.4.0 (`docs/development/release.md`) | — | 10–25 | — |
| 27 | `odd/tasks/library-benchmark.md` | T1 → T5 | 26 | No |
| 28 | `odd/tasks/markov-chains.md` | T1 → T8 | 10 (S1–S4), 11 (S5) | No |
| 29 | `odd/tasks/integral-outputs.md` | T1 → T7 | 11, 28 | No |
| 30 | `odd/tasks/network-models.md` | T1 → T8 | 10, 11, 29 | No |
| 31 | `odd/tasks/agent-based-models.md` | T1 → T8 | 30 | No |

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
- **Foundation before any new kind, engine or inference method.**
  `model-kind-foundation` gives the description, seeds, replicates and
  calibration with common random numbers. Every later ODD builds on them, and
  its builder change (D10) migrates every test once, so the inference ODDs
  are written against the final API.
- **Stochastic engine before the inference program.** It reuses the whole
  aggregate description, so it costs least; ABC (`smc-inference`), the
  synthetic likelihood and the particle filters (`state-space-inference`)
  need it, and `inference-foundation` splits a calibration objective that
  already holds replicates, so the split is done once.
- **`inference-foundation` first of the program.** It splits the calibration
  problem, removes the ensemble, and adds the sample, the predictive, the
  scores and the entry point that the thirteen later ODDs use. No release
  between it and `resampling-uncertainty` (its D2), so no release lacks an
  uncertainty method.
- **Identifiability and resampling next.** `identifiability` gives the
  sensitivity port that the Gibbs scale, profiles, Laplace and gradients
  read; `resampling-uncertainty` gives the blocks and the replicate runner
  that conformal prediction, the Gibbs scale, profiles and the sandwich use.
  Both need no noise model, the user's first concern (`inference-foundation`
  L3).
- **Methods without a noise model before methods with one.** Conformal
  prediction and the SMC core (Gibbs posterior, ABC) come before
  `observation-models`, which then adds the likelihood target to the same
  core.
- **Likelihood methods in dependency order.** Profiles and local
  approximations need the likelihood; MCMC reuses the SMC kernels and the
  unconstrained transform of local approximations; gradients need the MCMC
  runner for NUTS and PSIS for ADVI; state-space methods need the likelihood,
  the chains and the stochastic engine.
- **Analysis and comparison last of the program.** Sensitivity analysis
  needs only the distributions and the predictive; model comparison needs the
  evidence, pointwise likelihood, PSIS and kernels; emulation needs the
  quantities of interest and the chains.
- **One release, then the benchmark.** Steps 10–25 break the API in two
  places (the foundation's builder and the removal of the ensemble). One 0.4.0
  release gathers them, and the benchmark runs on that release, as the user
  asked (`library-benchmark` L7). A release after step 14 is allowed if a
  user needs it earlier.
- **Benchmark before the other modelling ODDs.** The user kept the benchmark
  before new kinds of modelling (Markov chains, networks, agents); only the
  stochastic engine moves ahead of it, because the inference program needs
  it.
- **Markov chains, then integral outputs.** Most of a chain fits the aggregate
  kind already; its builder, matrix and analysis (S1–S4) need only the
  foundation. Rewards are the main use of Markov chains, so integral outputs
  follow with no gap (Markov D4), before networks and agents, so those kinds
  add integrals of their counts with the declaration already in place.
- **Networks before agents.** Both deliver one structure, `Agents`
  (`agent-based-models.md` D6). The network ODD delivers it first, with a
  graph and no attributes, and the variant `Network`: the first real second
  kind, which tests the generic layers with less to build. The agent ODD then
  adds attributes, `Complete`, `Groups` and Python rules on the same
  structure and engines.

## When decisions are made

The decisions of an ODD of the uncertainty program (steps 12–25) are settled
when its implementation starts, not before (user, 2026-10-15). A decision
that changes what an earlier ODD built, or the architecture of
`inference-foundation.md` (its D1, D3, D4, D7), updates the later ODDs that
depend on it in the same PR, with a line in their Log.

## Not in parallel

Steps 1–8 change the same tests and modules (`tests/support/`,
`api/reporting.py`, `context/`), so they run one after the other. Each starts
from `develop` after the previous PR merges. Steps 10–25 and 28–31 also run
one after the other, in the order of the table: each builds on the
interfaces the one before adds. Inside the program (12–25), steps 13 and 14,
and steps 23 and 22, touch different modules and could run in parallel on two
branches; the table keeps one order so each ODD is written against the merged
code of the one before.
