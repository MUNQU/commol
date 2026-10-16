# Feature: Markov chains as a structure of a process

Issue: [#142](https://github.com/MUNQU/commol/issues/142), which holds the status, the dependencies and the pull request.

Goal: a user writes a process once with `ModelBuilder` (from a matrix if they
wish), chooses `MarkovChain` as its structure when building it (foundation
D10), and gets its distribution over time from the deterministic engines,
its trajectories from the stochastic engines, and its analysis (transition
matrix, stationary distribution, absorption), with an error at build when
the process is not a chain.
Motivation: the user asked to add Markov chains to the kinds of modelling
(L1). Most of a Markov chain already fits the aggregate kind; what is
missing is a way to choose it as a modelling, a check, an analysis and the
documentation that shows it.
Branch: `feat/markov-chains` from `develop`, after
`odd/model-kind-foundation.md` and
`odd/stochastic-compartmental.md` are merged. S1 to S4 need only the
foundation and could go first; S5 needs the stochastic engines.
Engram mirror: topic `odd/markov-chains/tasks` (project `commol`).
Runners: the Rust tests, clippy and rustdoc (from the root), `maturin develop
--release --features test-kind`, then `pytest`, `pytest -m docs`, `ruff check
.`, `ty check commol` and `vulture` (from `py-commol/`), `mkdocs build
--strict` (from the root).

## Current state (verified 2026-10-14 on `develop` at `901236d`)

### What already works

| Markov chain | Commol today |
|---|---|
| States | Bins, or the compartments of a stratified model |
| Transition probability `p_ij` per step (discrete time) | A proportional rate `p_ij` from `i` to `j` in a `DifferenceEquations` model: the flow is `p_ij × X_i` per step (`docs/guide/transitions.md`) |
| Transition rate `q_ij` (continuous time) | A proportional rate `q_ij` in a `DifferentialEquations` model: `dX/dt = Σ flows` is the Kolmogorov forward equation |
| The distribution `π(n)` | The state with `total=1.0` (`InitialConditions.total` is a real number since PR #108) |
| `N` units | `total=N`: the expected number in each bin |
| Non-homogeneous chain | A rate that reads `t`, a series parameter or an `AxisPattern` |
| Tunnel states (time in a state) | A chain of bins, or a stratification by time in state |
| An absorbing state | A bin with no outgoing transition |

### What is missing

- **Probabilities out of a compartment.** If the proportional rates out of
  one compartment sum to more than 1, its amount goes negative.
  `odd/difference-negative-amounts.md` makes every difference step that
  leaves an amount below zero an error, whatever its cause, before this ODD
  runs; a chain needs no check of its own.
- **No analysis.** Commol plans every flow (`transition_flows`,
  `commol-model/src/flows.rs`) but cannot give the transition matrix, the
  stationary distribution or the absorption probabilities.
- **No trajectories.** One path of a unit through the chain is a stochastic
  run, which `odd/stochastic-compartmental.md` adds.
- **No way to choose it.** A chain is not a modelling the user can pick, and
  a chain of ten states needs one transition per entry of its matrix in
  `ModelBuilder`.
- **No guide.** Nothing in `docs/` says that a Markov chain is a compartmental
  model with proportional rates.

## Specs

Every spec follows the [documentation definition of done](../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — A chain is a structure chosen at build

- Fix (D1): `MarkovChain(total, bin_fractions, stratification_fractions)` is a
  structure (foundation D10) that the user passes to `build`, as `Aggregate`,
  `Network` or `Agents`: `process.build(MarkovChain(total=1.0,
  bin_fractions={"A": 1.0, "B": 0.0, "C": 0.0}), engine="DifferenceEquations")`.
  It is a variant of `Aggregate` (foundation D11), the class
  `MarkovChain(Aggregate)`: same kind, fields and engines, with `"variant":
  "markov_chain"` in the model file, so it loads back as `MarkovChain` and the
  engines do not change. With the variant, `build` asks the engine to check
  that the process is a chain (S3): every flow proportional, one source and at most one target, no
  rate that reads the state, no inflow and no impulse; else it raises with
  the reason. `total=1.0` gives a distribution, `total=N` the expected number
  in each bin.
- Fix (D1): `ModelBuilder.add_transition_matrix(matrix, bins, rate_names=None)`
  adds one transition for each entry off the diagonal that is not zero (a
  number, or the name of a parameter when `rate_names` gives one), for any
  structure. The diagonal is what stays, and is not a transition.
- Fix (D1): a rate keeps one word, `rate`; the engine sets what it means
  (a per-step probability in a discrete engine, a rate per unit of the axis
  in a continuous one), as for every structure.
- Check: a process written with `add_transition_matrix` and the same process
  written transition by transition give equal models; `MarkovChain` builds
  with each engine of the aggregate kind; a process with an absolute rate, a
  rate that reads `neighbors_B` or two sources fails at `build` with
  `MarkovChain`, names the transition and the reason, and builds with
  `Aggregate`; a stratified chain with one category-specific rate gives the
  matrix written by hand over the expanded bins.

### S2 — Probabilities that leave a compartment

- Fix: none in the engine. The check moved to
  `difference-negative-amounts.md` (L2), which covers every cause of a
  negative amount, not only probabilities that sum above 1. `matrix(at)` (S3)
  and the analysis (S4) reject a discrete chain whose `P` has an entry
  outside `[0, 1]` at a position they read, since the analysis needs a
  stochastic matrix, and name the flows.
- Check: a chain whose outflows from one state sum to `1.2` fails in
  `matrix()` and names the flows.

### S3 — The chain of a model

- Fix: `Model.markov_chain()`, on a model whose structure is `MarkovChain`,
  returns a `MarkovAnalysis` bound to the model (another structure raises
  and names `MarkovChain`), with the states (the expanded compartments, plus one absorbing `outside`
  state when a flow leaves the system), the time (`"discrete"` for
  `DifferenceEquations` and `StochasticSteps`, `"continuous"` for the others,
  from `EngineType::rate_meaning`), `homogeneous` (false when a rate reads
  `t`, `step`, a series parameter or an `AxisPattern`, which the engine knows
  from the compiled flows), and `matrix(at=0.0)`: `P = I + K` per step, or the
  generator `Q = K − diag(K·1)`, where `K_ij` is the rate from `i` to `j` at
  the position `at`. The engine builds `K` from the planned flows (D3).
- Fix: the engine rejects a model that is not a chain, with the reason: an
  absolute rate, a flow with several sources or targets, a rate that reads the
  state (a compartment, a bin total, `total`), or an inflow.
- Check: a three-state chain gives the matrix written by hand; each row of
  `P` sums to 1 and each row of `Q` to 0; an absolute rate is rejected and
  named; a rate that reads `t` makes `homogeneous` false and `matrix(at)`
  differ between two positions as written by hand.

### S4 — Analysis

- Fix (D3): `MarkovAnalysis` gives, from each starting state:
  - `classes()`: communicating classes, which are closed, and which states
    are absorbing or transient (of `matrix(at)` for a chain that is not
    homogeneous, with `at` required);
  - `absorption_probabilities(*, start=0.0, horizon=None)`,
    `expected_time_in_states(…)` and `expected_time_to_absorption(…)`, each
    returning one result type with the values and the mass not yet absorbed
    at the horizon (zero with no horizon);
    - homogeneous chain: with no horizon, from the fundamental matrix
      `N = (I − T)⁻¹` (discrete) or `−T⁻¹` (continuous), where `T` is the
      transient block; with a horizon, as below;
    - chain that is not homogeneous: `horizon` is required, with no default;
      discrete time multiplies the matrices of each step from `start`
      (`P(s)·P(s+1)·…`, or the backward recursion `h(k) = P(k)·h(k+1)`);
      continuous time solves the Kolmogorov forward equation of the
      transition matrix `P(s, t)` with the ODE engine's solver;
  - `stationary_distribution(*, period=None)`: the solution of `πP = π` or
    `πQ = 0` for a homogeneous chain, with an error that names the classes
    when it is not unique; with `period`, the distribution of each phase from
    the product of the matrices over one period, after commol checks that the
    matrices repeat over two periods; a chain that is not homogeneous with no
    `period` raises and says that it has no stationary distribution;
  - `to_numpy(at=0.0)` and `to_dataframe(at=0.0)`, with state labels.
  The linear algebra runs in Rust (`nalgebra`, already a dependency of
  `commol-differential`), so the numbers come from the engine. A `P` with an
  entry outside `[0, 1]` at a position the analysis reads raises (S2).
- Check: the gambler's ruin with `p = 0.4` and a capital of `3` out of `5`
  gives the ruin probability of its closed form; a two-state continuous chain
  gives `π = (b, a)/(a + b)`; a chain with two closed classes raises from
  `stationary_distribution()` and names them; a homogeneous chain gives the
  same absorption with a long horizon as with none, within the unabsorbed
  mass; a pure death chain with probability `q(k)` at step `k` gives the
  survival `∏(1 − q(k))` to the horizon; a continuous death rate `a + b t`
  gives the survival `exp(−(a s + b s²/2))` over `[0, s]`; a chain with a
  two-step period gives the stationary distribution of `P(0)·P(1)` and of
  `P(1)·P(0)`, and raises when the matrices do not repeat; a chain that is
  not homogeneous raises without `horizon` and without `period`.

### S5 — Trajectories

- Fix: none beyond `stochastic-compartmental.md`: a chain run with
  `StochasticSteps` or `StochasticEvents` and `total=N` gives `N` units moving
  at random; with `total=1` it gives one path. The guide shows both.
- Check: the share of 10 000 paths absorbed in each state matches
  `absorption_probabilities()` within its binomial tolerance.

### S6 — Description

- Fix: for a model whose structure is `MarkovChain`, the `dynamics` section of the
  description (foundation S1) adds a `Matrix` block with `P` or `Q`, next to
  the equations; the state graph labels each edge with its probability or
  rate.
- Check: the Markdown rendering of a chain holds its matrix with state labels.

### S7 — Documentation

- Fix: a guide page "Markov chains" (`docs/guide/markov-chains.md`):
  a process written with `ModelBuilder` and `add_transition_matrix`, then
  built with `MarkovChain` and with `Aggregate`, how a chain maps
  to a model (the table of "What already works"), stratified chains and the
  names of their expanded states, `N` units and
  distributions, why probabilities out of a state must sum to at most 1 (the
  error of `difference-negative-amounts.md` and of S2), the analysis of S4,
  trajectories, and a worked example with abstract states.
  `design-decisions.md`: "A Markov chain is a structure of a process".
  Changelog.
- Check: `pytest -m docs`, `mkdocs build --strict`.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1 | How does a user choose a Markov chain? | As a structure passed to `build`, `MarkovChain`, a variant of `Aggregate` that checks at build that the process is a chain and gives the analysis; the process is written with `ModelBuilder`, from a matrix with `add_transition_matrix` if the user wishes. No new `ModelKind`, no builder of its own. | `MarkovChainBuilder` (approved first, L3, L4): a second syntax for one process, which could not move to a network or to agents without a rewrite (foundation D10, L7). The analysis on any aggregate model (second proposal): the check would come at analysis, not when the user picks the modelling, and every aggregate model would show methods that rarely apply (L8). A `markov` kind: engines would run two kinds, and the chain would lose or copy what `Aggregate` has. Cost: a user with an `Aggregate` model who wants its stationary distribution changes one line, as for a network or agents. | approved (L8) |
| D2 | S2 | Where to check the probabilities | Moved to `difference-negative-amounts.md` D1 (L2). | — | superseded (L2) |
| D3 | S3, S4 | Non-homogeneous chains | `matrix(at)` gives the matrix at one position. The analysis takes `horizon`: optional for a homogeneous chain (none means infinite), required with no default otherwise, and every result gives the mass not yet absorbed at the horizon. `stationary_distribution(period=T)` for a periodic chain, after a check over two periods; with no period, a chain that is not homogeneous raises, since it has no stationary distribution. | Analysis of homogeneous chains only (first recommended): chains whose probabilities change along the axis, a usual case, would get no absorption or time in states (L5). Analyse `matrix(at)` as if it were homogeneous: wrong results without notice. A default horizon: the horizon is a modelling choice. Cost: `O(n³ H)` for `n` expanded states and `H` steps (about 8·10⁹ operations for 200 states and 1000 steps, seconds in Rust); in continuous time an ODE of `n²` unknowns; the check of a period over two periods does not prove that it holds for ever, so the guide says that the user states it. | approved (L5) |
| D4 | — | Rewards (a weight per bin and unit of the axis, summed along a run) | Out of this ODD, in `integral-outputs.md`, written now and run right after this one: an output that integrates an expression of the state along the axis, for every kind. Until then, undiscounted totals come from `expected_time_in_states` times the rewards. | Inside this ODD: rewards for chains only, though the operation is the same for every kind; generalizing them later would break the API. Never: Markov chains without rewards miss a main use. Later with no date: the same cost to the user as never. | approved (L6) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S2 | Rejection of a `P` with an entry outside `[0, 1]` in `matrix(at)` and the analysis | direct | |
| T2 | S3 | The matrix `K` from the planned flows; `Model.markov_chain` and its rejections | direct | |
| T3 | S1 | `MarkovChain` structure and its check at build; `ModelBuilder.add_transition_matrix` | direct | |
| T4 | S4 | Analysis in Rust (homogeneous, horizon, periodic); bindings; `MarkovChain` in Python | direct | |
| T5 | S5 | Trajectory tests against the analysis | direct | |
| T6 | S6 | Matrix in the description | direct | |
| T7 | S7 | Documentation | direct | |
| T8 | all | Final audit: every runner | direct | |

## Delivery

- Strategy: one PR to `develop` from `feat/markov-chains`, squash-merged
  (`feat(model): …`).
- Order: T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8.
- Version impact: added (`MarkovChain`, `add_transition_matrix`,
  `markov_chain`); not breaking.

## Evidence

(Filled during implementation.)

## Log

- L1 (user, 2026-10-14): added Markov chains to the kinds of modelling to
  plan, and asked for one ODD per kind on top of `model-kind-foundation.md`.
- L2 (user, 2026-10-14): a scan of every difference run of the suite found
  seven negative runs, none from probabilities above 1, so the check of S2 was
  too narrow. The user asked for the correction in its own ODD or in the one
  that adds the check; it went to `difference-negative-amounts.md`, with the
  check, and the engine part of S2 and D2 left this ODD.
- L3 (user, 2026-10-14): asked whether a class for chains, on the existing
  code, would serve users better than none. It would: D1 keeps no new model
  kind and adds `MarkovChainBuilder`.
- L4 (user, 2026-10-14): asked whether the builder could take
  stratifications. It can: a stratified chain is still a chain over the
  expanded states, and moves between categories exist in `ModelBuilder`.
  Approved D1 with stratifications.
- L5 (user, 2026-10-14): found too high the cost of analysing only
  homogeneous chains. Over a finite horizon every quantity but the
  stationary distribution exists for any chain, and a periodic chain has one
  per phase. Approved D3 with `horizon` and `period`.
- L6 (user, 2026-10-14): approved D4: rewards in their own ODD,
  `integral-outputs.md`, written at once and run right after this one.
- (2026-10-14) Foundation D10 (`model-kind-foundation.md` L12): one process,
  structures apart, neutral terms (`neighbors_<bin>` for `contacts_<state>`).
  The decisions marked "to revisit" are reviewed under it.
- L7 (user, 2026-10-14): under foundation D10, `MarkovChainBuilder` would be
  a second syntax for one process.
- L8 (user, 2026-10-14): asked whether the choice of modelling made after
  the process should include Markov chains. It should: approved D1 with
  `MarkovChain` as a structure chosen at build, its check at build, and
  `add_transition_matrix` in `ModelBuilder`.
