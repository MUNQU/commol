# Feature: integral outputs

Issue: [#143](https://github.com/MUNQU/commol/issues/143), which holds the status, the dependencies and the pull request.

Goal: a model can declare an output that is the integral, along the axis, of
an expression of its state, such as the time spent in a state, the cost or
weighted sum of the bins of a chain, or the area under a curve. Every engine computes it
during the run, so it can be observed, calibrated, reported and described as
any other output.
Motivation: the Markov chain ODD left out rewards (a weight per bin and unit
of the axis), a main use of Markov chains, because the operation is not
specific to chains (`markov-chains.md` D4, L6). The user asked for this ODD
to be written at once and to run right after the Markov ODD.
Branch: `feat/integral-outputs` from `develop`, after
`odd/markov-chains.md` is merged.
Engram mirror: topic `odd/integral-outputs/tasks` (project `commol`).
Runners: the Rust tests, clippy and rustdoc (from the root), `maturin develop
--release --features test-kind`, then `pytest`, `pytest -m docs`, `ruff check
.`, `ty check commol` and `vulture` (from `py-commol/`), `mkdocs build
--strict` (from the root).

## Current state (verified 2026-10-14 on `develop` at `901236d`)

- An accumulator (`ModelBuilder.add_accumulator`,
  `docs/guide/building-models.md`, "Accumulators") counts what transitions
  move. It is a state entry after the compartments: `CompiledModel::derivatives`
  (`commol-model/src/system.rs`) adds each flow to the accumulators it names,
  the difference engine adds it once per step, and the ODE engine integrates
  it with the solver. With stratifications, an accumulator expands like a bin
  (`expand_accumulators`, `commol-model/src/builder.rs`).
- Nothing integrates what stays in a compartment. Person-time, the cost of a
  chain, or the area under a curve must be computed by the user
  from the trajectory, outside the engine, so they cannot be observed in a
  calibration.
- Calibration already observes cumulative outputs over windows
  (`docs/guide/calibration-observations.md`, `window`), and the reports read
  accumulators as cumulative (`api/reporting.py`).
- After the earlier ODDs: `describe` and its `outputs` section (foundation),
  the stochastic engines, and `MarkovChain` with `expected_time_in_states`
  (Markov S4).

## Specs

Every spec follows the [documentation definition of done](../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — The declaration

- Fix (D1, D4): `ModelBuilder.add_integral(id, name, expression,
  per_compartment=None, stratified_expressions=None, counting=None)`, part
  of the process (foundation D10). `expression` is a rate expression (the language of
  `docs/guide/mathematical-expressions.md`) that may read compartments, bin
  totals, `total`, `total_<category>`, parameters, series, `t` and `step`.
  A bin id means what it means in a rate: the total of the bin. By default an
  integral is one output. With `per_compartment=True` (the flag of
  `add_transition`, with the same meaning, `docs/guide/transitions.md`) it
  expands like a bin, and in each copy a bin id means the compartment of that
  copy's categories; `stratified_expressions` then gives some categories
  their own expression, with the matching rules of `stratified_rates`, and
  raises without `per_compartment=True`. Rust `Integral` in `Process`; `check_model`
  validates the variables of every expression and its security, in Rust.
- Fix (D2): there is no discount argument; a discount is a factor in the
  expression, such as `c_A * A * (1 + d) ^ (-t)` or `c_A * A * exp(-r * t)`.
- Check: a model saved and loaded back is equal; an expression with an
  unknown variable is rejected and named; in a stratified model, an integral
  of `A` gives one output that integrates the bin total, and with
  `per_compartment=True` one output per category, each reading its own
  compartment, whose sum equals the single output;
  `stratified_expressions` without `per_compartment=True` raises.

### S2 — The engines

- Fix: an integral is a state entry after the accumulators. `derivatives`
  writes the value of each expression at the current state, so the ODE engine
  integrates it with its tolerance and its breakpoints.
- Fix (D3): in the discrete engines (`DifferenceEquations`,
  `StochasticSteps`) the user chooses how a step counts, with no default, and
  `build` raises for an integral without it, listing the options:
  `"start"` (the value at the start of each step, `Σ_{k<n} f(k)`), `"end"`
  (`Σ_{1≤k≤n} f(k)`), or `"trapezoid"` (the mean of both, the half-cycle
  correction of discrete-time Markov models). A continuous engine raises when `counting` is
  given, since its integral is exact.
- Fix: `StochasticEvents` integrates the expression between events, where the
  state is constant: exactly when the expression does not read `t`, otherwise
  with the quadrature and the tolerance of the time rescaling
  (`stochastic-compartmental.md` D4).
- Fix: integrals are outside `state_outputs`, so they do not count towards
  `total`, as accumulators.
- Check: the integral of `A` under `A' = −k A` is `A₀ (1 − e^(−k t)) / k`
  (ODE, within the solver tolerance); in a difference model with
  `A(n) = A₀ (1 − p)ⁿ`, `"start"` gives `A₀ (1 − (1 − p)ⁿ) / p`, `"end"` that
  sum shifted by one step, and `"trapezoid"` their mean; the mean over many
  `StochasticEvents` runs equals the ODE integral within its tolerance; an
  expression with a discount factor gives the discounted closed form.

### S3 — Results, calibration, reports and plots

- Fix: an integral is a cumulative output, as an accumulator: `Simulation`
  results hold it, an observation can name it with or without `window`, the
  reports read it as cumulative, and the plotter draws it.
- Check: a calibration of `k` from observations of the integral of `A`
  recovers it; a windowed observation of an integral compares the increase
  over its window.

### S4 — Description

- Fix: the `outputs` section of the description (foundation S1) lists each
  integral with its expression (in LaTeX, as `∫ f dt` from `0`), its
  stratified expressions and, in a discrete engine, its counting.
- Check: the Markdown rendering of a model with one integral holds its
  expression and its counting.

### S5 — Documentation

- Fix: `docs/guide/building-models.md` (a section "Integrals", next to
  "Accumulators", with person-time, a discounted cost and an area under a
  curve, all abstract); `docs/guide/markov-chains.md` (rewards of a chain
  as integrals, and their relation to `expected_time_in_states`);
  `design-decisions.md` ("A discount is part of the expression", "Discrete
  counting is the user's choice"); API pages; changelog.
- Check: `pytest -m docs`, `mkdocs build --strict`.

## Out of scope

- The network and agent kinds: each of their ODDs adds integrals of its
  counts (the same declaration, its own engines).
- Rewards in the analysis of `MarkovChain` (an expected reward from the
  fundamental matrix): the integral of a run with `total=1` gives the same
  number with the counting the user chose.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1 | What does the output integrate? | Any expression of the state: `add_integral(id, name, expression)`. Occupancy is the integral of `A`, a reward the integral of `c_A * A + c_B * B`. | Weights per state only (`add_occupancy(weights={"A": "c_A"})`, the name first proposed): a linear reward, but no area under a curve of a ratio, no utility that depends on the total, and a name that describes one use. A reward feature for chains only: the same operation, limited to one kind (`markov-chains.md` D4). | approved (L2) |
| D2 | S1 | Discounting | A factor in the expression; no argument. | A `discount=` argument: it must choose between compounding per step (`(1 + d)^(−t)`) and continuous (`e^(−r t)`), and between rates per year or per unit of the axis, a modelling choice the library would make for the user. Cost: the user writes the factor; the guide gives both forms. | approved (L3) |
| D3 | S2 | How a step counts in a discrete engine | `counting` required, with no default: `"start"`, `"end"` or `"trapezoid"`. | `"start"` by default (what the difference step gives with no extra work): discrete-time Markov models differ by the half-cycle correction, which changes results by about half a step of reward, and the literature does not agree on one rule, so it is the user's choice. A fixed `"trapezoid"`: the same objection. | approved (L4) |
| D4 | S1 | Stratified integrals | A bin id means the bin total, as in a rate. One output by default; `per_compartment=True` (the flag and meaning of `add_transition`) expands one output per category, with bin ids meaning the compartment of each copy; `stratified_expressions` needs it. | Expand by default with bin ids meaning the compartment of each copy (first recommended): `A` would mean one thing in a rate and another in an integral (L5). A new syntax for the own compartment (`$A`): a second way to say what `per_compartment` says, and more syntax for the parser and its security check. Change the meaning in rates too (a bin id as its own compartment by default): fixes the trap of `difference-negative-amounts.md` at its source, but breaks every stratified model that reads bins, and needs its own ODD; if it comes, integrals follow it unchanged. Cost: one flag for outputs per category. | approved (L5) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1 | `Integral` in Rust and Python; builder methods; validation; save and load | direct | |
| T2 | S2 | Integrals in `derivatives` and in the ODE and difference engines; counting | direct | |
| T3 | S2 | Integrals in `StochasticSteps` and `StochasticEvents` | direct | |
| T4 | S3 | Results, observations, reports and plots | direct | |
| T5 | S4 | Description | direct | |
| T6 | S5 | Documentation | direct | |
| T7 | all | Final audit: every runner | direct | |

## Delivery

- Strategy: one PR to `develop` from `feat/integral-outputs`, squash-merged
  (`feat(model): …`), with this ODD archived in it.
- Order: T1 → T2 → T3 → T4 → T5 → T6 → T7.
- Version impact: added; not breaking.

## Evidence

(Filled during implementation.)

## Log

- L1 (user, 2026-10-14): approved `markov-chains.md` D4: rewards out of the
  Markov ODD, in an ODD of their own, written now and run right after it.
  This ODD names the feature "integral" instead of "occupancy" (D1), since an
  occupancy is one of its uses.
- L2 (user, 2026-10-14): approved D1, with the name "integral".
- L3 (user, 2026-10-14): approved D2 (a discount is a factor in the
  expression).
- L4 (user, 2026-10-14): approved D3 (`counting` required in discrete
  engines, an error when given to a continuous one).
- L5 (user, 2026-10-14): found too high the cost of a bin id meaning the
  compartment in an integral and the bin total in a rate. Approved D4 with the
  meaning of rates and the flag `per_compartment`.
