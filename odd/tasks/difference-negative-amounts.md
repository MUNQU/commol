# Fix: amounts that go negative in a difference step

Goal: a `DifferenceEquations` run stops with an error when a step leaves a
compartment with less than nothing, instead of returning negative amounts
without notice; calibration treats such a run as infeasible, not as a
failure; and the examples and tests that go negative today are corrected.
Motivation: while it reviewed D5 of `stochastic-compartmental.md`, the user
asked whether an error for per-step probabilities above 1 cost too much. An
instrumented run of the whole suite (L1) found seven difference runs that end
with negative amounts, four of them in published examples. None comes from
proportional rates that sum above 1, the case `markov-chains.md` S2 checked;
so the rule is wider than that ODD said, and it moves here.
Branch: `fix/difference-negative-amounts` from `develop`, after
`odd/archive/tasks/axis-scale-refactor.md` is merged and before the 0.3.0 release.
Engram mirror: topic `odd/difference-negative-amounts/tasks` (project `commol`).
Runners: the Rust tests, clippy and rustdoc (from the root), `maturin develop
--release --features test-kind`, then `pytest`, `pytest -m docs`, `ruff check
.`, `ty check commol` and `vulture` (from `py-commol/`), `mkdocs build
--strict` (from the root).

## Current state (verified 2026-10-14 on `develop` at `901236d`)

- `DifferenceEquations::step` (`commol-difference/src/simulation.rs`) adds
  the flows of every transition, computed from the state at the start of the
  step, and checks nothing. A step that moves more than a compartment holds
  leaves it negative, and the run goes on.
- Every engine returns `Result<_, String>` (`SimulationEngine`,
  `commol-core/src/simulation_engine.rs`). `CalibrationProblem::cost`
  (`commol-calibration/src/calibration_problem.rs`) aborts the whole
  calibration on any simulation error, and gives the base penalty for a run
  with values that are not finite.
- With stratifications, a bin id in a rate means the total of the bin
  (`docs/guide/transitions.md`, "per_compartment"). A rate such as
  `k1 * A * B / total` then moves, from each compartment of `A`, an amount
  computed from the bin total, so the flow out of one compartment can exceed
  what it holds.

### Runs that go negative (L1)

| Test or page | Model | Lowest amount | Cause |
|---|---|---|---|
| `docs/guide/examples.md`, Example 6 | Capacity Model | `C` −117.4 | `k2_max * (1 - max(0, (B - capacity) / capacity))` names `B`, so it is absolute and not multiplied by `B`; it turns negative when `B > 2 × capacity`, and the flow drains `C` |
| `docs/guide/examples.md`, Example 7 | Multi-Stratified Model | `A_g1_t2` −229.7 | `k1_low * A * B / total` and `k1_high * …` with bin totals |
| `docs/guide/examples.md`, Example 8 | Subgroup-Normalized Model | `A_g1` −165.0 | `k1 * A * B / total` with bin totals |
| `docs/guide/examples.md`, Example 9 | Conditional Stratification Model | `A_g2_s1` −528.0 | `k1 * A * B / total` with bin totals |
| `tests/building/test_stratifications.py`, `TestConditionalStratification::test_a_run_keeps_the_total` | `abc_builder` | `A_g2_s1` −162.8 | Default `ab` rate `k1 * A * B / total` with bin totals |
| `tests/building/test_stratifications.py`, `TestFullProduct::test_a_run_of_a_two_stratification_model_keeps_the_total` | `abc_builder` | `A_g2_t2` −50.9 | The same |
| `tests/reference_models/test_onchocerciasis.py`, `test_fractions_of_the_totals_follow_the_rescaled_system` | `flow_model` | `P` −29.9 | `increments` (`tests/support/flows.py`) reads the right-hand side as one difference step; `P` loses `(phi + gamma + mu) × P = 2.82 P` per year |

## Specs

Every spec follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — The check in the difference step

- Fix (D1): after the flows of a step and after the impulses of a step,
  `DifferenceEquations` checks every compartment. An amount below
  `−10⁻¹² × total` (the total at the start of the step, so rounding of flows
  that empty a compartment exactly passes) stops the run with an error that
  names the compartment, the step, its amount before the step, and the flows
  out of it with their amounts, largest first. Accumulators are not checked:
  a net flow may count negative.
- Fix: the check reads the state the step already wrote, one pass over the
  compartments, with no allocation.
- Check: a model that moves `1.2 × A` out of `A` in one step fails at step 0
  and names both flows; a rate that turns negative at step 7 fails at the
  step where the amount drops below zero; a proportional rate of exactly 1
  empties a compartment with no error; an impulse that moves more than its
  source holds fails at its step.

### S2 — A typed error for infeasible runs

- Fix (D2): `commol-core` gets `SimulationError`, with `Infeasible { … }`
  for a run whose state leaves the values the model allows, and `Failed(String)`
  for every other error. `SimulationEngine` returns it where it returns
  `String` today; the bindings turn `Infeasible` into a Python
  `InfeasibleStateError` (a subclass of `ValueError`) and `Failed` into the
  error they raise today.
- Fix: `CalibrationProblem::cost` gives an infeasible run the base penalty,
  as it does for values that are not finite, and records nothing in the
  history; any other error still aborts. The probabilistic calibration and its
  predictions do the same.
- Check: a calibration whose bounds include rates that empty a compartment
  converges to the rates that fit the data; a calibration with a rate that
  cannot be evaluated still fails with its message; every existing
  calibration test passes with the same results (T2 records any change).

### S3 — Examples and tests

- Fix: `docs/guide/examples.md`:
  - Example 6 writes the rate it describes, a flow that slows as the
    downstream compartment fills: `k2_max * B * max(0, 1 - C / capacity)`.
  - Examples 7, 8 and 9 add `per_compartment=True` to `flow_AB`, so each
    compartment of `A` loses an amount computed from its own amount. Each
    example says in one sentence why.
- Fix: the two tests of `test_stratifications.py` pass `ab="k1"`, a
  proportional rate; they check that the total stays constant, which does not
  depend on the form of the rate.
- Fix: `increments` (`tests/support/flows.py`) multiplies every rate by
  `h = 2⁻⁶` and divides the increments by `h`. A power of two scales a float
  exactly and keeps the kind of each rate (it names the same variables), so
  the result is the right-hand side, and the step stays inside the
  compartments.
- Check: the instrumented run of L1 (every `DifferenceEquations` run of
  `pytest -m "docs or not docs"`) finds no negative amount; the test of
  onchocerciasis passes with `ENGINE_REL`; the examples print the same kind
  of output as before.

### S4 — Documentation

- Fix: `docs/guide/transitions.md` (a section "Amounts below zero": what the
  difference engine checks, the error, and the two usual causes, an absolute
  rate on a bin total and a rate that turns negative, each with its fix);
  `docs/guide/calibration.md` (an infeasible run gets the penalty);
  `docs/codebase/design-decisions.md` ("A difference step never leaves an
  amount below zero", with D1 and D2); the API page of the error; changelog
  (Changed, breaking: a difference model that went negative now raises;
  Fixed: the four examples).
- Check: `pytest -m docs`, `mkdocs build --strict`.

## Out of scope

- The ODE engine. A solver may step below zero by its tolerance, and a model
  whose exact solution goes negative is a modelling choice there; a check
  needs its own tolerance and its own ODD.
- The meaning of a bin id in a stratified rate. It is the bin total by design
  (`per_compartment` exists for the other case), and the examples show it is
  easy to misuse; whether a rate that names its own source bin should default
  to the compartment is a separate decision. The check of S1 makes the misuse
  visible.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1 | What happens when a difference step leaves an amount below zero? | An error that names the compartment, the step and the flows out of it, for every `DifferenceEquations` model and every cause. | A check only of proportional rates that sum above 1 (`markov-chains.md` D2 as written): none of the seven runs found breaks that rule (L1). A warning: a negative amount is never a correct state, and a warning inside a loop of calibrations is lost. Clamp the amount to zero: the total is no longer kept and the flows are no longer those of the model, without notice. Scale the outflows down to what the compartment holds: a different model, chosen by the library, against "modelling choices belong to the user". | approved (L2) |
| D2 | S2 | How does calibration treat such a run? | A typed error; calibration gives the run the base penalty and goes on. | Abort, as for any error today: an optimizer that explores the bounds of a rate would stop on the first infeasible point, though the fit lies elsewhere. Every error as a penalty: a rate that cannot be evaluated would hide behind a poor fit. Parse the message: a contract kept by the wording of an error. Cost: the error type of `SimulationEngine` changes in every engine; commol is alpha. | approved (L2) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1 | The check in `DifferenceEquations::step` and after impulses; Rust tests | direct | |
| T2 | S2 | `SimulationError`, the trait and the engines; penalty in calibration; Python error | direct | |
| T3 | S3 | Examples 6–9, the two tests, `increments`; the instrumented run | direct | |
| T4 | S4 | Documentation | direct | |
| T5 | all | Final audit: every runner | direct | |

## Delivery

- Strategy: one PR to `develop` from `fix/difference-negative-amounts`,
  squash-merged (`fix(difference): …`), with this ODD archived in it.
- Order: T1 → T2 → T3 → T4 → T5. T3 must land with T1: the docs tests run the
  examples, so they fail until the examples are corrected.
- Version impact: breaking for a difference model that goes below zero today,
  which now raises; ships in 0.3.0 with the other breaks.

## Evidence

- L1 scan: a `pytest` plugin that wraps `Simulation.run` and `run_at` and
  records every `DifferenceEquations` run whose state has a value below
  `−10⁻⁹`; 994 passed, 3 xfailed, 7 runs found (the table above).
- The fixes of S3 were tried on 2026-10-14: Examples 6–9 with them end with no
  amount below zero, and the onchocerciasis and residence-time tests pass with
  `increments` scaled by `2⁻⁶` at `ENGINE_REL = 1e-9`.

## Log

- L1 (user, 2026-10-14): asked whether the cost of D5 of
  `stochastic-compartmental.md` was acceptable. The scan of all runs found the
  seven negative runs above; the user approved D5 with an error for negative
  rates too, and asked for an ODD that corrects the examples, or for the
  correction in the ODD that adds the check. Both go here: the check makes the
  examples fail, so they must change in the same PR, and a bug that exists
  today should not wait for the Markov ODD.
- L2 (user, 2026-10-14): approved D1 (an error for every amount below zero)
  and D2 (a typed error; calibration gives the run the base penalty).
