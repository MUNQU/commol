# Feature: stochastic engines for the aggregate structure

Goal: a process built with `Aggregate` (foundation D10) runs as a stochastic
process with no change to the process: whole units move between compartments
at random,
with the rates of the model as probabilities or hazards. The user gets
variability between runs, extinction and the behaviour of small amounts,
and the trajectories of a Markov chain (`odd/tasks/markov-chains.md`).
Motivation: the user asked which kinds of modelling commol should add (L1).
A stochastic engine reuses the whole process and its `Aggregate` structure,
so it is the cheapest addition, and the closest libraries have it: epydemix
(stochastic simulation) and odin2 + dust2 (deterministic or stochastic)
(`odd/tasks/library-benchmark.md`, Context).
Branch: `feat/stochastic-compartmental` from `develop`, after
`odd/archive/tasks/model-kind-foundation.md` is merged.
Engram mirror: topic `odd/stochastic-compartmental/tasks` (project `commol`).
Runners: the Rust tests, clippy and rustdoc (from the root), `maturin develop
--release --features test-kind`, then `pytest`, `pytest -m docs`, `ruff check
.`, `ty check commol` and `vulture` (from `py-commol/`), `mkdocs build
--strict` (from the root).

## Current state (verified 2026-10-14 on `develop` at `901236d`)

- Two engines run the compartmental kind, both deterministic
  (`EngineType`, `commol-core/src/types/engine.rs`).
- `CompiledModel::derivatives` (`commol-model/src/system.rs:129`) computes
  every planned flow with `flow_amount` and adds it to its sources, targets
  and accumulators. Each flow is one `transition_flows` entry: a source
  compartment set, a target set, a rate and whether it is proportional
  (`plan_transition_flows`, `commol-model/src/flows.rs`).
- A proportional rate moves `rate × source`; an absolute rate is the flow
  itself (`docs/guide/transitions.md`).
- Impulses jump the state before the flows of their step
  (`CompiledModel::apply_impulses`).
- The total is a real number (`InitialConditions.total`), so initial amounts
  need not be whole.
- The foundation ODD adds `set_seed`, replicates, calibration with common
  random numbers and `dynamics_form::StochasticEvents`.

## How the model maps to a stochastic process

Each planned flow is one **event channel**. An event moves one unit: it takes
one from each source compartment, adds one to each target compartment and
one to each accumulator of the transition. The rate of the channel follows
from the rate the model already has:

| Rate kind | Continuous time (hazard) | Discrete time (per step) |
|---|---|---|
| Proportional, `k` | Each unit of the source leaves at hazard `k`: propensity `k × source` | Each unit leaves with probability `k`: `Binomial(source, k)` units move |
| Absolute, `r` | Events at rate `r`: propensity `r` | `Poisson(r)` units move, capped by the source |
| No source (inflow), `r` | Propensity `r` | `Poisson(r)` units enter |

The expected change of the state in continuous time is the right-hand side
of the ODE engine, and the expected change in one step of the discrete
engine is the flow of the difference engine when no cap acts. So both
deterministic engines are the mean-field limits of these engines, which
gives the main test oracle.

## Specs

Every spec follows the [documentation definition of done](../../../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — The rates of every flow, not only their sum

- Fix: `CompiledModel::flow_rates(step, t, state, out)` returns the value of
  each planned flow (the term `derivatives` adds), and
  `CompiledModel::flow_changes()` the change of state of one event of each
  flow (sources −1, targets +1, accumulators +1). `derivatives` calls
  `flow_rates`, so the three engines read one computation ("One right-hand
  side, two engines" becomes "one set of flows, every engine").
- Check: `derivatives` equals `flow_changes`ᵀ · `flow_rates` on the models of
  `crates/commol-model/tests/`.

### S2 — The continuous-time engine

- Fix: `StochasticEvents` and `StochasticSteps` are the names of the methods
  for every structure (foundation S0, `agent-based-models.md` D3); this ODD
  implements them for `Aggregate`, and the network and agent ODDs for their
  structures.
- Fix (D1, D2): a crate `commol-stochastic`, on `commol-model` and the
  foundation's `commol-random` (generator, samplers, sum tree of rates), with
  the engine
  `StochasticEvents` (name in D3): Gillespie's direct method by default,
  with `method = "tau_leaping"` (a fixed `tau` or the adaptive choice of Cao,
  Gillespie and Petzold 2006) as a setting. When a rate reads `t` or `step`,
  the exact method rescales time (D4): between events the state is fixed, so
  it draws `u ~ Exp(1)` and finds the `τ` at which the total rate integrated
  from the last event reaches `u`, within a tolerance of the engine's
  settings, and picks the event from the rates at that position. The engine
  stops at every position where a rate can jump (the breakpoints of the ODE
  engine), so no integral crosses a discontinuity; tau-leaping never leaps
  across them either. Impulses apply at
  their positions as binomial draws (proportional) or whole amounts
  (absolute), before the events of that position. Outputs at the requested
  positions are the state at that time (the last event before it).
- Check: in a pure death process the time to extinction of `n` units has the
  mean `Σ 1/(k i)`; a pure death process with the rate
  `k (1 + sin t)` keeps each unit with probability `exp(−∫ k (1 + sin s) ds)`,
  a closed form that checks time rescaling; a birth-death process with birth `b` and death `d < b`
  goes extinct with probability `(d/b)^n` (Kendall 1948); the mean of many
  runs of a model with linear rates equals the ODE solution within the
  tolerance the number of runs gives; tau-leaping agrees with the direct
  method in distribution (two-sample test) on a model where both apply.

### S3 — The discrete-time engine

- Fix: `StochasticSteps` (name in D3): the chain binomial. At each step,
  for each source compartment, the units that leave by its proportional
  flows are one multinomial draw (each flow with probability `k`, staying
  with `1 − Σk`), so a unit never leaves twice and an amount never goes
  negative. Absolute flows draw a Poisson number capped by what the source
  still holds, in the planned order. Rates are per-step probabilities
  (`RateMeaning::PerStep`); a negative rate, a proportional rate above 1,
  or a sum of proportional rates of one source above 1 is an error that
  names the compartment, the step and the flows (D5). It is the
  `SimulationError::Infeasible` of `difference-negative-amounts.md`, so
  calibration gives such a run its penalty.
- Check: the mean of many runs equals the difference engine; the variance of
  one step of a single outflow equals `n k (1 − k)`; a negative rate and a sum
  of `1.2` each fail at their step and name the flows.

### S4 — Whole units

- Fix (D6): a stochastic engine needs whole amounts, and raises when `total`
  is not a whole number. When every initial amount is whole, nothing more is
  needed. Otherwise the engine raises unless its settings name a rounding,
  with no default, and the error lists the two:
  - `"largest_remainder"`: the same whole amounts in every replicate, which
    sum to the total;
  - `"random"`: each replicate draws its amounts from a multinomial of the
    total with the fractions of the model, so the mean start equals the
    fractional amounts.
  Calibrated initial conditions (`InitialConditionPlan`) go through the
  rounding the user chose.
- Check: a model whose amounts are whole needs no setting; a stratified model
  whose fractions do not give whole amounts raises with no setting and lists
  both options; with `"largest_remainder"` it starts from whole amounts that
  sum to the total; with `"random"` the mean start of many replicates equals
  the fractional amounts within its tolerance; a total of `1000.5` raises and
  names the engine.

### S5 — Description

- Fix: the `dynamics` section of a stochastic model is an event table (form
  `StochasticEvents`): one row per flow with its change of state and its
  propensity (continuous) or its draw (discrete), and the same state graph as
  the deterministic model. The `schedule` section says the method and its
  settings.
- Check: the description of a model run by each engine lists every flow with
  its propensity or its draw.

### S6 — Simulation and calibration

- Fix: nothing new: the foundation gives `seed`, `Simulation.replicates`,
  calibration with common random numbers and a `replicate_summary` the user
  chooses, plots and reports of replicates. This ODD
  checks them with a real stochastic engine.
- Check: a calibration of one rate on data from 200 runs recovers the rate
  with `replicate_summary="mean"` and with `"crps"`; with a fixed seed it is
  reproducible.

### S7 — Documentation

- Fix: a guide page "Stochastic simulation" (`docs/guide/stochastic.md`):
  when to use it, the table of how rates map to events, whole units, seeds
  and replicates, the cost (events grow with the total; tau-leaping), and
  calibration with replicates. `design-decisions.md`: "Every engine reads one
  set of flows", "Whole units in a stochastic engine". `repository-map.md`,
  `AGENTS.md` (the new crate), changelog.
- Check: `pytest -m docs`, `mkdocs build --strict`.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S2, S3 | Which engines? | Two entries: a continuous-time engine (direct method, tau-leaping as a setting) and a discrete-time chain binomial. | Continuous only: users of the difference engine, whose rates are per-step probabilities, would have no stochastic version of their model. Discrete only: misses the exact process that the ODE engine is the limit of. One engine per method (direct, tau-leaping, next reaction): three entries for one process; the method is a setting, as `tsit45` and `bdf` are. | approved (L2) |
| D2 | S2 | Where does the engine live? | The stochastic engines of the aggregate kind in a new crate, `commol-stochastic`, beside `commol-difference` and `commol-differential`, on `commol-model`. What any stochastic engine of any kind shares (generator, seeds, samplers, the sum tree that picks the next event) is in `commol-random`, which the foundation makes. Another algorithm for the same process is a setting of an engine here; another stochastic process of the aggregate kind (a stochastic differential equation, a hybrid engine) is a new entry, here when it shares code, else in its own crate; the engines of other kinds go in their kind's crate. | The samplers inside `commol-stochastic` (first recommended): the network and agent engines would depend on a compartmental engine crate to draw numbers (L3). Inside `commol-difference`: mixes two engines in one crate, against the layout of the others. Inside `commol-model`: the compiler would carry an engine. | approved (L3) |
| D3 | S2, S3 | Names | `StochasticEvents` (continuous) and `StochasticSteps` (discrete). | `Gillespie`: names one algorithm, and tau-leaping is not it. `CTMC` and `DTMC`: exact for linear rates only; with rates that read the state the process is still Markov but the names suggest the chains of `markov-chains.md`. `StochasticDifferentialEquations`: wrong, an SDE is a diffusion. The guide names the terms of the literature (Gillespie's SSA, chain binomial) so they can be searched. | approved (L4) |
| D4 | S2 | A rate that reads `t` between events | Rescale time in the exact method (Anderson 2007, modified next reaction method): find where the integrated total rate reaches an exponential draw, with a tolerance in the engine's settings; stop at every breakpoint of the ODE engine. A model whose rates do not read `t` keeps the plain direct method at no cost. Tau-leaping does not leap across a breakpoint; the discrete engine evaluates rates once per step, as the difference engine does. | Freeze the rates at the last event (first recommended): an approximation, hidden from the user, for every rate that changes continuously, such as a sinusoid (L5). Thinning (Lewis and Shedler 1979): exact, but needs an upper bound of every rate, which a formula does not give. Ignore `t`: wrong for every rate that changes along the axis. Cost: an integral and a root per event when a rate reads `t`. | approved (L5) |
| D5 | S3 | Per-step probabilities above 1, or below 0 | An error that names the compartment, the step and the flows. | Clamp to `[0, 1]`: hides a model error and changes the rates without notice. Renormalize: changes the model. Cost: none for existing models, since the engine is new; a binomial draw has no meaning outside `[0, 1]`. | approved (L6) |
| D6 | S4 | Amounts that are not whole | Raise for a total that is not whole. For whole amounts, nothing. Otherwise the user names the rounding in the engine's settings, with no default: `"largest_remainder"` or `"random"` (a multinomial draw per replicate). | Largest remainder by default (first recommended): with small amounts the rounding is a modelling choice, not a numerical detail; 0.5 units of a seed in each of ten groups starts five groups with one unit and five with none, which decides which groups can start a process at all, and the library would choose it without notice, against the rule of foundation D8 (modelling choices belong to the user, with no default). Raise for any amount that is not whole: most stratified models would fail with no way out but to compute whole amounts by hand. Random rounding by default: the same objection, the other way. Cost: a stratified model whose amounts are not whole needs one setting. | approved (L7) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1 | `flow_rates` and `flow_changes`; `derivatives` on them | direct | `924753e` |
| T2 | S4 | Whole initial amounts | direct | `987507f` |
| T3 | S2 | `commol-stochastic`: direct method, tau-leaping, impulses, outputs at positions; registry entry, bindings | direct | `987507f` |
| T4 | S3 | Chain binomial; the check of per-step probabilities | direct | `987507f` |
| T5 | S5 | Description of the event table | direct | `0f363bd` |
| T6 | S6 | Calibration and replicates against the oracles; reference models | direct | `0f363bd` |
| T7 | S7 | Documentation | direct | `0707a34` |
| T8 | all | Final audit: every runner | direct | the archive commit |

## Delivery

- Strategy: one PR to `develop` from `feat/stochastic-compartmental`,
  squash-merged (`feat(engine): …`).
- Order: T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8.
- Version impact: added (two engines); not breaking.

## Evidence

Verified 2026-10-16 on `feat/stochastic-compartmental`, every runner of
`AGENTS.md`: Rust 332 tests (336 with `cargo test --all-features
--workspace`), `cargo fmt --check`, `clippy -D warnings` (with and without
`--all-features`), `rustdoc -D warnings` with `commol-stochastic`; Python 1157
tests (1 skipped: `pydot` is not installed), 74 docs tests, `ruff`, `ty`,
`vulture`; `mkdocs build --strict`. The 17 oracle tests of
`crates/commol-stochastic/tests/oracles.rs` run in about 5 s in a debug build.

| Spec | Evidence |
|---|---|
| S1 | `CompiledModel::{flow_rates, transition_rates, flow_changes, impulse_rates, impulse_changes, breakpoints, reads_time}` and `FlowChange` (`crates/commol-model/src/{system,types}.rs`); the differential engine reads `breakpoints` (`solve.rs`). Tests: `crates/commol-model/tests/flows.rs` (`derivatives_are_the_changes_of_one_event_times_each_flow` on four models with proportional, absolute, multi-piece, inflow, outflow, category and accumulator flows, and a rate of `t`); every deterministic test unchanged. |
| S2 | `StochasticEvents` (`crates/commol-stochastic/src/events.rs`); `EngineType::StochasticEvents`, `StochasticEventsSettings`, `EventMethod` (`crates/commol-core/src/types/engine.rs`); `Engine` (`py-commol/src/engine.rs`). Tests (`oracles.rs`): `a_pure_death_goes_extinct_after_the_sum_of_its_mean_waits`, `a_rate_that_reads_t_keeps_each_unit_with_its_integrated_hazard` (five units at `k (1 + sin t)`), `a_rate_that_switches_on_at_a_threshold_waits_for_it`, `a_birth_death_process_goes_extinct_with_the_probability_of_kendall`, `the_mean_of_many_runs_is_the_solution_of_the_ode`, `tau_leaping_agrees_with_the_direct_method_in_distribution` (two-sample Kolmogorov–Smirnov at 0.001, adaptive and `tau = 0.01`), `a_run_depends_on_its_seed_and_not_on_where_it_reports`, `impulses_move_whole_units_at_their_positions`, `an_impulse_that_is_not_a_probability_or_a_whole_number_fails`. |
| S3 | `StochasticSteps` (`crates/commol-stochastic/src/steps.rs`). Tests: `the_mean_of_many_chain_binomial_runs_is_the_difference_engine`, `one_step_of_an_outflow_has_the_binomial_variance`, `a_unit_never_leaves_twice_and_an_absolute_flow_is_capped`, `a_negative_rate_or_probabilities_above_one_fail_at_their_step`; `tests/stochastic/test_engines.py::TestProbabilities`. |
| S4 | `check_whole_units` and `as_whole` (`crates/commol-model/src/builder.rs`), `Rounding` (`engine.rs`), `Units` (`crates/commol-stochastic/src/units.rs`). Tests: `oracles.rs` (`amounts_that_are_not_whole_need_a_rounding`, `whole_amounts_need_no_rounding_and_a_total_must_be_whole`, `the_largest_remainder_starts_every_run_from_the_same_whole_amounts`, `a_random_rounding_starts_on_average_from_the_amounts_of_the_model`, `calibrated_initial_conditions_go_through_the_rounding`), `units.rs` tests, `tests/stochastic/test_whole_units.py`. |
| S5 | `DynamicsDescription.events` (`context/_model/equations.py`), `schedule_blocks` (`context/_model/describe.py`), `EngineType::schedule`, `FlowDescription::flow`. Tests: `tests/description/test_events.py`. |
| S6 | `tests/stochastic/test_calibration.py` (one rate from the mean of 200 runs, `"mean"` and `"crps"`, both engines; one seed gives one result), `tests/stochastic/test_engines.py` (means against the deterministic engines, plots and reports of replicates). |
| S7 | `docs/guide/stochastic.md`; `design-decisions.md` ("Every engine reads one set of flows", "Whole units in a stochastic engine"); `repository-map.md`, `mental-model.md`, `conventions.md`, `AGENTS.md`, `core-concepts.md`, `building-models.md`, `describing-models.md`, `simulations.md` and `calibration.md` (their skipped stochastic examples now run), `model-structure.md`, `docs/api/simulation.md`, changelog. |

## Log

- L1 (user, 2026-10-14): asked which kinds of modelling commol should add.
  The recommended order was a stochastic compartmental engine, then
  networks, then agents; the user added Markov chains, and asked for one ODD
  per kind on top of `model-kind-foundation.md`.
- L2 (user, 2026-10-14): approved D1 (a continuous-time and a discrete-time
  engine; the algorithm is a setting).
- L3 (user, 2026-10-14): asked whether one crate fits when other, different
  stochastic engines come later. It did not: the samplers would sit in a
  compartmental engine crate. Approved D2 with `commol-random` (made by the
  foundation) for what every stochastic engine shares, and
  `commol-stochastic` for the compartmental ones.
- L4 (user, 2026-10-14): approved D3 (`StochasticEvents`, `StochasticSteps`).
- L5 (user, 2026-10-14): noted that the problem of D4 belongs to the exact
  event-by-event methods, not to tau-leaping or the discrete engine. Approved
  D4 with time rescaling (Anderson 2007) instead of frozen rates.
- L6 (user, 2026-10-14): asked whether the cost of D5 was acceptable. It
  was: the engine is new and a draw needs a probability. A scan of every
  difference run of the suite found seven that go negative today, none from
  probabilities above 1; that fix went to `difference-negative-amounts.md`.
  Approved D5, with a negative rate as an error too.
- L7 (user, 2026-10-14): approved D6 as revised: the rounding of amounts that
  are not whole is a modelling choice, so the user names it, with no default
  (largest remainder was the first recommendation).
- L8 (implementation, 2026-10-16): `derivatives` adds `changes × flow` piece by
  piece, in the order of the old right-hand side, so every deterministic run
  is the same to the bit. `transition_rates` gives the rate alone, which the
  chain binomial needs as a probability. The breakpoints of the ODE engine
  moved to `CompiledModel::breakpoints`, so both continuous-time engines stop
  at the same positions.
- L9 (implementation): a run of `StochasticEvents` does not depend on the
  positions it reports at: each wait is measured from the last change of the
  state (an event, a leap or a breakpoint), never from a reported position,
  the flow of an event is drawn only when it happens, and a leap past the end
  of a run is kept for the next. When a rate reads `t`, the integral grows in
  pieces from the mean wait at the last change, which do not depend on the end.
- L10 (implementation): in continuous time a flow has no events while a
  source holds fewer units than an event takes; without it an absolute rate
  that does not vanish with its source would take units it does not have. An
  impulse moves at most what its sources still hold, and an absolute impulse
  that is not a whole number of units, or a proportional one that is not a
  probability, is `SimulationError::Infeasible`; a negative propensity too.
- L11 (implementation): an adaptive leap bounds the change of every
  compartment, with the order 2 of Cao, Gillespie and Petzold (2006), since a
  formula has no known order; a leap of fewer than ten events is one exact
  event, and a leap that leaves an amount below zero is halved up to 30 times.
  A fixed `tau` of 0.05 failed the two-sample test against the direct method
  on the chain model: its bias is of first order; the test uses 0.01.
- L12 (implementation): the whole-unit check is in `check_model`, so `build`
  raises, not only the engine; an amount is whole within a relative `1e-9`,
  the rounding of a fraction times the total. The engine makes the start whole
  at the first run after a reset, not when it is built, so the values
  calibration writes go through the rounding and `"random"` draws after them.
- L13 (implementation): the engine recomputes every propensity after each
  event, because a rate may read any compartment through a total, so it picks
  the event by a linear scan; the `SumTree` of `commol-random` pays only with
  a dependency graph of the rates, out of scope here.
- L14 (implementation): the settings of `ModelEngine` are read with the class
  of its engine (`settings_of_the_engine`), since `{"rounding": "random"}`
  fits both stochastic classes and a model must load back as the class it was
  saved with. `transition_flows` gives each piece the index of its flow, so
  the event table groups the pieces of one event; its `Change` column lists
  the compartments, not the accumulators, which Python cannot expand per flow.
- L15 (implementation): delivered by PR #118, from `feat/stochastic-compartmental`.
