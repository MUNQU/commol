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
`odd/tasks/model-kind-foundation.md` is merged.
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

Every spec follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done).

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
| T1 | S1 | `flow_rates` and `flow_changes`; `derivatives` on them | direct | |
| T2 | S4 | Whole initial amounts | direct | |
| T3 | S2 | `commol-stochastic`: direct method, tau-leaping, impulses, outputs at positions; registry entry, bindings | direct | |
| T4 | S3 | Chain binomial; the check of per-step probabilities | direct | |
| T5 | S5 | Description of the event table | direct | |
| T6 | S6 | Calibration and replicates against the oracles; reference models | direct | |
| T7 | S7 | Documentation | direct | |
| T8 | all | Final audit: every runner | direct | |

## Delivery

- Strategy: one PR to `develop` from `feat/stochastic-compartmental`,
  squash-merged (`feat(engine): …`).
- Order: T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8.
- Version impact: added (two engines); not breaking.

## Evidence

(Filled during implementation.)

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
