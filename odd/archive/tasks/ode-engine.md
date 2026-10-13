# Feature: An ODE engine for differential-equation models

Goal: run a model whose typology is `DifferentialEquations` as a system of
ordinary differential equations, `dX/dt = Σ flows(t, X)`, with the same model
definition, the same flows and the same Python API (`Simulation`, `Calibrator`,
reports, plots) as a difference-equations model.
Motivation: continuous-time models (class C in
`odd/tasks/reference-model-tests.md`) can only be approximated today.
Branch: `feat/ode-engine` from `develop`, after `test-suite-refactor.md`
merges and before `reference-model-tests.md` (D1 of that ODD).
Engram mirror: topic `odd/ode-engine/tasks` (project `commol`).
Runners: `maturin develop --release` then `pytest` and `pytest -m docs` (from
`py-commol/`), `cargo test -p commol-core -p commol-difference -p commol-calibration`
plus the new crates,
`cargo clippy --workspace --all-targets -- -D warnings`,
`RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`, `mkdocs build --strict`
(from the root).

## Current state (verified 2026-10-13 on `fix/code-findings` at a887f81)

- `ModelTypes::DifferentialEquations` exists in Rust
  (`crates/commol-core/src/types/dynamics.rs:13`) and Python
  (`commol/constants.py:14`, `Dynamics.typology`). `ModelBuilder` accepts it
  (`_validate_typology`), and `print_equations` writes `dX/dt = …`
  (`context/_model/equations.py:279`).
- No engine runs it: `crates/commol-difference/src/builder.rs:57` rejects every
  typology but `DifferenceEquations`; `docs/codebase/mental-model.md` (L24)
  says so.
- `DifferenceEquations::step` (`simulation.rs` L79–223) first computes every
  flow from one snapshot of the state into `compartment_flows`, then adds them
  to the population. That is one explicit Euler step with `h = 1`: the loop up
  to L212 is already the right-hand side of the ODE.
- `SimulationEngine` (`commol-core/src/simulation_engine.rs`) already allows
  continuous time (`current_step` "represents the current time value"), and
  `CalibrationProblem<E: SimulationEngine>` is generic.
- The bindings are not: `py-commol/src/calibration.rs` takes
  `&PyDifferenceEquations` in 8 functions, and Python calls
  `commol_rs.difference.*` for expansion, flows and names even when it only
  needs the model (`context/_model/engine_state.py` L100, L122, L193, L315,
  L346).
- Time-series parameters are evaluated at `current_step as u64`
  (`simulation.rs` L135), and time patterns are formulas of `step`: both are
  piecewise constant between integer steps.

## Specs

Every spec follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — One right-hand side for both engines

- Fix: the flow computation of `step` becomes one function,
  `derivatives(t, state, out)`, that both engines call. The difference engine
  is `state += derivatives(n, state)`; the ODE engine integrates it.
- Check: a Rust test per model in `crates/commol-difference/src/simulation.rs`
  tests: one difference step equals `state + derivatives(0, state)` exactly;
  the existing Rust and Python suites pass unchanged.

### S2 — Model compilation shared by both engines (D2)

- Fix: expansion, layout, flow plan, rate evaluation (JIT, fast path,
  evalexpr fallback) and the right-hand side move out of
  `commol-difference` into a crate that both engines use. The Python helpers
  that need only the model (`initial_state`, `expanded_outputs`,
  `transition_flows`, `check_names`) move from `commol_rs.difference` to that
  crate's module.
- Check: `commol-difference` holds only the stepper; `engine_state.py` no
  longer imports `difference` for expansion; no public Python name changes.

### S3 — The solver (D1)

- Fix: `diffsol` (pinned minor version, default features) solves the system;
  commol keeps no numerical method of its own. The right-hand side of S1 is the
  `rhs` closure. Methods: `tsit45` (explicit, the default) and `bdf` (stiff).
  The engine integrates in pieces and restarts the solver at the start of each
  one (D12): a piece ends at every impulse step and at the last output, and
  also at every integer when a rate or a formula parameter reads `step` or a
  time series (`CompiledModel::depends_on_step`). So every discontinuity of the
  right-hand side is the start of a piece. Outputs inside a piece come from
  the dense output of `diffsol`.
- Facts (probe on diffsol 0.17.1, 2026-10-13): it builds with `maturin`'s
  toolchain and needs no system library; `tsit45` and `bdf` agree on an SIR
  run to 8 digits (final-size residual 6.7e-8). `rhs_implicit` also takes a
  Jacobian-vector product, which `bdf` needs (D6). The `nalgebra` and `faer`
  features both compile `nalgebra` and `faer` in 0.17.1 (`diffsol-la`
  enables `faer/sparse-linalg` under either), so a feature choice saves
  nothing; the choice is which matrix type the engine uses (D7).
- Matrix type (D7): `bdf` uses dense `NalgebraMat`/`NalgebraLU` below a
  compartment-count threshold and sparse `FaerSparseMat`/`FaerSparseLU` above
  it. The threshold comes from a benchmark (T7), not a guess. Open risk: how
  `diffsol` gets the Jacobian sparsity pattern from closures with a
  finite-difference `J·v` (D6) is not verified; T3 starts with a probe of it.
  The pattern is known in advance: compartment `j` enters the derivative of
  `i` only if a flow out of or into `i` has a rate that reads `j`, or `j` is
  its source (`flows.rs` plan).
- Check: Rust tests against closed forms, error below the tolerance:
  exponential decay, logistic growth, a linear 2×2 system, the SIR final-size
  relation, an SIS endemic equilibrium. A run with a stiffness ratio of 1e5
  over 100 time units finishes and meets the tolerance. Cost recorded in
  Evidence next to the difference engine.

### S4 — Time inside a step (D3)

- Fix: on `[n, n + 1)`, `step` is `n` and `t` is the continuous time. A rate
  that depends on `step` or on a time-series parameter is constant on the
  interval.
- Check: a pulse pattern on step `n` gives the flow of a constant rate on
  `[n, n + 1)` (closed form `1 − e^{−p}` for a per-capita pulse `p`), not the
  difference-engine flow `p`; `design-decisions.md` states this difference.

### S5 — Accumulators are integrals

- Fix: each accumulator is an extra state component, `dA/dt = Σ counted flows`,
  so a windowed count is exact to the solver tolerance.
- Check: an accumulator of an exponential decay equals the closed-form
  integral.

### S6 — Python API

- Fix: `Simulation` builds the engine that the typology names; the protocol
  `DifferenceEquationsProtocol` becomes `EngineProtocol`. Solver settings
  live where D4 says. `Calibrator`, `ProbabilisticCalibrator`, reports and
  plots accept either engine; the bindings in `py-commol/src/calibration.rs`
  take either engine (one enum, no duplicated functions).
- Check: every public workflow test (simulate, calibrate, probabilistic
  calibration, report, plot) runs with an ODE model; calibration recovers known
  parameters; `_commol_rs.pyi` matches the bindings.

### S7 — Units and output times (D5, D10)

- Integer time is assumed today in `Simulation.run(num_steps: int)`,
  `ObservedDataPoint.step: int` (`context/calibration.py:52`), `windows.py`
  (`window_steps: int`) and `SimulationEngine::run_at_steps_into_buffer(&[u32])`.
- Fix (D5): the ODE engine reports at any list of times, from the dense output
  of `diffsol`. `Simulation.run(num_steps)` keeps its meaning (times 0…n);
  a new `Simulation.run_at(steps)` returns the outputs at the given sorted
  positions on the axis (the independent variable, whatever it is). The engine trait gains `run_at_positions_into_buffer(&[f64])`; the
  difference engine accepts only integer steps there and names the first
  non-integer one in its error. Observations, windows, reports and plots
  take non-integer steps for an ODE model (D10).
- The one-step-unit rule (`design-decisions.md`, "One step unit in the unit
  check") applies unchanged: its inferred unit is the unit of `t`.
  `TimeScale` converts non-integer steps as it converts integer ones.
- Check: `run_at(steps=[0, 0.5, 2.25])` on exponential decay matches the closed
  form; a calibration with observations at non-integer steps recovers known
  parameters; a window over `[0.5, 3.5]` matches the closed-form integral;
  the difference engine rejects `run_at(steps=[0.5])` with a clear error; a unit
  test for an ODE model in `1/day`.

### S9 — Instant jumps (impulses) (D3, D8, D9)

- Today a per-capita `TimePattern.pulse` of amount `p` on step `n` moves
  `p · X` in one difference step: in the difference engine a pulse already is
  an instant jump. Numeric pulses are stored as time-series parameters
  (`docs/guide/time-patterns.md`, "How patterns are stored"), other patterns
  only as formula strings, so the engine cannot tell a pulse from a rate.
- Fix (D8): a transition with `impulse_at=[t1, t2, …]` or
  `impulse_every=(period, offset)` is an impulse: at those times its `rate`
  moves a fraction (per-capita) or an amount (absolute) from source to target,
  in both engines. Source, target, stratified rates and accumulators work as
  for any transition. Impulse steps may be non-integer for an ODE model (D5); a
  difference model requires integers, as for observations (D10). The ODE
  engine integrates between breakpoints (integer times and impulse times),
  applies the jump at each impulse time and restarts the integration, so no
  `diffsol` event is needed. The order of the jump and the flows of the same
  step is D9.
- Check: an impulse of fraction `p` on a model with no other flow gives
  `X(n) = (1 − p) X(n⁻)` in both engines; pulse vaccination
  (`S' = μK − μS` between jumps, `S → (1 − p) S` at `t = kT`) reaches the
  closed-form periodic orbit `S*`; the impulse is serialized and loaded back;
  an accumulator counts the jumped amount.

### S10 — The unit check accepts any axis (D11)

- Cause: `context/_model/units.py` (`TIME_DIMENSIONS = ("[time]", "[step]")`,
  `UnitChecker._step_unit`) takes the step unit only from parameter units of
  dimension `[time]` or `[step]`. A model whose axis is a length, with rates in
  `1/meter`, gets the default `step` and every flow `<bin>/meter` fails
  against the expected `<bin>/step`. The engines do not care which dimension
  the axis has (D10).
- Fix (D11): the axis unit may have any single dimension (time, length, …).
  The rule of "One step unit in the unit check" is unchanged: every flow must
  be exactly `<bin unit>/<step unit>`, and the engine converts nothing.
  `TimeScale` stays for a time axis only; using it with a non-time step unit
  is an error that names the unit.
- Check: a model in `1/meter` passes; `1/meter` mixed with `1/day` fails and
  names both; a model in `1/day` behaves as today (`TestStepUnit` unchanged);
  `TimeScale` with a `meter` step unit raises.

### S8 — Documentation

- `docs/guide/simulations.md`: a section on differential equations (typology,
  solver settings, the time rule of S4) with a runnable example.
- `docs/guide/units-and-equations.md` and `design-decisions.md` ("Physical time
  is not in the model", "One step unit in the unit check"): the axis may be
  any independent variable, not only time (S10).
- `docs/guide/transitions.md` (or a new page): impulses, in both engines, and
  how they differ from a pulse rate.
- `docs/codebase/mental-model.md`: the "An ODE solver" row of "What Commol is
  not" goes; the layer diagram shows both engines.
- `docs/codebase/repository-map.md`, `design-decisions.md` (time inside a
  step, solver choice), `AGENTS.md` ("If you need to…" rows), API reference,
  rustdoc of every new public item, changelog "Added".

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S3 | Which solver? | (first draft) Our own Dormand–Prince 5(4). | The user chose `diffsol`: no numerical method to maintain, stiff methods (`bdf`, `sdirk`) and events included. Cost: a 0.x dependency whose API changes often (0.12 in April 2026, 0.17.1 in October 2026); pin the minor version and upgrade on purpose. `russell_ode` needs non-Rust libraries (hard for wheels); `ode_solvers` has explicit methods only. | approved (L2): `diffsol`, no numerical method of our own |
| D2 | S2 | Where does the code go? | A new crate for model compilation and the right-hand side; `commol-difference` keeps the stepper; a new `commol-differential` holds the ODE stepper. | `commol-differential` depends on `commol-difference`: smaller change, but one engine depends on another and Python keeps asking `difference` to expand an ODE model. Solver inside `commol-difference`: the smallest change, but the crate name becomes false and the crate grows. | approved (L3): `commol-model` (expansion, flows, rate evaluation, `derivatives`); `commol-difference` steps; `commol-differential` wraps `diffsol` |
| D3 | S4 | What do `step` and pulses mean inside a step? | `step` = floor of `t`; a pulse is a constant rate on its unit interval. Instant jumps of state (impulses) are a later feature with their own ODD. | Instant jumps now: exact pulse vaccination, but a new model concept (events) for both engines and its serialization. `step` = `t` (continuous): a pulse `step == n` is true on a set of zero length, so it never acts. | approved (L4): instant jumps in this PR (S9); rates still see `step` = floor of `t` |
| D4 | S6 | Where do solver settings live? | In the model: `Dynamics.solver` (tolerances, maximum step), only for `DifferentialEquations`, with defaults. | Argument of `Simulation`: a saved model file would not reproduce a run ("Model files are written in full" in `design-decisions.md`), and `Calibrator` and reports would need the settings passed again. Fixed in the engine: users cannot trade speed for accuracy. | approved (L5): `Dynamics.solver` (`OdeSolver`: method, `rtol`, `atol`), set through `ModelBuilder(..., solver=...)` |
| D5 | S7 | At which times does the engine report? | (first draft) Integer times only. | The user chose any list of times (dense output of `diffsol`), so observations, windows, reports and plots accept real times for an ODE model. | approved (L6): any list of times |
| D6 | S3 | How does `bdf` get the Jacobian-vector product `J·v`? | Finite differences: `J·v ≈ (f(x + εv) − f(x)) / ε`, one extra right-hand-side call. | Symbolic derivative of each rate expression: exact, but a differentiator over the expression tree is new numerical code to maintain, which D1 wants to avoid. Explicit methods only (no `bdf`): stiff models become slow or fail. | approved (L7) |
| D7 | S3 | Which matrix type does the engine use? | Dense `nalgebra` (`NalgebraMat`, `NalgebraLU`); sparse `faer` only if a measured stiff model with many compartments needs it. | Sparse `faer` from the start: wins for large stiff systems, but needs the Jacobian sparsity pattern, which the closure interface must be shown to give (not verified). Both behind a generic engine: two code paths to test now for a need not yet measured. Explicit `tsit45` uses no matrix solve, so the choice affects only `bdf`. | approved (L8): by size — dense `nalgebra` below a measured threshold, sparse `faer` above it |
| D8 | S9 | What is the API of an impulse? | A transition option: `add_transition(..., impulse_at=[...] or impulse_every=(period, offset), rate=<fraction or amount>)`, serialized in `Transition`. It reuses source, target, stratified rates, accumulators and per-capita/absolute rules. | A separate `add_impulse` and `Impulse` type: clearer name, but it duplicates source/target/stratification/accumulator handling. Read pulse formulas as impulses: no new API, but only numeric pulses are detectable, and a pulse inside `combine` or `from_formula` would silently behave differently in the two engines. | approved (L9): `impulse_at` / `impulse_every=(period, offset)` on `add_transition` |
| D9 | S9 | In step `n`, does the jump come before or after the flows? | Before: at time `n` the state jumps, then the step `[n, n + 1)` runs from the new state, in both engines. | After: matches "the flows of step n see the old state", but then the ODE output at `n` would show the state before the jump while the difference output shows it after, so the two engines disagree at the output times. | approved (L10) |
| D10 | S7 | How does an observation give a non-integer position on the axis? | `ObservedDataPoint.step` stays and accepts a float for an ODE model; a difference model requires an integer, and the error names the observation. `run_at(steps=[…])`. | The axis is the independent variable, not only time: `design-decisions.md` ("Physical time is not in the model") measures it in steps, so `step` is already neutral. `time`: false when the axis is not time. `at`: neutral too, but a rename with a deprecated alias for no gain. | approved (L12): `step: float`, no rename (replaces L11) |
| D11 | S10 | How does the check know the axis unit? | Declared: `ModelBuilder(..., step_unit="meter")`, stored in the model, any single dimension; when absent, the current inference (`[time]` or `step`) applies, so existing models do not change. | Infer it from any dimension: ambiguous, because a parameter in `1/(person·meter)` may be a density, not a rate along the axis, and the check would guess. Always declare it: clear, but breaks every model with units today. | approved (L14): optional `step_unit` on `ModelBuilder`, inferred when absent |
| D12 | S3 | Where does the engine restart the solver? | At impulse steps and the last output always, and at every integer only when the right-hand side reads `step` or a time series (`CompiledModel::depends_on_step`). | At every integer always (the first draft of S3): simpler to state, but a model whose rates never jump at integers pays a restart per unit, and `bdf` falls back to order 1 each time; results are the same within the tolerance. | approved (L18) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1 | Extract `derivatives` from `step`; equality tests | direct | #101 (`3c13102`) |
| T2 | S2 | New compilation crate (D2); move expansion, flows and Python helpers; suites unchanged | direct | #101 (`3c13102`) |
| T3 | S3, S5 | Probe of the sparse path (pattern with closures, D7); then `commol-differential` on `diffsol` (`tsit45`, `bdf` with finite-difference `J·v`), unit intervals, accumulators as state; closed-form tests | direct | #101 (`3c13102`) |
| T4 | S4, S7 | Time rule, units, `run_at` and non-integer-step observations, windows, reports and plots (D5, D10); tests | direct | #101 (`3c13102`) |
| T4b | S9 | Impulses (D8, D9): model type, builder, serialization, both engines; tests incl. pulse-vaccination orbit | direct | #101 (`3c13102`) |
| T4c | S10 | Axis unit of any dimension (D11) in `units.py`, builder, model, `TimeScale` guard; tests | direct | #101 (`3c13102`) |
| T5 | S6 | Bindings (engine enum), `Simulation`, protocols, solver settings (D4), `.pyi`; workflow tests | direct | #101 (`3c13102`) |
| T6 | S8 | Documentation and changelog | direct | #101 (`3c13102`) |
| T7 | all | Final audit: every runner, `pytest -m docs`, cost table in Evidence, dense–sparse threshold benchmark (D7) | direct | #101 (`3c13102`) |

## Delivery

- Strategy: one PR to `develop` from `feat/ode-engine`, squash-merged
  (`feat(engine): …` commits; T1–T2 may use `refactor(engine): …`).
- Order: T1 → T2 → T3 → T4 → T4b → T4c → T5 → T6 → T7. T1 and T2 change no behavior and
  can be checked against the unchanged suites.
- Version impact: new features (ODE engine, impulses, `run_at`), changelog
  "Added"; `ObservedDataPoint.step` accepts a float for an ODE model,
  changelog "Changed" (D10).
- Rebuild the extension after every Rust change, before `pytest`.

## Evidence

Verified 2026-10-13 on `feat/ode-engine` before the merge, Apple M-series,
`--release`.

- Runners: `cargo fmt --check`, `cargo clippy --workspace --all-targets
  --all-features -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc` (5
  crates), `cargo test` of the 5 crates (202 passed, 2 benchmarks ignored),
  `pytest` (803 passed), `pytest -m docs` (56 passed), `ruff check`, `ruff
  format --check`, `ty check commol`, `mkdocs build --strict`: all clean.
- Probe (T3, D7): `FaerSparseMat` with closure `J·v` by finite differences
  finds the exact pattern by its NaN probe (bidiagonal chain of 50 states: 99
  non-zeros). The NaN arrives in `v`, so the finite-difference step must not
  depend on `‖v‖` when it is NaN (`RightHandSide::jac_mul`).
- S1: `one_step_adds_the_derivatives_exactly` (4 models) passes with `assert_eq`.
- S3: closed forms (decay with both methods, logistic, linear 2×2, SIR final
  size residual < 1e-7, SIS equilibrium) within tolerance; stiff ratio 1e5
  over 100 time units with `bdf`: 1.2 ms, error < 1e-6 relative.
- Piece rule (D12, approved in L18): the engine ends a piece of integration
  at every integer only when a rate reads `step` or a time series
  (`CompiledModel::depends_on_step`), and always at impulse steps and at the
  last output. Results are the same within the tolerance; a model without
  `step` restarts `bdf` once, not once per unit.
- D7 benchmark (`benchmark_dense_against_sparse_bdf`, 50 time units, best of 3):

| model | states | dense bdf (ms) | sparse bdf (ms) |
|---|---|---|---|
| stiff chain | 16 | 0.31 | 0.32 |
| stiff chain | 32 | 0.64 | 0.50 |
| stiff chain | 64 | 1.94 | 1.07 |
| stiff chain | 128 | 9.78 | 2.01 |
| stiff chain | 256 | 81.92 | 4.79 |
| stiff chain | 512 | 652.67 | 12.81 |
| coupled groups | 24 | 0.30 | 0.79 |
| coupled groups | 48 | 0.75 | 2.94 |
| coupled groups | 96 | 2.79 | 17.58 |
| coupled groups | 192 | 20.00 | 60.08 |
| coupled groups | 384 | 142.86 | 423.82 |

  Size alone does not decide: sparse wins on a sparse pattern (17× at 256) and
  loses about 3× from 192 states when every row reads a total. `SPARSE_THRESHOLD
  = 128` bounds the loss of either choice to about 5×. A rule on the density of
  the pattern would remove both losses (open question for the user).
- Cost next to the difference engine (100 steps, best of 5):

| model | states | difference (ms) | tsit45 (ms) | bdf (ms) |
|---|---|---|---|---|
| coupled groups, 100 steps | 3 | 0.025 | 0.079 | 0.103 |
| coupled groups, 100 steps | 30 | 0.083 | 0.185 | 0.556 |
| coupled groups, 100 steps | 300 | 0.700 | 1.387 | 224.721 |

  `bdf` on a non-stiff model of 300 states pays a dense Jacobian of 300
  right-hand-side calls per factorization; `tsit45` (the default) costs about
  twice the difference engine.

## Log

- L1 (user, 2026-10-13): during D1 (then D3) of `reference-model-tests.md`, chose to
  implement the ODE in commol, in its own ODD and PR between the test refactor
  and the reference tests.
- L2 (user, 2026-10-13): D1 → `diffsol` directly, "no quiero tener que preocuparme de mantener métodos numéricos" (after a comparison of `diffsol`, `russell_ode` and `ode_solvers`; no Rust crate exists for difference equations). Then asked why not `faer` too: a probe showed both features compile anyway in 0.17.1.
- L3 (user, 2026-10-13): D2 → "Crate común + dos motores" (`commol-model`, `commol-difference`, `commol-differential`).
- L4 (user, 2026-10-13): D3 → "Saltos instantáneos ya" (S9, D8, D9 added).
- L5 (user, 2026-10-13): D4 → "En el modelo".
- L6 (user, 2026-10-13): D5 → "Cualquier lista de tiempos" (S7 rewritten; D10 added).
- L7 (user, 2026-10-13): D6 → "Diferencias finitas".
- L8 (user, 2026-10-13): D7 → "Elegir según el tamaño".
- L9 (user, 2026-10-13): D8 → "Opción de add_transition".
- L10 (user, 2026-10-13): D9 → "Antes de los flujos". Note: a pulse rate today is neither before nor after: the difference engine computes it with the other flows from the same old state (`simulation.rs` L163–212).
- L11 (user, 2026-10-13): D10 → "Renombrar a time: float" with a deprecated `step` alias for one minor version.
- L12 (user, 2026-10-13): D10 revised — the axis of a compartment model may be space, not time, so `time` is incoherent. → "step: float"; `run_at(steps=…)`. L11 is replaced.
- L13 (user, 2026-10-13): add the unit check for a non-time axis to this ODD (S10, D11, T4c). Found while revising D10: `TIME_DIMENSIONS` only accepts `[time]` and `[step]`.
- L14 (user, 2026-10-13): D11 → "Declarada, inferida si falta".
- L15 (agent, 2026-10-13): implementation of T1–T7 on `feat/ode-engine`, not
  committed. Two points for the user: the piece rule of S3 (Evidence,
  "Deviation from S3") and the D7 finding that size alone does not choose the
  matrix well (Evidence, D7 benchmark). Plots raise for a non-whole
  `window_steps` (the plotted series holds integer steps); ensemble statistics
  read non-integer observation steps from the engine, not by interpolation.
- L16 (user, 2026-10-13): the axis is not "time or space": space was only an
  example of an axis that is not time. The axis is whatever independent
  variable the model differentiates over. Docs and docstrings say so.
- L17 (user, 2026-10-13): `TimeScale` rate methods are discrete-time; this PR
  only warns (docstrings, `core-concepts.md`, `simulations.md`). The refactor
  is `odd/tasks/axis-scale-refactor.md`.
- L18 (user, 2026-10-13): accepted the piece rule of the implementation (D12):
  restart at every integer only when a rate reads `step` or a time series.
  S3 rewritten. D7 (size threshold 128) stays as approved; a density-based
  matrix rule is not part of this PR.
- L19 (user, 2026-10-13): PR #101 squash-merged into `develop` as `3c13102`;
  branch `feat/ode-engine` deleted; this ODD archived.
