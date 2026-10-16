# Feature: Fix the code findings F1–F12

Goal: fix the eleven code defects that the documentation overhaul found
(`odd/documentation-overhaul.md`, Evidence T3–T6, T14), plus the
Nelder-Mead note of T6 (F12), and remove the "known issue" notes that the docs
carry for them.
Branch: `fix/code-findings` from `develop`, one PR, squash-merged (see Delivery).
Engram mirror: topic `odd/code-findings/tasks` (project `commol`).
Runners: `maturin develop --release` then `pytest` (from `py-commol/`),
`cargo test -p commol-core -p commol-difference -p commol-calibration`,
`cargo clippy --workspace --all-targets -- -D warnings`, `mkdocs build --strict`
(from the root).

## Specs

Each spec names the finding, the code behind it (verified 2026-10-12 on
`develop` at 32439e4) and the check that closes it. Every spec also follows the
[documentation definition of done](../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — `pi` and `e` work in every rate (F5)

- Cause: `crates/commol-difference/src/builder.rs::resolve_var_slot` maps an
  unknown name to `VarSlot::Parameter(name)`; `simulation.rs::resolve_slot`
  then looks up `pi`/`e` in the context and fails ("Variable 'pi' not found").
  Only the JIT (`jit/codegen.rs::load_variable`) knows them as constants.
- Fix: the engine resolves `pi` and `e` as constants in every path (fast path,
  JIT slots, evalexpr fallback).
- Check: a Rust test and a Python test run a rate `pi * k * A` and `e * k * A`
  and match the analytic value. The known-issue warning in
  `docs/guide/mathematical-expressions.md` (L79) is removed.

### S2 — Population variables and operators fail loudly (F6)

- Cause: `jit/compiler.rs` (L95) sums every compartment whose name ends with the
  suffix of an `N_…` name, so `N_t1_g1` (suffixes out of declaration order)
  gives 0. `math_expression/mod.rs::RateMathExpression::from_string` does not
  list `%` as an operator, so `k % 2` becomes a parameter name.
- Fix: an `N_…` name that names no subpopulation is an error at build time
  (Python validation and engine). `RateMathExpression::from_string` lists
  `%` as an operator: the JIT already supports it (`jit/ast.rs`
  `BinaryOperator::Mod`, `jit/codegen.rs`) and `security.py` allows it; only
  a rate whose sole operator is `%` (`step % 7`) is misread.
- Check: tests for `N_t1_g1` (error that lists the valid names) and for `%`.

### S3 — Reserved names cannot be parameter ids (F4)

- Cause: `py-commol/commol/context/model.py::validate_parameter_names_not_reserved`
  rejects only bin ids.
- Fix: parameter ids cannot be `N`, an `N_…` subpopulation name, `step`, `t`,
  `pi`, `e`, or an expanded compartment name. The engine owns the expanded names
  (golden rule), so the compartment and `N_…` check lives in the Rust builder;
  the fixed names are also rejected in Python for an early error.
- Check: one test per reserved kind; the error names the conflict.

### S4 — `ModelBuilder.reset` and `clone` keep the builder consistent (F3)

- Cause: `py-commol/commol/api/model_builder.py::reset` (L956) does not clear
  `_accumulators`. Found while planning: `clone` (L936) does not copy
  `_accumulators` either.
- Fix: `reset` clears every collection; `clone` copies every collection.
- Check: tests that `reset` leaves no accumulator and that a clone builds the
  same model as the original.

### S5 — `plot_series` accepts bin ids of a stratified model (F7)

- Cause: `py-commol/commol/api/plotter.py::plot_series` takes output names; a
  bin id of a stratified model raises `KeyError` ("Missing result series for
  aggregate components").
- Fix (D5): `plot_series` and `plot_cumulative` take `series` instead of
  `bins`. Each entry is an output name (exact match first) or a bin id; a
  bin id plots the sum of `simulation.outputs_for(bin)` through
  `_sum_series`. No alias for `bins`.
- Check: tests plot a bin id of a stratified model, an output name, and an
  unknown id (clear error); the guide, the docstrings and the examples use
  `series`.

### S6 — A clear error for a non-binary category (F10)

- Cause: an `initial_condition` id that names a category of a non-binary
  stratification fails with "not found". The engine already has the right
  message in `crates/commol-calibration/src/initial_conditions.rs::paired_category`
  ("needs a binary stratification"), so an earlier check rejects the id first.
- Fix: the first check that sees the id gives the binary-stratification message.
- Check: a test asserts the message; `calibration-initial-conditions.md` states
  the rule.

### S7 — Loss functions use weights as documented (F11)

- Cause: `crates/commol-calibration/src/calibration_problem.rs` (L672) computes
  `SumSquaredError` and `WeightedSSE` with one arm: `Σ (w · r)²`, so the weight
  is squared and the two losses are the same. `RootMeanSquaredError` and
  `MeanAbsoluteError` ignore weights.
- Fix (D4): `sse` = `Σ w·r²`, `rmse` = `√(Σ w·r² / Σ w)`, `mae` =
  `Σ w·|r| / Σ w`; `LossConfig::WeightedSSE`, `LossConfig.weighted_sse` and
  the `"weighted_sse"` name are removed. The ensemble metric
  (`probabilistic/ensemble/mod.rs` L55) follows the same rule.
- Check: Rust tests for each loss with weights ≠ 1; `calibration.md` gives the
  formulas.

### S8 — `Simulation.run` is repeatable (F8)

- Cause: `crates/commol-difference/src/simulation.rs::DifferenceEquations::run`
  never resets, and `py-commol/commol/api/simulation.py::_run_raw` does not call
  `reset`, so a second `run` continues the first one.
- Fix (D2): `Simulation.run` calls `engine.reset()` before it runs; continuing
  is `simulation.engine.step()`.
- Check: two calls to `run` give the same result, `step` restarts at 0;
  `simulations.md` loses the `engine.reset()` workaround.

### S9 — `Simulation` does not use a stale model (F9)

- Cause: `Simulation.__init__` serializes the model once
  (`_initialize_engine`). After `update_parameters` it fails with
  `Parameter index '2' not found` or silently uses old values.
- Fix (D3): `Simulation` keeps the JSON it was built from; `run` and the
  `engine` property compare it with `model_dump_json()` and rebuild the
  engine (and the cached compartments and outputs) when it differs. The
  check is in `engine` because `Calibrator` takes `simulation.engine` at init
  (`calibrator.py:69`). Measured: dump 0.018 ms, `run(100)` 0.27 ms, rebuild
  0.8 ms (60 compartments).
- Check: tests for a parameter set after the `Simulation` exists (from `None`
  and from a value); the warning in `simulations.md` is replaced.

### S10 — Transition conditions do what the API says (F2)

- Cause: `Transition.condition` and `ModelBuilder.create_condition` are stored
  and serialized, but no engine code reads them (`flows.rs` reads only
  stratified-rate conditions). A condition that is never true still moves
  population.
- Fix (D1): remove `Transition.condition`, `ModelBuilder.create_condition`,
  the `condition` argument of `add_transition`, and the `Condition`/`Rule`
  types if nothing else uses them, in Python, the bindings and the Rust types.
- Check: no reference to transition conditions is left (`rg`); passing
  `condition` raises `TypeError`; `transitions.md` and the `add_transition` docstring
  match the code.

### S11 — The unit check matches what the engine computes (F1)

- Cause: `py-commol/commol/context/_model/units.py::_infer_time_unit` matches
  unit strings against a fixed list of physical time units and the caller
  defaults to `day` (L157), so `1/step` always fails with "expected
  `<bin unit>/day`".
- Also found (2026-10-12, run on `develop`): `utils/equations.py::check_equation_units`
  accepts any compatible unit (`is_compatible_with`), so `k1` in `1/day` and
  `k2` in `1/week` pass, but the engine never converts values: `k2` is used
  per step like `k1`. A bin without a unit fails with "'None' is not defined
  in the unit registry".
- Fix (D6): one step unit, inferred from the rate parameters by
  dimensionality (`[time]` or the `step` dimension); exact equality, not
  `is_compatible_with`; a mix is an error naming each parameter and unit;
  default `step`. The time unit comes from the dimensionality of the
  parameter units, not from substring matches. A bin without a unit gives an
  error that names the bin.
- Check: tests for a model in `1/step`, one in `1/day`, one in `1/week`,
  mixes `1/day`+`1/week` and `1/day`+`1/step` (errors), a bin without a
  unit (error naming it); `units-and-equations.md` (L67) loses the restriction.

### S12 — `converged` means converged, and Nelder-Mead respects bounds (F12)

- Note in `documentation-overhaul.md` (T6): Nelder-Mead from the midpoint can
  stop at a bound and still report `converged=True`.
- Causes (verified):
  - `crates/commol-calibration/src/optimization.rs` sets `converged:
    state.termination_status.terminated()` (L650 Nelder-Mead, L863 PSO).
    argmin's `terminated()` is true for every reason, `MaxItersReached`
    included, so every finished run reports `converged=True`.
  - The objective clamps values to the bounds
    (`initial_conditions.rs::clamp_to_bounds`), so the loss is flat outside
    them. A simplex that crosses a bound collapses there, meets
    `sd_tolerance` and stops with `SolverConverged`.
  - `build_nelder_mead_solver` (L545) builds each vertex as
    `value * simplex_perturbation`: a start value of 0 gives a degenerate
    simplex, and the step ignores the bound width.
- Mechanism of the false convergence: argmin's Nelder-Mead stops when the
  standard deviation of the vertex costs is below `sd_tolerance`; vertices in
  the flat clamped region have equal costs.
- Fix: `converged` is true only for `SolverConverged` and `TargetCostReached`
  (both optimizers). Nelder-Mead uses the variable transform of D7, an
  additive initial simplex in `u`, and `initial_step` instead of
  `simplex_perturbation`; `best_parameters` are reported in `x`.
- Check: tests for a run that hits `max_iterations` (`converged=False`), a
  start value of 0, and a one-unknown problem whose optimum is inside the
  bounds (recovered without `initial_guess`). `calibration-optimizers.md`
  states what `converged` and `termination_reason` mean and loses the
  workaround note.

## Decisions (the user's to make)

Each row has a recommendation. All are approved (L4–L10).

| ID | Spec | Question | Recommendation | Trade-off | Status |
|---|---|---|---|---|---|
| D1 | S10 | Implement or remove transition conditions? | Deprecate `condition` and `create_condition` (`DeprecationWarning` in 0.3, removal in 0.4). Rates already express gates: `if(A > 0, k, 0)` and stratified rates. | Implementing needs per-step rule evaluation in the engine and a new semantics for `strat:` rules that stratified rates already cover. | approved (L4): remove in 0.3, no deprecation period |
| D2 | S8 | Should `Simulation.run` start from the initial state? | Yes: `run` resets the engine first. A user who wants to continue uses `simulation.engine.step()`. | Breaking for anyone who chains `run` calls on purpose; the docs and the changelog say so. | approved (L5) |
| D3 | S9 | Rebuild or fail when the model changed? | Rebuild: `Simulation` keeps the JSON it was built from and rebuilds the engine in `run` when `model_dump_json()` differs. It also catches direct field edits. | One serialization per `run` (small next to a run); an error instead would be cheaper but forces users to recreate the object. | approved (L6) |
| D4 | S7 | What do the losses compute with weights? | One meaning of a weight in every loss: `sse` = `Σ w·r²`, `rmse` = `√(Σ w·r² / Σ w)`, `mae` = `Σ w·|r| / Σ w`; remove `weighted_sse`. With every weight at 1 nothing changes. | Results change only for users who set weights. | approved (L7) |
| D5 | S5 | What does the plot selection accept, and under which name? | Output names and bin ids under a parameter renamed `series` (in `plot_series` and `plot_cumulative`); an exact output name first, else a bin id, whose total comes from `outputs_for`. No `bins` alias. | Breaking rename; changelog "Changed". | approved (L8) |
| D6 | S11 | How does the unit check treat time? | One step unit, inferred from the dimensionality of the rate parameters (a physical time unit or `step`); every rate parameter must use exactly that unit (no conversion, because the engine converts nothing); a mix is an error that names each parameter and unit; `step` is the default. No new API. | A model that is all `1/week` is accepted: a step is then a week. | approved (L9) |
| D7 | S12 | How does Nelder-Mead treat bounds? | Variable transform (MINUIT, lmfit): Nelder-Mead searches an unbounded `u`, the model gets `x = lo + (hi − lo)·(sin u + 1)/2`; the initial simplex adds a step in `u`; `initial_step` (fraction of the bound width, default 0.1) replaces `simplex_perturbation`, no alias. | Every Nelder-Mead result changes; an optimum exactly at a bound is reached only asymptotically; more iterations near bounds. | approved (L10) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1 | `pi`/`e` as constants in `resolve_var_slot` / `resolve_slot` and the evalexpr path; Rust and Python tests; remove the known-issue note | direct | 982985d |
| T2 | S2 | Reject unknown `N_…` names (Python validation and `jit/compiler.rs`); add `%` to `from_string`; tests | direct | b81be1c |
| T3 | S3 | Reserved parameter ids: Rust builder check for compartments and `N_…`, Python check for the fixed names; tests; `building-models.md` | direct | b81be1c |
| T4 | S4 | `ModelBuilder.reset` and `clone` cover every collection; tests | direct | 7794a29 |
| T5 | S6 | Binary-stratification error for a non-binary category; test; `calibration-initial-conditions.md` | direct | 25638d3 |
| T6 | S5 | `series` parameter in `plot_series` and `plot_cumulative` (D5); tests; docstrings, `simulations.md`, every `bins=` in docs and tests, changelog "Changed" | direct | dccc48a |
| T7 | S7 | Weighted losses and removal of `weighted_sse` (D4), calibration and ensemble; Rust tests (weights ≠ 1, all weights 1 equal to the old values); `calibration.md`, `calibration-observations.md`, `_commol_rs.pyi`, changelog "Changed" and "Removed" | direct | 96d7f25 |
| T8 | S8 | `Simulation.run` resets first (D2); tests; `simulations.md`, changelog "Changed" | direct | 1d817b3 |
| T9 | S9 | Rebuild on model change (D3); tests (parameter from `None`, from a value, direct field edit, `Calibrator` after a change); `simulations.md`, `calibration.md` warning | direct | 1d817b3 |
| T10 | S10 | Remove transition conditions (D1); tests; `transitions.md`, API reference, `add_transition` docstring, changelog "Removed" | direct | e3cfd77 |
| T11 | S11 | Unit check with one inferred step unit (D6); tests; `units-and-equations.md`, `check_unit_consistency` docstring, changelog "Fixed" | direct | 75ed951 |
| T12 | S12 | `converged` from the termination reason (Nelder-Mead and PSO); variable transform and `initial_step` (D7); tests; `calibration-optimizers.md`, `NelderMeadConfig` docstring, `_commol_rs.pyi`, changelog "Fixed" and "Changed" | direct | c8655f3 |
| T13 | S1–S12 | Final audit: every check above, full suites, `mkdocs build --strict`, `pytest -m docs`, changelog `[Unreleased]` (Fixed, Changed, Deprecated), no "known issue" note left for F1–F12 | direct | (this commit) |

## Delivery

- Strategy: `single-pr`, like the documentation overhaul. Every task is one
  work-unit commit on `fix/code-findings`; the PR to `develop` is squash-merged
  once T12 passes.
- Order: T1–T12 in any order (every decision is approved), T13 last.
- Version impact: D1 (removal), D2, D4, D5 (rename), D6 and D7 change public
  behavior, and so does the `converged` fix (runs that hit `max_iterations`
  now report `False`). The changelog has an entry for each. The next release is
  0.3.0, and the changelog has a "Changed" entry for each.
- Rebuild the extension after every Rust change, before `pytest`.

## Evidence

Runners on `fix/code-findings` at T13: `cargo fmt --check`, `cargo clippy
-D warnings`, `cargo doc -D warnings` clean; Rust 183 passed; `pytest` 719
passed (656 at the start); `pytest -m docs` 55 passed; `ruff`, `ty` clean;
`mkdocs build --strict` clean.

| Task | Check | Docs |
|---|---|---|
| T1 | `simulation.rs` `pi_and_e_are_constants_in_every_rate_path` (fails without the fix); `test_rate_expressions.py::TestConstants` | `mathematical-expressions.md` loses the known issue |
| T2 | `builder.rs` `subpopulation_variable_out_of_declaration_order_is_a_build_error`, `modulo_is_an_operator_in_a_rate`; `TestSubpopulationVariables*`, `TestOperators` | `mathematical-expressions.md` (`%`, pitfalls), `design-decisions.md` |
| T3 | `builder.rs` `parameter_id_cannot_be_a_compartment_or_a_subpopulation`; `test_reserved_names.py` | `building-models.md` "Names the engine sets" |
| T4 | `test_model_builder.py::TestCloneAndReset` (2 of 3 fail without the fix) | `building-models.md` builder table |
| T5 | `test_initial_condition_ids.py` | `calibration-initial-conditions.md` error table |
| T6 | `test_plotter.py::TestSeriesSelection` | `simulations.md` |
| T7 | `types.rs` `loss_tests`, `ensemble` weighted test, `test_central_fit_loss_uses_weights_in_every_metric` | `calibration.md`, `calibration-observations.md`, `probabilistic-calibration.md`, `design-decisions.md` |
| T8, T9 | `test_simulation_runs.py` | `simulations.md`, `calibration.md`, `design-decisions.md` |
| T10 | `test_transition_flows.py::TestNoTransitionConditions`; `git grep` finds no transition condition | `transitions.md`, `building-models.md`, `model-structure.md` |
| T11 | `test_units.py::TestStepUnit` | `units-and-equations.md`, `design-decisions.md` |
| T12 | `optimization::tests`; `test_optimizer_convergence.py` | `calibration-optimizers.md`, `calibration.md`, `design-decisions.md` |

Found while implementing:

- T2: a rate read from JSON is `RateMathExpression::Parameter(<whole string>)`
  (untagged enum), so `check_names` parses it again before it lists variables.
- T5: the Python check also rejected bin prefixes (`B_g1`) and qualified
  categories (`group:g1`) that the guide documents; the engine check
  (`check_initial_conditions`) accepts them.
- T7: Python `central_fit_loss` had its own formula; it now calls
  `LossConfig.aggregate`.
- T11: most `test_units.py` models had no `bin_unit` and failed for that
  reason only; they now set `bin_unit="person"`.
- T12: the verbose path (`python_observer.rs`) built its own simplex with a
  fixed 10% perturbation; it now runs `optimize_with_observer`.

## Log

- L1 (user, 2026-10-12): "Crea un ODD para solucionar los hallazgos F1–F11 de odd/documentation-overhaul.md."
- L2: code paths of F1–F11 re-checked on `develop` (32439e4) with CodeGraph;
  `ModelBuilder.clone` missing `_accumulators` found and added to S4.
- L3 (user, 2026-10-12): "Añade Nelder-Mead al alcance". Added as F12 (S12,
  D7, T12); the final audit moves to T13.
- L4 (user, 2026-10-12): D1 → "Eliminar ya en 0.3" (after asking why the
  recommendation was to deprecate).
- L5 (user): every recommendation must say why it beats each alternative.
  D2 → "Reiniciar siempre".
- L6 (user): D3 → "Reconstruir".
- L7 (user): D4 → "Pesos en todas" (recommendation revised from the first
  draft: `sse` without weights).
- L8 (user): D5 → bin ids and output names, "adecua el nombre del parámetro
  para que sea entendible" → `series`, no alias.
- L9 (user): D6 → "Unidad única inferida". Before asking, a run on `develop`
  showed `1/day` + `1/week` passing the check (added to S11).
- L10 (user): D7 → "Transformación" (recommendation revised from the first
  draft: keep clamping and warn).
- L11 (user, 2026-10-13): "implementa @odd/code-findings.md". T1–T13
  done on `fix/code-findings`; T2+T3 and T8+T9 share one commit each.
