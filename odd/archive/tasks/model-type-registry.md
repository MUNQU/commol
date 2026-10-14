# Refactor: one registry of model types

Goal: a model type (today `DifferenceEquations` and `DifferentialEquations`)
is declared in one place, in the engine: its engine, the model kind it runs,
its settings, the form of its equations and the meaning of a rate. Adding a type adds an entry, not
edits in five places, and Python asks the engine instead of branching on the
typology.
Motivation: audit T0 of `odd/tasks/axis-scale-refactor.md` (theme D, and the
"meaning of a rate" of theme E that `axis-scale-refactor` S2 needs). The goal
of commol includes other model types later (for example stochastic, delay or
spatial models).
Branch: `refactor/model-type-registry` from `develop`, after
`odd/archive/tasks/axis-positions.md` is merged.
Second of four ODDs; the order of all pending work is in
`odd/execution-order.md`.
Engram mirror: topic `odd/model-type-registry/tasks` (project `commol`).
Runners: the Rust tests, clippy and rustdoc (from the root), `maturin develop
--release`, then `pytest`, `pytest -m docs`, `ruff check .`, `ty check commol`
and `vulture` (from `py-commol/`), `mkdocs build --strict` (from the root).

## Current state (verified 2026-10-13 on `develop` at `34d75eb` with PR #102)

The findings, with their places, are in `odd/tasks/axis-scale-refactor.md`,
section "Audit T0":

- D-1: two typologies, declared in Python (`constants.ModelTypes`) and in Rust
  (`commol-core/src/types/dynamics.rs`).
- D-2: a new type edits `Engine`, `EngineArg` and `dispatch!`
  (`py-commol/src/engine.rs`), the `if/else` that picks the engine class
  (`context/_model/engine_state.py:106`) and the `if/else` of the left-hand
  side of `print_equations` (`context/_model/equations.py:318`).
- D-3: the settings of one engine (`OdeSolver`) live in the generic
  `Dynamics.solver` and in `ModelBuilder(solver=…)`, which takes them before
  `build(typology)` chooses the type; `context/dynamics.py:282` rejects them
  for a difference model in Python.
- E (part): what a rate means per type (a per-step amount or a continuous
  rate) is written in the guides, not asked from the type.

## Specs

Every spec follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — The engine declares the model types

- Fix: one list in Rust (D1). Each entry is an engine, named by its typology
  (`DifferenceEquations`, `DifferentialEquations`), and names the model kind
  it runs; today one kind, compartmental. Each entry builds its engine from a
  model, validates its settings, and says the form of its equations and the
  meaning of a rate. The bindings expose the list, so Python builds an engine,
  prints equations and checks settings by asking it: the `if/else` of D-2 go.
- Check: a Rust test that every entry names its kind, builds, prints and
  rejects wrong settings; the Python tests of both typologies unchanged.

### S2 — Settings per model type

- Fix: `Model.engine = {type, settings}` replaces `Dynamics.typology` and
  `Dynamics.solver` (`model-kind-interface` D1; this ODD makes the `engine`
  half of it, that ODD the `structure` half). The settings of an engine travel
  with it, and the builder takes them where it takes the engine (D3).
- Check: settings saved and loaded back; settings of one type given to
  another type raise an error from the engine that names both.

### S3 — How to add a model type

- Fix: a checklist in `docs/codebase/maintainer-playbook.md` and an entry in
  `design-decisions.md`.
- Check: `mkdocs build --strict`.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1 | Where is the list of model types, and what is an entry? | In Rust (`commol-core`). An entry is an engine, named by its typology, that names the model kind it runs (today only compartmental), with everything that depends on it: the engine it builds, how it validates its settings, the form of its equations and the meaning of a rate. Python keeps the typology names only as the `Literal` of `Dynamics.typology`, for static typing, and a test checks that it equals the engine's list; for the rest Python asks the engine, so the `if/else` of D-2 go. | Python reads the list only at runtime: `typology` becomes `str`, and `ty` checks and completion are lost. The whole list with behavior in both languages: D-1 today, against the golden rule. Generate the `Literal` from Rust at build time: no duplication, but a code-generation step for two names; the test catches the same drift. An entry per typology without its model kind: a typology then means a kind and an engine at once, and another kind (an agent network) forces a second breaking change (`model-kind-interface` L1). (The crates are not published, so there is no Rust-only user.) | approved (L2, L5) |
| D2 | S2 | Shape of the settings | Replaced by `odd/tasks/model-kind-interface.md` D1 (its L6): `Model.engine = {type, settings}`; `Dynamics` goes. The settings of an engine are still defined and validated by that engine in Rust and typed in Python as a union of one class per engine. | — | replaced (L6) |
| D3 | S2 | Where does the builder take the engine and its settings? | In `build(engine="DifferentialEquations", settings=None)`, where the engine is chosen, so one builder builds the model with each engine and its own settings. The name matches `Model.engine.type` and `Model.engine.settings`; the engine names stay; the kind is set by the builder (`ModelBuilder` builds compartmental models). | Keep `typology`: the model says `engine` and the builder `typology` for one thing, and "typology" reads as the model kind, the mix `model-kind-interface` D1 separates. One `Engine(type, settings)` argument: an extra import for the most common call. In `ModelBuilder(…)`: one builder cannot build both engines with their own settings; that is why `tests/support/models.py` `build_ode` replaces `solver` with `model_copy`. | approved (L4, L7) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1 | List of engines in Rust, each with its model kind (D1); bindings; Python asks it | direct | |
| T2 | S2 | `Model.engine = {type, settings}` in Python and Rust, replacing `Dynamics.typology` and `solver`; `build(engine=…, settings=None)` (D3) | direct | |
| T3 | S3 | Maintainer checklist and design decision | direct | |
| T4 | all | Final audit: every runner | direct | |

## Delivery

- Strategy: one PR to `develop` from `refactor/model-type-registry` (PR #105),
  squash-merged (`refactor(engine): …`).
- Order: T1 → T2 → T3 → T4.
- Version impact: none for released names: `solver` and
  `ModelBuilder(solver=…)` are unreleased, so the changelog `[Unreleased]`
  entries of the ODE engine change instead. It can ship in one minor version
  with the two other ODDs.

## Evidence

Branch `refactor/model-type-registry` from `develop` at `e997d58`
(2026-10-14).

- T1 (S1, D1): `crates/commol-core/src/types/engine.rs` holds the list,
  `EngineType` (`ALL`, `name`, `from_name`, `kind`, `rate_meaning`,
  `equation_lhs`, `settings`), with `ModelKind` (`compartmental`),
  `RateMeaning` (`per_step`, `continuous`), `EngineSettings` and
  `OdeSolver` (moved from `dynamics.rs`, now `deny_unknown_fields` and
  `validate`). An entry cannot build its engine in `commol-core`, which the
  engine crates depend on, so `Engine::build` lives in the bindings
  (`py-commol/src/engine.rs`); an exhaustive `match` makes the compiler name
  every place a new entry needs. One Python class `commol_rs.engine.Engine`
  replaces `DifferenceEquations`, `DifferentialEquations`, `EngineArg` and
  `engine_class!`; `commol_rs.engine.EngineType` exposes the list.
  `build_engine` and `print_equations` ask it: the two `if/else` of D-2 are
  gone, and `commol.constants.ModelTypes` is removed.
- Rust tests of S1: `engine.rs` (core) checks for every entry its name, kind,
  rate meaning, equations, defaults and the rejection of wrong settings;
  `py-commol/src/engine.rs` checks that every entry builds its engine and
  runs, and that the engine takes the model's settings. The binding crate's
  tests run with `cargo test -p commol` (added to the test commands; CI
  already runs `cargo test --workspace`).
- T2 (S2, D2 → `model-kind-interface` D1, D3): `Model.engine = {type,
  settings}` (`ModelEngine`, Python `context/engine.py` and Rust); `Dynamics`
  keeps only `transitions`. In Rust, `ModelEngine` reads its settings through
  the entry, so a model with settings of another engine does not load. Python
  `ModelEngine` asks `EngineType.check_settings`; the error names both
  engines. `ModelBuilder(solver=…)` is gone; `build(engine, settings=None)`.
  `engine` has no default (as `typology` had none).
  `EngineSettings = OdeSolver`: the union has one member, since the
  difference engine takes no settings; Rust finds the owner of foreign
  settings by trying the other entries.
- `print_equations` keeps its header `Model Type: <engine>`, so the
  print-equation tests are unchanged except the `build` call.
- T3 (S3): checklist "Model type (engine)" in
  `docs/codebase/maintainer-playbook.md`; decision "One list of engines" in
  `design-decisions.md`; `repository-map.md`, `mental-model.md`,
  `AGENTS.md`, the guides, `advanced/model-structure.md` and the changelog
  (`[Unreleased]`: Added, Changed with a mapping table, Removed).
- T4: `cargo fmt --check`, `cargo clippy --workspace --all-targets -D
  warnings`, the Rust tests (238 passed, `-p commol` included), rustdoc with
  `-D warnings`, `maturin develop --release`, `pytest` (954 passed, 4
  xfailed), `ruff check`, `ruff format --check`, `ty check commol`,
  `vulture` and `mkdocs build --strict`: all pass.

## Log

- L1 (user, 2026-10-13): split the audit T0 of
  `odd/tasks/axis-scale-refactor.md` into three ODDs; this one takes theme D
  and the meaning of a rate per type.
- L2 (user, 2026-10-13): approved D1, corrected: the crates are not
  published, so the argument of a Rust-only user does not hold; Python keeps
  the names as a `Literal` for typing, checked by a test against the engine.
- L3 (user, 2026-10-13): approved D2 (`Dynamics.settings` per type;
  `typology` stays).
- L4 (user, 2026-10-13): approved D3 (`build(typology, settings=None)`).
- L5 (user, 2026-10-13): D1 extended so it does not go stale against
  `odd/tasks/model-kind-interface.md` D1: an entry is an engine that names the
  model kind it runs. The rest of D1 stands.
- L6 (user, 2026-10-13): commol is alpha, so a decision weighs no backward
  compatibility. D2 kept `Dynamics.typology` because it was released; it is
  replaced by `model-kind-interface` D1 (`Model.engine`). D3 is to be
  reviewed under the same rule.
- L7 (user, 2026-10-13): D3 reviewed under the alpha rule: the argument is
  `engine`, not `typology` (`build(engine=…, settings=None)`).
- L8 (user, 2026-10-13): while ordering the pending tasks, split the layout
  of `model-kind-interface` D1 between the two ODDs: this one makes
  `Model.engine` (T2), that one `Model.structure` (its T0).
- L9 (agent, 2026-10-14): implemented T1–T4 (see Evidence). `Engine::build`
  is in the bindings, not `commol-core`, because the core cannot depend on
  the engine crates; its tests run with `cargo test -p commol`.
