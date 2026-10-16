# Commol — Agent Guide

Commol is a compartment-modeling library: a Python API (`py-commol/commol/`)
on top of Rust engines (`crates/`): a model compiler (`commol-model`), a
difference engine and an ODE engine on `diffsol`, two stochastic engines of
whole units (`commol-stochastic`), reproducible random numbers for stochastic
engines (`commol-random`), joined by PyO3 bindings
(`py-commol/src/`). This file is an index. The details live in `docs/`, and
the code is the final authority.

**Golden rule: the Rust engine is the single source of truth.** Python
validates input and presents results. It never re-implements how the engine
expands, initializes, plans or simulates a model. It asks the engine.

**Documentation is mandatory.** Every change to this repository, including
yours, ships with the documentation it needs, in the agreed format and at the
agreed quality. Before you call a change done, follow the
[documentation definition of done](docs/codebase/conventions.md#documentation-definition-of-done).

## Before you change code

1. Read [Mental model](docs/codebase/mental-model.md) (5 minutes).
2. Find the owner of the behavior in [Repository map](docs/codebase/repository-map.md).
3. Read the [Design decisions](docs/codebase/design-decisions.md) that touch it.
4. Follow the checklist for your change in the [Maintainer playbook](docs/codebase/maintainer-playbook.md).
5. Update the documentation as the [definition of done](docs/codebase/conventions.md#documentation-definition-of-done) requires.

## Commands

Run them from `py-commol/` with the virtual environment `py-commol/.venv`.

| Task | Command |
|---|---|
| Install dependencies | `uv sync --extra dev --extra docs` |
| Build the extension | `source .venv/bin/activate && maturin develop --release --features test-kind` (the tests need the test model kind; a release never has it) |
| Python tests | `pytest` |
| Documentation checks (examples, API coverage) | `pytest -m docs` |
| Python lint, types and dead code | `ruff check . && ruff format --check . && ty check commol && vulture commol .vulture_allowlist.py --min-confidence 80` |
| Rust tests (from the root) | `cargo test -p commol-core -p commol-model -p commol-difference -p commol-differential -p commol-calibration -p commol-random -p commol-stochastic -p commol && cargo test -p commol --features test-kind` |
| Rust lint (from the root) | `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings` |
| Documentation (from the root) | `mkdocs build --strict` |
| Rust docs (from the root) | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p commol-core -p commol-model -p commol-difference -p commol-differential -p commol-calibration -p commol-random -p commol-stochastic` |

Rebuild the extension after every Rust change, before you run `pytest`.
`cargo build --workspace` does not link the PyO3 crate outside `maturin`; use
`clippy` and the per-crate tests instead.

## Rules

| Area | Rule |
|---|---|
| Commits | `type(scope): description`: scope required, one short phrase, no period, no body, no co-author or AI attribution lines. |
| Local files | Never stage or commit `CLAUDE.md` or `AGENT.md`; they are local context files. |
| Branches | Work on a branch from `develop`. PRs to `develop` are squash-merged; `develop` → `main` uses a merge commit. See [Development workflow](docs/development/workflow.md). |
| Rust toolchain | `rust-toolchain.toml` pins the one Rust version of local builds, hooks and CI. Never use a floating `stable` in a workflow; Dependabot proposes each newer stable release in its own PR: [The Rust toolchain](docs/development/workflow.md#the-rust-toolchain). |
| Tests | Run the Rust and the Python suites before you call a change done. Put a Python test in the package of its guide page, build models with `tests/support/`, and check a result: [Tests](docs/codebase/conventions.md#tests). |
| Comments | Describe the code as it is, never how it came to be. See [Conventions](docs/codebase/conventions.md). |
| Docs | **Mandatory.** Every change ships with its documentation, in the agreed format and quality: the [documentation definition of done](docs/codebase/conventions.md#documentation-definition-of-done). |
| Planning | Work is an ODD (`odd/<name>.md`, never moved) with its GitHub issue (label `odd`), whose blockers and milestone set the order and the release. A pull request says `Closes #N`. See [`odd/README.md`](odd/README.md). |
| Releases | Follow [Release process](docs/development/release.md): a release issue (label `release`) holds the checklist, and the notes come from the changelog (`.github/scripts/release_notes.py`). A published PyPI version is final. |

## If you need to…

| Task | Start with | Then check |
|---|---|---|
| Change how a model expands into compartments | `crates/commol-model/src/builder.rs` (`expand`) | `layout.rs`, `py-commol/commol/context/_model/engine_state.py` |
| Change how transitions become flows | `crates/commol-model/src/flows.rs` | `helpers.rs`, `system.rs` (`flow_rates`, `flow_changes`, `derivatives`), `context/_model/equations.py` (`DynamicsDescription`) |
| Change the difference step | `crates/commol-difference/src/simulation.rs` | `crates/commol-core/src/simulation_engine.rs` |
| Change the ODE engine or its solver settings | `crates/commol-differential/src/solve.rs` | `lib.rs`, `crates/commol-core/src/types/engine.rs` (`OdeSolver`), `py-commol/commol/context/engine.py` |
| Add a model type (engine) | `crates/commol-core/src/types/engine.rs` (`EngineType`) | `py-commol/src/engine.rs` (`Engine`), `py-commol/commol/context/engine.py`, the [checklist](docs/codebase/maintainer-playbook.md#model-type-engine) |
| Add a model kind | `py-commol/commol/context/structure.py` (`ModelStructure`) | `crates/commol-core/src/types/structure.rs`, `engine.rs` (`ModelKind`), `py-commol/src/engine.rs`, the [checklist](docs/codebase/maintainer-playbook.md#model-kind) |
| Change how a rate reads the neighbors of a unit | `crates/commol-model/src/flows.rs` (`resolve_neighbors`) | `crates/commol-core/src/types/aggregate.rs` (`Neighbors`), `builder.rs` (`check_model`), `py-commol/commol/context/neighbors.py` |
| Change impulses | `crates/commol-model/src/system.rs` (`apply_impulses`) | both engines, `py-commol/commol/context/transition.py` |
| Change which positions a model type accepts | `crates/commol-core/src/simulation_engine.rs` (`check_position`) | each engine's `check_position`, `crates/commol-calibration/src/calibration_problem.rs` (`check_positions`) |
| Change how results, windows or reports read positions | `py-commol/commol/api/results.py` | `api/windows.py`, `api/reporting.py`, `api/plotter.py`, `api/probabilistic/statistics_calculator.py` |
| Add a function or variable to rate expressions | `crates/commol-core/src/math_expression/` | `py-commol/commol/utils/security.py`, `docs/guide/mathematical-expressions.md` |
| Change initial conditions | `py-commol/commol/context/initial_conditions.py` | `crates/commol-model/src/builder.rs` (`initial_state`), `py-commol/commol/context/aggregate.py`, `docs/guide/building-models.md` |
| Change a stochastic engine of the aggregate kind (events, chain binomial, whole units) | `crates/commol-stochastic/src/` (`events.rs`, `steps.rs`, `units.rs`) | `crates/commol-core/src/types/engine.rs` (`StochasticEventsSettings`, `Rounding`), `crates/commol-model/src/system.rs` (`flow_changes`, `breakpoints`), `py-commol/commol/context/engine.py`, `docs/guide/stochastic.md` |
| Add or change a stochastic engine, seeds or samplers | `crates/commol-random/` | `crates/commol-core/src/simulation_engine.rs` (`set_seed`, `run_replicates`), `engine.rs` (`is_stochastic`), `py-commol/commol/api/simulation.py` (`replicates`) |
| Change calibration | `crates/commol-calibration/src/calibration_problem.rs` | `values.rs`, `crates/commol-model/src/initial_conditions.rs`, `py-commol/commol/api/calibrator.py` |
| Change probabilistic calibration | `py-commol/commol/api/probabilistic_calibrator.py` | `api/probabilistic/`, `crates/commol-calibration/src/probabilistic/` |
| Change the Python API | `py-commol/commol/api/` | `docs/api/`, the guide page of that feature |
| Expose a Rust function to Python | `py-commol/src/` (`engine.rs` for a function that takes an engine) | `py-commol/commol/commol_rs/_commol_rs.pyi` |
| Change how a model is described | `py-commol/commol/context/description.py` (blocks), `context/_model/describe.py` (layouts) | `ModelStructure.describe`, `context/_model/aggregate_description.py`, `api/description.py` (renderers), `docs/guide/describing-models.md` |
| Change reports | `py-commol/commol/api/reporting.py` | `ModelStructure.report_sections`, `context/_model/report.py`, [Design decisions](docs/codebase/design-decisions.md#reports) |

## Documentation map

| You are… | Read |
|---|---|
| Using commol | `docs/getting-started/`, `docs/guide/`, `docs/api/` |
| Changing commol | `docs/codebase/` (start at [the index](docs/codebase/index.md)) |
| Releasing or merging | `docs/development/` |
