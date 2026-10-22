# Commol — Agent Guide

Commol is a compartment-modeling library: a Python API (`py-commol/commol/`)
on top of Rust engines (`crates/`): a model compiler (`commol-model`), a
difference engine and an ODE engine on `diffsol`, two stochastic engines of
whole units (`commol-stochastic`), reproducible random numbers for stochastic
engines (`commol-random`), the calibration problem and its optimizers
(`commol-calibration`), the uncertainty methods with their parameter sample,
predictive and scores (`commol-inference`), joined by PyO3 bindings
(`py-commol/src/`). This file is an index. The details live in `docs/`, and
the code is the final authority.

**Golden rule: the Rust engine is the single source of truth**
([Mental model](docs/codebase/mental-model.md#the-golden-rule)).

**Documentation is mandatory.** A change, yours too, is done only when it meets
the [documentation definition of done](docs/codebase/conventions.md#documentation-definition-of-done).

## Before you change code

1. Read [Mental model](docs/codebase/mental-model.md) (5 minutes) and look at
   the diagrams of [Architecture](docs/codebase/architecture.md).
2. Find the owner of the behavior in [Repository map](docs/codebase/repository-map.md).
3. Read only the page of your area in [Design decisions](docs/codebase/design-decisions.md).
4. Follow the checklist for your change in the [Maintainer playbook](docs/codebase/maintainer-playbook.md).
5. Update the documentation as the [definition of done](docs/codebase/conventions.md#documentation-definition-of-done) requires.

## Commands

<!-- --8<-- [start:commands] -->
Run them from `py-commol/` with the virtual environment `py-commol/.venv`.

| Task | Command |
|---|---|
| Install dependencies | `uv sync --extra dev --extra docs` |
| Build the extension | `source .venv/bin/activate && maturin develop --release --features test-kind` (the tests need the test model kind; a release never has it) |
| Python tests | `pytest` |
| Documentation checks (examples, API coverage) | `pytest -m docs` |
| Coverage studies of the uncertainty methods (slow; CI runs them when what they run changes, and before a release) | `pytest -m coverage_study` |
| Python lint, types and dead code | `ruff check . ../.github/scripts && ruff format --check . ../.github/scripts && ty check commol && vulture commol .vulture_allowlist.py --min-confidence 80` |
| Rust tests (from the root) | `cargo test -p commol-core -p commol-model -p commol-difference -p commol-differential -p commol-calibration -p commol-inference -p commol-random -p commol-stochastic -p commol && cargo test -p commol --features test-kind` |
| Test coverage, as CI measures it | `pytest --cov=commol --cov-fail-under=100`; from the root, `.github/scripts/rust_coverage.sh --summary-only` (the Rust tests and `pytest` on an instrumented extension, failing on a line that no test runs and that has no `// coverage: <reason>`; needs `cargo-llvm-cov` and the `llvm-tools-preview` component). Policy: `docs/codebase/coverage.md` |
| Rust lint (from the root) | `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings` |
| Documentation (from the root) | `mkdocs build --strict` |
| Rust docs (from the root) | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p commol-core -p commol-model -p commol-difference -p commol-differential -p commol-calibration -p commol-inference -p commol-random -p commol-stochastic` |

Rebuild the extension after every Rust change and every `uv sync`, before you
run `pytest`; uv never builds it (`package = false`).
`cargo build --workspace` does not link the PyO3 crate outside `maturin`; use
`clippy` and the per-crate tests instead.
<!-- --8<-- [end:commands] -->

## Rules

| Area | Rule |
|---|---|
| Commits | `type(scope): description`: scope required, one short phrase, no period, no body, no co-author or AI attribution lines. |
| Local files | Never stage or commit `CLAUDE.md` or `AGENT.md`; they are local context files. |
| Branches | One branch, `main` (trunk-based). Work on a short branch from `main`; a PR to `main` goes through the merge queue and is squash-merged. A release is a tag on `main`; a branch `release/X.Y` is cut from its tag only to fix an old version. See [Development workflow](docs/development/workflow.md). |
| Rust toolchain | `rust-toolchain.toml` pins the one Rust version of local builds, hooks and CI. Never use a floating `stable` in a workflow; Dependabot proposes each newer stable release in its own PR: [The Rust toolchain](docs/development/workflow.md#the-rust-toolchain). |
| Tests | Run the Rust and the Python suites before you call a change done. Put a Python test in the package of its guide page, build models with `tests/support/`, and check a result: [Tests](docs/codebase/tests.md). |
| Comments | Describe the code as it is, never how it came to be. See [Conventions](docs/codebase/conventions.md). |
| Docs | **Mandatory.** Every change ships with its documentation, in the agreed format and quality: the [documentation definition of done](docs/codebase/conventions.md#documentation-definition-of-done). |
| Planning | Work is a GitHub issue written from `.github/ISSUE_TEMPLATE/`: a proposal (label `proposal`) holds the requirements with their tests, the decisions and the tasks; its blockers and milestone set the order and the release, and the [project](https://github.com/orgs/MUNQU/projects/1) shows its status. Two proposals run in parallel when no link joins them and neither changes what the other touches (`.github/scripts/parallel.py`). The next one is `.github/scripts/next.sh`; start it with `.github/scripts/start.sh <number> <type>/<name>`, never by hand. A bug gets the milestone of the nearest release it fits when it is opened; at the close of a release, every issue still open in its milestone moves to another, with a comment that says why. A change updates every open issue it affects (body and a comment; `python .github/scripts/affected.py --closes <number>` lists them), and work it leaves undone goes to an open issue, never only to a comment of the closed one. A pull request says `Closes #N`. See [Planning](docs/development/workflow.md#planning). |
| Releases | Follow [Release process](docs/development/release.md): a release issue (label `release`) holds the checklist, and the notes come from the changelog (`.github/scripts/release_notes.py`). A published PyPI version is final. |

## If you need to…

<!-- --8<-- [start:tasks] -->
| Task | Start with | Then check |
|---|---|---|
| Change how a model expands into compartments | `crates/commol-model/src/builder.rs` (`expand`) | `layout.rs`, `py-commol/commol/context/_model/engine_state.py` |
| Change how transitions become flows | `crates/commol-model/src/flows.rs` | `helpers.rs`, `system.rs` (`flow_rates`, `flow_changes`, `derivatives`), `context/_model/equations.py` (`DynamicsDescription`) |
| Change the difference step | `crates/commol-difference/src/simulation.rs` | `crates/commol-core/src/simulation_engine.rs` |
| Change the ODE engine or its solver settings | `crates/commol-differential/src/solve.rs` | `crates/commol-differential/src/lib.rs`, `crates/commol-core/src/types/engine.rs` (`OdeSolver`), `py-commol/commol/context/engine.py` |
| Add a model type (engine) | `crates/commol-core/src/types/engine.rs` (`EngineType`) | `py-commol/src/engine.rs` (`Engine`), `py-commol/commol/context/engine.py`, the checklist "Model type (engine)" in `docs/codebase/maintainer-playbook.md` |
| Add a model kind | `py-commol/commol/context/structure.py` (`ModelStructure`) | `crates/commol-core/src/types/structure.rs`, `crates/commol-core/src/types/engine.rs` (`ModelKind`), `py-commol/src/engine.rs`, the checklist "Model kind" in `docs/codebase/maintainer-playbook.md` |
| Change how a rate reads the neighbors of a unit | `crates/commol-model/src/flows.rs` (`resolve_neighbors`) | `crates/commol-core/src/types/aggregate.rs` (`Neighbors`), `builder.rs` (`check_model`), `py-commol/commol/context/neighbors.py` |
| Change impulses | `crates/commol-model/src/system.rs` (`apply_impulses`) | both engines, `py-commol/commol/context/transition.py` |
| Change which positions a model type accepts | `crates/commol-core/src/simulation_engine.rs` (`check_position`) | each engine's `check_position`, `crates/commol-calibration/src/calibration_problem.rs` (`check_positions`) |
| Change how results, windows or reports read positions | `py-commol/commol/api/results.py` | `api/windows.py`, `api/reporting.py`, `api/plotting/`, `api/inference/predictive.py` |
| Change a figure | `py-commol/commol/api/plotting/` (`panels.py`, the data; `figures.py`, the drawing) | `api/plotting/accessors.py`, `api/plotting/compared.py` (`plot_compared`), the `plot` method of the result, `docs/guide/simulations.md` (Plotting) |
| Add a function or variable to rate expressions | `crates/commol-core/src/math_expression/` | `py-commol/commol/utils/security.py`, `docs/guide/mathematical-expressions.md` |
| Change initial conditions | `py-commol/commol/context/initial_conditions.py` | `crates/commol-model/src/builder.rs` (`initial_state`), `py-commol/commol/context/aggregate.py`, `docs/guide/building-models.md` |
| Change a stochastic engine of the aggregate kind (events, chain binomial, whole units) | `crates/commol-stochastic/src/` (`events.rs`, `steps.rs`, `units.rs`) | `crates/commol-core/src/types/engine.rs` (`StochasticEventsSettings`, `Rounding`), `crates/commol-model/src/system.rs` (`flow_changes`, `breakpoints`), `py-commol/commol/context/engine.py`, `docs/guide/stochastic.md` |
| Add or change a stochastic engine, seeds or samplers | `crates/commol-random/` | `crates/commol-core/src/simulation_engine.rs` (`set_seed`, `run_replicates`), `crates/commol-core/src/types/engine.rs` (`is_stochastic`), `py-commol/commol/api/simulation.py` (`replicates`) |
| Change calibration | `crates/commol-calibration/src/calibration_problem.rs` and its layers (`forward.rs`, `observation.rs`, `feasibility.rs`, `discrepancy.rs`, `objective.rs`); `crates/commol-calibration/src/starts.rs` | `values.rs`, `crates/commol-model/src/initial_conditions.rs`, `py-commol/commol/api/calibrator.py`, `py-commol/src/inference.rs` (`CalibrationProblem`) |
| Add or change an observation model or the likelihood | `crates/commol-calibration/src/likelihood/` (`likelihood/mod.rs`, `density.rs`) | `crates/commol-inference/src/predictive.rs` (`draw_observation`), `py-commol/src/calibration.rs` (`ObservationModel`), `py-commol/commol/context/observation_models.py`, `context/_model/engine_state.py` (`rust_observation_models`), `docs/guide/uncertainty/observation-models.md` |
| Add an uncertainty method | `crates/commol-inference/` | `py-commol/src/inference.rs`, `py-commol/commol/context/inference/`, `api/inference/registry.py`, the checklist "Uncertainty method" in `docs/codebase/maintainer-playbook.md` |
| Change a profile, its thresholds or a subproblem that holds a quantity | `crates/commol-inference/src/profiles/` (`tracer.rs`, `threshold.rs`, `verdict.rs`, `resets.rs`), `crates/commol-calibration/src/subproblem.rs` | `crates/commol-calibration/src/optimization.rs` (`optimize_subproblem`), `py-commol/src/profiles.rs`, `py-commol/commol/context/inference/profiles.py`, `api/inference/profiles.py`, `docs/guide/uncertainty/profiles.md` |
| Change sensitivities or identifiability | `crates/commol-inference/src/sensitivity/`, `crates/commol-inference/src/identifiability/` | `crates/commol-core/src/simulation_engine.rs` (`output_accuracy`), `py-commol/src/identifiability.rs`, `py-commol/commol/context/inference/identifiability.py`, `api/inference/identifiability.py`, `docs/guide/uncertainty/identifiability.md` |
| Change a bootstrap, the jackknife, blocks or residual noise | `crates/commol-inference/src/resampling/` | `crates/commol-calibration/src/calibration_problem.rs` (`with_data`), `py-commol/src/inference.rs` (`bootstrap`, `jackknife`), `py-commol/commol/context/inference/resampling.py`, `api/inference/resampling.py`, `docs/guide/uncertainty/resampling.md` |
| Change a sequential Monte Carlo method (the sampler, a target, a move kernel) or a prior | `crates/commol-inference/src/smc/` (`sampler.rs`, `gibbs.rs`, `abc.rs`, `kernel.rs`); `distribution.rs`, `priors.rs`, `transform.rs` | `py-commol/src/smc.rs`, `py-commol/commol/context/inference/smc.py`, `context/inference/distributions.py`, `api/inference/smc.py`, `docs/guide/uncertainty/smc.md` |
| Change a conformal method, its scores or its quantile | `crates/commol-inference/src/conformal/` | `py-commol/src/conformal.rs`, `py-commol/commol/context/inference/conformal.py`, `api/inference/conformal.py`, `docs/guide/uncertainty/conformal.md` |
| Change a sensitivity method, a design or a quantity of interest | `crates/commol-inference/src/global_sensitivity/`; `design.rs`, `crates/commol-inference/src/quantity.rs`, `batch.rs` | `crates/commol-calibration/src/forward.rs` (`unobserved`), `py-commol/src/sensitivity.rs`, `py-commol/src/quantity.rs`, `py-commol/commol/context/inference/quantities.py`, `context/inference/global_sensitivity.py`, `api/sensitivity.py`, `api/inference/global_sensitivity.py`, `api/results.py` (`quantities`), `docs/guide/uncertainty/sensitivity.md` |
| Change a local approximation (the curvature, Laplace, the sandwich, the Laplace mixture) or PSIS | `crates/commol-inference/src/local/`; `psis.rs`, `importance.rs` | `crates/commol-calibration/src/likelihood/density.rs` (`information`, `scores`), `discrepancy.rs` (`term_derivatives`), `py-commol/src/local.rs`, `py-commol/commol/context/inference/local.py`, `api/inference/local.py`, `docs/guide/uncertainty/local.md` |
| Change the parameter sample, the predictive, the scores or the quantile | `crates/commol-inference/src/sample.rs`, `predictive.rs`, `scores.rs`, `statistics.rs` | `py-commol/commol/context/inference/sample.py`, `api/inference/`, `docs/guide/uncertainty/index.md` |
| Change the Python API | `py-commol/commol/api/` | `docs/api/`, the guide page of that feature |
| Expose a Rust function to Python | `py-commol/src/`; `py-commol/src/engine.rs` for a function that takes an engine | `py-commol/commol/commol_rs/_commol_rs.pyi`, `py-commol/commol/context/_model/engine_state.py` (the only Python caller of the engine queries) |
| Change how a model is described | `py-commol/commol/context/description.py` (blocks), `context/_model/describe.py` (layouts) | `ModelStructure.describe`, `context/_model/aggregate_description.py`, `api/description.py` (renderers), `docs/guide/describing-models.md` |
| Change reports | `py-commol/commol/api/reporting.py` (the blocks) | the methods that call it (`api/results.py`: `SimulationResults.report`, `SimulationReplicates.report`; `api/inference/predictive.py`: `Predictive.report`; `context/model.py`: `Model.applied_parameters`; `context/inference/identifiability.py`: `IdentifiabilityReport.report`), `ModelStructure.report_sections`, `context/_model/report.py`, `docs/codebase/decisions/results.md` (Reports) |
<!-- --8<-- [end:tasks] -->

## Documentation map

| You are… | Read |
|---|---|
| Using commol | `docs/getting-started/`, `docs/guide/`, `docs/api/` |
| Changing commol | `docs/codebase/` (start at [the index](docs/codebase/index.md)) |
| Releasing or merging | `docs/development/` |
