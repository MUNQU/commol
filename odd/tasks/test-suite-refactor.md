# Feature: Refactor of the Python test suite

Goal: refactor `py-commol/tests/` so that a reader finds the tests of a feature
in one place, every test checks a result, and the new tests of
`odd/tasks/reference-model-tests.md` have a structure and shared helpers to
build on. Tests and contributor docs only; no change to `commol/` or `crates/`.
Branch: `test/suite-layout` from `develop`, after PR #99 merges.
Engram mirror: topic `odd/test-suite-refactor/tasks` (project `commol`).
Runners: `maturin develop --release` then `pytest`, `pytest -m docs` and
`pytest --cov=commol` (from `py-commol/`), `ruff check . && ruff format --check .
&& ty check commol`, `mkdocs build --strict` (from the root).

Baseline (2026-10-13, `fix/code-findings` at a887f81): 719 tests pass in 20 s;
26 test files, 14,729 lines.

## Specs

### S1 — One package per feature, as the user guide

- Cause: files mix three criteria. Some follow an API object (`test_model.py`,
  2,087 lines, one `TestModel` class of 1,900 lines), some a feature
  (`test_time_patterns.py`), and two directories nest one file each
  (`test_models_execution/test_difference_equations/test_difference_equations_sir.py`,
  `test_transition_rate/`). Stratification tests live in the SIR execution file.
- Fix (D2): one package per page of `docs/guide/`, so "where is the test of X"
  has the same answer as "where is the guide of X".
- Check: every test file sits in the package of its feature; no test module is
  over ~600 lines; `docs/codebase/repository-map.md` (L54) describes the new
  layout.

### S2 — Shared model factories and assertions

- Cause: there is no `conftest.py`. Tests call `ModelBuilder(` more than 140
  times, and more than 40 private helpers (`_model`, `_problem`,
  `_stratified_model`, …) repeat the same small models in different files.
- Fix: `tests/support/` holds model factories (SIR, SEIR, stratified SIR, A→B)
  with keyword overrides, a run helper, and assertions
  (`assert_population_conserved`, `assert_close` with a stated tolerance).
  `tests/conftest.py` exposes them as fixtures where a fixture helps.
- Check: no two test files define the same model by hand; `rg "def _model"`
  finds only models that are specific to one test.

### S3 — Every test checks a result

- Cause: some tests check only that a call does not fail
  (`TestStratifiedSIRSimulation::test_stratified_sir_runs_successfully`).
  A wrong engine passes them.
- Fix: each such test gets an oracle: conservation of population, an analytic
  value, or a value computed by hand in the test.
- Check: review list in Evidence; no test whose only assertion is "no
  exception" unless an exception is the subject of the test.

### S4 — No duplicate tests and no deprecated pytest use

- Cause: parameter-name validation is tested in
  `test_difference_equations_sir.py::TestParameterNameValidation` and in
  `test_reserved_names.py`. A class-scoped fixture defined as an instance
  method raises `PytestRemovedIn10Warning`; `pyproject.toml` allows
  `pytest<10`, so it fails in pytest 10.
- Fix: one home per behavior; fixtures follow pytest 9 rules.
- Check: `pytest -W error::pytest.PytestRemovedIn10Warning` passes; the
  duplicate pairs and their kept test are listed in Evidence.

### S5 — The refactor does not lose coverage

- Fix: this ODD changes tests only, never `commol/` or `crates/`.
- Check: line coverage of `commol/` (`pytest --cov=commol`) is not lower than
  the baseline; every removed test names the kept test that covers it
  (Evidence).

### S6 — Documentation

Follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done):
`docs/codebase/repository-map.md` (test layout), a "Tests" section in
`docs/codebase/conventions.md` (packages, factories, the oracle rule,
tolerances), and `AGENTS.md` if a command changes. No changelog entry:
nothing visible to users changes.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | all | Order and number of PRs? | The refactor is its own PR, first; the ODE engine (`ode-engine.md`) and the reference tests (`reference-model-tests.md`) follow. Start after PR #99 merges. | One PR with the reference tests: a reviewer cannot check that the refactor lost no test when many new tests arrive with it. Reference tests first: they would use the old helpers and move again in the refactor. Start before #99: #99 edits many test files, so the refactor would conflict. | approved (L2) |
| D2 | S1 | Layout of `tests/`? | One package per guide page (`building/`, `transitions/`, `rates/`, `time_patterns/`, `initial_conditions/`, `simulation/`, `units/`, `calibration/`, `probabilistic/`, `reporting/`, `plotting/`, `docs/`) plus `support/`; `reference_models/` comes with `reference-model-tests.md`. | Mirror the source modules (`api/`, `context/`): behavior crosses modules and the engine owns it (golden rule), so one feature would spread over several packages. Keep flat files: the 2,000-line files stay and the reader still searches. | approved (L3) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S5 | Baseline: test count, `pytest --cov=commol` report, warnings; record in Evidence | direct | |
| T2 | S2 | `tests/support/` factories and assertions, `conftest.py` | direct | |
| T3 | S1, S2 | Move and split files into the D2 packages; replace hand-built models with factories | direct | |
| T4 | S3, S4 | Oracles for smoke tests; merge duplicates; fix deprecated fixtures | direct | |
| T5 | S5, S6 | Coverage and count check against T1; `repository-map.md`, "Tests" section in `conventions.md`; PR | direct | |

## Delivery

- Strategy: one PR to `develop` from `test/suite-layout`, squash-merged
  (`test(suite): …` commits).
- Order: after PR #99 merges; T1 → T5. Then `ode-engine.md`, then
  `reference-model-tests.md`, then `library-benchmark.md`.
- Version impact: none (tests and contributor docs only).

## Evidence

(Filled during implementation.)

## Log

- L1 (user, 2026-10-13): asked for a refactor of the test suite, together with
  the reference-model tests.
- L2 (user, 2026-10-13): D1 → "Dos PRs, refactor primero".
- L3 (user, 2026-10-13): D2 → "Paquetes de la guía", after asking why not one
  package per source module (answer: most tests check engine behavior with no
  Python module to mirror; a feature crosses several modules; modules couple
  tests to internals). Pure-Python helpers (`security.py`, `time_scale.py`,
  `api/probabilistic/`) go in the package of their feature.
- L4 (user, 2026-10-13): the refactor and the reference tests are separate
  deliveries with the ODE engine between them, so they are separate ODDs.
  This ODD holds the former Part 1 of `reference-model-tests.md`.
