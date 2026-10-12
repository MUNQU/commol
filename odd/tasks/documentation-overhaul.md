# Feature: Documentation overhaul

Goal: documentation as complete, correct and well referenced as gentle-ai's,
useful to people and to agents, and a mandatory rule that keeps it that way.
Branch: `docs/documentation-overhaul`, one PR (#98) merged at the end (see Delivery).
Engram mirror: topic `odd/documentation-overhaul/tasks` (project `commol`).
Runners: `pytest` (from `py-commol/`), `cargo test -p commol-core -p commol-difference -p commol-calibration`, `mkdocs build --strict` (from the root).

## Specs

### S1 — Useful to people and agents

User: "Quiero una documentación que sea como la de gentle-ai, es decir, que pueda ser útil tanto a humanos como a agentes."

- Every reader has one entry point: users the site, maintainers `docs/codebase/`,
  agents `AGENTS.md`, user agents `llms.txt`.
- Every page has one job and follows the agreed page shape (S4).
- Check: the README has a "Documentation" table (where to go / what you will
  find); every docs page links back to its index.

### S2 — As complete as possible

User: "Quiero que la documentación sea lo más completa posible. Básate en la documentación de gentle-ai para evaluar cuál es la calidad que busco"

- Every name in `commol.__all__` has an API reference entry (baseline 2026-10-12: 20 of 32).
- Every public Python function, method and class has a NumPy-style docstring
  (baseline: 23 of 316 missing).
- Every public Rust item has rustdoc (baseline: about 150 missing under
  `-W missing_docs`).
- The user guide covers every public option: each argument, default, and error.
- Check: the audit script reports zero gaps; `missing_docs` is denied in CI.

### S3 — Correct and well referenced

User: "¿Has comprobado que toda la documentación es correcta, está bien referenciada y que todo el código está bien documentado?"

- Every Python code block in `docs/` runs in CI, top to bottom per page, unless
  it is marked as a non-runnable fragment with a reason (baseline: 82 of 190
  blocks fail).
- Every internal link and anchor resolves (`mkdocs build --strict`; baseline:
  `#time-series-parameters` is broken). External links are checked.
- Every claim in `docs/codebase/` names the file, function or test behind it.
- Check: the doc-example job and `mkdocs build --strict` pass in CI.

### S4 — Mandatory documentation for every change

User: "Añade también una obligatoriedad de que cualquier nueva modificación del repositorio debe de complimentar la documentación necesaria, con el formato acordado y la calidad adecuada."

- A written, mandatory "documentation definition of done" in
  `docs/codebase/conventions.md`:
  - which docs each type of change must update;
  - the agreed format: page shape, simple technical English, abstract names in
    examples, tables and checklists, claims that name their source, NumPy
    docstrings, rustdoc;
  - the quality checks every change must pass.
- `AGENTS.md` states the rule for agents; `contributing.md` and the maintainer
  playbook state it for people.
- A pull request template asks for the documentation checklist.
- CI enforces what a tool can check: strict docs build (S3), doc examples (S3),
  docstring coverage and `missing_docs` (S2).
- Check: a pull request without the required docs fails CI or the review
  checklist.

### S5 — User guide at the gentle-ai level

- Long pages are split so each has one job: `guide/calibration.md` (1411 lines)
  and `guide/building-models.md` (1056 lines).
- Every guide page starts with a quick path, then details, then a checklist or
  next step.
- Diagrams (mermaid) for the model lifecycle and calibration flow.
- A version note on every page (`dev` describes unreleased code).

### S6 — Machine-readable documentation

- The site publishes `llms.txt` and `llms-full.txt`, versioned with mike.
- Check: both files exist in the built site and include the API reference.

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S4 | Documentation definition of done: `conventions.md`, `AGENTS.md`, `contributing.md`, maintainer playbook, PR template | direct | b2db454 |
| T2 | S3, S4 | Doc-example test harness and CI job; mark non-runnable fragments | direct | this commit (see Evidence) |
| T3 | S3 | Fix examples: `quickstart.md`, `core-concepts.md`, `index.md` | direct | this commit (see Evidence) |
| T4 | S3 | Fix examples: `building-models.md`, `mathematical-expressions.md`, `time-patterns.md` | direct | this commit (see Evidence) |
| T5 | S3 | Fix examples: `simulations.md` | direct | this commit (see Evidence) |
| T6 | S3 | Fix examples: `calibration.md`, `probabilistic-calibration.md`; investigate the engine error `Parameter index '2' not found` (calibration.md L934) | direct | this commit (see Evidence) |
| T7 | S3 | Fix the broken anchor; add an external link check | direct | pending |
| T8 | S2 | API reference pages for the 12 missing public names | direct | pending |
| T9 | S2, S4 | Python docstrings for the 23 missing definitions; docstring coverage gate in CI | direct | pending |
| T10 | S2, S4 | Rust docs for the public items; deny `missing_docs` in CI | direct | pending |
| T11 | S5 | Split and restructure `calibration.md` and `building-models.md`; quick paths, version notes, back links | direct | split done in T4 and T6; version notes in T12 |
| T12 | S1, S5 | README documentation table; home page entry points; mermaid diagrams | direct | pending |
| T13 | S6 | `llms.txt` and `llms-full.txt` in the site build | direct | pending |
| T14 | S1–S6 | Final audit: re-run every baseline measurement and record zero gaps | direct | pending |

## Delivery

- Strategy: `single-pr`, at the user's request (L6). Every task is one or more
  work-unit commits on `docs/documentation-overhaul`; PR #98 carries all of them
  and is merged into `develop` once T14 passes.
- Order: T1 first (the rule applies to the rest of this work), then T2 (no
  example fix can be verified before the harness exists), then T3–T14.
- Every task closes with a work-unit commit and its evidence below.

## Evidence

- Baseline audit (2026-10-12): `run_doc_blocks.py` per page →
  index 1/1, quickstart 3/9, core-concepts 3/21, building-models 32/44,
  mathematical-expressions 16/25, time-patterns 15/19, simulations 0/21,
  calibration 24/33, probabilistic-calibration 2/5, examples 12/12.
- Real API mismatches found by the baseline: `add_bin(description=)`
  (building-models L54), `Population` validation (building-models L714),
  positional Pydantic init (calibration L222), `plot(scale_values=)`
  (calibration L611), engine `RuntimeError` (calibration L934),
  `CalibrationParameter` validation (calibration L1024).

- T1 (b2db454): `docs/codebase/conventions.md#documentation-definition-of-done`
  states the mandatory rule, the change → docs table, the agreed format and the
  quality checks. `AGENTS.md` states it before "Before you change code" and in
  its rules table; `contributing.md`, the maintainer playbook and the new
  `.github/PULL_REQUEST_TEMPLATE.md` require it. `mkdocs build --strict` passes,
  the anchor exists in the built site, and every relative link in `AGENTS.md`
  resolves. The CI gates are added by T2, T9 and T10, and the table in
  `conventions.md` marks them as "being added" until then.

- T2: `py-commol/tests/test_docs_examples.py` runs every page with `python`
  blocks as one script (one namespace, temporary working directory, Agg
  backend, `plt.show` disabled), including indented fences; `<!-- doc-test:
  skip: <reason> -->` skips a fragment and requires a reason. Marker `docs`
  registered in `pyproject.toml`. Code Quality runs it with the full suite; the
  Build matrix runs `pytest -m "not docs"`. First run: 5 pages pass, 10 fail
  (fixed by T3–T6). The rule, `AGENTS.md` and `workflow.md` document the
  harness and the marker.

- T3: `quickstart.md` and `core-concepts.md` rewritten in the agreed shape (back
  link, quick path or concept table, runnable examples, tables, next steps) with
  abstract names only; `index.md` already passed. `pytest -m docs` passes for
  the three pages; every printed value claimed in a comment was checked against
  a run (expansion names, head counts, `subgroup_population`,
  `get_conditioning_categories`, `$compartment` ids, `print_equations` showing
  `k2 * B`, `TimeScale` values).
- Finding F1 (code, not fixed here): `check_unit_consistency` infers the time
  unit only from known time units (`UnitChecker._infer_time_unit`) and defaults
  to `day`; a parameter in `1/step` therefore always fails with "expected
  `<bin_unit>/day`". Per-step units are natural in Commol (a model has no
  physical time). The docs use `1/day`; the user decides on a code fix.

- T4 (also the `building-models.md` part of T11): `building-models.md` split
  into `building-models.md` (builder, bins, parameters including time series,
  stratifications, accumulators, initial conditions, build, save/load),
  `transitions.md` and `units-and-equations.md`; `mathematical-expressions.md`
  and `time-patterns.md` rewritten. Every argument table comes from
  `inspect.signature`; every printed value and every rule was checked by running
  it (time-series modes, validation timing, `$compartment` ids, category
  transitions, `to`, `per_compartment`, stratified-rate selection, build errors,
  unit errors, `print_equations` text and LaTeX output, every function and
  operator, subpopulation variables, every `TimePattern` and schedule).
  `mkdocs.yaml` now validates anchors (`validation:` block), and the moved
  anchors were fixed. Wrong claims removed: a working `condition` on
  transitions, `add_bin(description=)`, `add_transition(description=)`,
  `(B > 100)` as a factor, the LaTeX output format, a 500-character limit,
  performance claims.
- Finding F2 (code): `Transition.condition` / `create_condition` are stored but
  never evaluated by the engine (a condition that is never true still moves
  population). The docs now say so; implement or remove is the user's call.
- Finding F3 (code): `ModelBuilder.reset()` does not clear accumulators.
- Finding F4 (code): only bin ids are rejected as parameter ids; `N`, `N_*`,
  `step`, `t`, `pi`, `e` and compartment names such as `A_g1` are accepted
  silently.
- Finding F5 (code, user-visible): in transition rates, `pi` and `e` outside a
  function argument fail at run time ("Variable 'pi' not found"):
  `crates/commol-difference/src/simulation.rs::resolve_slot` looks them up in
  the context, but they are JIT constants (`jit/codegen.rs::load_variable`).
  Documented as a known issue in `mathematical-expressions.md`.
- Finding F6 (code): an `N_…` variable with suffixes out of declaration order
  (`N_t1_g1`) evaluates to 0 instead of failing; `<bin>_…` fails correctly.
  `%` passes the security check but makes the whole rate a parameter name
  (`MathExpression::from_string` does not list it as an operator).

- T5: `simulations.md` rewritten: `Simulation` member table from
  `inspect`, output formats, accumulators, stratified outputs through
  `outputs_for` / `group_outputs` / `total_series`, category totals through
  `Model.initial_state()` labels (no name matching), windowed totals, plotting
  with the real `PlotConfig` fields. Every printed value checked by a run.
  Wrong claims removed: `SeabornStyleConfig` (does not exist), a run that
  restarts from the initial state, name-prefix filtering by category.
- Finding F7 (code): `SimulationPlotter.plot_series(bins=...)` takes output
  names; a bin id of a stratified model raises a confusing `KeyError` ("Missing
  result series for aggregate components"). The `plot_series` docstring also
  omits `step_to_label`, `tick_every` and `x_label` (fixed in T9).
- Finding F8 (code, user-visible): `Simulation.run` continues from the engine's
  current state; a second call does not start from the initial state
  (`DifferenceEquations::run` never resets and `Simulation._run_raw` does not
  call `reset`). Calibration is not affected: it resets before every
  evaluation. Documented with `simulation.engine.reset()`; whether `run` should
  reset is the user's call.

- T6 (also the `calibration.md` part of T11): `calibration.md` split into
  `calibration.md` (workflow, problem, observations and unknowns tables,
  losses, results, write-back), `calibration-initial-conditions.md`,
  `calibration-observations.md`, `calibration-optimizers.md`,
  `calibration-constraints.md` and `reports.md`; `probabilistic-calibration.md`
  rewritten with complete configuration tables from `model_fields` and a fast
  runnable ensemble (about 5 s). Every calibration example generates its data
  from a known model and prints the recovered values (checked: 0.4 / 0.1, 0.02,
  0.01, 0.3). The navigation groups the guide into Building Models and
  Calibration. Mermaid diagrams enabled in `mkdocs.yaml`. The design document's
  type sketch is marked as a fragment. All docs examples run in 6.6 s.
  Wrong claims removed: constraint penalty `Σ w max(0, -c)²` (it replaces the
  loss), `plot_series(scale_values=)`, `SeabornStyleConfig`, `sse` ignoring
  weights, the plain-weighted formula.
- The engine error `Parameter index '2' not found` (old calibration.md L934) is
  Finding F9 (code, user-visible): `Simulation` snapshots the model when it is
  created. Built while a parameter is `None`, it fails after
  `update_parameters` with that obscure `RuntimeError`; built before a value
  changes, it silently keeps the old value. Documented as a warning with the
  rule "create a new Simulation after changing the model".
- Finding F10 (code): an `initial_condition` id naming a category of a
  non-binary stratification fails with "not found" instead of saying that only
  binary stratifications can be calibrated as a split.
- Finding F11 (code): `LossConfig::SumSquaredError` and `WeightedSSE` compute
  the same `Σ (weight × r)²`; the weight is squared. Documented as is.
- Note: Nelder-Mead from the midpoint can stop at a bound and still report
  `converged=True` (seen on a one-unknown problem); documented with the advice
  to check bounds and give `initial_guess`.

## Log

- L1 (user, 2026-10-12): "¿Qué habría que hacer para mejorar la documentación del proyecto? Quiero una documentación que sea como la de gentle-ai, es decir, que pueda ser útil tanto a humanos como a agentes. Antes de implementar nada, dime qué harías."
- L2 (user): "¿Has comprobado que toda la documentación es correcta, está bien referenciada y que todo el código está bien documentado?"
- L3 (user): "Quiero que la documentación sea lo más completa posible. Básate en la documentación de gentle-ai para evaluar cuál es la calidad que busco"
- L4 (user): "Genera un documento odd para realizar la tarea. Añade también una obligatoriedad de que cualquier nueva modificación del repositorio debe de complimentar la documentación necesaria, con el formato acordado y la calidad adecuada."
- L5: phases 1–2 of the earlier plan are done: `AGENTS.md` and `docs/codebase/` (#96), repository cleanup (#97).
- L6 (user): "¿puedes realizar todo el ODD e integrar al final?"
