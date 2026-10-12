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
| T3 | S3 | Fix examples: `quickstart.md`, `core-concepts.md`, `index.md` | direct | pending |
| T4 | S3 | Fix examples: `building-models.md`, `mathematical-expressions.md`, `time-patterns.md` | direct | pending |
| T5 | S3 | Fix examples: `simulations.md` | direct | pending |
| T6 | S3 | Fix examples: `calibration.md`, `probabilistic-calibration.md`; investigate the engine error `Parameter index '2' not found` (calibration.md L934) | direct | pending |
| T7 | S3 | Fix the broken anchor; add an external link check | direct | pending |
| T8 | S2 | API reference pages for the 12 missing public names | direct | pending |
| T9 | S2, S4 | Python docstrings for the 23 missing definitions; docstring coverage gate in CI | direct | pending |
| T10 | S2, S4 | Rust docs for the public items; deny `missing_docs` in CI | direct | pending |
| T11 | S5 | Split and restructure `calibration.md` and `building-models.md`; quick paths, version notes, back links | direct | pending |
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

## Log

- L1 (user, 2026-10-12): "¿Qué habría que hacer para mejorar la documentación del proyecto? Quiero una documentación que sea como la de gentle-ai, es decir, que pueda ser útil tanto a humanos como a agentes. Antes de implementar nada, dime qué harías."
- L2 (user): "¿Has comprobado que toda la documentación es correcta, está bien referenciada y que todo el código está bien documentado?"
- L3 (user): "Quiero que la documentación sea lo más completa posible. Básate en la documentación de gentle-ai para evaluar cuál es la calidad que busco"
- L4 (user): "Genera un documento odd para realizar la tarea. Añade también una obligatoriedad de que cualquier nueva modificación del repositorio debe de complimentar la documentación necesaria, con el formato acordado y la calidad adecuada."
- L5: phases 1–2 of the earlier plan are done: `AGENTS.md` and `docs/codebase/` (#96), repository cleanup (#97).
- L6 (user): "¿puedes realizar todo el ODD e integrar al final?"
