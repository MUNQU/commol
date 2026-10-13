# Conventions

[Back to Codebase Guide](index.md)

How code, comments, documentation and commits are written in Commol. The tools
check some of these rules. The rest depend on the author and the reviewer.

## Quick reference

| Area | Rule | Checked by |
|---|---|---|
| Python style | Ruff, line length 88 | `ruff check`, `ruff format` (pre-commit, CI) |
| Python types | Annotations on every signature; never `Any` | `ty check` (pre-commit, CI) |
| Python dead code | No unused code; a name used only in a string annotation goes in `py-commol/.vulture_allowlist.py` | `vulture` (pre-commit, CI) |
| Rust style | `cargo fmt`; no Clippy warnings | pre-commit, CI (`-D warnings`) |
| Rust docs | Every public item documented (`missing_docs`); no broken doc links | clippy and `cargo doc` (CI) |
| Docstrings | NumPy style on every public module, class, function and argument | `ruff` rules `D1` and `D417` (pre-commit, CI) |
| Comments | Describe the code as it is | Review |
| Documentation | Mandatory with every change: [definition of done](#documentation-definition-of-done) | CI and review |
| Doc examples | Abstract names; every block runs | `pytest -m docs` (CI) and review |
| Tests | One package per guide page; shared factories; every test checks a result | Review; pytest 10 deprecations fail the run (`filterwarnings`) |
| Commits | `type(scope): description` | Review |

## Python

- Every function and method signature has type hints. Never use `Any`: use a
  precise type, a `TypeVar`, a `Protocol` or a union.
- Public functions and methods have a NumPy-style docstring: a one-line
  summary, then `Parameters`, `Returns` and `Raises` as needed. Private helpers
  may use a one-line docstring.
- Context data structures are Pydantic v2 models.
- `ModelBuilder` methods return `Self`, so calls chain.
- Log with the standard `logging` module.

## Rust

- Edition 2024.
- Public items have rustdoc comments, and a new module starts with a `//!`
  summary.
- Hot paths avoid allocation: reuse buffers, access by index, keep evaluation
  contexts alive.

## Comments and docstrings

Describe the code as it is now. Never describe how it came to be.

| Do not write | Write instead |
|---|---|
| "This replaces the old prefix matching" | "Matches compartments by label." |
| "Previously this was computed in Python" | Nothing: the history is in git. |
| "This is what makes calibration work" | The rule the caller must satisfy. |
| "Kept in sync with `builder.rs`" | A test that fails when they differ. |

Design reasons and invariants that cross components go to
[Design decisions](design-decisions.md), not to the source.

## Documentation definition of done

!!! warning "Mandatory"
    **Every change to this repository ships with the documentation it needs, in
    the agreed format and at the agreed quality.** A change without it is not
    done, and its pull request is not merged. This applies to people and to
    agents, to features, fixes, refactors, dependency updates and CI changes.

### 1. Update the docs the change affects

| If the change… | Update |
|---|---|
| Adds or changes a public Python name, argument, default or error | Its docstring; a `::: path` entry in `docs/api/` for a new name in `commol.__all__`; the guide page that uses it |
| Changes model behavior (expansion, rates, initial conditions, simulation) | The guide page in `docs/guide/`; [Model structure](../advanced/model-structure.md) if a type changes |
| Changes calibration or ensembles | `guide/calibration.md` or `guide/probabilistic-calibration.md` |
| Changes an expression function or variable | `guide/mathematical-expressions.md` |
| Moves code, adds a module, or changes who owns what | [Repository map](repository-map.md); `AGENTS.md` if a "If you need to…" row changes |
| Adds or changes a rule that is not obvious from the code | [Design decisions](design-decisions.md), with the test that guards it |
| Changes a workflow, a command, CI or the release | `AGENTS.md`; `docs/development/` |
| Is visible to users | `docs/about/changelog.md`, under `[Unreleased]` |
| Adds or changes a public Rust item | Its rustdoc comment |

When in doubt, search `docs/` and `AGENTS.md` for the changed name and update
every match.

### 2. Use the agreed format

**Pages**

- One page, one job. The title says the outcome.
- Start with one paragraph (what it is, who it is for), then a quick path, then
  details, then a checklist or the next step.
- Prefer tables, checklists and short sections to long prose.
- Start every page with a link back to its index, and end it with
  previous/next links when it belongs to a sequence.

**Text**

- Simple technical English: short sentences, one idea per sentence, active
  voice, the answer first.
- Every claim about the code names the file, function or test behind it.
- Describe the present. History belongs in git and in the changelog.

**Examples**

- Use abstract names only: bins `A`, `B`, `C`; categories `group1`, `cat1`;
  parameters `k1`, `alpha`. No names from epidemiology, chemistry or any other
  field.
- Every Python block runs, top to bottom within its page, in one namespace and
  from a temporary directory (`py-commol/tests/docs/test_examples.py`). Define
  what a block needs in an earlier block of the same page.
- A block that cannot run alone (a signature, a fragment) is marked on the line
  just before its fence, with a reason:

    ```markdown
    <!-- doc-test: skip: shows only the signature -->
    ```

  Use the marker only when the block cannot run; never to hide a failing
  example.

**Code documentation**

- Python: a NumPy-style docstring on every public function, method and class
  (`Parameters`, `Returns`, `Raises`).
- Rust: a rustdoc comment on every public item; a `//!` summary at the top of
  every new module.
- Comments follow [Comments and docstrings](#comments-and-docstrings).

### 3. Pass the quality checks

| Check | Command | Enforced by |
|---|---|---|
| The site builds with no broken internal link or anchor | `mkdocs build --strict` (with `validation:` in `mkdocs.yaml`) | CI (Documentation) |
| External links answer | lychee (`.github/workflows/links.yaml`) | CI on documentation changes, weekly; not a required check |
| The documentation examples run | `pytest -m docs` (from `py-commol/`) | CI (Code Quality) |
| Every name in `commol.__all__` has an API entry (`::: …` under `docs/api/`) | `pytest -m docs` (`tests/docs/test_reference.py`) | CI (Code Quality) |
| Public Python modules, classes, functions and arguments have docstrings | `ruff check .` (rules `D1`, `D417`, NumPy convention) | pre-commit, CI |
| Public Rust items have docs, without broken links | `#![warn(missing_docs)]` in every crate (denied by `clippy -D warnings`); `RUSTDOCFLAGS="-D warnings" cargo doc` | pre-commit (clippy), CI |
| The table in step 1 is complete for this change | — | Pull request checklist and review |
| The format in step 2 is followed | — | Review |

The pull request template repeats this checklist. A reviewer who finds a gap
asks for it before approving.

### Writing for people and agents

The same pages serve both. An agent reads the Markdown source by path, so:

- keep each page self-contained, with its links;
- name files and functions exactly, in code format;
- put the rule before the explanation.

## Tests

Where a Python test goes, how it builds its model, and what it must check.
The tests live in `py-commol/tests/`.

### Layout: one package per guide page

A test goes in the package of the guide page that documents the behavior it
checks, so "where is the test of X" has the same answer as "where is the guide
of X". A pure-Python helper (`utils/security.py`, `api/time_scale.py`,
`api/probabilistic/`) goes in the package of its feature. The one exception is
`reference_models/`: it checks published models across features (see
[Reference models](#reference-models)).

| Package | Tests of | Guide page |
|---|---|---|
| `building/` | The builder, parameters, stratifications, reserved names, save and load | [Building Models](../guide/building-models.md) |
| `transitions/` | Flows, `$compartment`, stratified rates, impulses | [Transitions](../guide/transitions.md) |
| `rates/` | Formulas, functions, variables, expression security | [Mathematical Expressions](../guide/mathematical-expressions.md) |
| `time_patterns/` | `TimePattern` formulas, runs and schedules | [Time Patterns](../guide/time-patterns.md) |
| `initial_conditions/` | Bin, stratification and compartment fractions; the initial state | [Building Models](../guide/building-models.md#initial-conditions) |
| `simulation/` | Runs, `run_at`, differential models, outputs, windowed totals, `TimeScale` | [Running Simulations](../guide/simulations.md) |
| `units/` | Unit checks and `print_equations` | [Units and Equations](../guide/units-and-equations.md) |
| `calibration/` | One file per calibration page: `test_problem.py`, `test_optimizers.py`, `test_constraints.py`, `test_observations.py`, `test_initial_conditions.py`, `test_write_back.py` | [Model Calibration](../guide/calibration.md) and its sub-pages |
| `probabilistic/` | Ensembles, their selection and intervals | [Probabilistic Calibration](../guide/probabilistic-calibration.md) |
| `reporting/` | Reports | [Reports](../guide/reports.md) |
| `plotting/` | `SimulationPlotter` | [Running Simulations](../guide/simulations.md#plotting) |
| `docs/` | The documentation examples and the API coverage (marker `docs`) | — |
| `reference_models/` | Published models against their stated results, one module per model family | — (across features) |
| `support/` | No tests: the code the tests share | — |

Keep a test module under about 600 lines. When it grows past that, split it by
topic inside its package.

### Models: use the factories

- Build models with the factories of `tests/support/models.py`:
  `sir_builder`, `seir_builder` and `a_to_b_builder`, and the special models
  `override_builder`, `shared_category_builder` and
  `conditional_pathway_builder`. A keyword overrides one part of the default
  model, so a test names only what it changes. `build` builds for difference
  equations; `build_ode` builds for differential equations, with the tight
  solver settings `TIGHT` by default; `run` builds and simulates.
- Build a model by hand only when one test needs it and no factory keyword
  expresses it. A model that a second file needs goes to `support/`.
- Fixtures: `sir_model` and `sir_simulation` (`tests/conftest.py`); `truth`
  and `truth_run` (`tests/calibration/conftest.py`). A fixture is
  function-scoped unless its cost is measured; a wider scope never holds an
  object a test can change. A class-scoped fixture is never an instance
  method (removed in pytest 10).

### Every test checks a result

A test compares a result with an oracle:

| Oracle | Example |
|---|---|
| A value computed by hand in the test | `reference_run` (`tests/support/reference.py`) evaluates the flows in plain Python, as the engine does: every flow from the state at the start of the step, no clamping. `tests/rates/test_formulas.py` compares whole trajectories with it. |
| An analytic value | A death rate `d` on every bin gives `N(t) = N(0) (1 - d)^t` (`tests/transitions/test_compartment_placeholder.py`). |
| An invariant | Conservation of the population (`assert_population_conserved`); two paths that must agree, such as a bin id and the explicit sum of its compartments. |
| What a stated result predicts about a run | `tests/support/oracles.py`: `final_size` solves the final size relation, `assert_threshold` checks that an infection grows at a reproduction number of `1.05` and dies out at `0.95`, `settled` reads a limit once doubling the horizon no longer moves it. |

"It runs" is not a result. A test whose only check is that no exception
occurs is allowed only when the exception is its subject, such as a validator
that accepts a valid input.

### Tolerances

- `assert_close(actual, expected, rel=…, abs=…)` (`tests/support/assertions.py`)
  needs a stated tolerance, and reports the key and the index of the first
  difference.
- `ENGINE_REL = 1e-9` compares the engine with a hand computation of the same
  arithmetic: only the order of the floating-point operations differs.
- `ODE_REL = 1e-6` compares a differential model with a closed form, a final
  size or an equilibrium. `flow_model` solves with `FINE` (`TIGHT` with
  `atol = 1e-12`), so the error of a value near zero stays below its own size.
- `pytest.approx` with its default (relative `1e-6`) suits a single value
  computed by hand.
- An optimizer result states the tolerance its convergence allows, for example
  `abs=1e-5` for Nelder-Mead on noise-free data.

### Reference models

`tests/reference_models/` shows that commol can state published models and
computes their stated results. A module holds one model family and is named
after its epidemiology (`test_pulse_vaccination.py`); a test checks one stated
result (`test_the_fixed_point_starts_the_periodic_orbit`).

- **Attribution.** The module docstring has a `References` section in the NumPy
  format with the original papers of the model and of its parameter values
  (authors, year, title, journal, DOI). A value or a result taken from a paper
  cites it in a comment next to the value. A model with no primary source is
  not added.
- **Independent oracle.** The expected value comes from the stated result,
  computed with `math` or `numpy`: a closed form, a root of the final size
  relation, a spectral radius. A test never compares commol with commol. A
  quantity that commol does not output, such as a reproduction number, is
  checked through what it predicts about a run (`tests/support/oracles.py`).
- **Equations.** To check that a model has the stated equations, `increments`
  reads its right-hand side at one state: a difference-equation step adds the
  flows of the state it starts from, and both engines compile the same flows.
- **Tolerances.** `ENGINE_REL` for a discrete model, `ODE_REL` for a
  differential one; an asymptotic result is read with `settled`, and the test
  states its horizon.
- **Gaps.** A model that commol cannot state, or computes wrong, gets a test
  marked `xfail(strict=True, reason="F<n>: …")` that names a finding of the
  task that added it, so a fix makes the test pass and forces the marker out.

To add a test, write the model with `Flow` and `flow_model` (`tests/support/flows.py`)
from the initial value of each bin, compute the stated result in the test,
and cite the papers.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):
`<type>(<scope>): <description>`.

- The scope is required.
- The description is one short phrase, with no period.
- No body, and no co-author or AI attribution lines.
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`,
  `ci`, `chore`.

```text
feat(transitions): add time-varying rates
fix(simulation): correct population calculation
docs(codebase): add repository map
```

See [Development workflow](../development/workflow.md) for branches and merge
methods.

## Navigation

Previous: [Design decisions](design-decisions.md) | Next: [Maintainer playbook](maintainer-playbook.md)
