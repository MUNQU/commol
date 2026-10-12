# Conventions

[Back to Codebase Guide](index.md)

How code, comments, documentation and commits are written in Commol. The tools
check some of these rules. The rest depend on the author and the reviewer.

## Quick reference

| Area | Rule | Checked by |
|---|---|---|
| Python style | Ruff, line length 88 | `ruff check`, `ruff format` (pre-commit, CI) |
| Python types | Annotations on every signature; never `Any` | `ty check` (pre-commit, CI) |
| Python dead code | No unused code | `vulture` (CI) |
| Rust style | `cargo fmt`; no Clippy warnings | pre-commit, CI (`-D warnings`) |
| Docstrings | NumPy style on public functions | Review |
| Comments | Describe the code as it is | Review |
| Documentation | Mandatory with every change: [definition of done](#documentation-definition-of-done) | CI and review |
| Doc examples | Abstract names; every block runs | `pytest -m docs` (CI) and review |
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
  from a temporary directory (`py-commol/tests/test_docs_examples.py`). Define
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
| The site builds with no broken link or anchor | `mkdocs build --strict` | CI |
| The documentation examples run | `pytest -m docs` (from `py-commol/`) | CI (Code Quality) |
| Every name in `commol.__all__` has an API entry (`::: …` under `docs/api/`) | `pytest -m docs` (`tests/test_docs_reference.py`) | CI (Code Quality) |
| Public Python names have docstrings | The docstring coverage check | CI (being added) |
| Public Rust items have docs | `missing_docs` | CI (being added) |
| The table in step 1 is complete for this change | — | Pull request checklist and review |
| The format in step 2 is followed | — | Review |

The pull request template repeats this checklist. A reviewer who finds a gap
asks for it before approving.

### Writing for people and agents

The same pages serve both. An agent reads the Markdown source by path, so:

- keep each page self-contained, with its links;
- name files and functions exactly, in code format;
- put the rule before the explanation.

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
