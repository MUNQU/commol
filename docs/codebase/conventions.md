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
| Doc examples | Abstract names only | Review |
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

## Documentation

- **A change is not done until its documentation is updated.** Check every page
  that mentions the changed concept: the guide page of the feature, the API
  page, [Model structure](../advanced/model-structure.md), and, for maintainers,
  the pages of this guide.
- **Examples are domain-agnostic.** Use abstract names: bins `A`, `B`, `C`;
  categories `group1`, `cat1`; parameters `k1`, `alpha`. Do not use names from
  epidemiology, chemistry or any other field.
- **Write for people and agents.** Put the answer first. Prefer tables,
  checklists and short sections to long prose. Name the file or function behind
  each claim, so a reader can check it.
- **Write in simple technical English**: short sentences, one idea per
  sentence, active voice.

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
