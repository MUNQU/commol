# Contributing Guidelines

[Back to Codebase Guide](../codebase/index.md)

Thank you for considering contributing to Commol! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

Before creating a bug report:

- Check the [issue tracker](https://github.com/MUNQU/commol/issues) for existing reports
- Verify the bug exists in the latest version

When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, Commol version)
- **Code samples** or test cases that demonstrate the problem

### Suggesting Features

Feature suggestions are welcome! Please:

- Check if the feature has already been requested
- Provide a clear use case and rationale
- Describe the desired API or interface
- Consider implementation complexity vs benefit

### Pull Requests

1. **Keep PRs focused**: One feature/fix per PR
2. **Write tests**: All new code should have tests
3. **Update docs (mandatory)**: Every change ships with its documentation. Follow the [documentation definition of done](../codebase/conventions.md#documentation-definition-of-done)
4. **Follow conventions**: Use the project's code style
5. **Be descriptive**: Write clear commit messages and PR descriptions

## Development Guidelines

The rules for code, comments, documentation and commits are in one place:
[Conventions](../codebase/conventions.md). In short:

| Area | Rule |
|---|---|
| Python | Ruff (line length 88), type hints on every signature and never `Any`, NumPy docstrings on every public item |
| Rust | `cargo fmt`, no Clippy warnings, rustdoc on every public item |
| Tests | pytest for Python, laid out as in [Tests](../codebase/conventions.md#tests); `#[cfg(test)]` modules and `tests/` for Rust; test both success and error cases |
| Documentation | Mandatory with every change: the [definition of done](../codebase/conventions.md#documentation-definition-of-done) |
| Commits | `type(scope): description`, one short phrase, no body; see [Commits](../codebase/conventions.md#commits) |

Run the checks before you open a pull request:

```bash
# Python, from py-commol/ with the virtual environment active
ruff check . ../.github/scripts && ruff format --check . ../.github/scripts && ty check commol
vulture commol .vulture_allowlist.py --min-confidence 80 && pytest

# Rust, from the repository root
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p commol-core -p commol-model -p commol-difference -p commol-differential -p commol-calibration -p commol-random -p commol-stochastic -p commol
```

The pre-commit hooks run most of them on every commit; see
[Pre-commit Hooks](workflow.md#pre-commit-hooks).

## Code Review Process

### For Contributors

1. **Respond to feedback** promptly and professionally
2. **Make requested changes** or discuss alternatives
3. **Keep the PR updated** with the target branch
4. **Test locally** before requesting re-review

### For Reviewers

1. **Be constructive** and respectful
2. **Explain reasoning** for requested changes
3. **Acknowledge good work** when you see it
4. **Test the changes** when possible

## Release Process

See [Release Process](release.md) for details on versioning and releases.

## Getting Help

- **Documentation**: Check the [docs](https://munqu.github.io/commol/latest/)
- **Questions, bugs and ideas**: Open a [GitHub issue](https://github.com/MUNQU/commol/issues)

## Recognition

Contributors will be recognized in:

- The project's AUTHORS file
- Release notes for significant contributions
- GitHub's contributor graph

Thank you for contributing to Commol! 🎉
