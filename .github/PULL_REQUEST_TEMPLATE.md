## Summary

<!-- What changes, and why. Put the outcome first. -->

## Testing

<!-- The commands you ran and their results. -->

- [ ] Rust: `cargo clippy --workspace --all-targets -- -D warnings` and the Rust tests pass
- [ ] Python: `maturin develop --release`, then `pytest`, `ruff` and `ty` pass

## Documentation (mandatory)

Every change ships with its documentation. See the
[documentation definition of done](https://github.com/MUNQU/commol/blob/develop/docs/codebase/conventions.md#documentation-definition-of-done).

- [ ] I updated every doc the change affects (step 1 of the definition of done), or this change affects no docs because: <!-- reason -->
- [ ] Docstrings (Python) and rustdoc (Rust) cover every new or changed public item
- [ ] Examples use abstract names and run top to bottom within their page
- [ ] `AGENTS.md` and `docs/codebase/` still describe the code (commands, ownership, design decisions)
- [ ] User-visible changes are in `docs/about/changelog.md` under `[Unreleased]`
- [ ] `mkdocs build --strict` passes
