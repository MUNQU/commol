## Summary

<!-- What changes, and why. Put the outcome first. -->

Closes #<!-- the issue this delivers; delete the line if there is none -->

- [ ] Every open issue this change affects is updated (body and a comment), and work it leaves undone is in an open issue: <!-- the issues that `python .github/scripts/affected.py --closes <number>` lists, each updated or why it needs no update; or "none" if it lists none -->

## Testing

<!-- Each requirement of the issue (R1, R2, …) and the tests that check it,
then the commands you ran and their results. -->

- [ ] Rust: `cargo clippy --workspace --all-targets --all-features -- -D warnings` and the Rust tests pass
- [ ] Python: `maturin develop --release --features test-kind`, then `pytest`, `ruff` and `ty` pass

## Documentation (mandatory)

Every change ships with its documentation. See the
[documentation definition of done](https://github.com/MUNQU/commol/blob/main/docs/codebase/conventions.md#documentation-definition-of-done).

- [ ] I updated every doc the change affects (step 1 of the definition of done), or this change affects no docs because: <!-- reason -->
- [ ] Docstrings (Python) and rustdoc (Rust) cover every new or changed public item
- [ ] Examples use abstract names and run top to bottom within their page
- [ ] `AGENTS.md` and `docs/codebase/` still describe the code (commands, ownership, design decisions, the diagrams of `architecture.md`)
- [ ] User-visible changes are in `docs/about/changelog.md` under `[Unreleased]`
- [ ] `mkdocs build --strict` passes
