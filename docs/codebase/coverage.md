# Coverage and mutation testing

[Back to Codebase Guide](index.md)

How much of the code the tests run, the thresholds that keep it there, the
lines that may stay out and why, and how to find the changes no test notices.
The rules of a single test are in [Tests](tests.md).

## Test coverage

Every line of the Python package and of the Rust code, the bindings
included, runs in a test, or it is of a kind this section lists. A test that
only runs a line, and checks nothing, does not count
([Every test checks a result](tests.md#every-test-checks-a-result)). A line that no
input reaches is removed, not tested.

| What | Command | Threshold |
|---|---|---|
| Python (`py-commol/commol/`) | `pytest --cov=commol --cov-fail-under=100`, from `py-commol/` | 100 % of the statements, in the job Python Coverage |
| Rust (`crates/`, `py-commol/src/`) | `.github/scripts/rust_coverage.sh`, from the root, with the virtual environment of `py-commol` active | `RUST_LINES` of the script (98.65 % of the lines), in the job Rust Code Quality |

The bindings have no Rust tests: `pytest` runs them. So the script builds the
Rust tests and the extension with the instrumentation of `cargo llvm-cov`,
runs both, and writes one report. It builds in the debug profile, so
`debug_assert!` runs and each line keeps its own count; the time budget
(`test_slow_tests_are_listed`) stays out, as the job Test of Build times the
release extension. It puts the release extension back when it ends. It needs `cargo-llvm-cov`
(`cargo install --locked cargo-llvm-cov`) and the component
`llvm-tools-preview` (`rustup component add llvm-tools-preview`). Options
after the script go to `cargo llvm-cov report`, such as `--summary-only` or
`--lcov --output-path lcov.info`.

An instrumented binary that runs while `LLVM_PROFILE_FILE` is not set writes
`default_<binary>_<pid>.profraw` in the directory it runs in. Git ignores
`*.profraw` and `*.profdata` (`.gitignore`), and
`tests/suite/test_rules.py::test_no_coverage_profile_is_tracked` fails when
one is tracked.

A threshold is the coverage reached. It rises when coverage rises; a pull
request that lowers it says why.

### What Python leaves out

| Exclusion | Where | Why |
|---|---|---|
| `if TYPE_CHECKING:` | `[tool.coverage.report] exclude_also` in `pyproject.toml` | Imports for the type checker only; Python never runs them. |
| `@overload` | the same | The signatures of an overload; the implementation after them runs. |

A line that no pattern covers and that a test cannot reach takes
`# pragma: no cover: <why>` on the same line. A pattern has a comment above
it that says why. `test_every_coverage_exclusion_has_a_reason`
(`tests/suite/test_rules.py`) checks both.

### What Rust leaves out

`cargo llvm-cov` has no exclusion by line on the stable toolchain, so the
threshold counts every line of a product file, and these are the kinds of
line it allows to stay uncovered:

| Kind | Example | Why no test runs it |
|---|---|---|
| A file of the tests | `py-commol/src/test_kind.rs`, the test model kind (`--ignore-filename-regex` of the script) | It is not part of the product: only the feature `test-kind` builds it. |
| An arm for the test model kind | The `#[cfg(feature = "test-kind")]` arms of `Engine` and `PyAggregate::from_json` that reject a plan or a structure of the other kind | The kind check runs first, so no model reaches them; they exist so that the `match` covers the kind of the tests. |
| An invariant | `unreachable!`, as the arm of `evaluate_rate` that no flow of the builder takes; an `expect` with the reason it holds | Only a broken invariant of the code reaches it. |
| An error that only a dependency returns | The `?` after `add_class`, `add_submodule` or `m.add` of a binding module; the setup of the Cranelift JIT; a step or an interpolation that `diffsol` cannot take | A failed interpreter, JIT or solver, not an input that the engine builds. |
| Code that a PyO3 attribute generates | A `#[pyclass]` or `#[pymethods]` line whose generated conversion no call uses | The macro writes it; no line of commol does. |
| The failure of a test | The `panic!` or the message of an `assert!` that reports what went wrong | It runs only when the test fails. |

A `?` or an arm that no input reaches and that is none of these kinds is
removed: the binding, the engine or the check before it already rules the
input out. A check that the Python layer makes first is not repeated in the
binding.

## Mutation testing

Coverage says that a line ran, not that a test notices when it changes.
Mutation testing changes the code, one small change at a time, and reports
the changes that no test fails on. It runs on demand, not in CI, because a
full run takes hours:

```bash
cargo install --locked cargo-mutants
cargo mutants -j 2 --timeout 120 -p commol-model --test-package commol-model \
  --test-package commol-difference --test-package commol-differential \
  --test-package commol-stochastic \
  --exclude-re 'replace \+= with .* in CompiledModel::breakpoints'
```

A mutant can loop without end and fill the memory: the loop of
`CompiledModel::breakpoints` never ends when its step does not grow, so the
command leaves those mutants out, and `--timeout` stops the others. Run
`cargo-mutants` and `mutmut` one at a time.

For Python, `mutmut` (`uv pip install mutmut`) with `[tool.mutmut]`
`paths_to_mutate = ["commol/context/", "commol/api/"]` in a copy of
`py-commol/`. A missed mutant gets a test that fails on it, or a reason: an
equivalent mutant changes no behavior. The Python suite also runs the Rust
code, so check a Rust mutant that the Rust tests miss against `pytest`
before you write a test for it. `mutmut` imports the package before it
activates a mutant, so a mutant of code that runs at import, such as the
registration of a model kind, survives every test: check it with
`MUTANT_UNDER_TEST=<name>` set before Python starts.

## Navigation

Previous: [Tests](tests.md) | Next: [Maintainer playbook](maintainer-playbook.md)
