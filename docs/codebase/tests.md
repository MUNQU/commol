# Tests

[Back to Codebase Guide](index.md)

Where a Python test goes, how it builds its model, what it must check, and
how long it may take. The Python tests live in `py-commol/tests/`; the Rust
tests are `#[cfg(test)]` modules and the `tests/` folder of each crate.

## Layout: one package per guide page

A test goes in the package of the guide page that documents the behavior it
checks, so "where is the test of X" has the same answer as "where is the guide
of X". A pure-Python helper (`utils/security.py`, `context/axis_units.py`,
`api/inference/intervals.py`) goes in the package of its feature. The exceptions are
`reference_models/`, which checks published models across features (see
[Reference models](#reference-models)), and `model_kinds/`, which runs a model
kind that is not aggregate through every generic layer.

| Package | Tests of | Guide page |
|---|---|---|
| `building/` | The builder, parameters, stratifications, reserved names, save and load | [Building Models](../guide/building-models.md) |
| `transitions/` | Flows, `$compartment`, stratified rates, impulses | [Transitions](../guide/transitions.md) |
| `rates/` | Formulas, functions, variables, expression security | [Mathematical Expressions](../guide/mathematical-expressions.md) |
| `axis_patterns/` | `AxisPattern` formulas, runs and schedules | [Axis Patterns](../guide/axis-patterns.md) |
| `initial_conditions/` | Bin, stratification and compartment fractions; the initial state | [Building Models](../guide/building-models.md#the-structure-aggregate) |
| `simulation/` | Runs, `run_at`, results and their positions, differential models, outputs, the window grid and windowed totals, `AxisUnits` | [Running Simulations](../guide/simulations.md) |
| `stochastic/` | The stochastic engines of an aggregate model through the layers: their means against the deterministic engines, whole units, calibration with replicates (the closed forms are in `crates/commol-stochastic/tests/oracles.rs`) | [Stochastic Simulation](../guide/stochastic.md) |
| `units/` | Unit checks | [Units](../guide/units-and-equations.md) |
| `description/` | `Model.describe`, its sections and blocks, the renderers and their snapshots (`description/snapshots/`), the equations of the dynamics (`equation_forms` in `support/description.py`) | [Describing a Model](../guide/describing-models.md) |
| `calibration/` | One file per calibration page: `test_problem.py`, `test_optimizers.py`, `test_starts.py`, `test_constraints.py`, `test_observations.py`, `test_initial_conditions.py`, `test_write_back.py` | [Model Calibration](../guide/calibration.md) and its sub-pages |
| `uncertainty/` | The parameter sample, the one quantile, the predictive, the scores, the entry point `Calibrator.uncertainty`, and the coverage harness (`test_coverage.py`, marker `coverage_study`) | [Uncertainty](../guide/uncertainty/index.md) |
| `reporting/` | Reports | [Reports](../guide/reports.md) |
| `plotting/` | `SimulationPlotter` | [Running Simulations](../guide/simulations.md#plotting) |
| `docs/` | The documentation examples and the API coverage (marker `docs`) | — |
| `reference_models/` | Published models against their stated results, one module per model family | — (across features) |
| `model_kinds/` | The test model kind (`support/kinds.py`) through a simulation, a calibration, a report and a plot; the check that an engine runs its kind; the variants of a structure (`UnstratifiedVariant`). It needs the extension built with `--features test-kind`. | — (across features; [Design decisions](decisions/model-kinds.md#one-interface-for-every-model-kind)) |
| `suite/` | The rules of this page that a program can check (`test_rules.py`), and the plugin that records how long each test takes (`durations.py`) | — |
| `support/` | No tests: the code the tests share | — |

Keep a test module under about 600 lines. When it grows past that, split it by
topic inside its package.

## Models: use the factories

- Build models with the factories of `tests/support/models.py`:
  `abc_builder` (A → B → C), `abcd_builder` (A → B → C → D) and
  `a_to_b_builder`, and the special models
  `override_builder`, `shared_category_builder` and
  `conditional_pathway_builder`. A factory returns a `FactoryBuilder`: a
  `ModelBuilder` of the process and the `Aggregate` that lays it out
  (`builder.structure`, which `set_structure` replaces). A keyword overrides
  one part of the default model, so a test names only what it changes. `build` builds for difference
  equations; `build_ode` builds for differential equations, with the tight
  solver settings `TIGHT` by default; `build_stochastic` builds for a
  stochastic engine (`STOCHASTIC_ENGINES`, and `MEAN_FIELD`, the deterministic
  engine of each); `run` builds and simulates.
- Build a model by hand only when one test needs it and no factory keyword
  expresses it. A model that a second file needs goes to `support/`.
- Test models use abstract names, as examples do: bins `A`, `B`, `C`,
  parameters `k1`, `k2`, categories `g1`, `t1`. Only `tests/reference_models/`
  and the support code that only it uses (`support/oracles.py`,
  `support/flows.py`) keep the names of their published source.
- Fixtures: `abc_model` and `abc_simulation` (`tests/conftest.py`); `truth`
  and `truth_run` (`tests/calibration/conftest.py`). A fixture is
  function-scoped unless its cost is measured; a wider scope never holds an
  object a test can change. A class-scoped fixture is never an instance
  method (removed in pytest 10).

## Every test checks a result

A test compares a result with an oracle:

| Oracle | Example |
|---|---|
| A value computed by hand in the test | `reference_run` (`tests/support/reference.py`) evaluates the flows in plain Python, as the engine does: every flow from the state at the start of the step, no clamping. `tests/rates/test_formulas.py` compares whole trajectories with it. |
| An analytic value | An outflow rate `d` on every bin gives `total(t) = total(0) (1 - d)^t` (`tests/transitions/test_compartment_placeholder.py`). |
| An invariant | A total that the model keeps (`assert_total_conserved`); two paths that must agree, such as a bin id and the explicit sum of its compartments. |
| What a stated result predicts about a run | `tests/support/oracles.py`, for the reference models: `final_size` solves the final size relation, `assert_threshold` checks that a model grows at a reproduction number of `1.05` and decays at `0.95`, `settled` reads a limit once doubling the horizon no longer moves it. |

"It runs" is not a result. A test whose only check is that no exception
occurs is allowed only when the exception is its subject, such as a validator
that accepts a valid input; it calls `assert_accepted(check, *args)`
(`tests/support/assertions.py`), which fails with the message of the
rejection.

A test checks the behavior of commol. It does not check:

- the behavior of Python, pydantic, NumPy or another library, such as a
  `str` path that `pathlib` accepts, or an unknown keyword that Python rejects;
- a default of a pydantic model or of a binding;
- an attribute read back as the test set it;
- the absence of an API that was removed: that is history, not behavior.

`tests/suite/test_rules.py` checks the part of this that a program can read:
every Python test has an `assert`, a `pytest.raises`, `pytest.warns` or
`pytest.fail`, a call of `np.testing.assert_*`, or a call of a helper of
`tests/` that has one; every Rust test has an `assert*!`, `panic!` or
`unreachable!`, calls a function of its crate that has one, is
`#[should_panic]`, or returns a `Result`. Review checks the rest.

## Duplicates

Two tests do not check the same behavior with the same inputs. Candidates
come from the coverage of each test:

```bash
COVERAGE_CORE=ctrace pytest --cov=commol --cov-context=test --cov-report=
```

The `sysmon` core, the default of coverage on Python 3.14, records a line only
for the first test that runs it, so it gives most tests no context. Two tests of one module whose covered lines are equal,
or one inside the other, are a candidate. Coverage does not decide: two tests
can run the same lines and check different results. Read both, and keep one
only when the other checks nothing that it does not: the same output for the
same input, or a weaker check of the same output (`"+" in formula` against
the exact formula).

## Tolerances

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
- `np.testing.assert_allclose` skips masked values. Read the points of a
  matplotlib collection as floats (`np.asarray(offsets, dtype=float)`, as
  `_scatter` in `tests/plotting/test_plotter.py` does), so a point at `None`
  is NaN and fails.

## Reference models

`tests/reference_models/` shows that commol can state published models and
computes their stated results. A module holds one model family and is named
after the published model (`test_pulse_vaccination.py`); a test checks one stated
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
  marked as [Skips, expected failures and ignored tests](#skips-expected-failures-and-ignored-tests)
  says, with the open issue of the finding.

To add a test, write the model with `Flow` and `flow_model` (`tests/support/flows.py`)
from the initial value of each bin, compute the stated result in the test,
and cite the papers.

## Skips, expected failures and ignored tests

- An `xfail` is `strict=True` and its reason starts with the open issue that
  fixes it (`reason="#196: …"`): the fix makes the test pass, and the strict
  marker then fails the run until the fix removes it.
- A test that skips when an optional dependency is missing runs in at least
  one CI job: the `dev` extra installs `pydot`, and Python Coverage and the
  job Test of Build on Ubuntu install `latexmk`.
- A Rust `#[ignore]` gives a reason with the `cargo test` command that runs it,
  such as the benchmarks of `crates/commol-differential/src/tests.rs`. A
  benchmark also asserts that the methods it times give the same result.

`tests/suite/test_rules.py` checks the `xfail` markers and the reasons of
`#[ignore]`.

## Time budget

A test takes at most 1 s, setup and teardown included, in the job Test of
Build on Ubuntu, which runs every test without coverage. Coverage slows a test
that runs much Python up to ten times, so the run under coverage (Python
Coverage) skips the check, and so does a run off Linux. A test over 0.5 s stays only if it checks what no faster test checks,
with the smallest inputs that still check it: a shorter horizon, fewer steps,
fewer replicates. `tests/suite/durations.py` records the time of each test,
and `test_slow_tests_are_listed` runs last and fails on a test over the budget
that the table below does not list.

The coverage studies (`coverage_study`) are the one exception. A study counts
how often an interval covers the truth over hundreds of synthetic data sets;
fewer data sets widen its tolerance until it checks nothing, so it cannot
shrink to the budget. They do not run less often than the code they check:
Build runs them for every pull request that changes what they run,
and the release runs them before it publishes ([CI pipelines](../development/workflow.md#continuous-integration-pipelines)).
`test_every_coverage_study_runs_in_a_group_of_the_workflow` checks that
`coverage.yaml` runs each of them, and
`test_the_studies_run_for_a_change_of_any_crate_they_use` that a change of
`commol-inference`, of a crate it depends on or of the difference engine runs
them.

The check reads the wall time of the run, so a machine that runs other work
at the same time can exceed the budget. `pytest --durations=20` shows the
slowest tests; Python Coverage prints them, slowed by coverage. A row names
a test, or one case of a parametrized test with its parameter in brackets.

| Test | Why it takes longer |
|---|---|
| `tests/description/test_rendering.py::TestLatex::test_the_latex_document_compiles` | It compiles the LaTeX description with `latexmk`; the first TeX run of the job loads its formats. No other test checks that the document compiles. |
| `tests/docs/test_examples.py::test_page_examples_run[docs/guide/simulations.md]` | It runs every example of the page, which draws and saves four figures with matplotlib, one label per plotted position; the figures are what the page documents. |

## Test coverage

Every line of the Python package and of the Rust code, the bindings
included, runs in a test, or it is of a kind this section lists. A test that
only runs a line, and checks nothing, does not count
([Every test checks a result](#every-test-checks-a-result)). A line that no
input reaches is removed, not tested.

| What | Command | Threshold |
|---|---|---|
| Python (`py-commol/commol/`) | `pytest --cov=commol --cov-fail-under=100`, from `py-commol/` | 100 % of the statements, in the job Python Coverage |
| Rust (`crates/`, `py-commol/src/`) | `.github/scripts/rust_coverage.sh`, from the root, with the virtual environment of `py-commol` active | `RUST_LINES` of the script, in the job Rust Code Quality |

The bindings have no Rust tests: `pytest` runs them. So the script builds the
Rust tests and the extension with the instrumentation of `cargo llvm-cov`,
runs both, and writes one report. It builds in the debug profile, so
`debug_assert!` runs and each line keeps its own count, and it puts the
release extension back when it ends. It needs `cargo-llvm-cov`
(`cargo install --locked cargo-llvm-cov`) and the component
`llvm-tools-preview` (`rustup component add llvm-tools-preview`). Options
after the script go to `cargo llvm-cov report`, such as `--summary-only` or
`--lcov --output-path lcov.info`.

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
| An invariant | `unreachable!`, as the arm of `evaluate_rate` that no flow of the builder takes | Only a broken invariant of the code reaches it. |
| An error that only the host returns | The `?` after `add_class` or `add_submodule` of a binding module; the setup of the Cranelift JIT | A failed interpreter or JIT, not an input. |
| The failure of a test | The `panic!` of a test that reports what went wrong | It runs only when the test fails. |

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

Previous: [Conventions](conventions.md) | Next: [Maintainer playbook](maintainer-playbook.md)
