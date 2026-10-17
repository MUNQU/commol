# Tests

[Back to Codebase Guide](index.md)

Where a Python test goes, how it builds its model, and what it must check.
The tests live in `py-commol/tests/`.

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
that accepts a valid input.

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
  marked `xfail(strict=True, reason="F<n>: …")` that names a finding of the
  task that added it, so a fix makes the test pass and forces the marker out.

To add a test, write the model with `Flow` and `flow_model` (`tests/support/flows.py`)
from the initial value of each bin, compute the stated result in the test,
and cite the papers.

## Navigation

Previous: [Conventions](conventions.md) | Next: [Maintainer playbook](maintainer-playbook.md)
