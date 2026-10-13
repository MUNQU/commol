# Repository Map

[Back to Codebase Guide](index.md)

Use this page when you know what to change but not where it belongs.

## Rust crates

| Path | Owns | Do not put here |
|---|---|---|
| `crates/commol-core/src/types/` | The model data types shared by all crates and serialized from Python (`Model`, `Population`, `Transition`, `InitialConditions`, …). | Engine behavior. |
| `crates/commol-core/src/math_expression/` | Parsing, preprocessing and evaluation of rate expressions; the Cranelift JIT and its global cache (`jit/`). | Model-specific variables. |
| `crates/commol-core/src/simulation_engine.rs` | The `SimulationEngine` trait and `OutputLabels`: what calibration needs from any engine. | Code for one engine. |
| `crates/commol-difference/src/builder.rs` | Expansion of bins and accumulators over stratifications (`expand`, `expand_compartments`, `expand_accumulators`), the initial state (`initial_state`), and engine construction. | Flow planning. |
| `crates/commol-difference/src/layout.rs` | `CompartmentLayout`: compartments in engine order with their labels, and lookups by bin and categories. | Name parsing. |
| `crates/commol-difference/src/flows.rs` | Transition flow planning (`plan_transition_flows`) and its description for Python (`describe_flows`). | Rate evaluation at run time. |
| `crates/commol-difference/src/helpers.rs` | Stratified-rate resolution and rate rewriting helpers. | New public API. |
| `crates/commol-difference/src/simulation.rs` | The simulation step and the `SimulationEngine` implementation. | Model validation. |
| `crates/commol-calibration/src/calibration_problem.rs` | Loss, constraints and the objective function. | Optimizer setup. |
| `crates/commol-calibration/src/initial_conditions.rs` | How calibrated initial-condition values reach the engine (`InitialConditionPlan`, `calibrated_initial_population`). | Parameter values. |
| `crates/commol-calibration/src/optimization.rs` | Nelder-Mead and particle swarm setup on the argmin fork. | Loss definitions. |
| `crates/commol-calibration/src/probabilistic/` | Parallel calibration runs, deduplication, clustering and ensemble selection (NSGA-II). | Statistics for reports. |

## PyO3 bindings

| Path | Owns | Do not put here |
|---|---|---|
| `py-commol/src/core.rs` | Bindings of the model types. | Logic. |
| `py-commol/src/difference.rs` | `DifferenceEquations` and the engine queries `initial_state`, `expanded_outputs`, `transition_flows`. | Logic beyond type conversion. |
| `py-commol/src/calibration.rs` | Calibration types and functions. | Logic beyond type conversion. |
| `py-commol/commol/commol_rs/_commol_rs.pyi` | The type stub of the extension. Update it with every binding change. | — |

## Python package (`py-commol/commol/`)

| Path | Owns | Do not put here |
|---|---|---|
| `api/model_builder.py` | The fluent `ModelBuilder`. | Validation rules that belong to the model. |
| `api/simulation.py` | `Simulation`: engine construction, runs, output grouping. | Expansion rules. |
| `api/calibrator.py`, `api/probabilistic_calibrator.py` | Calibration entry points and conversion to Rust types. | Loss or optimizer logic. |
| `api/probabilistic/` | Ensemble statistics, normalization and confidence intervals (`intervals.py`). | New interval conventions outside `ci_percentiles`. |
| `api/reporting.py` | `simulation_report` and `applied_parameters_report`. | Domain vocabulary. |
| `api/time_patterns.py`, `api/time_scale.py`, `api/windows.py` | Time-varying rates, step/physical-time conversion, windowed totals. | Physical time inside the model. |
| `api/plotter.py` | Plots of results. | Computation of results. |
| `context/` | The Pydantic model: fields, validation, JSON I/O, update methods. | Engine rules. |
| `context/_model/engine_state.py` | The only Python access to the engine queries. | Python re-implementations of them. |
| `context/_model/` (other files) | Internals of `Model`: equation printer, unit checker, calibration write-back, JSON rendering. | Public API. |
| `utils/security.py` | Validation of every user expression (`SecurityConfig` limits). | Expression evaluation. |
| `utils/equations.py` | Dimensional analysis with `pint`. | — |

## Other paths

| Path | Owns |
|---|---|
| `py-commol/tests/` | Python tests: one package per page of the user guide (`building/`, `transitions/`, `rates/`, `time_patterns/`, `initial_conditions/`, `simulation/`, `units/`, `calibration/`, `probabilistic/`, `reporting/`, `plotting/`, `docs/`), and `support/` with the shared model factories, the reference solver and the assertions. See [Tests](conventions.md#tests). |
| `crates/*/tests/`, `#[cfg(test)]` modules | Rust tests. |
| `docs/` | This site. `docs/codebase/` is for maintainers; the rest is for users. |
| `odd/` | Task records and review ledgers of past work. History, not specification. |
| `.github/workflows/` | CI: code quality, build, docs, release. |

## If you need to…

| Task | Start with | Then check |
|---|---|---|
| Change how a model expands | `builder.rs` (`expand`) | `layout.rs`, `engine_state.py`, `tests/building/`, `tests/simulation/test_outputs.py` |
| Change how transitions become flows | `flows.rs` | `helpers.rs`, `context/_model/equations.py`, `tests/transitions/` |
| Change the simulation step | `simulation.rs` | `simulation_engine.rs` |
| Add an expression function or variable | `math_expression/` | `utils/security.py`, `docs/guide/mathematical-expressions.md` |
| Change initial conditions | `context/initial_conditions.py` | `builder.rs` (`initial_state`), `tests/initial_conditions/` |
| Change calibration | `calibration_problem.rs` | `initial_conditions.rs`, `api/calibrator.py`, `tests/calibration/` |
| Change ensembles | `api/probabilistic_calibrator.py` | `api/probabilistic/`, `crates/commol-calibration/src/probabilistic/` |
| Expose an engine rule to Python | `py-commol/src/` | `_commol_rs.pyi`, `engine_state.py` |
| Change a report | `api/reporting.py` | [Design decisions](design-decisions.md#reports), `tests/reporting/` |

## Placement rule

Place code at the narrowest owner that can hold it:

1. **A rule about how a model becomes numbers** → a Rust crate.
2. **Type conversion between Python and Rust** → `py-commol/src/`.
3. **Validation of user input** → the Pydantic model in `context/`.
4. **User-facing workflow** → `api/`.
5. **Presentation** (reports, plots) → `api/reporting.py`, `api/plotter.py`.

## Navigation

Previous: [Mental model](mental-model.md) | Next: [Design decisions](design-decisions.md)
