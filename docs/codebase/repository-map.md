# Repository Map

[Back to Codebase Guide](index.md)

Use this page when you know what to change but not where it belongs.

## Rust crates

| Path | Owns | Do not put here |
|---|---|---|
| `crates/commol-core/src/types/` | The model data types shared by all crates and serialized from Python: `Model` (header, `engine`, `process`, `structure`), `Process` (`process.rs`: bins, accumulators, stratifications, transitions; the names `NEIGHBORS` and `NEIGHBORS_PREFIX`), `ModelStructure` (one variant per kind, tagged by `kind`), `Aggregate` (`aggregate.rs`: `InitialConditions`, `Neighbors` and their checks, `AggregateVariant`), `Transition` (`transitions.rs`), …; `test_kind.rs` only with the `test-kind` feature. | Engine behavior. |
| `crates/commol-core/src/types/engine.rs` | `EngineType`, the one list of engines (model types): per entry, the model kinds it runs (`kinds`, `engines_of`; `ModelKind`, with the value types it accepts, `ModelValueType`), the form of its dynamics (`DynamicsForm`: its schedule and the left-hand side of its equations), its settings and their validation, and the meaning of a rate. `ModelEngine`, the engine and settings of a model. | The engines themselves. |
| `crates/commol-core/src/math_expression/` | Parsing, preprocessing and evaluation of rate expressions; the Cranelift JIT and its global cache (`jit/`). | Model-specific variables. |
| `crates/commol-core/src/simulation_engine.rs` | The `SimulationEngine` trait, for an engine of any kind: runs at positions (`check_position`), outputs (`output_names`, `state_outputs`, `output_sums`), parameters, reset, the seed of a stochastic engine (`set_seed`), and the values of its kind (`plan_values`, `apply_values`, `ValuePlan`); `Trajectory`, `check_positions`, `run_replicates` (one run per seed, in parallel). What calibration needs from any engine. | Code for one engine or one kind. |
| `crates/commol-random/` | What every stochastic engine shares and no model: the generator (`Generator`, `Xoshiro256PlusPlus` seeded by SplitMix64), `replicate_seed`, the samplers (`sample`: exponential, Poisson, binomial, multinomial) and `SumTree`, the rates of a set of events. | Anything of a model or an engine. |
| `crates/commol-model/src/builder.rs` | Expansion of the bins and accumulators of a process over its stratifications (`expand`, `expand_compartments`, `expand_accumulators`), the initial state (`initial_state`), the checks of an aggregate model (`check_model`: variant, neighbors, names, whole units for a stochastic engine; `as_whole`), the subgroup total names (`subgroup_total_names`), impulse schedules and the compilation into `CompiledModel` (`try_from_model`). | Flow planning; stepping. |
| `crates/commol-model/src/layout.rs` | `CompartmentLayout`: compartments in engine order with their labels, and lookups by bin and categories. | Name parsing. |
| `crates/commol-model/src/labels.rs` | `OutputLabels` (bin and categories of an output) and `output_sums` (bin totals and partial totals). | Name parsing. |
| `crates/commol-model/src/engine.rs` | `CompartmentalEngine`: what a compartmental engine gives besides `SimulationEngine` (compartments, labels, `state`, `total`, `set_initial_condition`). | Code for one engine. |
| `crates/commol-model/src/initial_conditions.rs` | How initial-condition values reach a compartmental engine (`InitialConditionPlan`: resolution, layers, remainder); both engines' `plan_values` and `apply_values` use it. | Parameter values. |
| `crates/commol-model/src/flows.rs` | Transition flow planning (`plan_transition_flows`), the sums the neighbors of a unit are (`resolve_neighbors`), and the description of the flows for Python (`describe_flows`). | Rate evaluation at run time. |
| `crates/commol-model/src/helpers.rs` | Stratified-rate resolution and rate rewriting helpers. | New public API. |
| `crates/commol-model/src/rates.rs` | Evaluation of a compiled rate: fast path, JIT with resolved slots, evalexpr fallback. | Flow logic. |
| `crates/commol-model/src/system.rs` | The flows every engine reads: `flow_rates` (the value of each flow), `transition_rates` (its rate), `flow_changes` (the change of one event of each flow), the right-hand side `derivatives` built on them; impulses (`apply_impulses`, `impulse_rates`, `impulse_changes`, `impulse_positions`); the positions where a rate can jump (`breakpoints`, `reads_time`); the accessors the engines use. | Stepping along the axis. |
| `crates/commol-model/src/types.rs` | `CompiledModel`, compiled flows, `FlowChange` (one event of a flow), impulse schedules, series lookup (`SeriesParameter`). | Python types. |
| `crates/commol-difference/src/simulation.rs` | `DifferenceEquations`: the discrete step (`state += derivatives`), impulses at integer steps, the positions it accepts (integers), and its `SimulationEngine` and `CompartmentalEngine` implementations. | Model compilation. |
| `crates/commol-differential/src/lib.rs` | `DifferentialEquations`: state, output positions, impulses, and its `SimulationEngine` and `CompartmentalEngine` implementations. | Numerical methods. |
| `crates/commol-differential/src/solve.rs` | The integration with `diffsol`: pieces between the breakpoints of the model, the finite-difference `J·v`, dense or sparse matrices (`SPARSE_THRESHOLD`). | Model compilation. |
| `crates/commol-stochastic/src/events.rs` | `StochasticEvents`: whole units one event at a time (the direct method, with a time change when a rate reads `t`) or in leaps (tau-leaping, fixed or chosen as Cao, Gillespie and Petzold 2006 do), stopping at the breakpoints; its `SimulationEngine` and `CompartmentalEngine` implementations. | Samplers; flows. |
| `crates/commol-stochastic/src/steps.rs` | `StochasticSteps`: the chain binomial (one multinomial per source, capped Poisson draws for absolute flows) and the check of its per-step probabilities. | Samplers; flows. |
| `crates/commol-stochastic/src/units.rs` | What both stochastic engines share: the generator, the whole start of a run (`largest_remainder`, `random_rounding`) and impulses that move whole units. | Anything of one engine. |
| `crates/commol-calibration/src/calibration_problem.rs` | Loss, constraints, the objective function (with the replicates of a stochastic engine, `with_replicates`), and the position check that names an observation or constraint (`check_positions`). It uses only `SimulationEngine`. | Optimizer setup; anything of one model kind. |
| `crates/commol-calibration/src/types.rs` | `LossConfig` (every loss, and `replicate_loss`), `ReplicateSummary`, `crps`, `Replicates`, the observation, parameter and constraint types. | Optimizer setup. |
| `crates/commol-calibration/src/values.rs` | How calibrated values reach any engine: parameters by id, values of the model kind through the engine (`ValueWriter`, `check_model_values`, `configured_engine`). | How a kind resolves its values. |
| `crates/commol-calibration/src/optimization.rs` | Nelder-Mead and particle swarm setup on the argmin fork. | Loss definitions. |
| `crates/commol-calibration/src/probabilistic/` | Parallel calibration runs, deduplication, clustering and ensemble selection (NSGA-II). | Statistics for reports. |

## PyO3 bindings

| Path | Owns | Do not put here |
|---|---|---|
| `py-commol/src/core.rs` | Bindings of the model types (`Model`, `Process`, `Aggregate`, …). | Logic. |
| `py-commol/src/model.rs` | The queries of `commol_rs.model` on a process and its aggregate structure: `initial_state`, `expanded_outputs`, `transition_flows`, `subgroup_total_names`; `check_model` on a model; the names `NEIGHBORS` and `NEIGHBORS_PREFIX`. | Logic beyond type conversion. |
| `py-commol/src/engine.rs` | `commol_rs.engine`: `Engine::build` (the four engines of the aggregate kind and those of the test kind), which checks the kind and builds the implementation of the pair of an `EngineType` entry and the kind of the structure, and the Python classes `Engine` (any engine, built from a model), `EngineType` (the list, for Python to ask) and `ModelKind`. | Engine logic. |
| `py-commol/src/test_kind.rs` | `TestEngine` and `TestStochasticEngine`, the engines of the test model kind; only with the `test-kind` feature. | Anything a release needs. |
| `py-commol/src/calibration.rs` | Calibration types and functions; each takes an `Engine`. | Logic beyond type conversion. |
| `py-commol/commol/commol_rs/_commol_rs.pyi` | The type stub of the extension. Update it with every binding change. | — |

## Python package (`py-commol/commol/`)

| Path | Owns | Do not put here |
|---|---|---|
| `api/model_builder.py` | The fluent `ModelBuilder`, which writes the process; `build(structure, engine)` joins it with a structure. | Validation rules that belong to the model; anything of one structure. |
| `api/simulation.py` | `Simulation`: runs (`run`, `run_at`, with the seed of a stochastic engine), replicates (`replicates`, `ReplicateRuns`), output grouping through `ModelStructure.outputs_by_source`. | Expansion rules; which positions a model type accepts; anything of one model kind. |
| `api/results.py` | `SimulationResults` (values with their positions and seed), `SimulationReplicates` (runs with one seed each, and their statistics), `index_at`, `sample`. | Engine rules. |
| `api/calibrator.py`, `api/probabilistic_calibrator.py` | Calibration entry points and conversion to Rust types. | Loss or optimizer logic. |
| `api/probabilistic/` | Ensemble statistics, normalization and confidence intervals (`intervals.py`). | New interval conventions outside `ci_percentiles`. |
| `api/reporting.py` | `simulation_report` and `applied_parameters_report`, with the sections the structure adds (`ModelStructure.report_sections`). | Domain vocabulary; sections of one model kind. |
| `api/patterns.py`, `api/windows.py` | Rates that change along the axis (`AxisPattern`), the window grid and windowed totals. | Physical units inside the model. |
| `api/plotter.py` | Plots of results. | Computation of results. |
| `api/description.py` | The renderers of a `ModelDescription`: text, Markdown (Mermaid), LaTeX, JSON and Graphviz DOT, and `save`. | Anything of one model kind: a renderer reads only blocks. |
| `context/` | The Pydantic model: fields, validation, JSON I/O, update methods. | Engine rules. |
| `context/model.py` | `Model`: the header, the engine, the process and the structure of any kind, generic in its structure class (`Model[Aggregate]`). | Anything of one model kind. |
| `context/process.py` | `Process`: bins, accumulators, stratifications and transitions, and the checks of the process by itself. | Checks that depend on a structure. |
| `context/structure.py` | `ModelStructure`, the interface every model kind gives the generic layers, and its registry of kinds and variants. | Code of one kind. |
| `context/aggregate.py` | `Aggregate`: the initial amounts and the neighbors, their checks against the process, and the aggregate side of the interface. | Generic code. |
| `context/initial_conditions.py` | `InitialConditions`, the fields and the fraction logic `Aggregate` holds. | Checks against the process. |
| `context/neighbors.py` | `Complete` and `Groups`, who the neighbors of a unit are. | How the terms resolve (the engine's). |
| `context/description.py` | `ModelDescription` and its blocks (`Text`, `Table`, `Equations`, `StateGraph`, `Matrix`): a description as data. | Rendering; code of one kind. |
| `context/transition.py` | `Transition` and `StratifiedRate`. | — |
| `context/axis_units.py` | `AxisUnits`: values in units to steps of the axis, and rates asked from the engine (`Model.axis_units`). | Formulas of a rate per engine (they are in `RateMeaning`). |
| `context/engine.py` | `ModelEngine` (`Model.engine`: the engine type and its settings, read with the class of that engine), the settings classes `OdeSolver`, `StochasticEventsSettings` and `StochasticStepsSettings`, `Rounding`, and the names of the engines as the `Literal` `EngineName`. | Settings validation or engine meanings: ask `commol_rs.engine.EngineType`. |
| `context/_model/engine_state.py` | The only Python access to the engine queries, and `build_engine`, which builds the engine the model names. | Python re-implementations of them. |
| `context/_model/describe.py` | The sections every kind shares and the layouts of a description (`"default"`, `"odd"`). | Sections of one kind. |
| `context/_model/` (other files) | Internals of an aggregate model: the sections of its description (`aggregate_description.py`, and the equations, event table and state graphs of `equations.py`), unit checker, calibration write-back, the `initial_conditions` report section (`report.py`); JSON rendering of any model. | Public API. |
| `context/report_entry.py` | The ensemble spread of a report entry (`Interval`, `entry_interval`), for every kind. | — |
| `utils/security.py` | Validation of every user expression (`SecurityConfig` limits). | Expression evaluation. |
| `utils/equations.py` | Dimensional analysis with `pint`. | — |

## Other paths

| Path | Owns |
|---|---|
| `py-commol/tests/` | Python tests: one package per page of the user guide (`building/`, `transitions/`, `rates/`, `axis_patterns/`, `initial_conditions/`, `simulation/`, `stochastic/`, `units/`, `calibration/`, `probabilistic/`, `reporting/`, `plotting/`, `docs/`); `reference_models/`, published models checked against their stated results; `model_kinds/`, the test model kind through every generic layer; and `support/` with the shared model factories, the test kind (`kinds.py`), the flow models, the reference solver, the oracles and the assertions. See [Tests](conventions.md#tests). |
| `crates/*/tests/`, `#[cfg(test)]` modules | Rust tests. |
| `docs/` | This site. `docs/codebase/` is for maintainers; the rest is for users. |
| `.github/ISSUE_TEMPLATE/` | The templates of a proposal, a bug or small change, and a release ([Planning](../development/workflow.md#planning)). |
| `.github/scripts/start.sh` | Start an issue: take it, set it to `In Progress`, and create its branch and worktree, linked to the issue. |
| `.github/scripts/next.sh` | The next proposals to implement: open, not blocked, not taken, by the field `Order` of the project. |
| `.github/scripts/parallel.py` | The proposals that can run in parallel, from the Touches and the links of the open issues. |
| `.github/scripts/release_notes.py` | The notes of a GitHub release, from its section of the changelog. |
| `.github/workflows/` | CI: code quality, build, docs, release. |

## If you need to…

| Task | Start with | Then check |
|---|---|---|
| Change how a model expands | `commol-model/src/builder.rs` (`expand`) | `layout.rs`, `engine_state.py`, `tests/building/`, `tests/simulation/test_outputs.py` |
| Change how transitions become flows | `commol-model/src/flows.rs` | `helpers.rs`, `system.rs` (`derivatives`), `context/_model/equations.py`, `tests/transitions/` |
| Change the difference step | `commol-difference/src/simulation.rs` | `simulation_engine.rs` |
| Change the ODE integration or the solver settings | `commol-differential/src/solve.rs` | `commol-differential/src/lib.rs`, `commol-core/src/types/engine.rs` (`OdeSolver`), `context/engine.py` |
| Change a stochastic engine of the aggregate kind | `commol-stochastic/src/` (`events.rs`, `steps.rs`, `units.rs`) | `commol-core/src/types/engine.rs` (`StochasticEventsSettings`, `Rounding`), `commol-model/src/system.rs` (`flow_changes`), `context/engine.py`, `tests/stochastic/` |
| Add a model type (engine) | `commol-core/src/types/engine.rs` (`EngineType`) | `py-commol/src/engine.rs` (`Engine`), `context/engine.py`, [checklist](maintainer-playbook.md#model-type-engine) |
| Add a model kind | `context/structure.py` (`ModelStructure`) | `commol-core/src/types/structure.rs`, `engine.rs` (`ModelKind`), `py-commol/src/engine.rs`, [checklist](maintainer-playbook.md#model-kind) |
| Change impulses | `commol-model/src/system.rs` (`apply_impulses`) | `types.rs` (`ImpulseSchedule`), both engines, `context/transition.py`, `tests/transitions/test_impulses.py` |
| Add an expression function or variable | `math_expression/` | `utils/security.py`, `docs/guide/mathematical-expressions.md` |
| Change initial conditions | `context/initial_conditions.py` | `builder.rs` (`initial_state`), `tests/initial_conditions/` |
| Change calibration | `calibration_problem.rs` | `values.rs`, `commol-model/src/initial_conditions.rs`, `api/calibrator.py`, `tests/calibration/` |
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
