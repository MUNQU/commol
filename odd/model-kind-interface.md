# Refactor: one interface for any kind of model

Goal: the layers that do not depend on how a model is built (simulation,
calibration, ensembles, reports, windows, plots) work through one interface
that any kind of model can give. Today the only kind is the compartmental
model (bins, stratifications, transitions with rates), run by a difference or
an ODE engine. After this ODD, another kind (for example an agent network)
is a new class that meets the interface, with its engines in the registry,
and nothing that exists breaks.
Motivation: the user wants the axis refactors to leave commol generic for the
next step, such as agent networks (L1). After `odd/axis-positions.md`,
`odd/model-type-registry.md` and `odd/axis-scale-refactor.md`, the
axis and the engine are generic, but `typology` names a model kind and an
engine at once, and the generic layers read compartmental fields.
Branch: `refactor/model-kind-interface` from `develop`, after
`odd/model-type-registry.md` is merged, and before
`odd/axis-scale-refactor.md`, whose converter reads `step_unit` and the
engine from the new `Model`.
Engram mirror: topic `odd/model-kind-interface/tasks` (project `commol`).
Runners: the Rust tests, clippy and rustdoc (from the root), `maturin develop
--release`, then `pytest`, `pytest -m docs`, `ruff check .`, `ty check commol`
and `vulture` (from `py-commol/`), `mkdocs build --strict` (from the root).

## Current state (verified 2026-10-13 on `develop` at `34d75eb` with PR #102)

### `typology` names a model kind and an engine at once

`DifferenceEquations` and `DifferentialEquations` both mean "a compartmental
model" and "how it advances along the axis". The typology lives in
`Model.dynamics.typology`, inside a compartmental `Model`:

| Part of `Model` | Fields | Kind |
|---|---|---|
| Header | `name`, `description`, `version`, `parameters`, `step_unit` | Any |
| Structure | `population` (`bins`, `accumulators`, `stratifications`, `transitions`, `initial_conditions`), `dynamics` (`typology`, `transitions`, `solver`) | Compartmental |

`model-type-registry` D1 (after its L5) makes each entry of the engine's list
an engine that names the model kind it runs; today every entry runs the
compartmental kind, so nothing yet checks the kind of a model.

### The generic layers read the compartmental structure

| File | Lines | Reads |
|---|---|---|
| `api/reporting.py` | 20, 65, 169, 200, 222–234, 257, 279, 288, 309–383 | Initial conditions (`population_size`, `bin_fractions`, `stratification_fractions`, `compartment_fractions`), stratifications |
| `api/plotter.py` | 58, 62, 351, 366, 668, 823, 917, 1023 | `population.bins` |
| `api/calibrator.py` | 462, 487, 495–498, 532 | `check_initial_conditions`, `population.bins`, `initial_conditions.population_size`, `dynamics.typology` |
| `api/probabilistic/calibration_runner.py` | 302–305 | `initial_conditions.population_size` |
| `api/probabilistic_calibrator.py` | 155, 185 | `check_initial_conditions`, `dynamics.typology` |
| `api/simulation.py` | 75 | `get_uncalibrated_initial_conditions` |

### The engine trait is compartmental

`compartments()`, `population()` and `set_initial_condition(compartment_index)`
(`commol-core/src/simulation_engine.rs`) assume the state is a vector of
compartment populations. The crates are not published, so changing the trait
breaks no user.

### What does not block another kind

- Results (`axis-positions` D1): a stochastic or agent engine returns a
  collection of the same result objects, one per realization; that is an
  addition.
- Settings per engine (`model-type-registry` D2) and `build(typology,
  settings)` (D3) hold for any kind.

## Specs

Every spec follows the [documentation definition of done](../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — An engine runs only its model kind

- Fix: the entries of the engine's list already name their model kind
  (`model-type-registry` D1). Building an engine checks that the model is of
  that kind (D1). `Dynamics.typology`, its names and the JSON of a saved model
  do not change.
- Check: a model of the test kind of S3 given to a compartmental engine, and
  the reverse, raise an error from the engine that names both kinds.

### S2 — The interface of a model kind

- Fix: one interface (D2) for what the generic layers may use: from `Model`,
  name, parameters, step unit and the engine built from the registry; from
  its `structure`, output names and labels, the values calibration may set
  (D4) and the report sections of the kind (D3). The compartmental structure
  meets it.
- Check: `ty check commol` checks that `Model` meets the interface; a kind
  that misses a method fails when it is made, naming the method; `Simulation`
  given a structure that is not a `ModelStructure` raises a clear error.

### S3 — The generic layers use only the interface

- Fix: `Simulation`, `Calibrator`, `ProbabilisticCalibrator`, the calibration
  runner, the reports and the plotter read only the interface. What only a
  compartmental model has (initial conditions, stratifications, bins) moves to
  a compartmental module behind the interface.
- Check: a test model kind, written only against the interface (in
  `tests/support/`), runs through `Simulation`, a calibration, a report and a
  plot. The compartmental tests pass unchanged.

### S4 — An engine trait for any state

- Fix: the compartmental methods of the engine trait (`compartments`,
  `population`, `set_initial_condition(compartment_index)`) move to the
  compartmental engines; the trait keeps what any kind gives (outputs,
  positions, parameters, reset).
- Check: the Rust tests of both engines and of calibration unchanged; the
  calibration crate uses only the trait.

### S6 — `Model` by composition

- Fix: `Model` is laid out as D1 says, in Python and in Rust (`commol-core`
  `types/model.rs`). `model-type-registry` T2 has already made `engine`
  (from `Dynamics.typology` and `Dynamics.solver`); here `structure` replaces
  `Population` and the rest of `Dynamics`, and transitions are held once. `ModelBuilder` builds a
  `Model` with a compartmental structure; reads of `model.population` and
  `model.dynamics` in `commol` move to `model.structure` and `model.engine`.
- Check: a model saved and loaded back keeps every field; the Python and Rust
  tests pass after the move; a search finds no `Population` or `Dynamics`
  outside the changelog.

### S5 — Documentation

- Fix: `docs/codebase/mental-model.md` and `repository-map.md` (model kind
  against engine), `design-decisions.md` (the interface and why), and a
  checklist "add a model kind" in `maintainer-playbook.md`.
- Check: `mkdocs build --strict`.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1, S6 | How are a model, its kind and its engine laid out? | By composition (L6). `Model` is one concrete class for any kind: `name`, `description`, `version`, `step_unit`, `parameters`, `engine` (`{type, settings}`) and `structure`, the part of its kind, tagged by `kind`. The compartmental structure holds `bins`, `accumulators`, `stratifications`, `transitions` (once) and `initial_conditions`. `Population` and `Dynamics` go. The registry holds engines, each naming the kind it runs, and checks `engine.type` against `structure.kind`. The engine type is part of the model, since it sets what a rate means. | Keep `Model`, `Population` and `Dynamics` and add only the interface (approved first, L1): transitions stay twice (`Population.transitions` and `Dynamics.transitions`; `population.rs:101` says the engine reads only the second), `population` stays a misnomer, and the engine stays mixed with the structure. Inheritance, an abstract `Model` with `CompartmentalModel`: direct access (`model.bins`), but two parallel hierarchies (models and builders) and the header repeated per kind; with composition `Simulation` and `Calibrator` take one type. The engine in the simulation, not the model: the meaning of a rate (per-step probability or continuous rate) depends on it. | approved (L6) |
| D2 | S2 | Form of the interface in Python | An abstract base class, `ModelStructure`, that each kind's structure inherits (D1, L6); `Model` holds one. A missing method fails when the object is made and names the method; optional methods have a default (no report sections, D3); `ty` checks it. Pydantic's metaclass derives from `ABCMeta` (checked, L3). | A `typing.Protocol`: a missing method shows only under `ty` or as an `AttributeError` mid-run, and it gives no defaults to a class that does not inherit; not inheriting is worth little, since the kinds are written in commol. | approved (L3, L6) |
| D3 | S3 | Where do the report sections that only a compartmental model has go? | The kind gives its own sections through one `ModelStructure` method, with the default `{}` (D2); the generic report builds the common sections and adds the kind's. The compartmental kind returns `{"initial_conditions": …}` from today's `_initial_conditions_report` (`reporting.py:288–383`), moved to a compartmental module; the output keeps its keys. The method takes what that section uses today (calibrated values, whether an override was calibrated, the confidence level), so the signature does not fall short for another kind. | Keep them in `reporting.py` behind a check of the kind: the branches by kind come back, the problem `model-type-registry` removes, and each new kind edits `reporting.py`. Drop them: `"initial_conditions"` is documented (`docs/guide/reports.md:99`) and used. Cost (L4): one method with a default, no new logic, report tests unchanged; the risk is the signature, fixed in T3 from today's inputs and exercised by the test kind with a section of its own. | approved (L4) |
| D4 | S3 | How does calibration know what it may set besides parameters? | `PARAMETER` and `FACTOR` stay generic, in the calibration. Each kind declares in `ModelStructure` the kinds of value of its own it accepts; the compartmental kind accepts `INITIAL_CONDITION`. In Rust, `crates/commol-calibration/src/initial_conditions.rs` (bin fractions, category splits, remainder) moves to the compartmental side (`commol-model`), and the engine applies a value of its kind through one trait method, so the calibration crate no longer knows compartments. A kind that does not accept a value type raises the error. | The calibration knows initial conditions (today): a compartmental notion in a generic layer, in Rust too, against S4; a new kind could not calibrate values of its own without editing calibration. Parameters and factors only: drops the calibration of initial conditions, documented and in use. Cost: the largest part of this ODD (about 300 lines of Rust move, one trait method), no new logic; the initial-condition calibration tests (Rust and Python) guard it. | approved (L5) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1, S2 | An engine checks the kind of its model (D1); the interface (D2); `Model` meets it | direct | `f513ed7` |
| T2 | S3 | `Simulation`, `Calibrator`, `ProbabilisticCalibrator` and the calibration runner on the interface (D4) | direct | `f513ed7` |
| T3 | S3 | Reports and plotter on the interface (D3); the test model kind | direct | `f513ed7` |
| T4 | S4 | Engine trait for any state | direct | `f513ed7` |
| T0 | S6 | `Model.structure` (D1): replaces `Population` and the rest of `Dynamics`, transitions once; Python, Rust, builder, every read | direct | `f513ed7` |
| T5 | S5 | Documentation | direct | `docs(api): document the model kind interface` |
| T6 | all | Final audit: every runner | direct | — |

## Delivery

- Strategy: one PR to `develop` from `refactor/model-kind-interface`,
  squash-merged (`refactor(api): …`).
- Order: T0 → T1 → T2 → T3 → T4 → T5 → T6.
- Version impact: breaking (D1): the saved model format, `model.population`,
  `model.dynamics`, `Population`, `Dynamics`. Commol is alpha, so the break is
  no cost; the changelog maps each old name to the new one ("Changed",
  "Removed").

## Evidence

Branch `refactor/model-kind-interface` from `develop` at `e3ed45a`
(2026-10-14).

- T0 (S6, D1): Rust `Model { name, description, version, parameters, engine,
  structure }` (`commol-core/src/types/model.rs`); `ModelStructure`
  (`structure.rs`), tagged by `kind`, with `CompartmentalStructure`
  (`compartmental.rs`, was `population.rs`) that holds the transitions once;
  `transitions.rs` (was `dynamics.rs`). `Model::compartmental` and
  `compartmental_mut` give the structure or an error that names the kind.
  `commol-model` functions take the structure (`expand_compartments`,
  `initial_state`, `describe_flows`), or the model when they need the
  parameters (`CompiledModel::try_from_model`, `check_names`). Python:
  `Model` is generic in its structure (`Model[CompartmentalStructure]`,
  `context/model.py`); `CompartmentalStructure` (`context/compartmental.py`)
  takes the validators of `Population`, `Dynamics` and the compartmental ones
  of `Model`, and its methods (`initial_state`, the `update_*` methods,
  `subgroup_population`, `get_conditioning_categories`, `outputs_by_source`);
  `context/transition.py` (was `dynamics.py`). `Population` and `Dynamics` are
  gone in both languages and in the bindings (`commol_rs.core.CompartmentalStructure`).
- T1 (S1, S2, D2): `ModelKind` (`ALL`, `from_name`, `value_types`) and
  `EngineType::check_kind`; `Model::check_kind` in `Engine::build`; Python
  `Model.validate_engine_runs_structure` asks `commol_rs.engine.EngineType.check_kind`.
  `ModelStructure` (`context/structure.py`) is an ABC with a registry of kinds
  (`of_kind`), abstract `check_model` and `outputs_by_source`, and defaults for
  `series_label`, `calibration_value_types` (asks `commol_rs.engine.ModelKind`),
  `uncalibrated_values`, `with_calibrated_values`, `report_sections`,
  `print_equations` and `check_unit_consistency`. `ModelEngine.type` is a `str`
  the list checks; the `Literal` is `CompartmentalEngineType`, checked against
  the compartmental entries.
- L8 (test kind): `ModelKind::Test`, `EngineType::TestEngine` and
  `TestStructure` behind the cargo feature `test-kind` (`commol-core`, and
  `commol` with `py-commol/src/test_kind.rs`); `tests/support/kinds.py`
  (`DecayStructure`). CI builds the extension with `--features test-kind`
  (`code-quality.yaml`, `build.yaml`); wheels do not.
- T2 (S3, D4): `Simulation`, `Calibrator`, `ProbabilisticCalibrator` and the
  runners read only the interface: `uncalibrated_values`, `outputs_by_source`,
  `engine_state.check_model_values` (the kind's value types, then the
  engine); no `initial_population_size` (the engine knows it). `Simulation`
  rejects a structure that is not a `ModelStructure` (`TypeError`).
- T3 (S3, D3): `applied_parameters_report` adds `ModelStructure.report_sections`;
  the compartmental `initial_conditions` section moved to
  `context/_model/report.py`, keys unchanged; `Interval` and `entry_interval`
  in `context/report_entry.py`. The plotter names a series with
  `series_label`. `tests/model_kinds/` runs the test kind through a
  simulation, a calibration, a report (with its own `outputs` section) and a
  plot.
- T4 (S4, D4): `SimulationEngine` keeps outputs (`output_names`,
  `state_outputs`, `output_sums`), positions, parameters and reset, and adds
  `plan_values` and `apply_values` with an associated `ValuePlan`. `compartments`,
  `output_labels`, `population`, `set_initial_condition` moved to
  `CompartmentalEngine` (`commol-model/src/engine.rs`); `OutputLabels` to
  `commol-model/src/labels.rs` with `output_sums`; `initial_conditions.rs` moved
  from `commol-calibration` to `commol-model` (`InitialConditionPlan`).
  `commol-calibration` uses only the trait (`values.rs`: `ValueWriter`,
  `check_model_values`, `configured_engine`) and depends on no engine crate
  outside its tests. D4 says "one trait method": the engine needs two, one to
  resolve the values once (`plan_values`) and one to write them per
  evaluation (`apply_values`), so ids are not resolved at every evaluation.
- Tests (S4 check, user request 2026-10-14): the tests of the
  initial-condition plan moved from `commol-calibration` to
  `crates/commol-model/tests/initial_conditions.rs`, on a fake compartmental
  engine; `crates/commol-calibration/tests/test_model_values.rs` checks the
  calibration with an engine of a kind that is not compartmental, so the crate
  has no `commol-model` dependency, not even in its tests. Each engine has a
  test of `plan_values`, `apply_values`, `reset`, `state_outputs` and
  `output_sums`. Python: the factories of `tests/support/` return
  `Model[CompartmentalStructure]` (no `ty` diagnostic on `model.structure` in
  the tests); new tests of the saved format, of `commol_rs.engine.ModelKind`
  and of the compartmental queries of a test engine.
- T5 (S5): `mental-model.md` ("Model kind and engine"), `repository-map.md`,
  `design-decisions.md` ("Model kinds"), `maintainer-playbook.md` ("Model
  kind"), `conventions.md` (`model_kinds/`), `advanced/model-structure.md`,
  `api/model.md`, the guides, `development/workflow.md`, `AGENTS.md` and the
  changelog (`[Unreleased]`: Added, Changed with a mapping table, Removed).

## Log

- L1 (user, 2026-10-13): asked whether the three axis ODDs leave commol
  generic for the next step, such as agent networks. They make the axis and
  the engine generic, not the model kind: `typology` names a kind and an
  engine at once, and the generic layers read the compartmental structure.
  Approved D1 (engines per model kind, an interface for the generic layers)
  and asked for it in a new ODD, without changing the existing ones.
- L2 (user, 2026-10-13): `model-type-registry` D1 now makes each entry an
  engine that names its model kind (its L5), so S1 here only checks the kind.
- L3 (user, 2026-10-13): approved D2 as an abstract base class `ModelKind`,
  not the `Protocol` first recommended: a Pydantic model can inherit an ABC
  (`ModelMetaclass` derives from `ABCMeta`), which gives an early error, defaults
  and an `isinstance` check.
- L4 (user, 2026-10-13): asked whether the cost of D3 is high. It is low: one
  method with a default, the code moves, the output keeps its keys. The risk
  is the signature; T3 takes it from what `_initial_conditions_report` reads.
  Approved D3.
- L5 (user, 2026-10-13): approved D4. The user then set a rule: commol is
  alpha, so a decision picks the best design without weighing backward
  compatibility; D1 and D2 here are to be reviewed again under it.
- L6 (user, 2026-10-13): under the alpha rule, reviewed D1 again: approved
  `Model` by composition (`engine` and `structure`, no `Population` or
  `Dynamics`, transitions once). D2's base class becomes `ModelStructure`.
  This replaces `model-type-registry` D2 and settles the `population` name
  that `axis-positions` S6 kept.
- L7 (user, 2026-10-13): while ordering the pending tasks, `Model.engine`
  moved to `model-type-registry` T2; T0 here makes only `Model.structure`.
- L8 (user, 2026-10-14): S1 and S3 need the test kind to have its own engine
  in the list, which is in Rust, and calibration runs in Rust; a kind written
  only in Python cannot have either. Approved: the test kind's engine is in
  Rust, in `EngineType`, compiled only with the cargo feature `test-kind`;
  its `ModelStructure` is in `tests/support/`. The tests build with
  `maturin develop --release --features test-kind`; published wheels do not
  have it. Rejected: an engine that calls Python (a new product feature, the
  GIL in calibration) and a test kind that becomes a compartmental model (S1
  cannot be tested).
- L10 (user, 2026-10-14): PR #106 merged into `develop` as `77aa9fd`; the
  ODD is archived.
- L9 (agent, 2026-10-14): implemented T0–T6 (see Evidence). T4 went before
  T2 in the code, because the test engine implements the final trait and the
  Python calibration lost `initial_population_size` with it.
