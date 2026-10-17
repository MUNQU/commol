# Maintainer Playbook

[Back to Codebase Guide](index.md)

Use this page before you make or review a change. It turns the
[Repository map](repository-map.md) and the
[Design decisions](design-decisions.md) into checklists.

## Every change

1. Read the user-facing page of the feature in the [User Guide](../guide/core-concepts.md).
2. Find the owner in the [Repository map](repository-map.md).
3. Read the tests next to that code before you change it.
4. Make the change in the narrowest owner.
5. Run both test suites (see [`AGENTS.md`](https://github.com/MUNQU/commol/blob/main/AGENTS.md#commands)).
6. Update the documentation as the mandatory
   [definition of done](conventions.md#documentation-definition-of-done) requires.
7. Run `python .github/scripts/affected.py --closes <number>` and update each
   open issue it lists whose current state, paths or assumptions the change
   alters: its body, and a comment that says what changed
   ([Planning](../development/workflow.md#lifecycle-of-a-proposal), step 4).

## Guardrails

- **Do not duplicate an engine rule in Python.** Add a binding instead, and
  call it from `context/_model/engine_state.py`.
- **Do not identify compartments by name.** Use labels (see
  [Compartments are labels](design-decisions.md#compartments-are-labels-with-a-name)).
- **Do not drop invalid input silently.** Reject it with a message that names
  the transition, compartment or parameter.
- **Do not change saved formats by accident.** Model JSON and report key order
  are part of the contract.
- **Do not trust a closed proposal as specification.** A closed issue records
  past work; the code and this guide describe the present.

## Checklists by type of change

### Process (bins, stratifications, accumulators) and the aggregate structure

- [ ] Change the Rust type in `crates/commol-core/src/types/` (`process.rs`,
      `aggregate.rs`) and the Pydantic model in `py-commol/commol/context/`
      (`process.py`, `aggregate.py`) together. A check of the process by
      itself goes in `Process`; a check against the structure goes in
      `Aggregate.check_model`.
- [ ] Update the PyO3 binding and `_commol_rs.pyi` if the type crosses the
      boundary.
- [ ] Update expansion in `builder.rs` (`expand`) if compartments change.
- [ ] Check that `test_expansion_matches_the_engine_output_names` still passes.
- [ ] Update [Building models](../guide/building-models.md) and
      [Model structure](../advanced/model-structure.md).

### Transitions and rates

- [ ] Plan the flows in `flows.rs`; reject what cannot be simulated.
- [ ] Check the [rate semantics](design-decisions.md#absolute-and-proportional-rates).
- [ ] Add a case to `tests/transitions/test_flows.py`, and check the `dynamics`
      section of `Model.describe` (`tests/description/`).
- [ ] Update [Building models](../guide/building-models.md).

### Expressions (functions, variables)

- [ ] Change `crates/commol-core/src/math_expression/`, including the JIT
      (`jit/`) and its tests.
- [ ] Add a new function to `SAFE_FUNCTIONS` in `utils/security.py`.
- [ ] Update [Mathematical expressions](../guide/mathematical-expressions.md).

### Initial conditions

- [ ] Change the Pydantic validation in `context/initial_conditions.py` and the
      engine rule in `builder.rs` (`initial_state`).
- [ ] Check that `Aggregate.initial_state(process)` reports the new state.
- [ ] If calibration can set it, update `initial_conditions.rs` in
      `commol-model` (`InitialConditionPlan`) and the write-back
      (`context/_model/calibration_write_back.py`).
- [ ] Check that `test_applied_model_reproduces_the_loss_the_optimizer_scored`
      still passes.

### Calibration

- [ ] Put the change in its layer of `crates/commol-calibration/`: running the
      engine in `forward.rs`, reading observations in `observation.rs`,
      bounds and constraints in `feasibility.rs`, the loss in
      `discrepancy.rs`, penalties only in `objective.rs`. Use only
      `SimulationEngine`: a rule of one model kind goes behind
      `plan_values`, `apply_values` or `output_sums` of its engines.
- [ ] Convert new options once, in `api/calibrator.py`.
- [ ] Check that `tests/simulation/test_windows.py::TestAgreementWithTheCalibrationLoss` still
      passes if the loss changes.
- [ ] Update [Calibration](../guide/calibration.md).

### Uncertainty method

A method is three parts, and plugs into `Calibrator.uncertainty` without
changing it ([Design decisions](design-decisions.md#methods-plug-into-one-entry-point)):

- [ ] The algorithm in `crates/commol-inference/`. It runs the engine only
      through `ForwardModel`, reads observations only through
      `ObservationOperator`, sees a `Violation` as outside the support
      (`discrepancy_in_support`), and draws every random number from a
      `commol_random::Generator` seeded from the problem's seed.
- [ ] The runner, bound in `py-commol/src/inference.rs`, and its stub in
      `_commol_rs.pyi`.
- [ ] The configuration, a subclass of `UncertaintyMethod[YourResult]`, and
      the result, in `context/inference/`; a result with draws holds a
      `ParameterSample`.
- [ ] `register(YourMethod, runner)` in `api/inference/registry.py` (import it
      from `commol.api.inference`).
- [ ] A coverage study (`tests/support/coverage.py`, marker `coverage_study`)
      that shows its intervals cover at their level.
- [ ] A guide page under `docs/guide/uncertainty/`, a row in the table
      "Which method answers which question" of its index, and an API entry.

An identifiability method follows the same steps with
`IdentifiabilityMethod[IdentifiabilityReport]` and `Calibrator.identifiability`;
it reads derivatives only through `SensitivitySource`, adds its section to
`IdentifiabilityReport`, and needs no coverage study.

### Model type (engine)

To add an engine to the list:

- [ ] Add the entry to `EngineType` in `crates/commol-core/src/types/engine.rs`
      and to `ALL`. The compiler then names every `match` to complete in that
      file: its name, the model kinds it runs (`kinds`), whether it is
      stochastic (`is_stochastic`), its settings (`EngineSettings`, parsed
      and validated with `deny_unknown_fields`), the form of its dynamics
      (`dynamics_form`: the schedule a description shows and the left-hand
      side of its equations) and the meaning of a rate.
- [ ] Write the engine in its own crate and implement `SimulationEngine`
      (for an engine of the aggregate kind, also `CompartmentalEngine` from
      `commol-model`); `try_from_model` rejects a model of another engine. A
      stochastic engine draws every random number from one
      `commol_random::Generator`, implements `set_seed`, and restores the
      generator in `reset`, so a run repeats from its seed.
- [ ] Add a variant to `Engine` in `py-commol/src/engine.rs`: one arm in
      `build` per kind it runs (the pair of the engine and the kind),
      `engine_type`, `dispatch!`, `plan_values` and `apply_values` (and
      `compartmental!` for an engine of the aggregate kind). Add the crate to
      `py-commol/Cargo.toml`.
- [ ] Add the name to the `EngineName` `Literal` in `context/engine.py`; if the
      engine has settings, add their Pydantic class to `EngineSettings`.
- [ ] Run the tests of `crates/commol-core/src/types/engine.rs`,
      `py-commol/src/engine.rs` and `tests/building/test_engine.py`; they run
      every entry of the list.
- [ ] Say what a rate means for it in the guides, and add it to
      [Model structure](../advanced/model-structure.md) and
      [Running simulations](../guide/simulations.md).
- [ ] Add the engine to [Inside Rust](architecture.md#inside-rust): a box in
      the diagram and a row in the table.

### Model kind

To add a kind of model, such as an agent network (see
[One interface for every model kind](design-decisions.md#one-interface-for-every-model-kind)):

- [ ] Add the kind to `ModelKind` in `crates/commol-core/src/types/engine.rs`
      and to its `ALL`: its name and the types of value calibration may set
      (`value_types`).
- [ ] Add the structure of the kind to `crates/commol-core/src/types/` and a
      variant to `ModelStructure` (`structure.rs`). The compiler names every
      `match` to complete, such as `Model::aggregate`. The structure reads
      the process of the model; it does not hold bins or transitions.
- [ ] Add its engines to the list, as in
      [Model type (engine)](#model-type-engine), or add the new kind to the
      `kinds` of an engine that runs it.
      Each engine implements `SimulationEngine`: `plan_values` and
      `apply_values` resolve and write the values of the kind, and
      `state_outputs` and `output_sums` name what an expression reads.
- [ ] Add a variant to `EngineValuePlan` in `py-commol/src/engine.rs`.
- [ ] Write a `ModelStructure` subclass in `py-commol/commol/context/`, with
      `kind` as a `Literal` default. Implement `check_model` and
      `outputs_by_source`; override `series_label`, `uncalibrated_values`,
      `with_calibrated_values`, `report_sections`, `describe` and
      `check_unit_consistency` when the kind has them. Reject in
      `check_model` a feature of the process the kind cannot run, and name
      the form that carries over. Import the module in `context/__init__.py`,
      so `Model.from_json` reads the kind.
- [ ] Do not write a builder: `ModelBuilder.build(structure, engine)` builds
      every kind. Do not change `Simulation`, `Calibrator`, the reports or the
      plotter: if one of them needs something of the kind, add a method to
      `ModelStructure` with a default.
- [ ] For a variant of a structure, subclass it with `variant` as a `Literal`
      default, and add the variant and its checks to Rust
      (`AggregateVariant` for `Aggregate`).
- [ ] Give the description of the kind: override `ModelStructure.describe`
      with the sections it fills (`entities`, `dynamics`, `initialization`,
      `design_concepts`), built from the blocks of
      `context/description.py`; the renderers need no change.
- [ ] Copy the tests of `tests/model_kinds/` for the new kind, and run the
      Rust and Python suites.
- [ ] Add the kind to [Model structure](../advanced/model-structure.md), the
      [Mental model](mental-model.md#model-kind-and-engine) and a guide page.
- [ ] Update [Architecture](architecture.md) if the kind adds a crate or
      changes the path of a simulation or a calibration.

### PyO3 bindings

- [ ] Keep logic out of `py-commol/src/`: convert types and call the crate.
- [ ] Update `_commol_rs.pyi`.
- [ ] Rebuild with `maturin develop --release --features test-kind` before you
      run `pytest`: the tests need the test model kind.

### Reports

- [ ] Keep the key order of existing entries.
- [ ] Put a section that only one model kind has in its
      `ModelStructure.report_sections`, not in `api/reporting.py`.
- [ ] Use `member_statistics` (with the weights of a `Predictive`) for values
      across members, and `ParameterSample.summary` for parameters.
- [ ] Update `tests/reporting/`.

### Dependencies

- [ ] Let Dependabot propose the update; it groups updates monthly.
- [ ] For a major update that needs code changes, open your own pull request
      and close the Dependabot one with a link to it.
- [ ] The argmin dependency is a fork (`MUNQU/argmin`, branch `development`):
      change the fork first, then update `Cargo.lock`.

### Releases

- [ ] Follow [Release process](../development/release.md).

## Review checklist

- [ ] The change lives in the owner that the [Repository map](repository-map.md) names.
- [ ] No engine rule is re-implemented in Python.
- [ ] The change meets the [documentation definition of done](conventions.md#documentation-definition-of-done):
      the affected docs are updated, in the agreed format, and the doc checks pass.
- [ ] Every new claim in the docs names a file, function or test.
- [ ] The diagrams of [Architecture](architecture.md) still describe the code.
- [ ] The tests that guard the touched [design decisions](design-decisions.md) still pass.
- [ ] Each open issue that `.github/scripts/affected.py` lists is updated, or does not depend on what changed.

## Navigation

Previous: [Conventions](conventions.md) | Back to: [Codebase Guide](index.md)
