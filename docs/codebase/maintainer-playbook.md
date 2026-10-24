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
5. Run both test suites ([Quick commands](../development/workflow.md#quick-commands)).
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
  [Compartments are labels](decisions/model-structure.md#compartments-are-labels-with-a-name)).
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

- [ ] Plan the flows in `commol-model/src/flows.rs`; reject what cannot be simulated.
- [ ] Check the [rate semantics](decisions/transitions-and-rates.md#absolute-and-proportional-rates).
- [ ] Add a case to `tests/transitions/test_flows.py`, and check the `dynamics`
      section of `Model.describe` (`tests/description/`).
- [ ] Update [Building models](../guide/building-models.md).

### Expressions (functions, variables)

- [ ] Change the language in `crates/commol-core/src/math_expression/`: a
      function is one entry of `FUNCTIONS` (`language.rs`), which the check,
      the JIT and Python (`commol_rs.core.FUNCTIONS`) read; an operator is a
      token and a level of the parser (`jit/parser.rs`), a case of `check`
      (`language.rs`) and of the JIT (`jit/codegen.rs`).
- [ ] Keep every walk of a tree (`Expr`) recursive only into the operands of
      a node: a chain of one level is one node, and `MAX_NESTING` bounds the
      depth. A new walk at the limit runs in
      `crates/commol-core/tests/expressions.rs::every_level_of_operators_at_the_limit_reads_on_a_worker_thread`.
- [ ] Give a new function its value in
      `crates/commol-core/tests/expressions.rs::every_function_of_the_guide_has_its_value`
      and its unit rule in `_FUNCTION_UNITS` (`utils/equations.py`).
- [ ] Update [Mathematical expressions](../guide/mathematical-expressions.md).

### Initial conditions

- [ ] Change the Pydantic validation in `context/initial_conditions.py` and the
      engine rule in `builder.rs` (`initial_state`).
- [ ] Check that `Aggregate.initial_state(process)` reports the new state.
- [ ] If calibration can set it, update
      `commol-model/src/initial_conditions.rs` (`InitialConditionPlan`) and the write-back
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
changing it ([Design decisions](decisions/uncertainty.md#methods-plug-into-one-entry-point)):

- [ ] The algorithm in `crates/commol-inference/`. It runs the engine only
      through `ForwardModel`, reads observations only through
      `ObservationOperator`, sees a `Violation` as outside the support
      (`discrepancy_in_support`), and draws every random number with the
      samplers of `commol-random`, from a `commol_random::Generator` seeded
      from a stream of the problem's seed: a new `Stream` of
      `crates/commol-random/src/streams.rs`, with a number of its own, and
      `Stream::replicate` for each draw or replicate; never the seed itself.
      It takes its numerical helpers and log densities from
      `commol_statistics::numeric` and its quantiles and weighted statistics
      from `commol_statistics::statistics`, and writes no copy of them.
- [ ] A method that samples a posterior is a `TemperedTarget` on
      `SmcSampler` (`crates/commol-inference/src/smc/`), not a sampler of its
      own, and its configuration takes required `priors` (`WithPriors`)
      ([Design decisions](decisions/uncertainty.md#one-smc-core-many-targets)).
- [ ] The runner, bound in `py-commol/src/` (`inference.rs` or the module
      of its family), which runs the method through `parallel`
      (`py-commol/src/parallelism.rs`), and its stub in `_commol_rs.pyi`
      (`tests/suite/test_rules.py::test_the_stub_declares_what_the_extension_has`); a
      row in the table of `docs/guide/parallelism.md` and a case of `METHODS`
      in `tests/parallelism/test_threads.py`.
- [ ] The configuration, a subclass of `UncertaintyMethod[YourResult]`, and
      the result, in `context/inference/`; a result with draws holds a
      `ParameterSample`.
- [ ] A method that fits several times fits its perturbations with
      `ReplicateRunner::fit_all` and checks them with `ReplicateFits::require`;
      its result has `failed` (the entries of `FitUnit::failed`, logged as
      warnings) and `not_converged` at the top
      ([Design decisions](decisions/results.md#fits-that-fail-have-one-place)).
- [ ] Every public function of the method returns `MethodError`, of the
      kind `Invalid` for a setting, an input, the data or the model that does
      not fit the method, and `Failed` for a computation that fails; the
      binding maps it with `method_error`, and the "Raises" section of its
      docstring lists `ValueError` and `RuntimeError` by that rule
      ([Design decisions](decisions/uncertainty.md#methods-raise-one-error-type)).
- [ ] The Python runner in the module of its family under `api/inference/`,
      which calls `register(YourMethod, runner)` (`api/inference/registry.py`);
      `api/inference/__init__.py` imports every such module.
- [ ] A coverage study (`tests/support/coverage.py`, marker `coverage_study`)
      that shows its intervals cover at their level.
- [ ] A guide page under `docs/guide/uncertainty/`, a row in the table
      "Which method answers which question" of its index, and an API entry.

An identifiability method follows the same steps with
`IdentifiabilityMethod[IdentifiabilityReport]` and `Calibrator.identifiability`;
it reads derivatives only through `SensitivitySource`, adds its section to
`IdentifiabilityReport`, and needs no coverage study.

A sensitivity method subclasses `SensitivityMethod[YourResult]` and runs
through `SensitivityAnalysis.run`. Its Rust side is a method of
`ForwardAnalysis` (`crates/commol-inference/src/global_sensitivity/`), which
builds the design from `InputSpace` (`design.rs`), runs it with
`BatchRunner` and calls the estimator; the binding and the Python runner only
convert. A method of independent inputs refuses a sample through
`InputSpace::independent`, and a random number comes from a stream of the
seed of the analysis (`Stream`, `crates/commol-random/src/streams.rs`).

### Model type (engine)

To add an engine to the list:

- [ ] Add the entry to `EngineType` in `crates/commol-core/src/types/engine.rs`
      and to `ALL`. The compiler then names every `match` to complete in that
      file: its name, the model kinds it runs (`kinds`), whether it is
      stochastic (`is_stochastic`), its settings (`EngineSettings`, parsed
      and validated with `deny_unknown_fields`), the form of its dynamics
      (`dynamics_form`) and the meaning of a rate, which sets the positions it
      reports at (`check_position`; see
      [One list of engines](decisions/engines.md#one-list-of-engines)).
- [ ] Write the engine in its own crate and implement `SimulationEngine`
      (for an engine of the aggregate kind, also `CompartmentalEngine` from
      `commol-model`: it gives its compiled model, `delegate_to_compiled_model!`
      writes the methods that read it, and `record_steps` runs an engine that
      reports only at integer steps); `check_position` is the rule of
      `EngineType`, `engine_type` returns its entry (so `is_stochastic` and
      the messages of a calibration read it), and `try_from_model` rejects a
      model of another engine. A
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
[One interface for every model kind](decisions/model-kinds.md#one-interface-for-every-model-kind)):

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
      `check_calibration_values`, `with_calibrated_values`,
      `report_sections`, `describe` and `check_unit_consistency` when the
      kind has them. Reject in
      `check_model` a feature of the process the kind cannot run, and name
      the form that carries over. Import the module in `context/__init__.py`,
      so `Model.from_json` reads the kind.
- [ ] Do not write a builder: `ModelBuilder.build(structure, engine)` builds
      every kind. Do not change `Simulation`, `Calibrator`, the reports or the
      figures: if one of them needs something of the kind, add a method to
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
- [ ] Take a count as `i64` (a seed as `i128`) and read it with its rule,
      `crate::counts::count(&RULE, value)` (`commol_core::SEED` for a seed),
      so a negative one raises the message of the rule; add the case to
      `tests/uncertainty/test_counts.py`.
- [ ] Update `_commol_rs.pyi`.
- [ ] Rebuild the extension with the test model kind
      ([Quick commands](../development/workflow.md#quick-commands)) before you run
      `pytest`.

### Reports

- [ ] Make a report a method of the object of results whose data it
      describes; its block is a private function of `api/reporting.py`.
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
      change the fork first, then update `Cargo.lock`. The fork keeps the
      numerical methods that know no model: the solvers, the sine transform
      of box bounds (`SineTransform`, `BoundedProblem`), the root finder
      (`NewtonRoot`) and the finite differences (`finitediff`). commol keeps
      what knows the model: the bounds, the start points, the bracket of a
      quantile.

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

Previous: [Coverage and mutation testing](coverage.md) | Back to: [Codebase Guide](index.md)
