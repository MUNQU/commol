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
5. Run both test suites (see [`AGENTS.md`](https://github.com/MUNQU/commol/blob/develop/AGENTS.md#commands)).
6. Update the documentation that describes the change.

## Guardrails

- **Do not duplicate an engine rule in Python.** Add a binding instead, and
  call it from `context/_model/engine_state.py`.
- **Do not identify compartments by name.** Use labels (see
  [Compartments are labels](design-decisions.md#compartments-are-labels-with-a-name)).
- **Do not drop invalid input silently.** Reject it with a message that names
  the transition, compartment or parameter.
- **Do not change saved formats by accident.** Model JSON and report key order
  are part of the contract.
- **Do not trust `odd/` as specification.** It records past work; the code and
  this guide describe the present.

## Checklists by type of change

### Model structure (bins, stratifications, accumulators)

- [ ] Change the Rust type in `crates/commol-core/src/types/` and the Pydantic
      model in `py-commol/commol/context/` together.
- [ ] Update the PyO3 binding and `_commol_rs.pyi` if the type crosses the
      boundary.
- [ ] Update expansion in `builder.rs` (`expand`) if compartments change.
- [ ] Check that `test_expansion_matches_the_engine_output_names` still passes.
- [ ] Update [Building models](../guide/building-models.md) and
      [Model structure](../advanced/model-structure.md).

### Transitions and rates

- [ ] Plan the flows in `flows.rs`; reject what cannot be simulated.
- [ ] Check the [rate semantics](design-decisions.md#absolute-and-per-capita-rates).
- [ ] Add a case to `test_transition_flows.py`, and check `print_equations`.
- [ ] Update [Building models](../guide/building-models.md).

### Expressions (functions, variables)

- [ ] Change `crates/commol-core/src/math_expression/`, including the JIT
      (`jit/`) and its tests.
- [ ] Add a new function to `SAFE_FUNCTIONS` in `utils/security.py`.
- [ ] Update [Mathematical expressions](../guide/mathematical-expressions.md).

### Initial conditions

- [ ] Change the Pydantic validation in `context/initial_conditions.py` and the
      engine rule in `builder.rs` (`initial_state`).
- [ ] Check that `Model.initial_state()` reports the new state.
- [ ] If calibration can set it, update `initial_conditions.rs` in
      `commol-calibration` and the write-back.
- [ ] Check that `test_applied_model_reproduces_the_loss_the_optimizer_scored`
      still passes.

### Calibration

- [ ] Keep the loss and the constraints in `calibration_problem.rs`.
- [ ] Convert new options in `api/calibrator.py` and
      `api/probabilistic/calibration_runner.py`.
- [ ] Check that `test_windows.py::TestAgreementWithTheCalibrationLoss` still
      passes if the loss changes.
- [ ] Update [Calibration](../guide/calibration.md) and, for ensembles,
      [Probabilistic calibration](../guide/probabilistic-calibration.md).

### PyO3 bindings

- [ ] Keep logic out of `py-commol/src/`: convert types and call the crate.
- [ ] Update `_commol_rs.pyi`.
- [ ] Rebuild with `maturin develop --release` before you run `pytest`.

### Reports

- [ ] Keep the key order of existing entries.
- [ ] Use `member_statistics` and `ci_percentiles` for ensemble values.
- [ ] Update `test_reporting.py`.

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
- [ ] Every new claim in the docs names a file, function or test.
- [ ] The tests that guard the touched [design decisions](design-decisions.md) still pass.

## Navigation

Previous: [Conventions](conventions.md) | Back to: [Codebase Guide](index.md)
