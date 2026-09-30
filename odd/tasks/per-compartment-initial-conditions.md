# Feature: Per-compartment initial conditions

Design doc: `docs/development/design-per-compartment-initial-conditions.md`
Branch: `feature/per-compartment-initial-conditions`

## Tasks

- [x] T1. Persist design doc and mkdocs nav entry (commit: pending)
- [x] T2. Python types: `CompartmentFraction` + `InitialConditions.compartment_fractions` with validators (`py-commol/commol/context/initial_conditions.py`)
- [x] T3. Python compartment-name generator with parity test (`py-commol/commol/context/compartments.py` + tests)
- [x] T4. Builder API: `set_initial_conditions(..., compartment_fractions=None)` with validation (`py-commol/commol/api/model_builder.py`)
- [x] T5. Rust types: `CompartmentFraction` + `InitialConditions.compartment_fractions` with serde default (`crates/commol-core/src/types/population.rs`)
- [x] T6. Rust engine: apply overrides in `initialize_population` + tests (`crates/commol-difference/src/builder.rs`)
- [x] T7. Model write-back: `update_compartment_fractions` + routing in `apply_calibration_parameters` (`py-commol/commol/context/model.py`)
- [x] T8. Integration tests: `IH_y60 = 0` end-to-end + calibration round-trip
- [ ] T9. Full test suites pass (Python pytest + cargo test); changelog entry

## Evidence

- T2/T3/T4: `uv run pytest tests/test_compartments.py tests/test_compartment_initial_conditions.py -q`
  -> all pass; generator output equals `engine.compartments` for unconditioned, plain, and
  conditional stratifications.
- T5: `cargo test -p commol-core types::population` -> 2 passed (old JSON deserializes with empty
  `compartment_fractions`; round-trip keeps the override). `CompartmentFraction` exported from
  `commol_core::types` and `commol_core` like every other population type.
- T6: `cargo test -p commol-difference -p commol-core` -> 90 + 21 passed; override applied,
  truncated conditional compartment overridden, unknown name errors, population conserved.
  The three test-module `InitialConditions` literals in `simulation.rs` and the one in
  `py-commol/src/core.rs` now include the new field (approved surface extension).
- T7: `apply_calibration_parameters` routes exact expanded compartments into
  `InitialConditions.compartment_fractions` via the new `Model.update_compartment_fractions`;
  bin and category routing unchanged. Superseded rejection test renamed to
  `test_per_compartment_initial_condition_is_written_back_not_silently_skipped` and an upsert
  test added.
- T8: `py-commol/tests/test_compartment_initial_conditions.py` covers the zero-override
  end-to-end scenario and the exact-compartment calibration round-trip (write-back + rebuild
  yields the calibrated initial population); no-override expansion preserved.
- Full verification: `uv run maturin develop`; `uv run pytest tests -x -q` -> 513 passed;
  `cargo test -p commol-difference -p commol-core` -> all passed;
  `cargo fmt --check && cargo clippy --all-targets -- -D warnings` -> clean;
  `uv run ruff format --check . && uv run ruff check .` -> clean; `uv run ty check commol` -> clean.
- Conservation semantics: uncalibrated (`None`) bin fractions are treated as an unknown base of
  0.0; the check is sound (never rejects a configuration whose calibrated total stays within
  `population_size`) but not complete for the unknown remainder. Documented in
  `model_builder.py` and covered by two tests.
