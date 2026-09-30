# Feature: Per-compartment initial conditions

Design doc: `docs/development/design-per-compartment-initial-conditions.md`
Branch: `feature/per-compartment-initial-conditions`

## Tasks

- [x] T1. Persist design doc and mkdocs nav entry (commit: pending)
- [ ] T2. Python types: `CompartmentFraction` + `InitialConditions.compartment_fractions` with validators (`py-commol/commol/context/initial_conditions.py`)
- [ ] T3. Python compartment-name generator with parity test (`py-commol/commol/context/compartments.py` + tests)
- [ ] T4. Builder API: `set_initial_conditions(..., compartment_fractions=None)` with validation (`py-commol/commol/api/model_builder.py`)
- [ ] T5. Rust types: `CompartmentFraction` + `InitialConditions.compartment_fractions` with serde default (`crates/commol-core/src/types/population.rs`)
- [ ] T6. Rust engine: apply overrides in `initialize_population` + tests (`crates/commol-difference/src/builder.rs`)
- [ ] T7. Model write-back: `update_compartment_fractions` + routing in `apply_calibration_parameters` (`py-commol/commol/context/model.py`)
- [ ] T8. Integration tests: `IH_y60 = 0` end-to-end + calibration round-trip
- [ ] T9. Full test suites pass (Python pytest + cargo test); changelog entry

## Evidence

(appended per task: commit id + test run)
