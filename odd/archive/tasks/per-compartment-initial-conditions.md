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
- [x] T9. Full test suites pass (Python pytest + cargo test); changelog entry
  (commit: 086e716 fix: harden per-compartment IC validation and replay calibration write-back)

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

## Judgment Day round 1 fixes

- `JD-A-001`: quoted the mkdocs nav title containing a colon-space; `yaml.safe_load` parses
  `mkdocs.yaml`.
- `JD-B-001`: `apply_calibration_parameters` now writes per-compartment overrides when an
  aggregate bin or stratification-category value intersects an existing override. The shared
  `context/compartments.py::uniform_compartment_fractions` computes the base expansion and
  weights, keeping the first entry per stratification id like the engine.
- WARNINGs: unknown compartments reach Python as `RuntimeError` from the PyO3 boundary; the
  Python generator treats a category-less condition as satisfied; `update_compartment_fractions`
  validates names and post-upsert totals; the initial-conditions report lists declared
  compartment overrides.
- Evidence: `uv run pytest tests -x -q` -> 524 passed; `cargo test -p commol-difference -p
  commol-core` -> 90 + 21 passed; `cargo fmt --check` and `cargo clippy --all-targets -- -D
  warnings` clean; `uv run ruff format --check . && uv run ruff check .` clean.

## Judgment Day round 2 fixes

- `JD-B-001`: `Model.apply_calibration_parameters` replays initial-condition values in problem
  parameter order over a working population that starts from the declared expansion plus the
  existing overrides. Bin values scale their covered compartments by weights taken from that
  starting population; category values split each complementary pair by the live pair total.
  A bin or category value whose covered compartments intersect an existing override or a
  compartment affected by an earlier value is persisted as per-compartment overrides holding the
  replayed values.
- W3-residual: `Model.update_compartment_fractions` validates the true post-upsert compartment
  total before storing, so a rejected call leaves the model unchanged; the coarse check in
  `InitialConditions.update_compartment_fractions` stays as an atomic fallback.
- A#5: override rows in `_initial_conditions_report` report `calibrated` when the compartment
  id is calibrated or a calibrated bin/category id covers the compartment.
- A#6: the duplicated condition predicates in `context/_model/helpers.py` and
  `context/_model/equations.py` now use `stratification_conditions_met`.
- A#7: removed the unreachable category branch in `_covered_compartments`; reworded the
  `uniform_compartment_fractions` docstring; updated the design doc interaction rules; added the
  `RuntimeError`/`ValueError` changelog notes.
- Evidence: `uv run pytest tests -x -q` -> 532 passed; `uv run ruff format --check . && uv run
  ruff check .` and `uv run ty check commol` clean; `mkdocs.yaml` parses unchanged.
- Engine replay check (throwaway script): for aggregate/category orderings over an overridden and
  a plain model, the reloaded engine population equals the population produced by applying the
  same values through `engine.set_initial_condition` in order (9/9 orderings match).
- T9 closed: full verification re-run on the uncommitted tree by the verify agent before commit
  086e716: `uv run pytest tests -x -q` -> 532 passed; `cargo test -p commol-difference -p
  commol-core` -> 90 + 21 + 1 passed; `cargo fmt --check`, `cargo clippy --all-targets -- -D
  warnings`, `uv run ruff format --check .`, `uv run ruff check .`, `uv run ty check commol` all
  clean; `mkdocs.yaml` parses.


## Residual resolution after Judgment Day ESCALATED (user decision: keep delta, resolve now)

Design decisions (recorded in odd/archive/judgment-day/per-compartment-initial-conditions-ledger.md):

- ESC-1 policy: the model definition may represent any state the engine can hold.
  Total guards live in authoring paths only (ModelBuilder construction, public
  update_* methods). Deserialization must accept whatever the write-back
  persisted, so 'a reload reproduces the scored head counts bit-identically'
  holds unconditionally.
- ESC-2 policy: engine parity is the primary invariant. The calibration
  write-back is a pure function of (current definition, problem values) and
  equals the engine's sequential replay exactly, including pre-existing
  overrides and multi-parameter orderings. Bin and unique-category replays are
  idempotent (pair totals are invariant under 0.4+0.6=1); a shared-category
  value is intentionally NOT idempotent on re-apply — re-applying re-scores,
  exactly like the engine. The round-2 declared-base branch is withdrawn.

Resolution (commit: 37bb0ab fix: resolve escalated JD residuals with
engine-parity write-back) — all escalated residuals closed:

- ESC-2/JD-G-1 fixed: declared-base branch withdrawn; live replay per parameter
  (engine parity incl. pre-existing overrides and mixed multi-parameter dicts);
  shared-category double-apply test rewritten to re-scoring semantics.
- ESC-1 fixed: round-2 Pydantic construction total guard withdrawn; reload of
  non-conserving states is unconditional; ModelBuilder and public update_*
  keep their guards.
- JD-G-3 fixed: non-conserving bin values divert to per-compartment overrides;
  bin conservation guard never bypassed; conserving values still update the
  fraction.
- ESC-3 fixed: conditional-category report intervals rescaled to the
  population unit (bracket the reported fraction).
- ESC-4 fixed: PyO3 bin_fractions constructor accepts Option<f64>.
- Verification: pytest 571 passed; cargo 90+25 passed; fmt/clippy/ruff/ty
  clean; independent verify agent 7/7 PASS before commit.
