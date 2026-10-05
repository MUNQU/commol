# Design: Per-compartment initial conditions

- **Status:** Proposed
- **Date:** 2026-09-30
- **Scope:** `commol-core` (types), `commol-difference` (engine build), `py-commol` (context, builder, model write-back)

## Problem

Model authors cannot express initial conditions for a specific expanded
compartment (for example `IH_y60 = 0` when the hospital pathway only exists
for the `oe60` age stratum). The current definition only supports:

- `bin_fractions`: fraction of the total population per bin, expanded
  uniformly across every stratification category combination.
- `stratification_fractions`: category splits applied to *every* bin.

The uniform expansion cannot express "this bin starts empty in one stratum
and seeded in another".

## What already exists (constraints discovered)

The low-level machinery already works per compartment; only the definition
layer is missing:

1. `SimulationEngine::set_initial_condition(compartment_index, value)`
   exists in the trait (`commol-core`) and is implemented
   (`commol-difference/simulation.rs`).
2. The Rust calibration layer already resolves `InitialCondition`
   parameters to targets by id (`calibration_problem.rs`):
   - exact expanded compartment name (`IH_y60`) → `InitialConditionTarget::exact`,
   - bin id or bin prefix → `aggregate`,
   - binary stratification category → `paired_category`.
   Values are fractions of the total population, converted to absolute
   head counts, with auto-correction of the fixed + calibrated fraction sum.
3. The Python `Calibrator` already accepts compartment-named
   `InitialCondition` parameters.
4. `CalibrationConstraint`/auto-correction reads the base engine population,
   so definition-level overrides compose coherently with calibrated
   aggregate parameters.

Missing: the model definition cannot *declare* per-compartment values, the
engine build cannot *apply* them, and Python's
`Model.apply_calibration_parameters` refuses to *write back* compartment
initial conditions.

## Proposal: `compartment_fractions` as post-expansion overrides

### Data model

```python
class CompartmentFraction(BaseModel):
    compartment: str   # expanded compartment name: "IH_y60", "S_oe60_v_c", ...
    fraction: float    # fraction of the TOTAL population, 0.0 <= fraction <= 1.0

class InitialConditions(BaseModel):
    population_size: int
    bin_fractions: list[BinFraction]
    stratification_fractions: list[StratificationFractions] = []
    compartment_fractions: list[CompartmentFraction] = []   # new
```

Mirrored in Rust (`commol-core/src/types/population.rs`) with
`#[serde(default)]` for backward compatibility with existing model JSON.

### Semantics

- **Fraction of the total population** (not of the bin, not absolute head
  counts). Rationale: mirrors `bin_fractions` naming and matches exactly the
  semantics the calibration layer already uses for exact-compartment targets
  (`absolute = fraction * population`).
- **Post-expansion override.** The uniform expansion — including the
  conditional-stratification logic that keeps some compartments unsplit —
  is unchanged. Overrides are applied afterwards, keyed by final
  compartment name. This works uniformly for fully suffixed
  compartments (`IH_oe60_v_c`) and truncated conditional ones (`IH_y60`)
  without special cases.
- **Precedence:** definition override wins over the uniform expansion;
  calibration wins over the definition at runtime (engine `reset()` +
  `set_initial_condition`); write-back persists calibration results into
  the definition. An empty list reproduces current behaviour exactly.
- **Backward compatible:** old model JSON files validate unchanged
  (`serde(default)` / Pydantic default).

### Application point (single place)

`crates/commol-difference/src/builder.rs::initialize_population`: after the
existing `distribution: HashMap<String, f64>` is built, apply each override
(`value = population_size * fraction`), erroring on an unknown compartment
name. Because `initialize_population` feeds `initial_population`, `reset()`
and the calibration auto-correction inherit overrides for free; neither
`simulation.rs` nor `calibration_problem.rs` changes.

### Validation

Two layers, one source of truth for compartment names:

- **Python (builder, early feedback):** `ModelBuilder.set_initial_conditions(...,
  compartment_fractions=None)` validates: (a) every compartment name exists
  in the expansion; (b) no duplicates; (c) `0.0 <= fraction <= 1.0`;
  (d) total population after overrides does not exceed `population_size`.
  For (a) a pure-Python compartment-name generator mirrors the Rust
  `generate_compartments` (including conditional-stratification behaviour).
  It becomes the shared helper for builder-side validation, with a parity
  test against `engine.compartments` guarding divergence.
- **Rust (engine, final authority):** unknown compartment names are a build
  error.

### Calibration write-back

`py-commol/commol/context/model.py::apply_calibration_parameters` replaces
its current rejection branch ("Per-compartment initial conditions ... cannot
be written back") with: an `InitialCondition` parameter whose id matches an
expanded compartment name is persisted as an upsert into
`compartment_fractions` (new `Model.update_compartment_fractions`, following
the existing `update_bin_fractions` / `update_stratification_fractions`
pattern). Reloading a calibrated run therefore rebuilds the exact initial
state.

### Interaction rules

- Initial-condition calibration values are replayed in problem parameter
  order over a working population that starts from the declared expansion plus
  the definition overrides. A bin-level value scales its covered compartments
  by weights taken from that starting population; a stratification-category
  value splits each complementary pair by the live total of the pair.
- A bin or category value whose covered compartments intersect a definition
  override, or a compartment affected by an earlier value in the replay, is
  persisted as per-compartment overrides holding the replayed values. Every
  other value updates the bin or stratification fraction it declares. An exact
  compartment value is always persisted as an override.

## Tests

1. **Pydantic:** validator coverage (duplicates, range, unknown compartment).
2. **Rust (`commol-difference` tests):** override application, truncated
   conditional compartment, unknown-name error, population conservation.
3. **Integration:** end-to-end `IH_y60 = 0` scenario (build → simulate →
   series stay zero); calibration round-trip (exact-compartment IC parameter
   → calibrate → apply parameters → rebuild engine → identical initial state).
4. **Expansion parity:** Python generator output equals `engine.compartments`
   for models with and without conditional stratifications.

## Non-goals

- Per-compartment initial values for *calibrated* fractions beyond what the
  exact-target path already supports.
- Changing the expansion or conditional-stratification semantics.
- Absolute (head-count) definition syntax; can be added later on top if
  needed.
