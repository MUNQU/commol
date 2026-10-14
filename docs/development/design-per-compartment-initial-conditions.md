# Design: Per-compartment initial conditions

[Back to Design decisions](../codebase/design-decisions.md)

- **Status:** Implemented (revised 2026-10-12: Rust is the single source of truth)
- **Date:** 2026-09-30
- **Scope:** `commol-core` (types), `commol-difference` (engine build), `py-commol` (context, builder, model write-back)
- **Names:** this record keeps the names of its date. Today `population_size`
  is `InitialConditions.total` (a real number), the initial population is the
  initial state, a head count is an amount, and `calibrated_initial_population`
  is `calibrated_initial_state`; see
  [State, amount, total and subgroup total](../codebase/design-decisions.md#state-amount-total-and-subgroup-total).

## Problem

Model authors cannot express initial conditions for a specific expanded
compartment (for example `B_g1 = 0` when the stratification `type` only
applies to group `g2`). The current definition only supports:

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
   - exact expanded compartment name (`B_g1`) → `InitialConditionTarget::exact`,
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

<!-- doc-test: skip: shows the shape of the types, not a runnable example -->
```python
class CompartmentFraction(BaseModel):
    compartment: str   # expanded compartment name: "B_g1", "A_g2_t1", ...
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
  compartments (`B_g2_t1`) and truncated conditional ones (`B_g1`)
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

The Rust engine is the only place that knows the expansion, so it is the only
place that validates names and totals:

- **Rust (`commol_difference::initial_state`):** expands the model and applies
  the overrides. It rejects an unknown or repeated compartment, a fraction
  outside `[0.0, 1.0]`, and overrides that make the initial population exceed
  `population_size` (relative tolerance `1e-9`). The engine constructor uses the
  same function.
- **Python:** keeps only the schema checks that need no expansion (fraction
  range, duplicate names). `Model` has a validator that asks the engine for the
  initial state when overrides exist, so `ModelBuilder.build`,
  `Model.model_validate_json` and `Model.from_json` all get the engine's
  verdict. `Model.update_compartment_fractions` validates a candidate copy the
  same way before it stores it.

The expansion itself is one Rust function (`expand`) shared by the engine
build, `initial_state` and the accumulator outputs. Python reads it through
`expanded_outputs` for printing equations and checking units.

`Model.initial_state()` returns each expanded compartment with its bin, its
applied categories and its initial population, as the engine builds it. The
report and `Model.subgroup_population` use it instead of a Python expansion.

### Applying calibrated initial conditions (engine)

`commol-calibration/src/initial_conditions.rs` owns how calibration values
reach the engine. `cost()`, `configured_engine()` and the public
`calibrated_initial_population` share it.

- **Targets.** An id resolves, in order, to an exact compartment, a qualified
  category (`group:g1`), a bin or bin prefix (its bin followed by its leading
  categories, matched on the labels), or the category of exactly one binary
  stratification. Category pairs come from the engine's
  `compartment_labels()` (bin and applied categories), not from compartment
  names. A bare name that two stratifications declare is an error that names
  the qualified alternatives.
- **Layers.** Values are written as bins, then category splits, then exact
  compartments, as the definition layers bin fractions, stratification
  fractions and overrides. Within a layer, parameter order applies.
- **Remainder.** `initial_condition_remainder` (a compartment, bin or bin
  prefix) is outside the optimizer. After the layers, it takes
  `population_size` minus every other compartment and keeps its own split. It
  must not overlap a calibrated bin or compartment. A negative remainder, or,
  without a remainder, a total above `population_size`, is infeasible: the
  objective returns a penalty and `configured_engine` an error.
- Results report the clamped values the objective scored.

### Calibration write-back

`Model.apply_calibration_parameters`:

1. asks the engine for the scored initial population;
2. gives every bin a calibrated value or the remainder covers its scored share,
   and every calibrated category the split the engine applied (an update the
   definition rejects is skipped);
3. asks the engine for the initial state of that candidate definition;
4. pins every compartment whose head count differs from the scored one as a
   `compartment_fractions` override.

The result rebuilds the scored population exactly. Python does not replay the
engine.

## Tests

1. **Pydantic:** validator coverage (duplicates, range, unknown compartment).
2. **Rust (`commol-difference` tests):** override application, truncated
   conditional compartment, unknown-name error, population conservation.
3. **Integration:** end-to-end `B_g1 = 0` scenario (build → simulate →
   series stay zero); calibration round-trip (exact-compartment IC parameter
   → calibrate → apply parameters → rebuild engine → identical initial state).
4. **Write-back parity:** for each mix of bin, category and exact-compartment
   values, the rebuilt population equals `calibrated_initial_population`.

## Non-goals

- Per-compartment initial values for *calibrated* fractions beyond what the
  exact-target path already supports.
- Changing the expansion or conditional-stratification semantics.
- Absolute (head-count) definition syntax; can be added later on top if
  needed.
