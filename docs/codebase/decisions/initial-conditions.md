# Initial conditions

[Back to Design decisions](../design-decisions.md)

How initial fractions compose, and how calibrated values reach the initial state.

## Subgroup totals compose down the condition chain

The fractions of a conditional stratification are relative to the subgroup that
its conditions select, not to the total. Subgroup totals therefore
multiply along the chain: `total × fraction(group1) × fraction(cat1)`.
`Aggregate.subgroup_total` and `Aggregate.get_conditioning_categories` own
this (`context/aggregate.py`); with per-compartment overrides,
`subgroup_total` sums the state the engine builds.

**Guard:** `tests/simulation/test_outputs.py::TestSubgroupTotal` compares the
result with the state the engine starts from.

## Calibrated initial conditions are applied in layers

`InitialConditionPlan` in `crates/commol-model/src/initial_conditions.rs`
applies calibrated values in a fixed order: bins, category splits, exact
compartments, and then the remainder named by
`CalibrationProblem.initial_condition_remainder`. Both engines of the aggregate
kind
resolve and write initial conditions through it
(`SimulationEngine::plan_values` and `apply_values`), and the calibration
writes every value through the engine (`ValueWriter` in
`crates/commol-calibration/src/values.rs`), so the objective, the predictions
and the write-back (`configured_engine`, then the binding
`calibrated_initial_state`) agree by construction.
`Aggregate.with_calibrated_values`
(`context/_model/calibration_write_back.py`) writes the values back as
fractions, and pins as per-compartment overrides every compartment that the
fractions cannot reproduce.

**Guards:**
`tests/calibration/test_write_back.py::TestApplyCalibrationParameters::test_applied_model_reproduces_the_loss_the_optimizer_scored`;
`crates/commol-model/tests/initial_conditions.rs` (the layers, the remainder
and the targets of an id).
See also [Per-compartment initial conditions](../../development/design-per-compartment-initial-conditions.md).
