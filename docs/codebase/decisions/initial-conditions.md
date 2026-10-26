# Initial conditions

[Back to Design decisions](../design-decisions.md)

How the fractions of the categories compose, how an amount of one compartment
overrides them, and how calibrated values reach the initial state. The initial
state itself is an amount per bin
([Model structure](model-structure.md#the-initial-state-is-an-amount-per-bin)).

## Category fractions compose down the condition chain

The fractions of a conditional stratification are relative to the subgroup that
its conditions select, not to the whole bin. The amount of a compartment
therefore multiplies along the chain:
`bin amount × fraction(group1) × fraction(cat1)` (`initial_state` in
`crates/commol-model/src/builder.rs`). `Aggregate.subgroup_total` sums the state
the engine builds, overrides included, and
`Aggregate.get_conditioning_categories` names the subgroup
(`_model/aggregate.py`).

**Guard:** `tests/simulation/test_outputs.py::TestSubgroupTotal` compares the
result with the state the engine starts from.

## A compartment amount overrides the expansion

`compartment_amounts` (`Aggregate`, and `InitialConditions` in
`crates/commol-core/src/types/aggregate.rs`) sets the amount of one expanded
compartment, such as `{"B_g1": 0}`, in the unit of its bin. `initial_state`
(`builder.rs`) applies it after the expansion, keyed by the name of the
compartment, so a conditional stratification that leaves a compartment
unsplit needs no special case, and the expansion does not change. An override
replaces what the bin amount and the fractions give, and it may be above the
amount of its bin. A calibrated value replaces the definition when a run
starts (`apply_values`), and the write-back writes overrides
([below](#calibrated-initial-conditions-are-applied-in-layers)).

**Rule:** only the engine knows the expansion, so only the engine checks an
override: `initial_state` rejects a name that is not an expanded compartment
and an amount that is not finite or is negative, and checks every override
before it applies one. `Aggregate.check_model` and
`Aggregate.update_compartment_amounts` (on a copy, so a rejected update
leaves the structure as it was) ask the engine through
`engine_state.initial_state`; Python keeps only the checks of the schema.

| Alternative | Why not |
|---|---|
| A fraction of a total | A model has no total: bins of different units start without a sum over them. |
| An amount per combination of categories before the expansion | It repeats the expansion in Python and needs every combination, also those a condition leaves out. |

**Guards:**
`crates/commol-model/src/builder.rs::override_on_truncated_conditional_compartment`,
`crates/commol-model/src/builder.rs::unknown_compartment_override_is_a_build_error`,
`tests/initial_conditions/test_compartment_amounts.py`.

## Calibrated initial conditions are applied in layers

`InitialConditionPlan` in `crates/commol-model/src/initial_conditions.rs`
applies calibrated values in a fixed order: bins, category splits, exact
compartments, and then the remainder named by
`CalibrationProblem.initial_condition_remainder`. A bin, prefix or compartment
value is an amount in the unit of its bin; a category value is a split.
Every engine of the aggregate kind (difference, differential and the two
stochastic engines) resolves and writes initial conditions through it
(`SimulationEngine::plan_values` and `apply_values`), and the calibration
writes every value through the engine (`ValueWriter` in
`crates/commol-calibration/src/values.rs`), so the objective, the predictions
and the write-back (`configured_engine`, then the binding
`calibrated_initial_state`) agree by construction.
`Aggregate.with_calibrated_values`
(`_model/calibration_write_back.py`) writes back the amount of every bin
whose every compartment a value or the remainder sets and the split of every
calibrated category, and pins as per-compartment overrides every compartment
that these cannot reproduce.

An id names, in this order: an expanded compartment (`A_g1`), a qualified
category (`group:g1`), a bin or a bin prefix (`A`, or `A_g1` over a deeper
stratification), or a category of exactly one binary stratification (`g1`)
(`InitialConditionPlan::new`). The plan matches categories on the labels of
the compartments, never on their names. A bare category that two
stratifications declare is an error that names the qualified ids.

**Rule:** the plan reads the initial state of the engine
(`CompartmentalEngine::initial_state`), never its current state: the split of
a bin, a prefix or the remainder and the pair totals of a category. A
`Calibrator` takes the engine of its `Simulation`, which holds the last state
of a run.

**Guards:**
`crates/commol-model/tests/initial_conditions.rs::a_bin_and_the_remainder_keep_the_split_of_the_initial_state_after_a_run`,
`crates/commol-model/src/initial_conditions.rs::category_shared_by_two_stratifications_is_ambiguous`,
`tests/calibration/test_initial_conditions.py::test_a_calibrator_on_a_simulation_that_has_run_splits_a_bin_by_the_initial_state`.

## The remainder keeps the sum the model declares

The remainder takes the sum of the amounts the model states
(`CompartmentalEngine::total`, fixed when the engine is built), less every
other compartment after the values are written. A compartment that no value
covers keeps its declared amount and cancels out, so the remainder keeps the
declared sum of the compartments the values and it cover, and a bin of another
unit does not enter. A bin left `None` counts as 0, so the model states the sum
on the remainder (`{"A": 1000, "B": None}`). Without a remainder every state is
feasible, because the engine rejects a negative amount. With one, a remainder
below zero is infeasible by its share of the amounts the values set, a number
with no unit that is never a division by zero.
`ForwardModel::configure` (`forward.rs`) puts the message of the plan of
values (`ValuePlan::infeasibility`) in the `Violation::InfeasibleStart` it
returns, so a calibration (`CalibrationProblem::configured_engine`) and a
predictive draw (`prediction_error` in `py-commol/src/inference.rs`) give the
same message, and the binding needs no access to the problem
(`tests/uncertainty/test_predictive.py::TestBands::test_a_draw_with_an_infeasible_start_names_its_set`).

Units live only in Python (the Rust `Bin` has none), so
`Aggregate.check_calibration_values` rejects a remainder whose unit is not the
unit of every compartment a value sets, from the coverage the engine resolves
(`InitialConditionPlan::coverage`, the binding `initial_condition_coverage`).

| Alternative | Why not |
|---|---|
| A total in the problem that the remainder keeps | It repeats a number of the model and can contradict it. |
| The sum of the bins of the unit of the remainder | The engine has no units, and the unit of a bin is optional free text. |
| The deficit as a share of the sum the remainder keeps | That sum can be 0, as when every bin starts empty. |

**Guards:**
`tests/calibration/test_write_back.py::TestApplyCalibrationParameters::test_applied_model_reproduces_the_loss_the_optimizer_scored`,
`tests/calibration/test_write_back.py::TestWriteBackReadability::test_an_amount_in_the_unit_of_its_bin_is_written_back`,
`tests/calibration/test_initial_conditions.py::TestRemainderUnits`;
`crates/commol-model/tests/initial_conditions.rs` (the layers, the remainder
and the targets of an id, and
`the_remainder_keeps_the_declared_sum_of_what_it_covers`).
