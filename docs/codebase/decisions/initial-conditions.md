# Initial conditions

[Back to Design decisions](../design-decisions.md)

How the fractions of the categories compose, and how an amount of one
compartment overrides them. How calibrated values reach the initial state is in
[Calibrated initial conditions](calibrated-initial-conditions.md). The initial
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

## A split can differ by the category of an earlier stratification

The fractions of a stratification are one split
(`StratificationFractions::Split`) or one split per category of a
stratification declared before it that its compartments have
(`StratificationFractions::ByCategory`, in
`crates/commol-core/src/types/aggregate.rs`). Its JSON names that
stratification under `"by"`, next to the splits:
`{"by": "group", "g1": {"t1": 0.25, "t2": 0.75}, "g2": {...}}`. The fraction
of a compartment is the fraction in the category of `by` it has
(`StratificationFractions::fraction`), so `A_g1_t1` starts with
`bin amount × fraction(g1) × fraction(t1 in g1)`, and the stratifications
stay apart in names, rates and conditions.

**Rule:** the engine checks the fractions against the stratifications
(`check_stratification_fractions` in `builder.rs`, from `check_model` and
`initial_state`): each stratification and each category named once, `by`
declared before the stratification it splits, every compartment of that
stratification with a category of `by`, one split per category of `by` that
those compartments have, so a split of a category that a condition leaves out
is refused, as an extra category is. A
field `"by"` that holds a text is the stratification, so a category may be
named `by` in one split, and the stratification `by` names cannot have a
category named `by`, whose split would take the key. Python sends the data and
reads the splits (`split_by`, `category_splits` in
`_model/initial_conditions.py`); an update of a category writes it into every
split, as the engine writes a calibrated category.

**Rule:** a calibrated category keeps every split by category
(`paired_category` in `crates/commol-model/src/initial_conditions.rs`, from
`CompiledModel::splits_by`). A category of the split stratification takes its
value inside every category of `by`. A category of `by` pairs the groups of
compartments that differ only in `by` and in the stratifications split by it,
directly or through another, so each side keeps the split of those
stratifications. Pairing one compartment with one would mix the splits of
`g1` and `g2`.

| Alternative | Why not |
|---|---|
| A new field of `Aggregate` | The split of `type` belongs where the reader looks for the fractions of `type`. |
| Keys that join the categories (`"g1:t1"`) | A nested split checks the sum of each category on its own, and leaves room for a split by two stratifications. |
| One stratification of every pair (`g1_t1`, …) | A rate of `g1` then sums every pair of it, and a condition names each pair. |

**Guards:**
`crates/commol-model/tests/initial_conditions.rs::a_split_by_category_names_an_earlier_stratification`,
`crates/commol-model/tests/initial_conditions.rs::the_fractions_of_each_category_sum_to_one`,
`crates/commol-model/tests/initial_conditions.rs::a_category_of_the_stratification_a_split_differs_by_keeps_the_split`,
`crates/commol-model/tests/initial_conditions.rs::a_category_keeps_a_chain_of_splits_that_differ_by_it`,
`crates/commol-model/tests/initial_conditions.rs::a_split_by_category_names_no_category_it_does_not_reach`,
`crates/commol-model/tests/initial_conditions.rs::a_side_that_starts_empty_keeps_its_declared_split`,
`tests/calibration/test_initial_conditions.py::test_the_write_back_of_a_split_by_category_rebuilds_the_scored_state`,
`tests/initial_conditions/test_stratification_fractions.py::test_a_stratification_splits_each_category_of_an_earlier_one`.

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
([Calibrated initial conditions](calibrated-initial-conditions.md#calibrated-initial-conditions-are-applied-in-layers)).

**Rule:** only the engine knows the expansion, so only the engine checks an
override: `initial_state` rejects a name that is not an expanded compartment
and an amount that is not finite or is negative, and checks every override
before it applies one. `Aggregate.check_model` and
`Aggregate.update_compartment_amounts` (on a copy, so a rejected update
leaves the structure as it was) ask the engine through
`engine_state.initial_state`; Python keeps only the checks of the schema.

**Rule:** an override states the amount of its compartment whatever its bin
holds, `None` too. Of a bin left `None`, the build marks as without a value
only the compartments that no override names (`unvalued_compartments`, filled
in `builder.rs`), and every reader takes that one record: `total` counts the
override, `check_values` names the other compartments of the bin, the
remainder keeps the override as an amount the model states (`states_none_of`),
and `set_initial_condition` gives a value as an override does
([A run needs every value](model-structure.md#a-run-needs-every-value)). With
`{"A": None, "B": 1000}` split in halves and `{"A_g1": 500}`, a run names
`A_g2`, and the remainder `A` keeps 500. A value or a remainder that also
sets a compartment without a value splits by the fractions, as over a bin of
which nothing is stated (`weights` in
`crates/commol-model/src/initial_conditions.rs`), not by the initial state,
where that compartment holds a 0 no one stated: the remainder `A` gives 250
to each half.

| Alternative | Why not |
|---|---|
| A fraction of a total | A model has no total: bins of different units start without a sum over them. |
| An amount per combination of categories before the expansion | It repeats the expansion in Python and needs every combination, also those a condition leaves out. |
| An override on a bin left `None` is a build error that names both | A run needs every value: a known amount of one compartment would need an amount of its bin that no one knows. DRY: the write-back of a calibration of part of such a bin (`calibrated_structure`, `_model/calibration_write_back.py`) keeps the bin `None` and pins the compartments the values set, so it would write a model the build refuses (`tests/initial_conditions/test_compartment_amounts.py::TestOverrideOnABinLeftNone::test_a_calibration_of_the_other_compartments_is_written_back`). |
| `total` counts such an override, and the record of values does not | The remainder keeps a sum that a run refuses, and the message asks for a value of a bin that holds one in part (DRY: two readings of one number). |
| No reader counts such an override | The model holds a number that only the split of a value reads, and an initial state that no run starts from (YAGNI). |
| One override gives its whole bin a value | The compartments it does not name start at `0`, which no one stated (A run needs every value). |
| A value over such a bin splits by the initial state | The same `0`: a value of `A` or a remainder `A` gives the compartments without a value nothing, and they then count as stated (A run needs every value). |

**Guards:**
`crates/commol-model/src/builder.rs::override_on_truncated_conditional_compartment`,
`crates/commol-model/src/builder.rs::unknown_compartment_override_is_a_build_error`,
`crates/commol-model/tests/initial_conditions.rs::an_override_gives_its_compartment_a_value_on_a_bin_left_none`,
`crates/commol-model/tests/initial_conditions.rs::a_value_over_an_override_and_compartments_without_one_splits_by_the_fractions`,
`tests/initial_conditions/test_compartment_amounts.py`.
