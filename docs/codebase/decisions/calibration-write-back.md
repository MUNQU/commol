# Calibration write-back

[Back to Design decisions](../design-decisions.md)

How `Model.apply_calibration_parameters` writes the values of a calibration
back onto a model, what a run of the model it writes holds, and how
`Model.applied_parameters` reports what it set. How calibrated values reach
the initial state during the calibration is in
[Calibrated initial conditions](calibrated-initial-conditions.md).

## A calibrated amount is written back as the optimizer found it

**Rule:** with a rounding, `check_model` builds every initial amount, whole or
not, of a bin as of a compartment (`check_whole_units` in
`crates/commol-model/src/builder.rs`), because a run rounds each bin at its
start (`Units::begin`) whether the model declares its amount or calibration
writes it. So `Model.apply_calibration_parameters` writes the number the
optimizer found, such as `B = 11.853…`, the model builds on the same engine,
and a run of it rounds the amounts the calibration scored by the same rule:
12 units of `B`. With no rounding, the engine refuses a known value that is
not whole when it writes it (`Units::set_initial_condition`), so the
write-back refuses `{"B": 12.3}` and the model never holds an amount it
cannot build.

The write-back rebuilds the scored amounts as it does on every engine, within
`_MATCH_TOLERANCE` (`_model/calibration_write_back.py`). For a bin of one
compartment it writes the amount itself, so a run of one seed is the run the
calibration scored. In a bin of several compartments the rebuilt amounts can
differ in their last bits. `"largest_remainder"` starts from the same units
unless two remainders tie within those bits. `"random"` can draw other units.
The last category with a positive share draws nothing
([Stochastic engines and seeds](engines.md#stochastic-engines-and-seeds)),
but every other conditional share of a bin still meets the branches of
`binomial` (`crates/commol-random/src/sample.rs`), a share of 0.5 and a mean
of 10, which an equal split always hits, or draws a count that its last bits
change. So the draw alone does not make a written-back model run as scored:
the pins of the write-back
([#474](https://github.com/MUNQU/commol/issues/474)) do, and where a rebuilt
amount still differs in its last bits, a replicate of the written model can
differ along its whole path, and the seed of the problem can give a loss
other than `result.final_loss`.

| Alternative | Why not |
|---|---|
| Write back the whole units a run starts from | KISS: `"random"` draws them again in each run, so no one state holds them, and the largest remainders of a rounded bin differ from those of the bin: `10.4 × (0.35, 0.65)` starts from `(3, 7)`, and `10 × (0.35, 0.65)` from `(4, 6)`. The golden rule and DRY: Python would repeat the rounding of the engine, and the model would hold a number the optimizer did not find. It is fragile too: the check it keeps sums the overrides into the bin, so a whole override `A_g1 = 4` in a bin of 100 split in thirds makes the bin 70.67, and the model is refused. |
| Round a calibrated amount when the engine writes it | S: the engine would round at the write and again at the start of the run. It does not remove the rule of the run: a value that sets only part of a bin, an exact compartment or a prefix, cannot make the bin whole by rounding itself, so `A_g1 = 3.7` written as 4 in a bin of 100 split in thirds still sums to 70.67. |
| Keep refusing a bin that is not whole when the settings name a rounding | DRY: see [Whole units in a stochastic engine](engines.md#whole-units-in-a-stochastic-engine). |

**Guards:**
`tests/stochastic/test_calibration.py::test_a_calibrated_amount_is_written_back_onto_a_model_that_builds`,
`tests/stochastic/test_calibration.py::test_a_stratified_amount_is_written_back_with_its_split`,
`crates/commol-stochastic/tests/oracles.rs::a_bin_that_is_not_whole_starts_as_calibration_writes_it`,
`crates/commol-model/src/builder.rs::a_bin_that_is_not_whole_builds_with_a_rounding`,
`crates/commol-stochastic/src/units.rs::a_bin_of_a_half_rounds_up_whatever_its_split`,
`tests/stochastic/test_whole_units.py::TestWholeUnits::test_a_rounding_makes_each_bin_its_nearest_whole_number`.

## A report marks what the calibration set, as the engine resolves it

**Rule:** `Model.applied_parameters(values, problem, ...)` takes the problem
the values came from, as `apply_calibration_parameters` does, and marks a row
`calibrated` only through it. An id does not say what a value set: a
parameter `g1` and the category `g1` of `group = [g1, g2]` share it, and the
problem declares which one each value is (`parameter_type`;
`declared_values` in `_problem/problem.py` refuses an id it does not
declare). A `parameter` value marks the parameter of its id. The structure
reads the values of its kind, from the problem and the values, as
`with_calibrated_values` does (`ModelStructure.report_sections`). For an
aggregate model, the plan of the write-back resolves the initial conditions
and the remainder on the model (`calibrated_initial_conditions` in
`_model/report.py`, through the binding `initial_condition_coverage` and
`InitialConditionPlan::coverage`). It reads the labels of the compartments,
which a write-back does not change, so it gives the written model the
compartments it gave the calibrated one
(`test_the_overrides_a_calibrated_prefix_pins_report_calibrated` reports on
the written model):

| Row | Calibrated when |
|---|---|
| A bin | The values and the remainder set every compartment of it: the bins whose amount the write-back writes (`covered_bins` in `_model/calibration_write_back.py`) |
| An override | The values and the remainder write its compartment (`TargetCoverage::written`): what a value sets, and both sides of each pair of a split. So each compartment the write-back pins: of a prefix such as `A_g1`, of the remainder, of a split |
| A category split | A value is a split of its stratification: the write-back sets the category and its complement, in every split by category |

Python reads the kind of a value from the engine, never from its id. A
sample gives its spread to the row a value names, and to no other.

| Alternative | Why not |
|---|---|
| `problem` stays optional, and without it the report reads the values by their ids | The golden rule and DRY: Python resolves an id a second way, and misreads it: the parameter `g1` marks the split `g1` and gives it its spread, and the overrides that a prefix `A_g1` pins are not calibrated. One idea, one name: one call would mean two things, with the problem or without it. |
| The engine resolves each id on the model alone, a parameter first | One idea, one name: an order chooses one of two targets in silence, which the plan refuses for an initial condition ([An initial-condition id names one target](calibrated-initial-conditions.md#an-initial-condition-id-names-one-target)). The problem already declares the type, and the report needs its remainder. |
| The write-back records in the model what it set | S and KISS: the model would hold the history of a calibration, save it in its file and keep it after an edit such as `update_bin_amounts`, and a draw reported on the model would read the record of another set. |
| A result carries the types of its values | DRY and YAGNI: a copy of the problem in each result and draw, and the remainder still needs the problem. |
| The complement of a split and the remainder are not calibrated | One idea, one name: `calibrated` would mean "a value names it" for a split and "the write-back wrote it" for an override that a prefix pins, which no value names either. Only the spread stays with the row a value names. |

**Guards:**
`tests/reporting/test_applied_parameters.py` (`test_a_parameter_with_the_id_of_a_category_marks_no_split`, `test_a_split_with_the_id_of_a_parameter_marks_no_parameter`, `test_the_overrides_a_calibrated_prefix_pins_report_calibrated`, `test_the_amounts_of_the_remainder_report_calibrated`, `test_an_override_a_calibrated_category_covers_reports_calibrated`, `test_a_value_the_problem_does_not_declare_is_refused`),
`crates/commol-model/tests/initial_conditions.rs::the_plan_gives_the_kind_and_the_compartments_of_each_value`,
`tests/calibration/test_initial_conditions.py::test_the_kind_of_each_target_comes_from_the_engine`.
