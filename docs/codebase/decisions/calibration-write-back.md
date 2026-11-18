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

A replicate of the written model, for the seed of the problem, is the run the
calibration scored
([A written-back model runs as the calibration scored it](#a-written-back-model-runs-as-the-calibration-scored-it)).

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

## A written-back model runs as the calibration scored it

**Rule:** on every engine, the write-back rebuilds the scored initial state
bit for bit, sign included, because a run can turn on the last bit of a start
amount. On a stochastic engine, its rounding to whole units does:
`"largest_remainder"` breaks a tie of remainders by their last bits
(`B = 12` split `0.7, 0.2, 0.1` gives the remainders 0.4, 0.4 and 0.2), and
`binomial` (`crates/commol-random/src/sample.rs`) branches at a conditional
share of 0.5 and at a mean of 10, which an equal split always hits, or draws
a count that the last bits of its share change, which shifts the stream of
the run; only the last category with a positive share draws nothing
([Stochastic engines and seeds](engines.md#stochastic-engines-and-seeds)).
On any engine, a rate that steps on a start value (`if`, a comparison,
`floor`, `ceil`, `round`, `%`, `1 / x`) does: `if(B_g1 >= 1.05, 0.2, 0.01)`
with `B = 1.5` split `0.7, 0.2, 0.1` scores `B_g1 = 1.05` and `B_g2 = 1.30`
at 5, and the bin and its split rebuild `B_g1 = 1.0499999999999998`, which
gives 1.11; a bin of `B = -0.0` rebuilds `0.0`, which `==` takes for `-0.0`
and `1 / B_g1` does not. `calibrated_structure`
(`_model/calibration_write_back.py`) drops the overrides of the compartments
that the values and the remainder set and the scored amounts of an earlier
write-back, writes the bins and the splits, and writes each compartment the
result starts other than in every bit, as the engine sorts it (`write_back`
of `crates/commol-model/src/builder.rs`, through `engine_state.write_back`;
Python only applies the result): a scored amount when its expansion lies
within the last bits of the scored amount, it has no override and its bin
has an amount
([A scored amount of the write-back lives in its own field](#a-scored-amount-of-the-write-back-lives-in-its-own-field)),
and an override otherwise, such as a value of an exact compartment, which the
bins and the splits do not hold, or a compartment the user overrides, whose
override the scored amount replaces. The engine reads both as Python writes
them (`float_roundtrip` of `serde_json`,
[A data object crosses as the JSON of its dump](python-models.md#a-data-object-crosses-as-the-json-of-its-dump)).
The written model, saved and loaded or not, then starts from the state the
calibration scored and gives the outputs it scored, for the seed of the
problem on a stochastic engine, so a run of it gives `result.final_loss`.

The cost is the scored amounts: the bins and the splits rebuild some
compartments of most splits only to their last bits, and a remainder bin
almost all of them, so a calibrated stratified model can hold a scored amount
for most of its compartments
(`tests/initial_conditions/test_scored_amounts.py::test_a_write_back_of_four_stratifications_writes_no_override`:
more than half of 162). They stay out of `compartment_amounts`, so
`describe()` gives them one line, and `Model.applied_parameters` no row.

A new calibration of the written model can score other units or amounts,
because the plan splits a value by the shares of the scored amounts.

A bin whose declared amount is subnormal splits a value by the fractions
([Amounts below the smallest normal number split by the fractions](calibrated-initial-conditions.md#amounts-below-the-smallest-normal-number-split-by-the-fractions)).

| Alternative | Why not |
|---|---|
| A sampler that does not turn on the last bits, such as the last category of `multinomial`, which takes the units left with no draw | O: it removes one branch of one sampler, and each rounding or sampler that branches on a value would need a fix of its own. `binomial` still branches at a share of 0.5 and at a mean of 10, so a bin split in thirds can still run otherwise, and `"largest_remainder"` still breaks a tie by the last bits. |
| Break a tie of the largest remainders within a tolerance | O and KISS: it covers one rounding, and it moves the edge without removing it: two remainders as far apart as the tolerance still split on their last bits. |
| Rebuild within a relative `1e-9` on a deterministic engine | The invariant "What the user saves, the engine reloads": a rate that steps on a start value runs otherwise, as above, and a model with no step runs off in its last bits, so its loss is `result.final_loss` only approximately. Two rules for one idea, against one comparison on every engine. |
| Correct only when a formula of the model steps, as the tree of the engine (`engine_state.formula_tree`) says | O and DRY: a list of the nodes that step (`Conditional`, a comparison of a `Chain`, `%`, `floor`, …) would copy a property of the language of the engine, and each new function would need a case. A step on any state at any time, not only on a start value, can flip on a last bit, so every formula of the model must be walked. A model with no step still runs off in its last bits. |
| Write a bin amount, a few units in the last place off, that rebuilds every compartment bit for bit | KISS: a search over the build of the engine, with no amount to find in general: one amount must round to the scored compartment for every product of fractions of its bin, each of which bounds it to its own interval, a split bounds every bin at once, and a remainder is the total less the others. Where the intervals do not meet, the scored amounts stay. |
| Make the plan split a value as the build does, so the bins and splits rebuild every bit | S: the plan splits a value by the initial state of the engine ([Calibrated initial conditions are applied in layers](calibrated-initial-conditions.md#calibrated-initial-conditions-are-applied-in-layers)), and the remainder and a category take their amounts from that state, not from a fraction. A change there changes the calibration of every engine. |
| The default parse of `serde_json` | The invariant "What the user saves, the engine reloads": it reads some numbers of 17 digits, such as a value a calibration found (those of `a_number_reads_back_bit_for_bit`), one unit in the last place off, so a scored amount would not hold what the calibration scored. The correct parse costs more per number, but the engine parses a model, a problem and its data once when it builds them, not in each run or evaluation: the profiles of `k1` and `k2` in `docs/guide/uncertainty/profiles.md` make the same 1096 and 1190 evaluations with either parse, in the same time. The parse moves the last bit of some numbers it reads, and with them the path of an optimizer: on the ridge of that page, where some refits never reach their `sd_tolerance` and run to `max_iterations`, the profile of `k1` made 80 764 evaluations with it against 65 174 without, which is why the page caps those refits at 1000 iterations. |

**Guards:**
`tests/stochastic/test_calibration.py::test_a_written_back_model_runs_as_the_calibration_scored_it`,
`tests/calibration/test_write_back_runs.py::test_a_deterministic_write_back_runs_as_the_calibration_scored_it`,
`tests/calibration/test_write_back_runs.py::test_a_write_back_keeps_the_sign_of_a_scored_zero`,
`tests/stochastic/test_calibration.py::test_a_stratified_amount_is_written_back_with_its_split`,
`tests/stochastic/test_calibration.py::test_successive_write_backs_end_with_the_scored_amounts_of_a_fresh_one`,
`tests/calibration/test_initial_conditions.py::test_a_write_back_drops_the_overrides_of_the_compartments_it_sets`,
`tests/calibration/test_initial_conditions.py::test_the_write_back_of_a_split_by_category_rebuilds_the_scored_state`,
`tests/calibration/test_write_back.py::TestApplyCalibrationParameters::test_applied_model_reproduces_the_loss_the_optimizer_scored`,
`crates/commol-core/tests/json.rs::a_number_reads_back_bit_for_bit`.

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
| An override | The values and the remainder write its compartment (`TargetCoverage::written`): what a value sets, and both sides of each pair of a split. So each compartment the write-back overrides: of a prefix such as `A_g1`, of the remainder, of a split |
| A category split | A value is a split of its stratification: the write-back sets the category and its complement, in every split by category |

Python reads the kind of a value from the engine, never from its id. A
sample gives its spread to the row a value names, and to no other.

| Alternative | Why not |
|---|---|
| `problem` stays optional, and without it the report reads the values by their ids | The golden rule and DRY: Python resolves an id a second way, and misreads it: the parameter `g1` marks the split `g1` and gives it its spread, and the overrides that a prefix `A_g1` writes are not calibrated. One idea, one name: one call would mean two things, with the problem or without it. |
| The engine resolves each id on the model alone, a parameter first | One idea, one name: an order chooses one of two targets in silence, which the plan refuses for an initial condition ([An initial-condition id names one target](calibrated-initial-conditions.md#an-initial-condition-id-names-one-target)). The problem already declares the type, and the report needs its remainder. |
| The write-back records in the model what it set | S and KISS: the model would hold the history of a calibration, save it in its file and keep it after an edit such as `update_bin_amounts`, and a draw reported on the model would read the record of another set. |
| A result carries the types of its values | DRY and YAGNI: a copy of the problem in each result and draw, and the remainder still needs the problem. |
| The complement of a split and the remainder are not calibrated | One idea, one name: `calibrated` would mean "a value names it" for a split and "the write-back wrote it" for an override that a prefix writes, which no value names either. Only the spread stays with the row a value names. |

**Guards:**
`tests/reporting/test_applied_parameters.py` (`test_a_parameter_with_the_id_of_a_category_marks_no_split`, `test_a_split_with_the_id_of_a_parameter_marks_no_parameter`, `test_the_overrides_a_calibrated_prefix_pins_report_calibrated`, `test_the_amounts_of_the_remainder_report_calibrated`, `test_an_override_a_calibrated_category_covers_reports_calibrated`, `test_a_value_the_problem_does_not_declare_is_refused`),
`crates/commol-model/tests/initial_conditions.rs::the_plan_gives_the_kind_and_the_compartments_of_each_value`,
`tests/calibration/test_initial_conditions.py::test_the_kind_of_each_target_comes_from_the_engine`.

## A scored amount of the write-back lives in its own field

**Rule:** `scored_amounts` (`InitialConditions` of
`crates/commol-core/src/types/aggregate.rs`, `ScoredAmount` in Python) holds,
by compartment, the amount a calibration scored and the amount the bins and
the fractions `built` for it when it was written back. The engine checks each
entry with `check_scored_amount` (`crates/commol-model/src/rules.rs`): two
initial amounts at most `SCORED_AMOUNT_ULPS`, 64, units in the last place
apart, `0.0` and `-0.0` the same: the bins and the fractions rebuild a
scored amount to a few of them, each fraction rounding once, and a value a
calibration sets lies further away. The engine also decides which
compartment takes one (`write_back`), so Python repeats no rule of it.
`initial_state` (`builder.rs`) applies a scored amount last, after
`compartment_amounts`, while it is current: the compartment expands to
`built` in every bit, has no `compartment_amounts` entry and is not of a bin
left `None` (`takes_a_scored_amount`, which `write_back` shares), so a
scored amount never gives a value to an unvalued bin. Any other entry is
stale and ignored, so an edit that changes the compartment, by an `update_*`
method, a direct assignment, `model_copy` or a file, drops its scored
amount, and an override wins on every path. `update_compartment_amounts`
drops the entry it makes stale, to keep the field tidy. A stale entry is
cleared by a new write-back or by `scored_amounts = {}`.

One idea, one name: `compartment_amounts` is a declaration of the user,
which every edit keeps, also a value a calibration set for an exact
compartment; `scored_amounts` is a record of the write-back, valid while the
model builds what it recorded, and no method of the user sets it. An
override that names no compartment is a build error, because the user wrote
a name the model does not have; a stale scored amount is ignored silently,
because it is valid data whose condition no longer holds. An edit that
rebuilds the same bits keeps a scored amount, which moves the amount by its
last bits only.

A file with a scored amount does not load in a commol older than the field
(`extra="forbid"`); a model with none writes no field
(`skip_serializing_if`, `exclude_if`), so it still does. A declared
override, a calibrated exact-compartment value among them, survives an
update of its bin
([A compartment amount overrides the expansion](initial-conditions.md#a-compartment-amount-overrides-the-expansion)).

| Alternative | Why not |
|---|---|
| The scored amounts in `compartment_amounts` | One idea, one name: `describe()`, `applied_parameters` and the user could not tell a scored amount from a declaration, and every edit kept a scored amount that froze the compartment at its old amount ([#483](https://github.com/MUNQU/commol/issues/483)): `update_bin_amounts({"B": 100})` left B at 31.05. |
| Invalidate a scored amount in each `update_*` method | Decide in the core, act at the edge: Python would need the process to map a bin or a stratification to the names of its compartments, which only the engine knows, and it would miss a direct assignment, `model_copy` and a file. |
| One fingerprint of the whole state for all the scored amounts | KISS and S: all or nothing, any edit drops every scored amount, also of the bins it does not touch, and a hash is opaque in the JSON a user reads. |
| A value of an exact compartment as a scored amount too | The invariant "What the user saves, the engine reloads": a coincidence of `built` would bring a value-sized amount back after an edit: `A_g1 = 700` in a bin of 600 split `0.5, 0.5` holds while the split is 0.5, and a bin edited to 0 or `None` would keep it. |

**Guards:**
`crates/commol-model/src/builder.rs::a_scored_amount_replaces_the_amount_its_compartment_still_builds`,
`crates/commol-model/src/builder.rs::a_scored_amount_is_stale_once_its_compartment_builds_another_amount`,
`crates/commol-model/src/builder.rs::an_override_wins_over_a_scored_amount_of_its_compartment`,
`crates/commol-model/src/builder.rs::a_scored_amount_never_gives_a_value_to_a_bin_left_none`,
`crates/commol-model/src/builder.rs::a_scored_amount_far_from_its_built_amount_is_refused`,
`crates/commol-model/src/builder.rs::a_write_back_corrects_the_last_bits_of_an_expansion_and_overrides_the_rest`,
`crates/commol-model/src/builder.rs::a_write_back_overrides_a_compartment_of_a_bin_left_none`,
`crates/commol-model/tests/value_rules.rs::a_value_inside_its_rule_is_accepted`,
`crates/commol-core/src/types/aggregate.rs::scored_amounts_read_and_write_back_bit_for_bit`,
`tests/calibration/test_write_back_runs.py::test_a_compartment_the_user_overrides_starts_as_the_calibration_scored_it`,
`tests/initial_conditions/test_scored_amounts.py` (no report row, an edit
that rebuilds the same bits, a new split, a bin left `None`, an override by assignment and by
the method, an update of a bin, the round trip, the refusal, a stochastic run
after a new split, four stratifications with no override),
`tests/description/test_description.py::TestSections::test_the_initial_state_says_how_many_compartments_start_from_a_scored_amount`.
