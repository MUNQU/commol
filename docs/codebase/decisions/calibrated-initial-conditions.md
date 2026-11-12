# Calibrated initial conditions

[Back to Design decisions](../design-decisions.md)

How calibrated values reach the initial state, which compartment keeps the sum
the model declares, and what an engine of whole units needs to vary an initial
condition. How the fractions of the categories compose and how an amount
overrides them is in [Initial conditions](initial-conditions.md).

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

An id names, in this order: an expanded compartment (`A_g1`), a bin or a bin
prefix (`A`, or `A_g1` over a deeper stratification) when it holds no `:`,
or a category of a binary stratification, qualified (`group:g1`) or bare
(`g1`) when one stratification has it (`InitialConditionPlan::new`). The plan
matches categories on the labels of the compartments, never on their names.
One function resolves a category id, `resolve_category` in
`crates/commol-model/src/initial_conditions.rs`, with the one separator
`CATEGORY_SEPARATOR`: the plan asks it among the categories the compartments
have, and `Aggregate.resolve_category` asks it through the binding
`commol_rs.model.resolve_category` (`resolve_declared_category`, which adds
the message of an unknown category) among the categories of the fractions.
A qualified id matches a whole stratification id, the separator and a whole
category id, never a split at one `:`, because ids have no rule on their
characters. A bare category that two stratifications declare is an error
that names them and the qualified ids. The write-back of a calibration
(`_model/calibration_write_back.py`) takes as categories the values the
engine resolves as category splits (`CalibratedInitialState.categories`).

**Rule:** the engine says the kind of each value (`InitialTargetKind`: `bin`,
`compartment` or `category`), together with the compartments it sets
(`InitialConditionPlan::coverage`), and Python reads the kind and infers
nothing from the compartments. The kind is what the id names in the order
above, so an id that names an expanded compartment is a compartment, also
when it is the id of a bin with one compartment. The remainder has no kind:
it is not a value, and the plan keeps it apart. The binding
`calibrated_initial_state` returns the state, the covered compartments and
the kinds from the one plan that writes them, so they agree by construction.

**Rule:** the plan reads the initial state of the engine
(`CompartmentalEngine::initial_state`), never its current state: the split of
a bin, a prefix or the remainder and the pair totals of a category. A
`Calibrator` takes the engine of its `Simulation`, which holds the last state
of a run. Compartments that start with nothing, such as those of a bin left
`None`, or a side of a category whose fraction is 0, split a value as the
fractions declare it within them: the product of the fractions that differ
between them (`CompiledModel::fractions`, `weights` in
`crates/commol-model/src/initial_conditions.rs`). A fraction they all take
scales every share alike, and when it is 0 it would empty them all, so it
does not count. They split evenly only when that product gives them
nothing.

**Guards:**
`crates/commol-model/tests/initial_conditions.rs::a_bin_and_the_remainder_keep_the_split_of_the_initial_state_after_a_run`,
`crates/commol-model/tests/initial_conditions.rs::a_bin_left_for_calibration_splits_a_value_by_its_fractions`,
`crates/commol-model/tests/initial_conditions.rs::a_prefix_that_starts_empty_keeps_its_declared_split`,
`tests/calibration/test_initial_conditions.py::test_a_calibrated_amount_keeps_the_split_by_category`,
`crates/commol-model/src/initial_conditions.rs::category_shared_by_two_stratifications_is_ambiguous`,
`tests/initial_conditions/test_stratification_fractions.py::test_an_ambiguous_category_is_refused`,
`crates/commol-model/tests/initial_conditions.rs::the_plan_gives_the_kind_and_the_compartments_of_each_value`,
`tests/calibration/test_initial_conditions.py::test_the_kind_of_each_target_comes_from_the_engine`,
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
It reads the compartments a value sets, never its kind, so a kind of value
that sets amounts is checked without a change there.

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

## Whole units need a rounding to vary an initial condition

A method says where the numbers of its values come from
(`ValueSource` of `crates/commol-calibration/src/values.rs`): it varies them,
such as an optimizer or a sampler, or it writes numbers it knows, such as the
draws of a sample. A varied source builds its writer with `ValueWriter::varied`:
it asks the engine whether it takes every number of the values of the plan
(`SimulationEngine::check_varied_values`, a required method that takes the
plan, so the values are resolved first and a new engine states its answer).
`StochasticEvents` and `StochasticSteps` answer through `Units`: with no
`rounding`, they refuse an initial condition, whatever its bounds, and the
message names it and the two roundings. The deterministic engines take every
number.

| Builder | Source |
|---|---|
| `CalibrationProblem::new` | Varied |
| `ForwardModel::unobserved` | Its argument: `ForwardAnalysis::new` passes varied for independent inputs, and known (every draw) for a sample, which is the design of the analysis |
| `configured_engine` (the write-back of `Model.apply_calibration_parameters`) | Known (`ValueWriter::known`): the engine checks each number when it writes it (`Units::set_initial_condition`) |
| `plan_model_values` | Known (`ValueWriter::known`), only to resolve the values: it writes no number |

`CalibrationProblem::new` and `ForwardModel::unobserved` then write numbers
of their source to a copy of the engine and check that no value is missing
([A run needs every value](model-structure.md#a-run-needs-every-value)).
A varied source writes the point of the bounds of each value nearest 0,
which is whole when 0 is within the bounds or the bound nearest 0 is whole,
so it does not show that every number is taken. A known source writes each of its sets, so every draw of a
sample is checked when the analysis is built, and the message names its set.

| Alternative | Why not |
|---|---|
| Check the bounds in `commol-calibration` | The calibration would hold a rule of the engine, against the golden rule. |
| Refuse when the plan is built (`InitialConditionPlan::new`) | Every plan is built, also for a write of known values, so a whole value written back would need a rounding for a reason that does not apply to it. |
| Treat every input of an analysis as varied | A sample writes only its draws: a sample of whole amounts would be refused for numbers it never writes. |
| A default `check_varied_values` that takes every number | A new engine of whole units that forgets it would take every number with no error. |
| Refuse only when a bound is not whole | Bounds `[0, 50]` pass, and every start fails on the first value that is not whole. |

**Guards:**
`crates/commol-calibration/tests/test_varied_values.rs`,
`crates/commol-stochastic/tests/oracles.rs` (`calibrated_initial_conditions_go_through_the_rounding`),
`tests/stochastic/test_calibration.py::test_a_calibrated_amount_needs_a_rounding`,
`tests/stochastic/test_calibration.py::test_every_initial_condition_that_varies_needs_a_rounding`,
`tests/stochastic/test_whole_units.py::test_an_analysis_that_varies_an_amount_needs_a_rounding`,
`tests/stochastic/test_whole_units.py::test_an_analysis_of_a_sample_writes_its_draws_as_known_amounts`,
`tests/stochastic/test_whole_units.py::test_every_draw_of_a_sample_is_checked_when_the_analysis_is_built`,
`tests/stochastic/test_whole_units.py::test_a_known_amount_is_written_back_when_it_is_whole`,
`tests/stochastic/test_calibration.py::test_a_calibrated_amount_with_a_rounding_is_calibrated`.
