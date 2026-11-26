# Time

[Back to Design decisions](../design-decisions.md)

How the axis works: its units, the time inside a step, impulses and positions.

## Physical units are not in the model

A `Model` records no step length: rates are per step and positions and periods
are step counts. The axis of a model is its independent variable, whatever
quantity it is (time is one case), so the engines never ask which dimension it
has. `Model.step_unit` is optional; only the unit check and `AxisUnits` read
it. `AxisUnits` (`_model/axis_units.py`) converts a value in any unit of the
step dimension to steps on `pint`, and `Model.axis_units()` builds it from
`step_unit` (or `step`) and the engine of the model. `steps` returns a `float`
and snaps a result within `1e-9` of an integer to it; whether a position must
be whole is the engine's check, not the converter's.

The axis owns the window grid: `AxisUnits.window_start`, `window_end`,
`window_index` and `window_ends` take the width as a span and a unit, as
`steps` does, and check that it is a positive whole number of steps. The
engine computes the grid in steps (`window_bounds`, `window_index`,
`window_ends`, `windows_in` of `crates/commol-calibration/src/observation.rs`)
that `windowed_totals`, `_observed_report` and the ticks of a figure read, with
a `window` of any positive length: the window of an observation of a
differential model need not be whole. No public function takes a width in
steps: a width of `n` steps is the span `n` in the step unit.

The rate conversions (`rate_from_mean_span`, `rate_from_probability`) ask the
engine: `RateMeaning::rate_from_mean_span` and `rate_from_probability`
(`crates/commol-core/src/types/engine.rs`) hold the formula of each meaning, so
a new engine picks a meaning and Python does not change. A per-step rate keeps
the survival (`1 - e^(-1/n)`) or the mean (`1/n`) of a residence, never both:
the caller passes `RateKeeps` (the JSON of its name), with no default, and the
engine rejects the mean of a span below one step instead of clamping the rate to 1.

**Rule:** no formula of a rate per engine nor of a window in Python; ask the engine.

**Guards:** `tests/simulation/test_axis_units.py`,
`crates/commol-core/src/types/engine.rs` (tests `a_rate_from_…`, `a_mean_shorter_…`),
`tests/simulation/test_windows.py::test_windows_come_from_the_engine`,
`tests/suite/test_engine_rules.py::test_only_the_engine_computes_a_window`.

## An origin puts the axis on the calendar

`AxisUnits.origin` (`_model/axis_units.py`) is the calendar moment of
position 0. It belongs to the axis, so `ModelBuilder` declares it next to
`step_unit`, `Model.origin` keeps it and `Model.axis_units()` passes it on. On
a `Simulation`, two runs of one model could read one position as two dates.
The engines never see it: it converts at the edges, as the step unit does.

The origin and every moment are naive (`naive_moment`): a `date` is its
midnight, and a moment with a time zone raises `ValueError` with the way to
convert it. A step has no time zone, so `position = (moment - origin) / step`
stays exact; converting to UTC would move the end of a day with summer time,
and dropping the zone would lose information.

An origin needs a step unit of time whose length is a whole number of days or
a whole fraction of a day (`seconds_per_step`). `pint` defines `month` and
`year` with a fixed length (a month is 1/12 of a year), so with an origin on
1 January, `moment_of(1)` in months would be 31 January 10:30, not
1 February. The test of a whole number of days, or of days per step, rejects
them and every unit made from them without a list of names.

A day or a week ends at the midnight that starts the next one. `day_ends` and
`week_ends` give these ends in `(start, end]`, the interval of a window, from
the first midnight after `start`; the first end can close a partial period.
`first_weekday` numbers days as `date.weekday` does and has no default: which
day starts a week is the user's choice. The number of a period is not on the
axis.

`plot.series` of results labels the ticks with `_date_label` (the date,
and the time when it is not midnight) when the model has an origin and the
call has no `position_label`. With no `tick_every`, it formats the ticks that
matplotlib places, so a long run does not get one label per position.

**Rule:** a conversion between dates and positions goes through `AxisUnits`;
an application keeps no origin of its own.

**Guards:** `tests/units/test_calendar_axis.py`,
`tests/plotting/test_run_figures.py::test_an_axis_with_origin_labels_ticks_with_dates`.

## One step unit in the unit check

The engine converts no value between units. So `UnitChecker._step_unit`
(`_model/units.py`) uses one step unit: the declared
`Model.step_unit`, of any single dimension (`check_step_unit`), or else `step`,
the default increment of any axis (dimension `[step]`). The check never infers
the step unit from the parameters: a parameter in `1/meter` may be a density,
not a rate along the axis, and inferring only from units of time gives time a
privilege that no other axis has. Without `step_unit`, a failed check says to
declare it. The check reads the flows the engine plans (`transition_flows`,
in `UnitChecker._collect_unit_errors`), so a proportional rate is checked as
its flow, `rate × source`, and never by a rule of its own in Python.
`check_formula_units` requires every flow to be **equal** to
`<bin unit>/<step unit>`, and every impulse to `<bin unit>`, not only
convertible to it (`_model/unit_checks.py`, `same_unit`).

A unit means what `pint` defines; there is no list of aliases. A name that
`pint` does not define becomes the one unit of its own dimension
(`define_units`). In the check, `t` (a position) has the step unit and `step`
(a count) is dimensionless. A number written in a rate and added to,
subtracted from or compared with a quantity is read in its unit, as the engine
reads it (`formula_unit`); a computed number stays dimensionless.

**Rule:** never accept a convertible unit as equal. `1/week` and `1/day` in one
model would read a weekly rate as a daily one.

**Guards:** `tests/units/test_model_units.py::TestConsistentModels`
(`test_a_proportional_rate_is_checked_as_its_flow`),
`tests/units/test_time_units.py::TestStepUnit`,
`tests/units/test_time_units.py::TestAxisUnit`,
`tests/units/test_time_units.py::TestPositionOnTheAxis`.

## Time inside a step

A differential model integrates in continuous time, but on `[n, n + 1)` its
rates see `step = n`: `t` is continuous, `step` is the floor of `t`. A rate that
reads `step` or a `"pulse"` series (and so `AxisPattern.pulse`, `pulses`,
`periodic`) is therefore constant on its unit interval. When the right-hand
side reads one of them (`CompiledModel::depends_on_step`), the engine ends a
piece of integration at every integer, so each jump of the rate is the start of
a piece (`CompiledModel::breakpoints`, `crates/commol-model/src/stops.rs`).
A series compares its positions as numbers, so a pulse at `-0.0` is at step 0.

A `"step_function"` or `"linear"` series reads `t`
(`SeriesParameter::evaluate`), so it holds or interpolates from any position.
The engine ends a piece at each of its points (`CompiledModel::series_points`),
where the series jumps or bends. A difference engine reads `t = step`, so a
series gives its value at each step, and it rejects a point between integers.

A rate can also jump where `t` reaches a value it compares `t` with
(`t >= t_on`, and so `AxisPattern.interval` and `linear_ramp`). The compiler
collects those values when they read only numbers, `pi`, `e` and parameters
that do not change during a run (`time_thresholds` in
`crates/commol-model/src/builder.rs`), and the engine evaluates them with the
current parameters and ends a piece there. Without that, an adaptive solver
whose rate is 0 before a short window grows its step and steps over the window.
A comparison with the state is a state event, which the engine does not detect.

Points that `same_position` calls one are one stop (`stops.rs`). An integer of a
model that reads `step` is a fixed stop that holds every point and run end
within the tolerance. Another stop is its earliest point. A run ends at the stop
that holds its last position, or else at that position, and the next run resumes
there, so a split run, also with `set_seed` or `set_parameter`, has the stops of
the whole run. Before a stop the rates see `t` below its earliest point
(`Stop::cut`, `RightHandSide::last_t`): a Runge–Kutta step evaluates them at the
end of its interval. Its impulses act once, before its rows (`Stop::impulses`).

**Alternatives:** (a) `step_of`, the integer that `same_position` calls the same
or else the floor, at every rate: far out an event before `n` reads `n`. (a+)
`step_of` at the start of a piece: a run that ends just below `n` resumes with
`n − 1` and drops an impulse at `n`, and a chain of stops acts it twice. With an
absolute band, a second rule, far out no floats are that close.

Whether an expression reads `step` comes from `reads_variable`: `get_variables`
lists the variables a model must define, so it leaves out the special ones.

**Consequence:** a proportional pulse `p` moves `p` of its source in a difference
model and `1 − e^(−p)` in a differential model. An impulse moves `p` in both.
The alternative, `step = t`, makes a pulse act on a set of zero length: never.

**Guards:** `tests/simulation/test_differential.py::TestTimeInsideAStep`,
`crates/commol-differential/src/tests.rs::a_pulse_rate_is_a_constant_rate_on_its_unit_interval`,
`a_rate_reads_the_step_of_its_unit_interval`,
`a_rate_that_switches_on_t_acts_only_on_its_window`,
`a_linear_series_matches_the_integral_of_its_interpolant`,
`a_step_function_series_holds_from_a_position_between_steps`,
`crates/commol-model/src/types.rs::a_pulse_at_minus_zero_is_read_at_step_zero`,
`crates/commol-stochastic/tests/test_state.rs::a_seed_set_at_minus_zero_acts_the_pulse_of_step_zero`,
`crates/commol-model/src/builder.rs::a_comparison_of_t_with_a_fixed_value_is_a_threshold`,
`crates/commol-differential/src/tests.rs::test_a_threshold_just_below_a_step_keeps_the_step`,
`a_rate_before_a_stop_sees_t_below_its_earliest_point`, `an_impulse_in_a_chain_of_points_acts_once`,
`a_run_split_next_to_a_stop_far_out_is_the_run_split_at_the_stop`, `a_model_that_reads_the_step_runs_below_5e8`,
`crates/commol-stochastic/tests/test_state.rs::a_threshold_just_below_a_step_keeps_the_step`,
`a_run_split_next_to_a_step_resumes_from_that_step`, `a_row_next_to_an_impulse_shows_the_state_after_it`,
`a_run_split_next_to_a_step_far_out_resumes_from_that_step`, `an_impulse_next_to_the_end_of_a_run_acts_before_a_new_seed`,
`crates/commol-model/tests/schedules.rs::a_point_next_to_a_step_is_in_the_stop_of_that_step`, `a_point_the_same_as_a_point_of_a_stop_is_in_that_stop`,
`tests/stochastic/test_engines.py::TestPositions::test_a_stop_next_to_a_step_keeps_the_step`,
`tests/axis_patterns/test_pattern_runs.py::TestDifferentialIntegration`.

## Impulses come before the flows of their step

At step `n` the state jumps first, then the flows of the step run from the new
state, in every engine of the aggregate kind (`CompiledModel::apply_impulses`;
`Units::apply_impulses` for the stochastic engines). Every jump of one
step is computed from the state before any of them. The output at `n` shows the
state after the jump. With the other order, the differential output at `n`
would show the state before the jump and the difference output the state after
it, so the two engines would disagree at the output steps. A difference model
needs integer impulse steps; a differential model stops its integration at each
impulse step and restarts from the new state, so `diffsol` needs no event.

**Guards:** `tests/transitions/test_impulses.py::TestJump`,
`crates/commol-differential/src/tests.rs::an_impulse_moves_its_fraction_at_once_in_both_engines`.

## The engine owns positions

A position on the axis is an `f64` in every engine call and a `float` in every
public Python name (`at`); a length is `window` or `period`; a count of steps
keeps `num_steps`. The `SimulationEngine` trait
(`crates/commol-core/src/simulation_engine.rs`) runs only at positions
(`run_at_positions_into_buffer`); `run(n)` runs on the grid the engine builds,
`grid(n)`, and returns it. One rule says which positions each engine accepts
(`EngineType::check_position` in `crates/commol-core/src/types/engine.rs`),
and each engine's `check_position` and the builder read it: an engine whose
rate is per step (`DifferenceEquations`, `StochasticSteps`,
`EngineType::reports_at_integer_steps`) non-negative integers, the others
(`DifferentialEquations`, `StochasticEvents`) any finite, non-negative
position. For a model that reads `step` these also read `CompiledModel::check_reach`,
as `breakpoints` does: below `5e8`, where the tolerance is half a step.
Its message starts with "The <engine> engine". The engine checks every
position where it receives it, and its message names the object that holds a
rejected one: the observation or constraint (`check_positions` in
`crates/commol-calibration/src/calibration_problem.rs`), the impulse transition
or series parameter when the model is built or its parameters change
(`check_model` in `crates/commol-model/src/builder.rs`, for an engine that
steps only at integers), or the positions and observations of a predictive before any
draw runs (`Predictive::new`; `observation_draws` through `ForwardModel::with_observations`).

**Rule:** Python passes positions on and shows the engine's message. It never
checks whether a model type accepts a position.

**Guards:** `crates/commol-calibration/tests/test_positions.rs`,
`crates/commol-core/tests/positions.rs::only_the_engines_of_rates_per_step_report_at_integer_steps_alone`,
`tests/simulation/test_runs.py::test_each_engine_names_itself_when_it_rejects_a_position`,
`crates/commol-difference/src/simulation.rs::integer_positions_are_accepted_and_others_rejected`,
`crates/commol-differential/src/tests.rs::a_difference_engine_names_a_series_point_between_steps`,
`crates/commol-model/tests/schedules.rs::an_engine_that_steps_at_integers_rejects_a_position_between_steps_at_the_check`,
`tests/building/test_parameters.py::test_a_point_the_engine_cannot_run_is_rejected_at_build_and_update`,
`tests/calibration/test_observations.py::TestPositionsAndEngines`,
`crates/commol-inference/tests/test_predictive.rs::positions_the_engine_cannot_run_are_rejected_before_any_draw`,
`crates/commol-inference/tests/test_predictive.rs::observation_draws_reject_a_position_the_engine_rejects`.

## Results carry their positions

`Simulation.run` and `run_at` return `SimulationResults`
(`_simulation/results.py`): a `dict` from output name to values, with the positions in
`positions` and the rows in `rows()`. A `dict` keeps `results["A"]`,
`json.dumps` and `pandas.DataFrame(results)` working; a reserved key could clash
with an output name. Windows, reports and plots read a value at a position
with the Rust `index_at` of an observation (`observation.rs`, a relative `1e-9`), never at a list index, so
a model type that reports between integers fits them. The engines take the
same tolerance (`same_position` in `crates/commol-core/src/position.rs`). A
predictive holds the positions it was asked for (`Predictive.positions`).

**Guards:** `tests/simulation/test_runs.py::TestResults`, `tests/simulation/test_runs.py::test_a_result_reads_a_position_with_the_tolerance_of_the_engine`,
`crates/commol-core/tests/positions.rs::same_position_is_relative_to_the_larger_magnitude`,
`tests/reporting/test_simulation_report.py::TestNonIntegerPositions`,
`tests/plotting/test_figures.py::TestDifferentialModel`.
