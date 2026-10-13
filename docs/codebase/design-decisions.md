# Design Decisions

[Back to Codebase Guide](index.md)

The rules on this page are not obvious from the code, and breaking one gives
results that are wrong but look correct. Each rule names where it lives and the
test that guards it. Read the ones of your area before you change it.

| Area | Decisions |
|---|---|
| Model structure | [Compartments](#compartments-are-labels-with-a-name), [stratification order](#stratification-order-is-part-of-the-model), [outputs](#outputs-are-resolved-by-label), [variable names](#the-engine-owns-the-variable-names) |
| Dynamics | [Rate semantics](#absolute-and-per-capita-rates), [stratified rates](#the-most-specific-stratified-rate-wins), [formula parameters](#formula-parameters-are-evaluated-every-step-in-order), [JIT cache](#compiled-expressions-are-cached-globally) |
| Initial conditions | [Subgroup head counts](#subgroup-head-counts-compose-down-the-condition-chain), [calibrated values](#calibrated-initial-conditions-are-applied-in-layers) |
| Time | [Physical time](#physical-time-is-not-in-the-model), [step unit](#one-step-unit-in-the-unit-check), [time inside a step](#time-inside-a-step), [impulses](#impulses-come-before-the-flows-of-their-step), [windowed observations](#windowed-observations) |
| Engines | [One right-hand side](#one-right-hand-side-two-engines), [solver](#the-solver-is-diffsol) |
| Calibration | [Weights](#one-meaning-of-an-observation-weight), [Nelder-Mead bounds](#nelder-mead-searches-inside-the-bounds), [simulations](#a-simulation-follows-its-model) |
| Results | [Confidence intervals](#one-confidence-interval-convention), [reports](#reports), [model files](#model-files-are-written-in-full) |
| Security | [Expressions](#every-user-expression-is-validated) |

## Model structure

### Compartments are labels with a name

A compartment is a bin plus one category for each stratification that applies
to it, in declaration order. Its name, `{bin}_{category}_{category}…`, is for
display only. Its identity is its label: `OutputLabels` in
`crates/commol-core/src/simulation_engine.rs` and `CompartmentLayout` in
`crates/commol-model/src/layout.rs`.

**Rule:** never parse a compartment name or match it by prefix. Bin `A` and bin
`A_x` both produce names that start with `A_`, and a conditional
stratification gives some compartments fewer suffixes than others. Use
`CompartmentLayout::find` or `of_bin` in Rust, and the labels that the engine
queries return in Python.

### Stratification order is part of the model

`expand` in `crates/commol-model/src/builder.rs` applies the
stratifications in declaration order. A stratification with `conditions` only
expands the compartments whose earlier categories satisfy all of them, so the
result can be a non-uniform product. A condition may only reference a
stratification declared before it (`ModelBuilder.add_stratification` checks
this).

**Rule:** changing the order of `add_stratification` calls changes which
compartments exist and what they are named.

### Outputs are resolved by label

`Model.get_outputs_by_source()` maps each bin or accumulator id to its output
names in engine order. It uses the engine expansion (`expanded_outputs`), so
`Simulation.outputs_for`, `group_outputs` and `total_series` group outputs by
label, not by name.

**Guards:** `tests/simulation/test_outputs.py::TestOutputsBySource::test_expansion_matches_the_engine_output_names`
(order) and `TestSourceIdSharingAPrefix` (a bin and an accumulator that share
a prefix).

### The engine owns the variable names

The engine sets one variable per expanded compartment and per subpopulation
total (`N_<categories>`, bin totals and partial bin totals).
`check_names` in `crates/commol-model/src/builder.rs` rejects a parameter
id that is one of those names, and an `N_…` variable in a rate or a formula
parameter that names no subpopulation. `Model.validate_names_with_engine`
calls it at `build`, and `DifferenceEquations::try_from_model` calls it again.

**Rule:** never list these names in Python. Ask the engine.

**Guards:** `builder.rs` tests
`parameter_id_cannot_be_a_compartment_or_a_subpopulation` and
`subpopulation_variable_out_of_declaration_order_is_a_build_error`;
`tests/building/test_reserved_names.py`.

## Dynamics

### Absolute and per-capita rates

The flow planner (`plan_transition_flows` in
`crates/commol-model/src/flows.rs`) decides for each flow whether the rate
is absolute or per-capita:

| Case | Rate is | Flow per step |
|---|---|---|
| A stratified rate sets `absolute` | As set | — |
| The rate references a compartment, a bin total or an `N…` variable | Absolute | `rate` |
| The transition has no source | Absolute | `rate` |
| Otherwise | Per-capita | `rate × source` |

A transition with several sources and a per-capita rate is rejected, because
it has no single source to multiply. `print_equations` shows the planned flows,
so a per-capita rate prints as `k1 * A`.

**Guard:** `tests/transitions/test_flows.py`.

### The most specific stratified rate wins

For each compartment, the stratified rate whose conditions match it with the
largest number of conditions wins (`helpers.rs`). On a tie, the first one in
the list wins. If none matches, the transition's default `rate` applies.
Building a model fails if a stratified rate matches no compartment that the
transition starts from.

### Formula parameters are evaluated every step, in order

Parameters with a formula are evaluated again at each step, in the order they
were declared (`simulation.rs`). A formula can use the value of an earlier
formula from the same step. Declare parameters in dependency order.

### Compiled expressions are cached globally

`JIT_CACHE` in `crates/commol-core/src/math_expression/mod.rs` maps each
preprocessed formula to one compiled function for the life of the process. The
same formula always runs the same machine code, so results are reproducible.

**Rule:** do not invalidate or bypass the cache.

## Initial conditions

### Subgroup head counts compose down the condition chain

The fractions of a conditional stratification are relative to the subgroup that
its conditions select, not to the whole population. Head counts therefore
multiply along the chain: `N × fraction(group1) × fraction(cat1)`.
`InitialConditions.subgroup_population` owns this; `Model.subgroup_population`
and `Model.get_conditioning_categories` delegate to it.

**Guard:** `tests/simulation/test_outputs.py::TestSubgroupPopulation` compares the
result with the population the engine starts from.

### Calibrated initial conditions are applied in layers

`InitialConditionPlan` in `crates/commol-calibration/src/initial_conditions.rs`
applies calibrated values in a fixed order: bins, category splits, exact
compartments, and then the remainder named by
`CalibrationProblem.initial_condition_remainder`. The objective, the predictions
and the write-back all use `calibrated_initial_population`, so they agree by
construction. `Model.apply_calibration_parameters`
(`context/_model/calibration_write_back.py`) writes the values back as
fractions, and pins as per-compartment overrides every compartment that the
fractions cannot reproduce.

**Guard:**
`tests/calibration/test_write_back.py::TestApplyCalibrationParameters::test_applied_model_reproduces_the_loss_the_optimizer_scored`.
See also [Per-compartment initial conditions](../development/design-per-compartment-initial-conditions.md).

## Time

### Physical time is not in the model

A `Model` records no step length: rates are per step and periods are step
counts. The axis of a model is its independent variable, whatever quantity it
is (time is one case), so the engines never ask which dimension it has. `Model.step_unit` is optional and only the unit check
reads it. `TimeScale` (`api/time_scale.py`) holds the length of a step of time
and converts durations and probabilities; the window grid is in
`api/windows.py`. Conversions **raise**
when a period is not a whole number of steps, instead of rounding. Months and
years are absent on purpose, because they have no fixed length in steps.
`TimeScale` applies only when that variable is time:
`TimeScale.from_step_unit` raises for a step unit that is not time.

**Guards:** `tests/simulation/test_time_scale.py::TestFromStepUnit`,
`tests/simulation/test_windows.py::TestWindowGrid`.

### One step unit in the unit check

The engine applies every flow once per step and converts no value between
units. So `UnitChecker._step_unit` (`context/_model/units.py`) uses one step
unit: the declared `Model.step_unit`, of any single dimension
(`check_step_unit`), or else the one unit of time that the parameter units use
(`step` when there is none). `check_equation_units` requires every flow to be
**equal** to `<bin unit>/<step unit>`, and every impulse to `<bin unit>`, not
only convertible to it (`utils/equations.py`, `same_unit`). Inference never
takes a unit that is not time: a parameter in `1/(item·meter)` may be a
density, not a rate along the axis, so the check does not guess.

**Rule:** never accept a convertible unit as equal. `1/week` and `1/day` in one
model would apply a weekly rate every day.

**Guards:** `tests/units/test_model_units.py::TestStepUnit`,
`tests/units/test_model_units.py::TestAxisUnit`.

### Time inside a step

A differential model integrates in continuous time, but on `[n, n + 1)` its
rates see `step = n`: `t` is continuous, `step` is the floor of `t`. A rate that
reads `step` or a `"pulse"` series (and so `AxisPattern.pulse`, `pulses`,
`periodic`) is therefore constant on its unit interval. When the right-hand
side reads one of them (`CompiledModel::depends_on_step`), the engine ends a
piece of integration at every integer, so each jump of the rate is the start of
a piece (`breakpoints` in `crates/commol-differential/src/solve.rs`).

A `"step_function"` or `"linear"` series reads `t`
(`SeriesParameter::evaluate`), so it holds or interpolates from any position.
The engine ends a piece at each of its points (`CompiledModel::series_points`),
where the series jumps or bends. A difference engine reads `t = step`, so it
gives the same values as before, and it rejects a point between integers.

A rate can also jump where `t` reaches a value it compares `t` with
(`t >= t_on`, and so `AxisPattern.interval` and `linear_ramp`). The compiler
collects those values when they read only numbers, `pi`, `e` and parameters
that do not change during a run (`time_thresholds` in
`crates/commol-model/src/builder.rs`), and the engine evaluates them with the
current parameters and ends a piece there. Without that, an adaptive solver
whose rate is 0 before a short window grows its step and steps over the window.
A comparison with the state is a state event, which the engine does not
detect.

On a piece `[start, end)` the right-hand side sees `t` no larger than the
float just below `end` (`RightHandSide::last_t`). A Runge–Kutta step evaluates
the right-hand side at the end of its interval; without the limit, a value that
jumps at `end` would already act on the last step of the piece.

Whether an expression reads `step` comes from `reads_variable`, not from
`get_variables`: `get_variables` leaves out the special variables, because it
lists the variables a model must define.

**Consequence:** a per-capita pulse `p` moves `p` of its source in a difference
model and `1 − e^(−p)` in a differential model. An impulse moves `p` in both.
The alternative, `step = t`, makes a pulse `step == n` true on a set of zero
length, so it would never act.

**Guards:** `tests/simulation/test_differential.py::TestTimeInsideAStep`,
`crates/commol-differential/src/tests.rs::a_pulse_rate_is_a_constant_rate_on_its_unit_interval`,
`a_rate_reads_the_step_of_its_unit_interval`,
`a_rate_that_switches_on_t_acts_only_on_its_window`,
`a_linear_series_matches_the_integral_of_its_interpolant`,
`a_step_function_series_holds_from_a_position_between_steps`,
`crates/commol-model/src/builder.rs::a_comparison_of_t_with_a_fixed_value_is_a_threshold`,
`tests/axis_patterns/test_pattern_runs.py::TestDifferentialIntegration`.

### Impulses come before the flows of their step

At step `n` the state jumps first, then the flows of the step run from the new
state, in both engines (`CompiledModel::apply_impulses`). Every jump of one
step is computed from the state before any of them. The output at `n` shows the
state after the jump. With the other order, the differential output at `n`
would show the state before the jump and the difference output the state after
it, so the two engines would disagree at the output steps. A difference model
needs integer impulse steps; a differential model stops its integration at each
impulse step and restarts from the new state, so `diffsol` needs no event.

**Guards:** `tests/transitions/test_impulses.py::TestJump`,
`crates/commol-differential/src/tests.rs::an_impulse_moves_its_fraction_at_once_in_both_engines`.

### Windowed observations

An observation at `at` with `window = W` is compared with
`output(at) - output(at - W)`, anchored at its own position. The Rust loss and
`windowed_totals` (`api/windows.py`) must give the same number, and so must the
ensemble statistics (`StatisticsCalculator._member_window_value`). Window `w`
covers `(w · W, (w + 1) · W]`; `window_index` places a position with
`max(ceil(at / W) − 1, 0)`, and `observed_report` calls it.

**Guards:** `tests/simulation/test_windows.py::TestAgreementWithTheCalibrationLoss` (the loss at
the generating parameters is zero),
`tests/reporting/test_simulation_report.py::test_observed_report_places_each_observation_by_window_index`
and
`tests/probabilistic/test_calibrator.py::test_windowed_bands_are_taken_at_the_observation_positions`.

### The engine owns positions

A position on the axis is an `f64` in every engine call and a `float` in every
public Python name (`at`); a length is `window` or `period`; a count of steps
keeps `num_steps`. The `SimulationEngine` trait
(`crates/commol-core/src/simulation_engine.rs`) runs only at positions
(`run_at_positions_into_buffer`); `run(n)` runs on the grid the engine builds,
`grid(n)`, and returns it. Each engine says which positions its model type
accepts (`check_position`): a difference engine non-negative integers, a
differential engine any finite, non-negative position. The engine checks every
position where it receives it, and its message names the object that holds a
rejected one: the observation or constraint (`check_positions` in
`crates/commol-calibration/src/calibration_problem.rs`), the impulse transition
or series parameter (`DifferenceEquations::from_compiled`).

**Rule:** Python passes positions on and shows the engine's message. It never
checks whether a model type accepts a position.

**Guards:** `crates/commol-calibration/tests/test_positions.rs`,
`crates/commol-difference/src/simulation.rs::integer_positions_are_accepted_and_others_rejected`,
`crates/commol-differential/src/tests.rs::a_difference_engine_names_a_series_point_between_steps`,
`tests/calibration/test_observations.py::TestPositionsAndTypology`.

### Results carry their positions

`Simulation.run` and `run_at` return `SimulationResults`
(`api/results.py`): a `dict` from output name to values, with the positions in
`positions` and the rows in `rows()`. A `dict` keeps `results["A"]`,
`json.dumps` and `pandas.DataFrame(results)` working; a reserved key could clash
with an output name. Windows, reports, plots and ensemble statistics read a
value at a position (`index_at`, which matches within a relative `1e-9`), never
at a list index, so a model type that reports between integers fits them. The
ensemble trajectories hold `StatisticsCalculator.positions`: the unit grid and
every observation position and window start.

**Guards:** `tests/simulation/test_runs.py::TestResults`,
`tests/reporting/test_simulation_report.py::TestNonIntegerPositions`,
`tests/plotting/test_plotter.py::TestDifferentialModel`,
`tests/probabilistic/test_differential.py`.

## Engines

### One right-hand side, two engines

`CompiledModel::derivatives` (`crates/commol-model/src/system.rs`) computes
every flow from one snapshot of the state: it is the right-hand side
`dX/dt = Σ flows`. The difference engine adds it once per step
(`X(n + 1) = X(n) + derivatives(n, X(n))`); the differential engine integrates
it. Neither engine has flow logic of its own, so both run the same model.
Accumulators are state components: a difference step adds the counted flows, a
differential engine integrates them.

**Guard:**
`crates/commol-difference/src/simulation.rs::one_step_adds_the_derivatives_exactly`.

### The solver is `diffsol`

The differential engine keeps no numerical method of its own: `diffsol`
(pinned to a minor version, `~0.17.1`) solves the system with `tsit45` or
`bdf`. `bdf` takes the Jacobian-vector product by finite differences
(`RightHandSide::jac_mul`), one extra right-hand-side evaluation, because a
symbolic derivative of every rate would be new numerical code. The matrix is
dense (`nalgebra`) below `SPARSE_THRESHOLD` state components and sparse
(`faer`) from there; `diffsol` finds the sparse pattern by passing NaN through
`J·v`. Solver settings live in the model (`Dynamics.solver`), so a saved model
reproduces a run.

**Guards:** `crates/commol-differential/src/tests.rs`
(`bdf_solves_a_stiff_model_within_the_tolerance`,
`a_large_stiff_model_uses_the_sparse_path`).

## Calibration

### One meaning of an observation weight

`LossConfig::term` and `LossConfig::finish` in
`crates/commol-calibration/src/types.rs` compute every loss: `sse` is
`Σ w·r²`, `rmse` is `√(Σ w·r² / Σ w)`, `mae` is `Σ w·|r| / Σ w`. The optimizer
(`calibration_problem.rs`), the ensemble central loss
(`probabilistic/ensemble/mod.rs`) and Python (`central_fit_loss`, through
`LossConfig.aggregate`) all call them.

**Rule:** never write a loss formula elsewhere. The central-fit gate compares
the ensemble loss with the optimizer loss, so both must measure the same way.

**Guards:** `types.rs` `loss_tests`;
`ensemble::tests::central_loss_uses_observation_weights_in_every_metric`;
`tests/probabilistic/test_selection.py::test_central_fit_loss_uses_weights_in_every_metric`.

### Nelder-Mead searches inside the bounds

The objective clamps values to their bounds, so the loss is flat outside them,
and a simplex that crosses a bound collapses there. Nelder-Mead therefore
searches an unbounded `u` and scores `x = lo + (hi − lo)·(sin u + 1)/2`
(`SineTransform` and `BoundedProblem` in `optimization.rs`). Its evaluations
and its result are recorded in `x`. `converged` is true only for
`SolverConverged` and `TargetCostReached` (`optimization::converged`).

**Rule:** report and record values in `x`, never in `u`. Every optimizer run,
the verbose one included, goes through `optimize_observed`.

**Guards:** `optimization::tests`; `tests/calibration/test_optimizers.py`.

### A `Simulation` follows its model

`Simulation` keeps the JSON its engine was built from. `run`, `engine` and
`simulation_outputs` compare it with `model_dump_json()` and rebuild the engine
when it differs (`Simulation._sync_engine`), and `run` resets the engine first.
`Calibrator` takes `simulation.engine` when it is created, so the check is in
the property, not only in `run`.

**Guard:** `tests/simulation/test_runs.py`.

## Results

### One confidence-interval convention

`ci_percentiles` (`api/probabilistic/intervals.py`) is the only place that turns
a confidence level into percentiles. `member_statistics` reduces each ensemble
member to the quantity of interest **first**, and then takes percentiles across
members. Taking percentiles first gives a band that describes no member.

**Rule:** every new interval goes through these two functions.

### Reports

`api/reporting.py` returns blocks of data, not documents:
`simulation_report` (observed values, series, accumulators) and
`applied_parameters_report` (calibrated parameters and initial conditions).

- A single run and an ensemble differ only at the leaves (`ReportLeaf`).
- The key order of each entry is part of the contract, because it changes saved
  JSON files.
- Labels and summary figures belong to the caller, because they carry domain
  vocabulary.

**Guard:** `tests/reporting/`.

### Model files are written in full

`Model.to_json` writes every field, nulls included. Dropping nulls is not
round-trip safe: an uncalibrated `Parameter.value` or `BinFraction.fraction`
would lose a required key and fail to reload. `render_json`
(`context/_model/serialization.py`) keeps files readable by writing long numeric
arrays on one line. It is internal on purpose: an output format in the public
API becomes a compatibility promise.

### Public models forbid unknown fields

Every public Pydantic class sets `model_config = ConfigDict(extra="forbid")`.
An unknown field, such as a misspelt or an old name (`window_steps`), raises a
`ValidationError` that names it, instead of being dropped with no error.

**Guard:** `tests/building/test_unknown_fields.py`.

## Security

### Every user expression is validated

Rate expressions (`context/dynamics.py`) and calibration constraints
(`context/calibration.py`) pass through `validate_expression_security`
(`utils/security.py`) before they reach the engine. The limits are in
`SecurityConfig`.

**Rule:** any new field that accepts an expression must call it.

## Navigation

Previous: [Repository map](repository-map.md) | Next: [Conventions](conventions.md)
