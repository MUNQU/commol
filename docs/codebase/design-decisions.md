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
| Time | [Physical time](#physical-time-is-not-in-the-model), [windowed observations](#windowed-observations) |
| Results | [Confidence intervals](#one-confidence-interval-convention), [reports](#reports), [model files](#model-files-are-written-in-full) |
| Security | [Expressions](#every-user-expression-is-validated) |

## Model structure

### Compartments are labels with a name

A compartment is a bin plus one category for each stratification that applies
to it, in declaration order. Its name, `{bin}_{category}_{category}…`, is for
display only. Its identity is its label: `OutputLabels` in
`crates/commol-core/src/simulation_engine.rs` and `CompartmentLayout` in
`crates/commol-difference/src/layout.rs`.

**Rule:** never parse a compartment name or match it by prefix. Bin `A` and bin
`A_x` both produce names that start with `A_`, and a conditional
stratification gives some compartments fewer suffixes than others. Use
`CompartmentLayout::find` or `of_bin` in Rust, and the labels that the engine
queries return in Python.

### Stratification order is part of the model

`expand` in `crates/commol-difference/src/builder.rs` applies the
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

**Guards:** `test_simulation_outputs.py::test_expansion_matches_the_engine_output_names`
(order) and `TestSourceIdSharingAPrefix` (a bin and an accumulator that share
a prefix).

### The engine owns the variable names

The engine sets one variable per expanded compartment and per subpopulation
total (`N_<categories>`, bin totals and partial bin totals).
`check_names` in `crates/commol-difference/src/builder.rs` rejects a parameter
id that is one of those names, and an `N_…` variable in a rate or a formula
parameter that names no subpopulation. `Model.validate_names_with_engine`
calls it at `build`, and `DifferenceEquations::try_from_model` calls it again.

**Rule:** never list these names in Python. Ask the engine.

**Guards:** `builder.rs` tests
`parameter_id_cannot_be_a_compartment_or_a_subpopulation` and
`subpopulation_variable_out_of_declaration_order_is_a_build_error`;
`test_reserved_names.py`.

## Dynamics

### Absolute and per-capita rates

The flow planner (`plan_transition_flows` in
`crates/commol-difference/src/flows.rs`) decides for each flow whether the rate
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

**Guard:** `test_transition_flows.py`.

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

**Guard:** `test_simulation_outputs.py::TestSubgroupPopulation` compares the
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
`test_calibrator.py::TestApplyCalibrationParameters::test_applied_model_reproduces_the_loss_the_optimizer_scored`.
See also [Per-compartment initial conditions](../development/design-per-compartment-initial-conditions.md).

## Time

### Physical time is not in the model

A `Model` records no step length: rates are per step and periods are step
counts. `TimeScale` (`api/time_scale.py`) holds the step length and converts
durations, probabilities and window grids. Conversions **raise** when a period
is not a whole number of steps, instead of rounding. Months and years are absent
on purpose, because they have no fixed length in steps.

**Guard:** `test_time_scale.py::TestWindowGrid`.

### Windowed observations

An observation with `window_steps = W` is compared with
`series[step] - series[step - W]`, anchored at its own step. The Rust loss and
`windowed_totals` (`api/windows.py`) must give the same number; the ensemble
statistics also use `windowed_totals`.

**Guards:** `test_windows.py::TestAgreementWithTheCalibrationLoss` (the loss at
the generating parameters is zero) and
`test_probabilistic_calibrator.py::test_windowed_bands_are_taken_at_the_observation_steps`.

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

**Guard:** `test_reporting.py`.

### Model files are written in full

`Model.to_json` writes every field, nulls included. Dropping nulls is not
round-trip safe: an uncalibrated `Parameter.value` or `BinFraction.fraction`
would lose a required key and fail to reload. `render_json`
(`context/_model/serialization.py`) keeps files readable by writing long numeric
arrays on one line. It is internal on purpose: an output format in the public
API becomes a compatibility promise.

## Security

### Every user expression is validated

Rate expressions (`context/dynamics.py`) and calibration constraints
(`context/calibration.py`) pass through `validate_expression_security`
(`utils/security.py`) before they reach the engine. The limits are in
`SecurityConfig`.

**Rule:** any new field that accepts an expression must call it.

## Navigation

Previous: [Repository map](repository-map.md) | Next: [Conventions](conventions.md)
