# Calibration write-back

[Back to Design decisions](../design-decisions.md)

How `Model.apply_calibration_parameters` writes the values of a calibration
back onto a model, and what a run of the model it writes holds. How
calibrated values reach the initial state during the calibration is in
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
unless two remainders tie within those bits. `"random"` can draw other units:
`multinomial` and `binomial` (`crates/commol-random/src/sample.rs`) branch on
the exact value of a share, and a share of `0.9999999999999999` instead of 1
makes `binomial_inversion` draw one more uniform, which shifts the stream of
the run. A replicate of the written model can then differ along its whole
path, and the seed of the problem can give a loss other than
`result.final_loss` ([#474](https://github.com/MUNQU/commol/issues/474)).

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
