# Calibration

[Back to Design decisions](../design-decisions.md)

What weights, bounds, the observation operator and constraints mean in a calibration.

## One meaning of an observation weight

`LossConfig::term` and `LossConfig::finish` in
`crates/commol-calibration/src/types.rs` compute every loss: `sse` is
`Σ w·r²`, `rmse` is `√(Σ w·r² / Σ w)`, `mae` is `Σ w·|r| / Σ w`. The
discrepancy (`discrepancy.rs`) calls them for the optimizer and for every
method that reads the loss.

**Rule:** never write a loss formula elsewhere.

**Guards:** `commol-calibration/src/types.rs` `loss_tests`; `discrepancy::tests`.

## Nelder-Mead searches inside the bounds

The objective clamps values to their bounds, so the loss is flat outside them,
and a simplex that crosses a bound collapses there. Nelder-Mead therefore
searches an unbounded `u` and scores `x = lo + (hi − lo)·(sin u + 1)/2`
(`SineTransform` and `BoundedProblem` in `optimization.rs`). Its evaluations
and its result are recorded in `x`. `converged` is true only for
`SolverConverged` and `TargetCostReached` (`optimization::converged`). An
error of the model at a vertex of the initial simplex stops the run with its
message (`NelderMead::init` of the argmin fork returns it).

**Rule:** report and record values in `x`, never in `u`. Every optimizer run,
the verbose one included, goes through `optimization::run`.

**Guards:** `optimization::tests`; `tests/calibration/test_optimizers.py`.

## A `Simulation` follows its model

`Simulation` keeps the JSON its engine was built from. `run`, `engine` and
`simulation_outputs` compare it with `model_dump_json()` and rebuild the engine
when it differs (`Simulation._sync_engine`), and `run` resets the engine first.
`Calibrator` takes `simulation.engine` when it is created, so the check is in
the property, not only in `run`.

**Guard:** `tests/simulation/test_runs.py`.

## One observation operator

`ObservationOperator` (`crates/commol-calibration/src/observation.rs`) is the
only code that turns a run into observed quantities: the row of each
observation, the sum of its outputs, its increment over its window and its
factor. The loss, the predictions of `PyCalibrationProblem`, the predictive's
`at_observations` and every score read it, and the plotter reads a windowed or
scaled series of a predictive through it. The calibration problem is split
into one layer per job, each in its own module: `ForwardModel` (`forward.rs`)
is the only code that clones, resets, writes values to or runs the engine;
`Feasibility` (`feasibility.rs`) checks bounds and constraints; `Discrepancy`
(`discrepancy.rs`) compares predictions with data; `OptimizationObjective`
(`objective.rs`) is the argmin cost function. `CalibrationProblem` holds them,
is built once and has no interior mutability; the buffers of an evaluation are
a `Workspace` per thread.

**Rule:** Python never transforms a prediction of a calibration problem or of
a sample: their windows, sums, factors, normalization, quantiles and scores
come from Rust. `Calibrator` builds the Rust problem once
(`commol_rs.inference.CalibrationProblem`) and every call reads it. The
helpers that present results already computed, such as `windowed_totals`
(`api/windows.py`) for a run or its replicates, are not predictions; they
must agree with the operator, which
`tests/simulation/test_windows.py::TestAgreementWithTheCalibrationLoss` checks.

**Guards:** `observation::tests`,
`crates/commol-inference/tests/test_predictive.rs::observation_draws_are_the_predictions_the_loss_compares`,
`tests/uncertainty/test_predictive.py::TestObservations`.

## Constraints bound the support

A point is feasible, or it violates the problem by an amount: a bound, a
constraint, an infeasible start or run, or a value that is not finite
(`Violation` in `feasibility.rs`). Only `OptimizationObjective` turns a
violation into a penalty, `base × (1 + amount)`, where `base` scales with the
largest observed value. An inference method sees the same violation as the
edge of the support: its log density is `−∞` (`discrepancy_in_support` in
`crates/commol-inference/src/target.rs`).

| Alternative | Why not |
|---|---|
| The penalty for every method | Its scale is arbitrary, so a sampler would give weight to infeasible points. |
| Reject and draw again, with no value | A profile or a Laplace fit needs the value at the edge. |

**Guards:** `feasibility::tests`,
`crates/commol-calibration/tests/test_infeasible_runs.rs`.
