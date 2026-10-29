# Calibration

[Back to Design decisions](../design-decisions.md)

What weights, bounds, the observation operator and constraints mean in a calibration, and how its work runs in parallel. How the optimizers search is on [Optimizers](optimizers.md).

## One meaning of an observation weight

`LossConfig::term` and `LossConfig::finish` in
`crates/commol-calibration/src/types.rs` compute every loss: `sse` is
`Σ w·r²`, `rmse` is `√(Σ w·r² / Σ w)`, `mae` is `Σ w·|r| / Σ w`. The
likelihood of observation models (`Likelihood::pointwise` in
`crates/commol-calibration/src/likelihood/mod.rs`) is `Σ w·log p`: a weight
is a power of its term, so a weight of 2 counts the observation twice in a
likelihood as in every loss. The discrepancy (`discrepancy.rs`) calls them
for the optimizer and for every method that reads the loss.

**Rule:** never write a loss formula or a log density of an observation
elsewhere.

**Guards:** `commol-calibration/src/types.rs` `loss_tests`; `discrepancy::tests`;
`crates/commol-calibration/tests/test_likelihood.rs::a_weight_of_two_is_the_observation_twice`.

## A problem has a loss or observation models

A `Discrepancy` measures predictions by a loss or by the negative
log-likelihood of one observation model per series label
(`DiscrepancyConfig`), never both: the fit and the posterior would answer
different questions about one problem. Neither has a default, because what
to fit is a modelling choice. A model takes the prediction of the
observation operator as its mean and only adds the scatter; its noise
arguments are numbers or parameters of type `Noise`, which the value writer
skips, as it skips factors. What only makes sense for a loss refuses
observation models: `normalize_observations`, the replicate summaries of a
stochastic engine (the likelihood of a summary ignores the noise of the
process), `LossDistance` of ABC and information matching of `η`.

| Alternative | Why not |
|---|---|
| Both, the loss for the optimizer and the likelihood for samplers | The fit and the posterior would answer different questions. |
| Keep `"sse"` by default and refuse only a loss written next to models | The rule would depend on whether a field was written, and the library would choose what to fit. |
| A noise parameter of type `parameter` | The value writer would send it to the engine, and reports would mix it with model parameters. |
| Densities from a dependency | Their values change with its version; SciPy reference values in a test check ours. |

**Guards:** `crates/commol-calibration/tests/test_likelihood.rs`,
`tests/uncertainty/test_observation_models.py`.

## A `Simulation` follows its model

`Simulation` keeps the JSON its engine was built from. `run`, `engine` and
`simulation_outputs` compare it with `model_dump_json()` and rebuild the engine
when it differs (`Simulation._sync_engine`), and `run` resets the engine first.
`Calibrator` takes `simulation.engine` when it is created, so the check is in
the property, not only in `run`.

**Guard:** `tests/simulation/test_runs.py`.

## The engine says which parameters a caller sets

`SimulationEngine::set_parameter` returns an error for an id the engine does
not let a caller set. An engine of the aggregate kind sets a constant or a
parameter to calibrate, and rejects any other id
(`CompiledModel::set_parameter` in `crates/commol-model/src/system.rs`): one
the model does not have, or a formula or a series parameter, which the model
writes again at each evaluation, so a value set on it is lost. A calibration
or any method on `ForwardModel` sets each parameter on a copy of the engine
when it is built (`ValueWriter::new` in `crates/commol-calibration/src/values.rs`),
so the engine's message reaches the user before any run, and Python checks no
parameter id (`check_values` in `_calibration/calibrator.py`). `ValueWriter::new` also
checks that a factor has positive bounds, the observed outputs and positions
are checked by `CalibrationProblem::new` alone, and Python passes each
message with no prefix.

**Guards:** `crates/commol-model/tests/parameters.rs`,
`tests/calibration/test_problem.py::TestProblemAgainstTheModel`.

## One observation operator

`ObservationOperator` (`crates/commol-calibration/src/observation.rs`) is the
only code of the engine that turns a run into observed quantities: the row of
each observation, the sum of its outputs, its increment over its window, its
scale and its factor. The loss, the predictions of `PyCalibrationProblem`, the
predictive's `at_observations` and every score read it, and the figures of a
predictive read a series with a window, a factor or a scale through it. The
figures of a run and of its replicates have no problem to read, and repeat
its rules in Python ([A known scale is a number of the
observation](#a-known-scale-is-a-number-of-the-observation)).
The calibration problem is split into one layer per job, each in its own
module: `ForwardModel` (`forward.rs`) is the only code that clones, resets,
writes values to or runs the engine; `Feasibility` (`feasibility.rs`) checks
bounds and constraints; `Discrepancy` (`discrepancy.rs`) compares predictions
with data; `OptimizationObjective` (`objective.rs`) is the argmin cost
function. `CalibrationProblem` holds them, is built once and has no interior
mutability; the buffers of an evaluation are a `Workspace` per thread.

**Rule:** Python never transforms a prediction of a calibration problem or of
a sample: their windows, sums, factors, normalization, quantiles and scores
come from Rust. `Calibrator` builds the Rust problem once
(`commol_rs.inference.CalibrationProblem`) and every call reads it. The
helpers that present results already computed, such as `windowed_totals`
(`_simulation/windows.py`) for a run or its replicates, are not predictions; they
read the rules of the operator through bindings (`index_at` and
`window_increments` of `observation.rs`), which
`tests/simulation/test_windows.py::TestAgreementWithTheCalibrationLoss` and
`tests/simulation/test_windows.py::test_windowed_totals_equal_the_prediction_of_a_windowed_observation`
check.

An observation is given by `at` and `window`, so its window starts at
`at − window` (`observation::window_start`), read as 0 when
`|at − window|` is at most `1e-9` times the larger of `|at|`, `window` and 1:
the rounding of the subtraction grows with its operands, and `same_position`
near 0 loses that scale, so a window of `1000 + 1e-7` has one verdict whether
it ends at 1000 or at 2000. `check_positions`, `schedule_of`, the windows of a
quantity, `window_increments` and the checks of Python (`ObservedDataPoint`,
`Increments`, through the binding) read this one rule. A quantity given by its start, `Gain(start, end)`, passes the
start itself (`ObservationOperator::with_starts`): `end − (end − start)` can
round away from `start`, as `0.4 − (0.4 − 0.1)` is `0.09999999999999998`.
The operator finds the row of each position and of each start with the
tolerance of the axis (`observation::index_at`: equal within `1e-9` of the
larger of their magnitudes and 1), so a start computed as `at − window`
(`window_start`) reads the row of the position it rounds away from. Results
read a position with the same `index_at`.

**Guards:** `observation::tests`,
`crates/commol-calibration/tests/test_positions.rs::a_window_that_starts_at_zero_within_the_rounding_of_its_end_is_accepted`,
`tests/calibration/test_observations.py::TestPositionsAndEngines::test_a_window_that_starts_at_zero_within_its_rounding_is_calibrated`,
`crates/commol-inference/tests/test_predictive.rs::observation_draws_are_the_predictions_the_loss_compares`,
`tests/uncertainty/test_predictive.py::TestObservations`,
`crates/commol-inference/tests/test_global_sensitivity.rs::a_gain_reads_its_start_exactly`,
`tests/uncertainty/test_propagation.py::TestPropagation::test_a_gain_reads_its_start_exactly`.

## A known scale is a number of the observation

A known share of the model output that an observation sees, such as the
share of the units that report, is `ObservedDataPoint.scale` (1 by default),
or the `scale` of a `PredictionSpec`. The operator multiplies the prediction
by it with the factor, so every reader in the engine (the loss, the
predictive, the scores, the profiles) applies it with no code of its own. The
figures of a `SimulationResults` and of its replicates have no calibration
problem to read: they repeat scale times factor in Python (`model_multiplier`
of `plotting/panels.py`). The cumulative figures of a run, of its replicates
and of a predictive divide each observation by it in Python, a predictive with
the weighted median of the factor; a predictive checks the factors of its
observations in the engine first, with no run
(`ForwardModel::with_observations`, the binding `check_observations`). A
target that copies the first observation of its series (conformal `targets`,
the dense design of finite differences) keeps its scale. Not a `factor` with
equal bounds: `CalibrationParameter` refuses them, and a parameter that
nothing fits would enter every method that reads the parameter vector. Not a
scale of the series: `factor_id`, `window` and `trials` change by
observation, and a loss has no observation model.

**Guards:** `observation::tests::a_scale_and_a_factor_multiply`,
`tests/calibration/test_observations.py::test_an_observation_reads_the_model_times_its_scale`,
`tests/uncertainty/test_profiles.py::TestPredictionProfiles::test_a_prediction_profile_reads_the_scale_of_its_prediction`,
`tests/plotting/test_figures.py::TestObservationOperator::test_a_scale_multiplies_the_line_and_each_point_of_its_own`,
`tests/plotting/test_run_figures.py::TestRunFigures::test_cumulative_observations_are_divided_by_their_scale`,
`tests/plotting/test_figures.py::TestObservationOperator::test_a_missing_factor_warns_once_per_factor`,
`tests/plotting/test_figures.py::TestObservationOperator::test_a_predictive_refuses_a_factor_the_problem_lacks`,
`tests/plotting/test_figures.py::TestObservationOperator::test_a_predictive_cumulative_figure_runs_no_draw_to_check_a_factor`.

## Constraints bound the support

A point is feasible, or it violates the problem by an amount: a bound, a
constraint, an infeasible start or run, a value that is not finite, or a
predicted mean outside the means its observation model allows (`Violation`
in `feasibility.rs`). Only `OptimizationObjective` turns a violation into a
penalty, `base × (1 + amount)`, where `base` (`Discrepancy::penalty_base`) is
above the discrepancy of a poor fit: `1000 · n · max_obs²` for a loss, and
`1000 · Σ wᵢ tᵢ` for observation models, with `tᵢ` the cost of observation
`i` at the narrowest noise the bounds allow and a mean `max_obs` from it;
at least `1e10`. An inference method sees the same violation as the
edge of the support: its log density is `−∞` (`discrepancy_in_support` in
`crates/commol-inference/src/target.rs`).

| Alternative | Why not |
|---|---|
| The penalty for every method | Its scale is arbitrary, so a sampler would give weight to infeasible points. |
| Reject and draw again, with no value | A profile or a Laplace fit needs the value at the edge. |
| The base of a loss for a likelihood | A normal model of a small `sd` costs more than it, so infeasible points would look better. |
| The worst feasible cost seen so far | State in the objective: the result would depend on the order of evaluations and on threads. |

**Guards:** `feasibility::tests`,
`crates/commol-calibration/tests/test_infeasible_runs.rs`,
`crates/commol-calibration/tests/test_likelihood.rs::the_penalty_is_above_every_feasible_cost`.

## A constraint reads a matrix at its point

A constraint reads `<M>_<i>_<j>` and `<M>_radius` of a matrix between
categories. `Feasibility::new` keeps the matrices it reads from
`SimulationEngine::matrices` and refuses one with a scale that has no value
and is not calibrated. Once per evaluation, before the run,
`Feasibility::context` scales each by the point (`ScaledMatrix::variables`);
one no calibrated parameter scales keeps its model values. The run reads the
same values (`CompiledModel::set_parameter`), so a radius bound holds for it.

| Alternative | Why not |
|---|---|
| Read the radius from the engine configured at the point, or from `commol-model` | The parameter constraints run before that copy of the engine exists; calibration works through `SimulationEngine` only. |

**Guards:** `crates/commol-calibration/tests/test_constraints.rs`,
`py-commol/tests/calibration/test_constraints.py::TestMatrixConstraints`.

## A result records its seed

Every random number of a calibration and of its uncertainty methods comes from
`Calibrator.seed`: `problem.seed`, or the seed the engine draws when the
calibrator is built (`commol_random::seed_or_drawn`), which the calibrator
logs. A particle swarm takes the seed of its start, drawn from it
([Optimizers](optimizers.md#a-swarm-takes-its-seed-from-its-caller)), so the
seed alone repeats a run. The
result records it, so a result read from a file repeats too:
`CalibrationResult.seed`, and `settings["seed"]` of the `SampleMethod` of every
sample a runner makes (`sample_method` in `resampling/runners.py`) or
`Calibrator.point_sample` makes.

**Guards:** `tests/calibration/test_problem.py`
(`test_a_result_records_the_seed_of_its_calibration`);
`tests/calibration/test_optimizers.py`
(`test_the_seed_of_a_result_repeats_a_particle_swarm`);
`tests/calibration/test_starts.py`
(`test_a_verbose_swarm_repeats_its_fit_from_the_seed`).

## Parallel work runs in the pool of the setting

Every binding that runs parallel work calls `parallel` (`py-commol/src/parallelism.rs`),
which releases the GIL and runs the job in `Parallelism::install`
(`crates/commol-core/src/parallelism.rs`): a `rayon` pool of the number of
threads of `commol.parallelism.threads`, built at its first use and kept for the next job
with the same setting. Every `par_iter` inside the job, of commol and of the
argmin fork, nested levels included, runs in that pool. The setting is a
`contextvars` variable that the extension owns, so each Python thread and
each `commol.parallelism.threads` block has its own. A pool is kept per number of
threads and levels: each thread of a pool holds the levels of its pool
(`start_handler`), because a nested job can run on any thread of the pool.

**Rule:** a binding never calls `py.detach` itself, and no code builds a pool
or reads `RAYON_NUM_THREADS`.

| Alternative | Why not |
|---|---|
| The global pool of `rayon` (`build_global`) | It is fixed at its first use, so a block could not change it, and other libraries of the process share it. |
| An argument of each method | One choice about the machine repeated in every call; a block could not set it for a whole script. |
| One pool per number of threads, the levels in a thread-local of the caller | A job that another thread steals reads the levels of that thread, which another setting may have left. |

**Guards:** `crates/commol-core/src/parallelism.rs` `tests`;
`tests/parallelism/test_threads.py::test_every_parallel_method_runs_on_the_threads_of_the_setting`.
They read the threads that ran the test kind (`threads_of_runs`), not the
CPU time, so the load of the machine does not change them.
