# Calibration

[Back to Design decisions](../design-decisions.md)

What weights, bounds and constraints mean in a calibration, and how its work runs in parallel. How an observation reads a run is on [Observations](observations.md); how the optimizers search is on [Optimizers](optimizers.md).

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

## The engine says what a fit of a problem is

A point of a problem, the best values of a fit or the end of one of its
starts, has a value of each parameter of the problem and of nothing else.
`values_of` (`crates/commol-calibration/src/settings.rs`) holds this rule
once: `FitRecord::values` and `StartRecord::end_values` give the values in
problem order, or refuse a point with a message that names the parameters it
lacks and those it has that the problem does not have. Each caller asks the
engine: `Calibrator.point_sample` (through `CalibrationProblem.fit_values`),
the residual noise of a fit, `Collinearity(at=result)` and
`StartsIdentifiability`. An extra parameter is refused, not ignored,
because it says that the fit is of another problem.

**Guards:** `crates/commol-calibration/tests/test_json.rs`
(`a_fit_gives_the_values_of_the_parameters_it_has`,
`a_start_gives_the_values_of_the_parameters_its_end_has`);
`tests/calibration/test_problem.py`
(`test_a_fit_of_another_problem_is_refused`);
`tests/uncertainty/test_identifiability.py`
(`test_a_result_holds_the_parameters_of_the_problem`,
`test_a_start_of_another_problem_is_rejected`).

## Parallel work runs in the pool of the setting

Every binding that runs parallel work calls `parallel` or `parallel_logged` (`py-commol/src/parallelism.rs`),
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
