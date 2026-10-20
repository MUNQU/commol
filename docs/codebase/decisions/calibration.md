# Calibration

[Back to Design decisions](../design-decisions.md)

What weights, bounds, the observation operator and constraints mean in a calibration.

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

## A subproblem holds one quantity fixed

`Subproblem` (`crates/commol-calibration/src/subproblem.rs`) holds one
parameter or one prediction while `optimize_subproblem` (`optimization.rs`)
fits the others with the problem's own optimizer. A held parameter leaves the
search space (`SearchSpace`): the optimizer searches one dimension less, and
the cost puts the fixed value back before it scores a point. A held
prediction `g(θ) = c` is an equality constraint met by the augmented
Lagrangian of the argmin fork, whose factory builds the problem's optimizer
at each outer iteration: Nelder-Mead on the sine transform, from the point of
the iteration before, or a swarm seeded by `replicate_seed(seed, k)`. The
prediction is read through the observation operator of a copy of the problem
whose runs also report at its positions (`CalibrationProblem::observing`), so
one run gives the objective and the constraint, and the cost keeps the
constraint of the last point it scored. A point that is not feasible keeps
its penalty and a constraint of 0. Each point starts from Algencan's scaled
penalty; the multiplier is the warm start.

| Alternative | Why not |
|---|---|
| Bounds `[v, v]` for a held parameter | Nelder-Mead keeps a vertex for a dimension that moves nothing. |
| A growing quadratic penalty on `g − c` | It meets the constraint only as `μ → ∞`, and a large `μ` makes a narrow valley where Nelder-Mead stops early. |
| A second run for the prediction | Twice the model runs of each evaluation. |
| `μ = 100·|Δλ/Δc|` from the two points before | In the experiment of #132 (the ABC model of `tests/support/` with a normal model, the problem of `the_multiplier_of_a_prediction_profile_is_minus_its_slope`), B at 30 took 5465 evaluations against 3179 with Algencan's `μ₀`, C at 24 5405 against 3175, for the same intervals and a mean error of the multipliers of 0.8e-4 against 1.2e-4; `10⁴·|Δλ/Δc|` gained 4 %. The stop rule reads `‖g − c‖`, which a small `μ` leaves large. |

A swarm without stop rules runs each outer iteration to its `max_iterations`,
so a held prediction stops at the outer limit and never reports convergence.

**Guards:** `crates/commol-inference/tests/test_profiles.rs`
(`fixing_a_parameter_at_the_fit_keeps_the_loss`,
`fixing_a_prediction_at_the_fit_keeps_the_parameters`,
`a_problem_observing_more_positions_scores_as_the_problem`,
`a_swarm_fits_a_subproblem_and_repeats_from_its_seed`).
