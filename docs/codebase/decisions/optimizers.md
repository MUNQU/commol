# Optimizers

[Back to Design decisions](../design-decisions.md)

How Nelder-Mead and the particle swarm of a calibration search, score and stop,
and how the starts of a calibration run and draw their seeds.

## Nelder-Mead searches inside the bounds

The objective clamps values to their bounds, so the loss is flat outside them,
and a simplex that crosses a bound collapses there. Nelder-Mead therefore
searches an unbounded `u` and scores `x = lo + (hi − lo)·(sin u + 1)/2`
(`SineTransform` and `BoundedProblem` of the argmin fork, in
`argmin::core::bounds`; `SearchSpace::transform` in `optimization.rs` builds
the transform and refuses an infinite bound). Its evaluations and its result
are recorded in `x`. `converged` is true only for
`SolverConverged` and `TargetCostReached` (`optimization::converged`). An
error of the model at a vertex of the initial simplex stops the run with its
message (`NelderMead::init` of the argmin fork returns it).

**Rule:** report and record values in `x`, never in `u`. Every optimizer run,
the verbose one included, goes through `optimization::run`.

**Guards:** `optimization::tests`; `tests/calibration/test_optimizers.py`.

## A swarm scores in parallel and keeps the order of its particles

The argmin fork runs with its feature `rayon`, so a particle swarm asks for
the costs of an iteration at once (`bulk_cost`). `OptimizationObjective`,
`Recorded` and `FixedParameter` answer with `OptimizationObjective::score_all`
(`objective.rs`): the points in parallel when the levels of the pool are
`"all"` (`inner_levels_in_parallel`), one after the other with `"outer"`,
and the costs in the order of the points. `Recorded` keeps the feasible
points after the scores, in the order of the points, under one lock, so the
history does not depend on the order in which the threads finish. With
several failing points, the error is that of the first in the order of the
points. A score takes the buffers of its thread out of the thread-local
`WORKSPACE` while it runs (`with_workspace`), so a job that the thread steals
meanwhile takes its own; `score_in` makes no `rayon` call. The swarm of a held
prediction scores `AugmentedLagrangianCost` of the fork, which runs in
parallel when `PredictionCost::parallelize` says so, at the same levels, and
also returns the error of the first failing point.

| Alternative | Why not |
|---|---|
| A `Mutex` around the history, filled by each `cost` | It records in the order the threads finish. |
| A parallel loop of our own in the swarm | It repeats the bulk map of the fork. |
| The levels as a field of `ParticleSwarmConfig` | The levels are a choice about the machine, next to the threads; a result does not depend on them. |

**Guards:** `crates/commol-calibration/src/optimization.rs` `tests`
(`a_swarm_gives_the_same_result_with_parallel_particles`,
`the_error_of_a_swarm_is_its_first_failing_particle`,
`a_swarm_on_a_subproblem_gives_one_fit_whatever_the_threads`,
`a_swarm_on_a_held_prediction_follows_the_levels`);
`objective::tests::a_score_holds_no_borrow_of_the_workspace`;
`tests/parallelism/test_threads.py::test_a_seed_gives_one_result_whatever_the_threads_and_the_level`.

## A swarm stops by a rule the modeller sets

`ParticleSwarmConfig` takes the two stop rules of the argmin fork
(`ParticleSwarm::with_improvement_stop`, `with_radius_stop`): the improvement
of the best cost over a window of iterations, at most
`tol_abs + tol_rel · |best cost|` (`ImprovementStop`), and the radius of the
swarm, the largest distance of a particle from the best position divided by
the diagonal of the search box (Zielinski and Laur, 2007). Both are `None`
until the modeller sets them, and the Python methods take every setting with
no default. A run that stops by a rule ends with `SolverConverged`, so
`converged` reports `True`. The rules read the costs and the positions only
and draw no random numbers, so a seed gives the same run up to the iteration
where a rule stops it. The improvement rule also gives the tolerance `τ` of a
profile under a swarm ([Profiles](profiles.md)).

| Alternative | Why not |
|---|---|
| A rule on by default | Its window and tolerances depend on the scale of the loss and the problem; a default would stop some runs early and hide the choice. |
| A rule in commol, around the fork's swarm | A rule that knows only numbers belongs to the fork, which also gives it to the augmented Lagrangian. |
| A default for `tol_abs` or `tol_rel` | A tolerance is a modelling choice; the modeller writes it. |

**Guards:** `crates/commol-calibration/tests/test_model_values.rs::a_particle_swarm_with_a_stop_rule_converges`;
`crates/commol-calibration/src/optimization.rs` `tests`
(`a_swarm_gathered_around_its_best_stops_by_the_radius_rule`,
`a_stop_rule_out_of_its_range_names_itself`);
`crates/commol-inference/tests/test_profiles.rs::a_swarm_fits_a_subproblem_and_repeats_from_its_seed`;
`tests/calibration/test_optimizers.py::TestConverged::test_a_swarm_that_stagnates_stops_and_converges`.

## One function runs the starts and chooses the best

`optimize_starts` (`crates/commol-calibration/src/starts.rs`) takes the start
points and one seed, runs the starts in parallel and returns every outcome
and the index of the best (`Starts`): the start of lowest loss among those
that did not fail, the first on a tie, a NaN loss counting as `+∞`. When every
start fails, its error lists the reason of each start. A calibration
(`PyCalibrationProblem::optimize`, which `Calibrator.run` calls) and every fit
of a replicate (`ReplicateRunner::fit_one`) call it, so their starts, seeds
and best start follow one rule; Python chooses no start. The seed gives two
streams of the registry of `commol-random` (`Stream`): `Stream::StartPoints`
draws the Latin hypercube of `start_points`, and start `i` takes
`Stream::StartSeeds.replicate(seed, i)`, so the starts draw from no stream of
another method with the same seed. A replicate
passes the seed `Stream::ReplicateFits.replicate(seed, b)`, apart from the
stream of its data, and its fit as start 0. `optimize` and
`optimize_with_history` are start 0, from the initial guess, and a verbose
calibration runs `optimize_starts_with_observer` with its one start.

| Alternative | Why not |
|---|---|
| The hypercube from the seed itself | A replicate draws its data from the seed of its replicate, so the hypercube could reuse that stream; each caller would need to know which seeds it must not pass. |
| Stream numbers of its own in `commol-calibration` | They can take a number of the registry: with one seed, the hypercube of the starts was the design of a sensitivity analysis. The registry lives in `commol-random`, below every crate that draws. |
| Start 0 on another seed than the others | Start 0 of a calibration and of a replicate would follow two rules. |
| The best start chosen in Python | A replicate chooses in Rust, so the two choices could differ on a tie or a NaN. |
| Only the reason of the first failed start | The other starts can fail for other reasons, which the message would hide. |

**Guards:** `crates/commol-calibration/tests/test_starts.rs`
(`the_best_start_is_the_first_of_lowest_loss`,
`every_failed_start_gives_its_reason`,
`the_calibration_from_the_guess_is_the_first_start`);
`crates/commol-calibration/src/starts.rs` `tests`
(`the_hypercube_comes_from_the_stream_of_the_start_points`,
`start_i_takes_seed_i_of_the_stream_of_the_start_seeds`,
`a_loss_that_is_not_a_number_is_the_worst`);
`crates/commol-inference/tests/test_resampling.rs::a_replicate_fit_is_the_calibration_of_its_data`;
`crates/commol-inference/tests/test_streams.rs::a_calibration_and_a_sensitivity_analysis_with_one_seed_draw_from_two_streams`;
`tests/calibration/test_starts.py::TestFailedStarts::test_every_start_failing_raises_with_each_reason`.

## A swarm takes its seed from its caller

A particle swarm takes its seed as an argument of every optimizer
(`optimize_from`, `optimize_subproblem`, `optimize_starts`), which every
caller gives; `ParticleSwarmConfig` has no seed. The swarm draws with
`commol_random::Generator`, which implements the generator trait of
`rand_core` that the swarm of the argmin fork takes (`with_rng_generator`),
so its numbers come from the one generator of commol and repeat from the
seed.

| Alternative | Why not |
|---|---|
| A seed in `ParticleSwarmConfig` and an argument that replaces it | Two seeds of one run; the configuration seed served only a test. |
| A seed from the system when none is given | A run that no seed repeats; every caller has a seed. |
| `rand::SmallRng` | Its sequence is not the one of the other draws of commol, and `rand` may change it in a version. |

**Guards:** `crates/commol-calibration/tests/test_starts.rs::a_swarm_repeats_its_fit_from_its_seed`;
`crates/commol-random/src/generator.rs` `tests`
(`the_trait_of_rand_core_draws_the_numbers_of_the_generator`);
`tests/calibration/test_starts.py::TestStarts::test_a_verbose_swarm_repeats_its_fit_from_the_seed`.
