# Engines

[Back to Design decisions](../design-decisions.md)

What every engine shares, and the rules of the difference, ODE and stochastic engines.

## One list of engines

`EngineType` (`crates/commol-core/src/types/engine.rs`) is the one list of
model types. An entry is an engine, named by its type
(`DifferenceEquations`, `DifferentialEquations`, `StochasticEvents`,
`StochasticSteps`): a method, not a kind. It names the model kinds it runs
(`EngineType::kinds`; today all four run `aggregate`). Each entry validates its settings and
says the form of its dynamics (`DynamicsForm`: the schedule a description
shows and the left-hand side of its equations) and the meaning of a rate (`per_step` or
`continuous`). `Engine::build` (`py-commol/src/engine.rs`) builds the engine of
an entry. A model holds its engine and the settings of that engine,
`Model.engine = {type, settings}`, because the engine sets what a rate means,
so a saved model reproduces its runs.

Python asks the list (`commol_rs.engine.EngineType`): it builds the engine,
writes the left-hand side of an equation and the schedule of a description,
and checks settings through it, and
never branches on an engine name. It keeps the names of the engines only as
the `Literal` `EngineName` (`context/engine.py`), for static typing and
completion of `ModelBuilder.build`; a test checks that it equals the entries of
the list (`test_the_python_names_equal_the_engines_of_the_list`). `ModelEngine.type` is a `str` that
the list checks, so a model can name an engine of any kind; its settings are read
with the class of that engine (`ModelEngine.settings_of_the_engine`), so a model
loads back with the class it was saved with. Generating the `Literal` from Rust would need
a code-generation step for four names; the test catches the same drift. An
entry names the kinds it runs, so another kind (for example an agent network)
adds kinds to an entry or new entries, not a second meaning of the type; see
[One interface for every model kind](model-kinds.md#one-interface-for-every-model-kind).

The builder takes the structure, the engine and its settings where it chooses
them, `build(structure, engine, settings=None)`, so one builder builds a model
for each structure and engine with its own settings. Settings of one engine given to another raise an error
from the engine that names both.

**Guards:** `crates/commol-core/src/types/engine.rs` (tests),
`py-commol/src/engine.rs::every_entry_builds_its_engine_and_runs`,
`tests/building/test_engine.py`.

## Every engine reads one set of flows

A compiled model is a set of flows (`crates/commol-model/src/system.rs`):
`CompiledModel::flow_rates` gives the value of each flow, its rate times its
source for a proportional rate (`transition_rates` gives the rate alone), and
`CompiledModel::flow_changes` gives the change of the state of one event of
each flow: `−1` for each source piece, `+1` for each target piece and each
accumulator. Every engine reads them, and none has flow logic of its own:

| Engine | Reads |
|---|---|
| `DifferenceEquations` | `derivatives`, `Σ changes × flow`, added once per step |
| `DifferentialEquations` | `derivatives`, integrated |
| `StochasticEvents` | `flow_rates` as propensities, and one event of `flow_changes` at a time |
| `StochasticSteps` | `transition_rates` as per-step probabilities, and the drawn number of events of each flow |

`derivatives` is `flow_changes`ᵀ · `flow_rates`, computed piece by piece in the
order of the old right-hand side, so the deterministic runs did not change by
a bit. The deterministic engines are the mean-field limits of the stochastic
ones, which gives the main test oracle. The positions where a rate can jump
come from the same model (`CompiledModel::breakpoints`), so the differential
and the continuous-time stochastic engines stop at the same positions.

A rate that is not a number stops the run in every engine, with a
`RuntimeError` that names the flow and the position: a calibration fails with
the cause instead of a penalty for a run of NaN. `derivatives` checks the flows
of both deterministic engines (`check_flows`); each stochastic engine checks
its own rates. A flow that is not a number is not an error when a compartment
of the state is not a number, because `diffsol` finds the sparse pattern of the
Jacobian by passing NaN through `J·v` ([The solver is `diffsol`](#the-solver-is-diffsol)).

**Guards:** `crates/commol-model/tests/flows.rs`
(`a_rate_that_is_not_a_number_names_its_flow_and_its_position`,
`a_compartment_that_is_not_a_number_gives_flows_that_are_not_numbers`),
`tests/simulation/test_runs.py::test_a_rate_that_is_not_a_number_stops_the_run`,
`crates/commol-difference/src/simulation.rs::one_step_adds_the_derivatives_exactly`,
`crates/commol-stochastic/tests/oracles.rs`
(`the_mean_of_many_runs_is_the_solution_of_the_ode`,
`the_mean_of_many_chain_binomial_runs_is_the_difference_engine`).

## A difference step never leaves an amount below zero

`DifferenceEquations` checks every compartment after the impulses of a step and
after its flows (`DifferenceEquations::step`, `settle` in
`crates/commol-difference/src/simulation.rs`). An amount below
`−10⁻¹² × total`, the total at the start of the step, stops the run with
`SimulationError::Infeasible` (`crates/commol-core/src/simulation_engine.rs`).
The message names the compartment, the step, its amount before the step and
the flows that drain it (`CompiledModel::outflows`, `impulse_outflows`). The
check reads the state and the right-hand side once, with no allocation, and the
state stays as it was before the step. Accumulators are not checked: a net flow
may count negative.

An error, for every cause, and not:

| Alternative | Why not |
|---|---|
| A check only of proportional rates that sum above 1 | None of the seven negative runs the test suite had broke that rule: they came from absolute rates on a bin total and from rates that turn negative. |
| A warning | A negative amount is never a correct state, and a warning inside a loop of calibrations is lost. |
| Clamp the amount to zero | The total is no longer kept, and the flows are no longer those of the model, without notice. |
| Scale the outflows down to what the compartment holds | A different model, chosen by the library; modelling choices belong to the user. |

Calibration tells such a run from a failure by its type, not by its message.
`SimulationEngine` returns `SimulationError`: `Infeasible { position, message }`
for a state outside the values the model allows, and `Failed(String)` for every
other error. The forward model turns an infeasible run into a
`Violation::InfeasibleRun` and a failed one into `EvaluationError::Failed`
(`forward.rs`); the optimization objective gives a violation a penalty and
records nothing in the history; a failed run still stops the calibration, so a
rate that cannot be evaluated does not hide behind a poor fit. An optimizer
that explores the bounds of a rate would otherwise stop on the first infeasible
point, though the fit lies elsewhere. A prediction of an infeasible draw raises
`InfeasibleStateError` (`prediction_error` in `py-commol/src/inference.rs`)
with the message of the run, which `Violation::InfeasibleRun` carries
(`tests/uncertainty/test_predictive.py::TestBands::test_an_infeasible_draw_of_a_predictive_names_the_compartment_and_step`).
Python receives `InfeasibleStateError`, a
`ValueError` (`simulation_error` in `py-commol/src/engine.rs`); a failed run
stays a `RuntimeError`.

The differential engine is not checked: a solver may step below zero by its
tolerance, and a model whose exact solution goes negative is a modelling choice
there.

**Guards:** `crates/commol-difference/src/simulation.rs`
(`flows_that_move_more_than_a_compartment_holds_stop_the_first_step`,
`a_rate_that_turns_negative_stops_the_step_where_an_amount_drops_below_zero`,
`proportional_rates_that_sum_to_one_empty_a_compartment`,
`an_impulse_that_moves_more_than_its_source_holds_stops_its_step`,
`an_accumulator_may_count_a_negative_net_flow`),
`crates/commol-calibration/tests/test_infeasible_runs.rs`.

## Stochastic engines and seeds

A stochastic engine draws every random number from one generator seeded by
`SimulationEngine::set_seed` (`crates/commol-core/src/simulation_engine.rs`);
`reset` restores it, so a run is reproducible from its seed. The default
`set_seed` does nothing, so the deterministic engines did not change.
`EngineType::is_stochastic` says which engines draw: `StochasticEvents` and
`StochasticSteps` (`crates/commol-stochastic/`). A run does not depend on the
positions it reports at: `StochasticEvents` measures each wait from the last
change of the state, never from a reported position, and draws the flow of an
event only when the event happens.

| Part | Rule |
|---|---|
| Generator | `Xoshiro256PlusPlus` of `rand_xoshiro`, seeded by four SplitMix64 numbers of the seed (`Generator` in `crates/commol-random/`). Its sequence is the same on every platform and version. |
| Samplers | Exponential, Poisson, binomial and multinomial written in `commol-random/src/sample.rs`, not taken from `rand_distr`, whose values may change in a minor version. A test holds the first values of each for a fixed seed. |
| Replicates | Replicate `i` of seed `s` takes `replicate_seed(s, i)`, the SplitMix64 of `s + i`. `run_replicates` runs them in parallel (`rayon`); a deterministic engine has no replicates. |
| Draws of a sample | One tree of seeds: draw `j` takes `replicate_seed(stream_seed(seed, DRAWS_STREAM), j)` (`draw_seeds` in `crates/commol-inference/src/predictive.rs`), and its replicate `i` takes `replicate_seed` of that seed and `i`, the rule of `Simulation.replicates`. `Predictive` and `Propagation` give the seed of each draw (`draw_seeds`), and the runs of a draw depend on neither the number of replicates nor the number of draws. |
| Calibration | Common random numbers: the same replicate seeds at every evaluation, so the loss is a deterministic function of the parameters (`CalibrationProblem::with_replicates`). The user chooses the summary (`ReplicateSummary`); `LossConfig` computes it. |

Reproducibility has three levels: the same machine and commol version give
the same run; another platform gives the same random numbers, but most rate
functions come from the system's `libm` (`address` in
`math_expression/jit/libm.rs`), which can differ in the last bit, so a run
there is equal in distribution and rarely differs draw by draw; another commol
version may change a run, and the changelog says so.

| Alternative | Why not |
|---|---|
| A random source passed to every call | The trait grows an argument that the deterministic engines ignore. |
| The seed in the model file | A model is a description; the seed belongs to a run, as `CalibrationProblem.seed` does. |
| `StdRng` or `SmallRng` | `StdRng` is slower and its security is not needed; `SmallRng` is documented as not portable. |
| A pure-Rust `libm` for every function of the JIT | `pow` takes 4 times as long (21 ns against 5 ns) and `exp`, `log` and `sin` about 40 % more, in every engine, for a difference in the last bit; the levels above are documented instead. The `libm` crate gives only the functions the standard library does not have (`erf`, `erfc`, `tgamma`, `lgamma`, the Bessel functions, `fdim`, `remainder`). |
| The JIT looks each function up in the process (`dlsym`) | A Linux binary links the system `libm` only for the functions Rust calls, so `floor` can be missing and the build fails; Windows names some functions otherwise (`_j0`). `address` gives the JIT the address of each function. |
| One run per calibration evaluation | The loss is noise, and Nelder-Mead stops anywhere. |
| The mean as the default summary | What to fit is a modelling choice; the mean of runs where half go extinct describes no run. |

**Guards:** `crates/commol-random/` (tests, with the stored first values of each
sampler in `tests/reference_values.txt`), `py-commol/src/engine.rs`
(`a_stochastic_engine_repeats_a_run_from_its_seed`,
`an_aggregate_stochastic_engine_repeats_a_run_from_its_seed`,
`a_deterministic_engine_ignores_the_seed`),
`crates/commol-stochastic/tests/oracles.rs::a_run_depends_on_its_seed_and_not_on_where_it_reports`,
`crates/commol-calibration/src/types.rs` (`replicate_tests`),
`tests/model_kinds/test_stochastic.py`.

## Whole units in a stochastic engine

A stochastic engine moves whole units, so it starts from whole amounts.
`check_model` (`crates/commol-model/src/builder.rs`, called by `build` and by
every engine) rejects a bin whose amount (the sum of its compartments) is not
whole, and initial amounts that are not whole unless the settings of the
engine name a `Rounding` (`crates/commol-core/src/types/engine.rs`); the
message lists both. There is no default. A rounding makes the compartments of
each bin whole and keeps the amount of the bin (`round_bins`), so a unit never
moves to another bin, such as from a concentration to a count:

| `rounding` | Start of each run (`Units::begin` in `crates/commol-stochastic/src/units.rs`) |
|---|---|
| `"largest_remainder"` | The same whole amounts: in each bin, each amount rounded down, the units left over to the largest remainders, the first compartment first on a tie |
| `"random"` | A multinomial draw of the amount of each bin with the fractions of the model, from the generator of the run |

The engine rounds at the start of a run, not when it is built, so a value that
calibration writes (`InitialConditionPlan`) goes through the same rounding, and
`"random"` draws after the values are written. With no rounding, a written
value that is not whole is an error.

| Alternative | Why not |
|---|---|
| Largest remainder by default | With small amounts the rounding is a modelling choice: half a unit in each of ten groups starts five groups with one unit and five with none, which decides which groups can start a process at all. The library would choose it without notice. |
| Random rounding by default | The same objection, the other way. |
| Reject every amount that is not whole | Most stratified models would fail, with no way out but to compute whole amounts by hand. |
| Round over every compartment at once | It keeps only the sum of every bin: a unit left over in one bin can start in another bin, of another unit. |
| Round each compartment on its own | No bin keeps its amount, and `"random"` changes it from one run to the next. |
| Round the amount of a bin that is not whole too | It changes a declared amount without notice. |

A per-step probability of `StochasticSteps` outside `[0, 1]`, or proportional
rates out of one compartment that sum to more than 1, are an error that names
the compartment, the step and the flows (`SimulationError::Infeasible`), not a
clamp: a binomial draw has no meaning outside `[0, 1]`, and a clamp would change
the model without notice.

**Guards:** `crates/commol-stochastic/tests/oracles.rs`
(`amounts_that_are_not_whole_need_a_rounding`,
`whole_amounts_need_no_rounding_and_each_bin_must_be_whole`,
`the_largest_remainder_starts_every_run_from_the_same_whole_amounts`,
`a_random_rounding_starts_on_average_from_the_amounts_of_the_model`,
`calibrated_initial_conditions_go_through_the_rounding`,
`a_negative_rate_or_probabilities_above_one_fail_at_their_step`),
`crates/commol-stochastic/src/units.rs` (`each_bin_keeps_its_amount_when_rounded`),
`builder.rs` (`a_bin_that_is_not_whole_is_a_build_error_of_a_stochastic_engine`),
`tests/stochastic/test_whole_units.py`.

## The solver is `diffsol`

The differential engine keeps no numerical method of its own: `diffsol`
(pinned to a minor version, `~0.17.1`) solves the system with `tsit45` or
`bdf`. `bdf` takes the Jacobian-vector product by finite differences
(`RightHandSide::jac_mul`), one extra right-hand-side evaluation, because a
symbolic derivative of every rate would be new numerical code. The matrix is
dense (`nalgebra`) below `SPARSE_THRESHOLD` state components and sparse
(`faer`) from there; `diffsol` finds the sparse pattern by passing NaN through
`J·v`. Solver settings live in the model (`Model.engine.settings`), so a saved
model reproduces a run.

A step that leaves the state not finite stops the run with an error that
names the last position the solver reached (`drive`), whatever the method:
`tsit45` accepts a step whose error estimate is NaN, where `bdf` fails on a
step that is too small.

**Guards:** `crates/commol-differential/src/tests.rs`
(`bdf_solves_a_stiff_model_within_the_tolerance`,
`a_large_stiff_model_uses_the_sparse_path`,
`a_solution_that_blows_up_stops_the_solver_where_it_fails`).
