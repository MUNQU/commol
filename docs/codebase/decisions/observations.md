# Observations

[Back to Design decisions](../design-decisions.md)

How an observation reads a run: the one operator of the engine, the known
scale of an observation, the coefficients of a weighted sum of outputs, the
parts of a window and the readers that refuse two observations of a series at
one position.
What a weight means and how a calibration runs are on
[Calibration](calibration.md).

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

## A coefficient weighs one output of a sum

An observation that sees the whole of one output and a known share of
another gives `ObservedDataPoint.coefficients`, one per output in the order
of `outputs`; `None` is 1 for each. The operator reads each output times its
coefficient (`ObservationOperator::apply`), before the window, the scale and
the factor, so every reader in the engine applies it with no code of its own.
The readers outside the operator read it too: the figures of a run and of its
replicates (`total_series` of `_simulation/series.py`, through `_read` and
`_model_line` of `_simulation/panels.py`), the band of a predictive with one
operator (`_predictive_band` of `uncertainty/panels.py`), the `observed`
block of a report (`coefficients`), the absolute error of a prediction in the
noise floor of a profile (the sum of the coefficients times the absolute
error, `prediction_error` of `profiles/tracer.rs`) and the series of the dense
design of finite differences (`dense_observations`).

**Rule:** a list of numbers next to `outputs`, not `outputs: dict[str,
float]`: every reader of `outputs` takes a list of names, and the type of a
public field would change for every user. Each coefficient is finite and 0 or
more, and one is above 0 (`ObservedDataPoint::check_coefficients`): a
negative one can give a count model a negative mean, and a 0 is an output the
measurement does not see. `coefficients` with `output` is refused
(`Observation::check`): the share of one output is `scale`. A coefficient is
known; an unknown share is a factor or a parameter of the model. The
operator checks the coefficients too, so a problem built in Rust has the
rules of one built in Python.

**Guards:** `crates/commol-calibration/tests/test_series.rs::weighted_sum_over_a_window`,
`crates/commol-calibration/tests/test_series.rs::an_operator_refuses_invalid_coefficients`,
`profiles::tracer::tests::the_absolute_error_of_a_prediction_counts_each_output_by_its_coefficient`,
`sensitivity::finite_differences::tests::observations_with_other_coefficients_are_another_series`,
`tests/calibration/test_coefficients.py`.

## The parts of a window are observations of every reader

An observation with `parts=K` (`ObservedDataPoint.parts`, 1 by default, the
rule `PARTS` of `crates/commol-calibration/src/ranges.rs`) is the sum of `K` independent observations over
equal parts of its window, with the boundaries
`b_k = start + k · (at − start) / K` (`ObservedDataPoint::part_boundaries`),
`b₀` the start of the window and `b_K` its end, so the parts share their
boundary rows. The operator keeps the rows of every boundary
(`ObservationOperator::apply_parts`) and gives the mean of each part, times
the coefficients, the scale and the factor; its prediction is their sum, so an
observation of one part is its window read whole. The operator is built wherever an operator is
(`ForwardModel::new`, `with_observations`, `on_schedule`,
`CalibrationProblem::observing`), so the `Workspace` holds the part means of
the problem's observations and of held-out ones, `ObservationNoise::observe`
gets them, and `Sensitivities.parts` holds their derivatives as a block of
its own, from the same runs, which the structural rank and the collinearity
do not read. `check_positions` refuses an interior boundary the engine does
not accept, naming it.

**Rule:** a count of equal parts, a field of the observation with one
spelling: in commol `period` is a length (`decisions/time.md`), and the window
changes by observation, as `window` and `trials` do. A label may mix any
parts, as it mixes windows: the window sets the scale of an observation, not
`parts`. A problem with a loss refuses `parts > 1`
([Calibration](calibration.md#a-problem-has-a-loss-or-observation-models)).

**Guards:** `crates/commol-calibration/tests/test_parts.rs`,
`crates/commol-inference/tests/test_identifiability.rs::the_structural_rank_reads_no_part`,
`tests/calibration/test_observations.py::TestParts`,
`tests/uncertainty/test_scores.py::TestLogScore::test_a_held_out_sum_reads_its_parts`,
`tests/uncertainty/test_profiles.py::TestPredictionProfiles::test_a_profile_of_a_sum_reads_its_parts`.

## A model reads the sum of the parts

`Noise::sum_log_density`, `sum_scores` and `sum_information`
(`crates/commol-calibration/src/likelihood/sums.rs`) read a term from the
means of its parts; with one part they are the functions of one mean. A
Poisson reads `Poisson(M)` and a normal `N(M, K · sd²)`, both exact. A
negative binomial series with parts needs `sum` (`Sum::Exact` or
`Sum::Moments`), and one without parts refuses it (`Likelihood::new`).
`"exact"` is the recursion of Panjer over the parts, in `O(K · y + y²)`, with
`log p(0)` kept apart and the terms scaled down past `1e150`; `"moments"` is
`NB(M, φ · Σμₜ²/M²)`. Every other model refuses parts. The draws read the sum
as the density does (`draw_sum` of `predictive.rs`): each part, summed, or
the matched negative binomial for `"moments"`.

**Rule:** a term of several parts reads the density of their sum, and its
draws, scores and information read that same density; a model with no closed
form for the sum refuses parts.

| Alternative | Why not |
|---|---|
| Moments only | They drift by up to 56 % of an interval with few effective parts, and bias `φ`. |
| Exact only | 1.3·10⁹ operations per evaluation at a total of 50 000. |
| A default `sum`, or a switch by the size of the total | A judgment hidden in the library. |
| Approximate sums of the log-normal and the Student t | No caller ([L8](https://github.com/MUNQU/commol/issues/498#issuecomment-6983148460)). |
| The Poisson below a small `φ` | A jump in the likelihood at the threshold; the recursion, with `log1p(φμ)/(φμ)` and no division by `φ`, reaches the Poisson at `φ = 0` within 6e-14 ([L9](https://github.com/MUNQU/commol/issues/498#issuecomment-6983149187)). |
| Truncating the recursion where `q_max^j < 1e-17` | It saves 10–100× at small `φ` and costs up to 7.6e-3 in `log p` far in the upper tail; only with a declared tolerance and a test of that tail ([L10](https://github.com/MUNQU/commol/issues/498#issuecomment-6983149799)). |

**Guards:** `crates/commol-calibration/tests/test_parts.rs::the_exact_sum_is_the_convolution_of_its_parts`,
`crates/commol-calibration/tests/test_parts.rs::the_moments_of_a_sum`,
`crates/commol-inference/tests/test_predictive.rs::draws_of_a_sum_have_the_variance_of_their_parts`,
`tests/uncertainty/test_observation_sums.py::TestSums`,
`tests/uncertainty/test_part_sums_coverage.py::TestPartSums`.

## A reader of positions refuses two observations at one

A label may hold two observations at one position (an increment and a total
that end there), each with its own window, scale and residual. The loss, the
likelihood, the scores, the predictive with `NoNoise` or `DistributionNoise`,
and the block methods of a given `block_length` that weigh blocks or
multiply residuals in place (the Bayesian and wild bootstraps, the block
jackknife, the sandwich, the conformal methods other than rolling origins)
read each observation alone and accept it. A reader that pairs
the observations of a series by position refuses it, with a message that
names the series and the position: `ResidualNoise` and `ResidualBootstrap`
(`SeriesResiduals::new`, one residual per position, and
`ResidualNoise::check` for the observations it serves), the automatic block
length (`automatic_block_length`, the residuals as consecutive values) and
`RollingConformal` (a horizon is a count of observations). They share one
check, `one_per_position` (`crates/commol-inference/src/resampling/blocks.rs`):
two observations share a position when their `at` are equal, as
`distinct_positions` counts the positions of the grid.
A new reader that pairs a series by position calls it, or reads each
observation with its own residual.

| Alternative | Why not |
|---|---|
| Keep one residual per position | It drops a residual with no message: a total gets the residuals of its increments. |
| A residual per observation at a position, matched by rank | The second observation of a position has no residual at most positions, so it would draw its own every time and no noise; the two have different scales, so a block mixes them. |
| Refuse at the calibration | A loss and a likelihood read the two well, and a total beside its increments is valid data. |

**Guards:** `crates/commol-inference/tests/test_resampling.rs::readers_of_a_series_by_position_refuse_two_observations_at_one_position`,
`crates/commol-inference/tests/test_resampling.rs::residual_noise_refuses_two_held_out_observations_of_a_series_at_one_position`,
`tests/uncertainty/test_resampling.py::TestResidualBootstraps::test_observations_at_one_position_are_refused`,
`tests/uncertainty/test_resampling.py::TestResidualNoise::test_held_out_observations_at_one_position_are_refused`.
