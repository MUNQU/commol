# Uncertainty

[Back to Design decisions](../design-decisions.md)

How the uncertainty methods share one sample, one registry, their ports and
one sampler, what a replicate may change, and why an analysis needs no data.

## A sample carries its weights

A method with draws returns its own result type, which holds a
`ParameterSample` (`crates/commol-inference/src/sample.rs`,
`py-commol/commol/context/inference/sample.py`): names, bounds, values, log
weights normalized to sum to 1, optional groups and the method that made it.
Plots, reports, the predictive and the scores read the sample, never the
result of one method. Every statistic is weighted and computed in Rust; equal
weights give the unweighted statistic.

| Alternative | Why not |
|---|---|
| One generic result for every method | Profiles, conformal intervals and jackknife estimates are not draws. |
| A union of result types | Every caller narrows it. |

**Guards:** `crates/commol-inference/src/sample.rs` tests,
`tests/uncertainty/test_sample.py`.

## Methods plug into one entry point

`Calibrator.uncertainty(method: UncertaintyMethod[R]) -> R` runs every
uncertainty method, `Calibrator.identifiability(method:
IdentifiabilityMethod[R]) -> R` every identifiability method, and
`SensitivityAnalysis.run(method: SensitivityMethod[R]) -> R` every sensitivity
method. The kinds subclass `InferenceMethod[R]` (`context/inference/method.py`)
and share one registry, which hands each runner the object of its entry
point; each entry point refuses another kind with a `TypeError` that names the
right one.
A method is three parts: a configuration that subclasses `UncertaintyMethod`
with its result type (`context/inference/`), a runner in Rust bound in
`py-commol/src/inference.rs`, and a result (`context/inference/`). It registers
its runner in `api/inference/registry.py`; the entry point looks the runner up
by the configuration's type and knows no method. A static type checker infers
`R` from the configuration's type.

| Alternative | Why not |
|---|---|
| One `Calibrator` method per technique | A class with thirty methods that every method edits. |
| Free functions | Each repeats the simulation and problem arguments. |
| `method.run(calibrator)` | The configuration, in `context/`, would import `api/`. |

Every random number of a method comes from the `commol-random` generator,
seeded from `CalibrationProblem.seed` (`replicate_seed`). Coverage studies
(`tests/support/coverage.py`, marker `coverage_study`) check that a method's
intervals cover the truth at their level; they run for a pull request that
changes what they run and before a release
(`.github/workflows/coverage.yaml`).

**Guards:** `tests/uncertainty/test_entry_point.py`,
`tests/uncertainty/test_coverage.py`.

## Sensitivities come from one port

Every reader of derivatives of the predicted observations asks
`SensitivitySource` (`crates/commol-inference/src/sensitivity/`); only a source
knows how its derivatives are made. `FiniteDifferences` is the first: central
differences, one-sided of second order at a bound, with the step
`h = η^{1/3} · max(|θ|, w)` (at most `w / 8`), where `η` is the output
accuracy of the engine (`SimulationEngine::output_accuracy`: machine epsilon,
or the `rtol` of the ODE engine) and `w` the width of the bounds (1 when infinite). Each entry
carries an estimated error, from the steps `h` and `2h` and the accuracy of
the runs; the structural rank counts a singular value only above the norm of
that error (Weyl's inequality). The scaled sensitivities
`S_ij = J_ij · |θ_j| / s_i` do not depend on units. A stochastic engine is
refused: its runs change in jumps with the parameters.

| Alternative | Why not |
|---|---|
| Each method differentiates on its own | The step, the error and the scaling would differ between methods. |
| Exact derivatives first | Symbolic differentiation of expressions and sensitivity equations in two engines; a later source behind the same port. |
| Machine epsilon as the accuracy of every engine | With `rtol = 1e-6` the solver's error over the step is a fifth of the derivative, and the rank reads noise. |
| A fixed rank threshold such as `1e-10` | Below the error of finite differences, so every model looks identifiable. |

**Guards:** `crates/commol-inference/tests/test_sensitivity.rs`,
`crates/commol-inference/tests/test_identifiability.rs`,
`tests/uncertainty/test_identifiability.py`.

## Two verdicts per parameter

An `IdentifiabilityReport` gives each parameter a structural verdict (what
any data of the design of the observations can determine) and a practical
verdict (what these data determine). Only `StructuralRank` on the design of
the observations gives the structural verdict; a dense design keeps its
verdicts in its section. Merging reports keeps a verdict the reports agree
on, and makes one they disagree on `conflicting`, with every reason.
Thresholds and tolerances of the methods are required arguments: they are
modelling judgments.

| Alternative | Why not |
|---|---|
| One verdict | A parameter structurally identifiable and poorly determined by the data, the usual case, would look like a conflict. |
| Defaults for the thresholds | A judgment hidden in the library. |

**Guards:** `tests/uncertainty/test_identifiability.py` (`TestReport`).
## A replicate changes only the data

A resampling method makes each replicate with
`CalibrationProblem::with_data(values, weights)`
(`crates/commol-calibration/src/calibration_problem.rs`): a copy of the problem
whose discrepancy is built as `new` builds it, so with normalization on its
factors come from the replicate's own values, as the fit's do. The engine, the
observation operator and the feasibility stay. A replicate starts from the fit
and, with more starts, from a Latin hypercube, and keeps its best end
(`ReplicateRunner`, `resampling/runner.rs`). Blocks hold every series in a
window of positions (`BlockScheme`), and `block_length` has no default.

| Alternative | Why not |
|---|---|
| `CalibrationProblem::new` per replicate | The problem does not keep its constraints, replicates or normalization, so a second copy of the arguments would live elsewhere. |
| The replicate's data in the `Workspace` | A per-thread buffer would become a second source of data. |
| Keep the fit's normalization factors | The replicates would apply another estimator than the fit. |
| Blocks per series | They break the dependence of series observed at the same time, so intervals of combined quantities come out too narrow. |
| Random starts only | Each replicate may land in another basin, which adds optimizer noise to the data's uncertainty. |

**Guards:** `crates/commol-calibration/tests/test_with_data.rs`,
`crates/commol-inference/tests/test_resampling.rs`,
`tests/uncertainty/test_resampling.py`.

## One SMC core, many targets

Every sequential Monte Carlo method runs on one sampler, `SmcSampler`
(`crates/commol-inference/src/smc/sampler.rs`), which knows no method. A
method is a `TemperedTarget` of two steps: `evaluate(θ, stream)` runs the
model once (one fresh replicate, seeded from the particle's stream, with a
stochastic engine) and gives the value the particle keeps, or `None` outside
the support (`discrepancy_in_support`); `log_kernel(value, level)` is a pure
function of that value. Reweighting and the choice of the next level read the
kept values, so they run no model: only a proposed move calls `evaluate`. The
Gibbs posterior keeps the loss (`−level · η · loss`, temperatures from 0 to
1); ABC keeps the distance (0 within the tolerance, `−∞` outside); the
posterior of observation models (`LikelihoodTarget`, `smc/likelihood.rs`)
keeps the log-likelihood (`level · log L`), so its log evidence is the log
marginal likelihood.

The sampler owns the rest, the same for every target: the next level by
bisection on the conditional effective sample size, to `target_cess`;
systematic resampling under `resample_threshold`; random-walk Metropolis on
the unbounded scale of `transform.rs`, with the covariance of the particles
times `2.38² / d` and the log Jacobian in the acceptance; and
`⌈log c / log(1 − a)⌉` moves per step, with `a` the acceptance of the first
move and `c` the required `stay_probability`. Every particle draws from its
own stream, so a run repeats from its seed whatever the threads. ABC reaches
the same rule with weights of 0 or 1: the next tolerance is the weighted
quantile of the distances, so `ABCSMC` takes `target_cess` and no quantile.
ABC resamples at every step, since a particle of weight 0 cannot move, and
when the first move of a step accepts less than `min_acceptance` it keeps the
particles of the step before, whose moves completed.

Every prior is a `Distribution` of one family (`distribution.rs`), truncated
to its parameter's bounds and drawn by inverting its distribution function
with special functions on `libm`, so a seed gives the same draws on every
platform. Each sampling method takes `priors` with no default, and `η` of a
Gibbs posterior has no default either: a number or a calibration.
Information matching reads the curvature and the meat of the sandwich;
bootstrap coverage interpolates the per-replicate thresholds of `η` at the
positions `i / (M + 1)`, because the largest `η` that covers at the level is
an order statistic whose expected coverage is below it.

| Alternative | Why not |
|---|---|
| One sampler per method (population Monte Carlo for ABC, a tempering sampler for the Gibbs posterior) | Two codes for resampling, moves and diagnostics, which drift apart. |
| `log_kernel(θ, level)` that runs the model | Every change of level would run every particle again, and with a stochastic engine a new simulation would replace the distance a particle was accepted with, so ABC would no longer sample its target. |
| The problem's replicates and their summary for ABC | ABC is valid only when each particle keeps the distance of one simulation. |
| Moves on the scale of the parameters | Proposals near a bound fall outside, each at the cost of a model run, and a parameter over orders of magnitude has a skewed posterior. |
| A fixed number of moves, or moves until the mean squared jump stabilizes | A cost hidden in the code, or no rule a test can check. |
| Distributions from a dependency | Their draws change with its version. |
| A uniform prior by default, or `η = 1` | A modelling choice made silently; the loss has no scale. |

**Guards:** `crates/commol-inference/tests/test_smc.rs`,
`crates/commol-inference/tests/test_distributions.rs`,
`tests/uncertainty/test_smc.py`,
`tests/uncertainty/test_coverage.py` (`TestGibbsPosterior`, `TestABCSMC`).

## Observation noise is an argument

Every reader of observations of a sample (`Predictive.at_observations`, every
function of `commol.scores`) takes the noise with no default: `NoNoise()` gives
a band of the model, `ResidualNoise(...)` and `DistributionNoise()` (the
observation models of the problem) a prediction interval. The log score needs
a density, so it takes only `DistributionNoise`; the parametric bootstrap
(`ParametricResampler`) draws its data sets with the same draws. The Rust port
`ObservationNoise` (`crates/commol-inference/src/predictive.rs`) sees one
draw's whole vector of observations with the draw's parameter values, so a
noise can draw blocks of residuals across observations, and a noise model can
read calibrated noise parameters. `ResidualNoise` serves only series the fit
has residuals of.

| Alternative | Why not |
|---|---|
| `NoNoise` by default | A band of the model would be scored against data with no notice, and cover less than its level. |
| A port per observation | It cannot draw one block for several observations, and sees no position or series. |
| Residual noise on the band at positions | A series may sum outputs with a factor: a residual has no correct split among outputs, and none for an output nobody observed. |

| A kernel density of the draws for the log score | A bandwidth to choose, and bias in the tails the log score weighs most. |

**Guards:** `crates/commol-inference/tests/test_resampling.rs::residual_noise_rejects_a_series_with_no_residuals`,
`tests/uncertainty/test_resampling.py::TestResidualNoise`,
`crates/commol-inference/tests/test_likelihood_target.rs`,
`tests/uncertainty/test_scores.py::TestLogScore`.

## Conformal variants share one score and one quantile

A conformal variant only chooses which predictions and which held-out values
feed a `NonconformityScore`; the score, its scale and the quantile
`⌈(n + 1)(1 − α)⌉` live once (`crates/commol-inference/src/conformal/score.rs`,
`quantile.rs`). `score`, `by` and `level` are required in every variant, and
`check_conformal_settings` rejects in Rust `"normalized"` per series. A horizon
of `RollingConformal` is a count of observations after the origin. `EnbPI`
draws its own block counts through `ReplicateRunner::fit_one`, and
`AdaptiveConformal` keeps its state in Rust (`AdaptiveLevel`) and returns the
same `ConformalResult` as every variant.

| Alternative | Why not |
|---|---|
| A default score or pooling | The choice sets the pattern of the widths, a modelling choice. |
| `"normalized"` per series | Its factor is one constant per series, so its intervals equal those of `"absolute"`. |
| A horizon as a distance on the axis | Positions are `f64` and can be irregular, so each point would have its own horizon and an unbounded interval. |
| A `CVPlus` class | Blocks of consecutive positions make the folds of CV+, so it is jackknife+ with a longer `block_length`. |
| EnbPI on a `BootstrapResult` | No bootstrap leaves a block out, so no replicate gives a leave-one-out prediction. |
| A mutable Python `AdaptiveConformal` | Configurations are frozen, and every variant returns one result type. |

**Guards:** `crates/commol-inference/tests/test_conformal.rs`,
`tests/uncertainty/test_conformal.py`, the studies of
`tests/uncertainty/test_coverage.py`.

## Scenarios pair the draws of one sample

`Scenarios` (`py-commol/commol/api/inference/scenarios.py`) holds one
`Predictive` per scenario, each on a copy of the model with its overrides,
from `source.with_model(model)` with the seed of the source. Run `k` of every
scenario is then the same draw with the same seed, so `compare` takes each
difference within a run before it reduces across runs. A quantity is one of
`Quantity`, read in Rust (`Predictive.quantities`); a fit is a sample of one
draw (`Calibrator.point_sample`), and one run gives no interval.

| Alternative | Why not |
|---|---|
| `predictive.under(overrides)` and `p1.compare(p2)` | Two predictives that must share the sample and the seeds: arguments that must agree. |
| A callable quantity | A second path in Python, per draw; `scenarios[name].runs()` gives the runs. |
| The difference of two intervals | It ignores the pairs, so it is too wide. |

**Guards:** `tests/uncertainty/test_scenarios.py::test_the_draws_of_every_scenario_share_their_seeds`,
`tests/uncertainty/test_scenarios.py::test_the_interval_of_a_difference_is_taken_draw_by_draw`.
