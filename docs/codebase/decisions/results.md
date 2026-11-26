# Results

[Back to Design decisions](../design-decisions.md)

How intervals and reports are made from a model and its runs. Descriptions,
figures and the files of models and results have pages of their own:
[Descriptions](descriptions.md), [Figures](figures.md) and [Files](files.md).

## One confidence-interval convention

Every interval is a weighted quantile: the inverse of the weighted empirical
distribution, interpolated between the midpoints of the cumulative weights
(`statistics::quantiles` in `crates/commol-statistics/src/statistics.rs`). With
equal weights it equals NumPy's `method="hazen"`; a weight of zero removes a
value. `member_statistics` reduces each member to the
quantity of interest **first**, and then takes weighted quantiles across
members; taking quantiles first gives a band that describes no member. The
replicates of a stochastic run (`SimulationReplicates.quantiles`), a sample,
a predictive and the summary of the replicates of a calibration
(`"median"` and `Quantile(q)`, `ReplicateSummary::summarize`) use the same
quantile. So a calibration with `Quantile(0.25)` compares the observations
with the value that `SimulationReplicates.quantiles([0.25])` gives for the
same runs.

The module lives in `commol-statistics`, a crate with no model and no
optimizer that calibration, inference and the bindings depend on and import
directly (`commol_statistics::statistics`), so there is one copy and one path. Every band and every summary of runs reduces a column with
its `summary` (the mean, the median, the central interval, the minimum and the
maximum), which Python calls as `weighted_summary`.

| Alternative | Why not |
|---|---|
| NumPy's default (`linear`) with equal weights and another rule with weights | Two conventions: a sample would change its interval when it gains weights. |
| Harrell–Davis | Smoother, but with no simple weighted form, and slower. |
| The module in `commol-core` | `commol-core` holds the model types, the engine contract and the threads. |
| The module in `commol-calibration` | A quantile is not calibration, and a crate that needs it would depend on the problem and its optimizers. |
| A copy of Hazen's rule in `commol-calibration` | Two quantiles again, which can drift apart. |

**Rule:** every new interval goes through `weighted_quantiles` or
`member_statistics`, and every quantile in Rust through
`statistics::quantiles`.

**Guards:** `crates/commol-statistics/src/statistics.rs` tests,
`crates/commol-calibration/tests/test_replicate_summary.rs`,
`tests/uncertainty/test_intervals.py`,
`tests/stochastic/test_calibration.py::test_a_quantile_summary_agrees_with_the_quantiles_of_the_replicates`,
and `tests/suite/test_rust_rules.py::test_commol_has_one_quantile`, which fails
when a Rust function named for a quantile sorts values outside
`statistics.rs`. The conformal quantile (`conformal/quantile.rs`) is the one
exception: it is a rank that gives an interval its coverage, not an estimate
of a quantile.

## A setting has one name and one check

A setting that means the same thing has the same name in every method, in
Python and in Rust. The level of an interval is `confidence_level`: in the
conformal methods, `Threshold`, `Sobol`, `BootstrapCoverageEta`, and in the
results that report it (`PredictionIntervals`, `ProfileSection`). A count is
a plain plural noun: `particles` (`ParticleSwarmConfig` and every SMC method),
`draws`, `replicates` (every bootstrap, `Sobol`, `EnbPI`), and `count` for the
argument of `Simulation.replicates`, `ParameterSample.resample` and
`Distribution.sample`. The rule that combines the replicates of a ratio is
`Ratio.replicate_rule`, so `replicates` is always a count. The level of a
quantile keeps the name `level` (`Quantile(level)`): it is not the level of an
interval.

The engine declares the range rule of each setting once, as data:
`Count::at_least("replicates", 1)` for a whole number, `Real::new(name, low,
high)` for a number with an open or a closed bound (`Real::share`,
`Real::positive`, …), in `crates/commol-core/src/range.rs`,
`crates/commol-calibration/src/ranges.rs` and
`crates/commol-inference/src/ranges.rs`. A rule has one message for every
value outside its range, a negative count included (`replicates is a whole
number of at least 1, got -1`); `statistics::check_levels` checks a quantile
level. Three paths read a rule: the Python model of an object, whose base
`RangeChecked` (`py-commol/commol/_data/ranges.py`) calls the one check of
the engine for its object when it is built (`commol_rs.core.check_ranges`
on the tables `RANGES`, each setting in its own type, so a parameter name
`"NaN"` is no number); the engine function that takes the setting; and a
binding, which reads a count (`i64`) with its rule (`py-commol/src/counts.rs`).

A caller checks a confidence level at its boundary, then computes the two
quantiles of its interval with `statistics::interval_levels`, a pure function
of `commol-statistics` that checks nothing: the crate of statistics depends on
no crate of commol. `BootstrapResult.interval_error`,
`PropagationResult.summary` and the scenario summaries call the binding
`commol_rs.inference.interval_levels`, which checks, then computes.

| Alternative | Why not |
|---|---|
| A pydantic constraint (`Field(gt=0.0, lt=1.0)`) in each model | The rule exists twice, the same mistake gives a different message in each method, and the message of the engine shows only where Python has none. |
| A message of its own in `counts.rs` | `-1` and `0` get two wordings of one rule. |
| An alias for an old name | Two names for one setting, which is what the rule removes. |
| The rules of the settings in `commol-statistics` | The settings of an engine depend on the crate of statistics, and a computation and a rule of a setting stay mixed. |

**Rule:** a new setting takes the name its meaning already has; a setting
with a range gets its rule, a row for its object in `RANGES` and the check in
the engine function that takes it.

**Guards:** `tests/suite/test_engine_rules.py` (`test_a_setting_has_one_name`,
`test_no_model_declares_a_range_of_a_setting`), `tests/uncertainty/test_ranges.py`,
`tests/uncertainty/test_counts.py`, `tests/uncertainty/test_intervals.py`,
`crates/commol-calibration/tests/test_ranges.rs`,
`crates/commol-inference/tests/test_ranges.rs`, `test_levels.rs`.

## A method takes its seed from its source and fits from its starts

Every uncertainty, identifiability and sensitivity method draws its random
numbers from the seed of its source: `Calibrator.seed` (`problem.seed`) or
`SensitivityAnalysis.seed`, split into streams (`Stream` in
`crates/commol-random/src/streams.rs`, each with its own number) and the
replicates of a stream (`Stream::replicate`); no method draws from the seed
itself. The starts of a calibration draw from two streams of the same
registry (`Stream::StartPoints`, `Stream::StartSeeds`). No method has a seed of its own, so one seed repeats a calibration
and every method on it. A score reads the seed its draws keep,
`ObservationDraws.seed` (`crates/commol-inference/src/predictive.rs`) or
`ObservationBounds.seed` (`scores.rs`): the place `v` of each observed value
within its mass `p(y)` comes from `Stream::PitTies` of it, one per
observation, and the PIT and the quantile residuals read the same `v`, so
`quantile_residuals` is `Φ⁻¹` of `pit`. Every method that fits the
data takes `starts`, at least 1 with the default 1, and fits with
`Calibrator.run(starts)`: the bootstraps, the jackknife, the conformal
methods, the profiles, `Laplace`, `Sandwich`, `LaplaceMixture`,
`InformationMatchingEta` and `BootstrapCoverageEta`.

A seed that its user does not set is drawn in one place,
`commol_random::seed_or_drawn` (`crates/commol-random/src/drawn.rs`, from the
entropy of the operating system). The bindings of a run, its replicates, a
calibration problem and a sensitivity analysis take `seed: Option<u64>`, and
each result gives the seed it ran with (`results.seed`,
`SimulationReplicates.seed`, `Calibrator.seed`, `SensitivityAnalysis.seed`),
so a run can be repeated. A method that needs the seed of its user, such as a
sensitivity analysis on a stochastic engine, refuses `None` with the message
of the engine (`ForwardModel::unobserved`); no type tells the engine where a
seed comes from.

| Alternative | Why not |
|---|---|
| Python draws the seed and passes it to the engine | Two places draw, and the engine cannot tell a seed its user set from one drawn for them, so it needs a second argument or type to refuse a drawn seed. |
| A seed type that says where it comes from (`Set` or `Drawn`) | The engine reads only whether a seed is set; `Option<u64>` says that. |
| A `seed` of a method that overrides the seed of the source | Two seeds choose the random numbers of one run, and `problem.seed` no longer repeats it. |
| A method that fits from one start, with no `starts` | A fit that ends at a local minimum gives a wrong center, and the user cannot ask for more starts. |

**Rule:** a new method takes no seed and reads the seed of its source; a new
method that fits the data takes `starts` with the default 1. A function with
no source, as `series_dependence` of any values, takes the seed, such as
`Calibrator.seed`.

**Guards:** `tests/uncertainty/test_identifiability.py::TestStructuralRank::test_structural_rank_takes_the_seed_of_the_calibrator`,
`crates/commol-random/tests/test_seeds.rs::a_seed_that_is_set_is_kept_and_one_that_is_not_is_drawn`,
`tests/model_kinds/test_stochastic.py` (`test_a_run_without_a_seed_keeps_the_seed_it_drew`, `test_replicates_without_a_seed_keep_the_seed_they_drew`),
`tests/uncertainty/test_sensitivity.py::TestAnalysis::test_with_no_seed_one_is_drawn_and_logged`,
`tests/uncertainty/test_scores.py::TestPit::test_the_pit_takes_the_seed_of_the_source`,
`crates/commol-inference/tests/test_predictive.rs::the_pit_of_ties_follows_the_seed_of_the_draws`,
`tests/uncertainty/test_local.py::TestStarts`,
`tests/uncertainty/test_smc.py::test_the_eta_calibrations_fit_from_their_starts`.

## Fits that fail have one place

Seven methods fit one perturbation of the data per replicate: the bootstraps,
the jackknife, jackknife+, rolling conformal, EnbPI, a bootstrap threshold and
the `Refitted` reference of a dependence analysis. They share one loop,
`ReplicateRunner::fit_all` (`resampling/runner.rs`), which returns
`ReplicateFits`, and one check, `FitUnit::require`, whose message says how many
of how many fits of a replicate, block or origin succeeded and how many the
method needs. Every Python result of several fits has `failed`, entries
`"<Start|Replicate|Block|Origin> <index> failed: <reason>"` logged as
warnings, and, for fits of replicates, `not_converged`, at its top.

Every refit starts from the fit, so `ReplicateRunner::new` refuses a fit that
has no run or is not feasible (`CalibrationProblem::evaluate`). The message
names the violated constraints of the first check that fails: the constraints
on the parameters come before those at positions. `ResidualNoise`,
`SplitConformal` and `AdaptiveConformal` do not refit, so they take a fit that
runs and is not feasible.

| Alternative | Why not |
|---|---|
| One loop per method, or `diagnostics.failed` | Counts and messages drift apart; three results have no diagnostics. |
| Refits from a fit that is not feasible | With no feasible point in the bounds, the fit ends on the penalty, and every refit starts and stays there: an interval of width 0, with no message. |
| The check in `fitted_values` | `ResidualNoise` reads a fit through it and must take one that is not feasible, and `RollingConformal` and `Refitted` with `NoNoise()` do not call it. |

**Guards:** `crates/commol-inference/tests/test_resampling.rs::every_replicate_method_counts_the_same_fits`,
`tests/uncertainty/test_resampling.py::test_every_method_lists_its_failed_replicates_in_failed`,
`crates/commol-inference/tests/test_errors.rs::a_method_that_refits_refuses_a_fit_that_is_not_feasible`,
`tests/uncertainty/test_resampling.py::test_a_fit_that_violates_a_constraint_is_refused`,
`tests/uncertainty/test_conformal.py::TestFailures::test_a_variant_that_refits_refuses_a_fit_that_is_not_feasible`,
`tests/uncertainty/test_dependence.py::TestReferences::test_refits_refuse_a_fit_that_is_not_feasible`,
`tests/uncertainty/test_resampling.py::TestResidualNoise::test_residual_noise_reads_a_fit_that_is_not_feasible`,
`tests/uncertainty/test_conformal.py::TestFailures::test_a_variant_that_reads_the_fit_takes_one_that_is_not_feasible`,
`tests/uncertainty/test_conformal.py::TestFailures::test_rolling_settings_come_before_the_check_of_the_fit`,
`tests/uncertainty/test_block_length.py::test_a_bootstrap_checks_the_length_before_the_run_of_the_fit`.

## Reports

A report is a method of the object of results whose data it describes:
`SimulationResults.report` (one run), `SimulationReplicates.report` (the
replicates of a stochastic engine), `Predictive.report` (the draws of a
sample), `Model.applied_parameters` (calibrated parameters, and the sections
of the model kind: `initial_amounts` and `category_splits` for an aggregate
model) and `IdentifiabilityReport.report`. Private functions in the package
of each object build its block, which is data, not a document:
`_simulation/reports.py` for a run and an ensemble,
`_model/applied_parameters.py` for the applied parameters and
`identifiability/report.py` for an identifiability analysis. No module
function reports anything.

| Rule | Why, against the alternative |
|---|---|
| The report of a run is on the results, not on `Simulation` | `Simulation.report(results, ...)` puts the report on the object that produces the data and takes the data as an argument; `results.report(simulation, ...)` takes two objects that must agree. |
| `SimulationResults` and `SimulationReplicates` hold a reference to their `Model`, which `Simulation.run`, `run_at` and `replicates` set; `Predictive` reads it through `source.simulation` | A report reads the outputs of each bin and accumulator (`ModelStructure.outputs_by_source`); a copy of that map would be a second source of truth, and the figures also need `series_label`. A report reads only the structure, never the values of the parameters, so a later `update_parameters` does not change it. `Model` holds no Rust object, so results still pickle. Results built by hand have no model, and a report that needs one raises `ValueError` that names the bin or accumulator. |
| `Predictive.report` takes no point estimate | Each leaf of an ensemble comes from its members only; the mean of the draws is not a run of the model. |
| The block of a report lives in the package of its object | A module of every block imports every object it reports, against the order of the layers ([Python models](python-models.md#the-layers-of-the-package)); the block changes with its object. |
| `Model.applied_parameters` is on the model | Most of the block is data of the model (one row per parameter, the sections of its structure); the values and the problem only add what calibration set. |
| `Model.applied_parameters` takes the problem | An id does not say what a value set; the problem declares it, and the engine resolves the values of the kind ([A report marks what the calibration set](calibration-write-back.md#a-report-marks-what-the-calibration-set-as-the-engine-resolves-it)). |

- A single run and an ensemble (replicates or a `Predictive`) differ only at
  the leaves (`_ReportLeaf`); a predictive's members carry their weights.
- The key order of each entry is part of the contract, because it changes saved
  JSON files.
- Labels and summary figures belong to the caller, because they carry domain
  vocabulary.

**Guard:** `tests/reporting/` (`test_reports.py` checks that each method
gives the block of its definition, with its key order).
