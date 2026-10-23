# Results

[Back to Design decisions](../design-decisions.md)

How intervals, descriptions, reports and model files are made from a model and its runs.

## One confidence-interval convention

Every interval is a weighted quantile: the inverse of the weighted empirical
distribution, interpolated between the midpoints of the cumulative weights
(`statistics::quantiles` in `crates/commol-calibration/src/statistics.rs`). With
equal weights it equals NumPy's `method="hazen"`; a weight of zero removes a
value. `ci_percentiles` (`api/inference/intervals.py`) turns a confidence level
into the two percentile points of `statistics::interval_levels`. `member_statistics` reduces each member to the
quantity of interest **first**, and then takes weighted quantiles across
members; taking quantiles first gives a band that describes no member. The
replicates of a stochastic run (`SimulationReplicates.quantiles`), a sample,
a predictive and the summary of the replicates of a calibration
(`"median"` and `Quantile(q)`, `ReplicateSummary::summarize`) use the same
quantile. So a calibration with `Quantile(0.25)` compares the observations
with the value that `SimulationReplicates.quantiles([0.25])` gives for the
same runs.

The module lives in `commol-calibration`, the lowest crate that takes a
quantile, and `commol-inference` re-exports it (`pub use
commol_calibration::statistics`), so there is one copy.

| Alternative | Why not |
|---|---|
| NumPy's default (`linear`) with equal weights and another rule with weights | Two conventions: a sample would change its interval when it gains weights. |
| Harrell–Davis | Smoother, but with no simple weighted form, and slower. |
| The module in `commol-core` | No crate below calibration takes a quantile, and `commol-core` holds the model types and the engine contract. |
| A copy of Hazen's rule in `commol-calibration` | Two quantiles again, which can drift apart. |

**Rule:** every new interval goes through `weighted_quantiles` or
`member_statistics`, and every quantile in Rust through
`statistics::quantiles`.

**Guards:** `crates/commol-calibration/src/statistics.rs` tests,
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

One Rust function checks each kind of setting, with one message:

| Check | Function | Message |
|---|---|---|
| A confidence level | `statistics::interval_levels` | `confidence_level is strictly between 0 and 1, got 1.5; use 0.95 for a 95% interval` |
| Any other setting strictly between 0 and 1 (`target_cess`, `stay_probability`, `min_acceptance`, the `quantile` of ABC, `correlation_threshold`) | `statistics::check_open_unit(name, value)` | `<name> is strictly between 0 and 1, got <value>` |

Python calls the first through the binding `commol_rs.inference.interval_levels`:
`ci_percentiles`, `BootstrapResult.interval_error`, `PropagationResult.summary`,
the scenario summaries, and every `confidence_level` field, whose type is
`ConfidenceLevel` (`context/inference/levels.py`).

| Alternative | Why not |
|---|---|
| A check in each method, with `Field(gt=0.0, lt=1.0)` | The same mistake gives a different message in each method, and the rule is written many times. |
| An alias for an old name | Two names for one setting, which is what the rule removes. |

**Rule:** a new setting takes the name its meaning already has, and a new
setting strictly between 0 and 1 is checked by one of the two functions.

**Guards:** `tests/suite/test_rules.py::test_a_setting_has_one_name`,
`tests/uncertainty/test_intervals.py::test_every_confidence_level_has_one_check`,
`crates/commol-inference/tests/test_levels.rs`.

## A method takes its seed from its source and fits from its starts

Every uncertainty, identifiability and sensitivity method draws its random
numbers from the seed of its source: `Calibrator.seed` (`problem.seed`) or
`SensitivityAnalysis.seed`, split into streams (`commol_random::stream_seed`,
`replicate_seed`). No method has a seed of its own, so one seed repeats a
calibration and every method on it. A score reads the seed its draws keep,
`ObservationDraws.seed` (`crates/commol-inference/src/predictive.rs`):
`scores::pit` randomizes ties from a stream of it. Every method that fits the
data takes `starts`, at least 1 with the default 1, and fits with
`Calibrator.run(starts)`: the bootstraps, the jackknife, the conformal
methods, the profiles, `Laplace`, `Sandwich`, `LaplaceMixture`,
`InformationMatchingEta` and `BootstrapCoverageEta`.

| Alternative | Why not |
|---|---|
| A `seed` of a method that overrides the seed of the source | Two seeds choose the random numbers of one run, and `problem.seed` no longer repeats it. |
| A method that fits from one start, with no `starts` | A fit that ends at a local minimum gives a wrong center, and the user cannot ask for more starts. |

**Rule:** a new method takes no seed and reads the seed of its source; a new
method that fits the data takes `starts` with the default 1.

**Guards:** `tests/uncertainty/test_identifiability.py::TestStructuralRank::test_structural_rank_takes_the_seed_of_the_calibrator`,
`tests/uncertainty/test_scores.py::TestPit::test_the_pit_takes_the_seed_of_the_source`,
`crates/commol-inference/tests/test_predictive.rs::the_pit_of_ties_follows_the_seed_of_the_draws`,
`tests/uncertainty/test_local.py::TestStarts`,
`tests/uncertainty/test_smc.py::test_the_eta_calibrations_fit_from_their_starts`.

## Fits that fail have one place

Six methods fit one perturbation of the data per replicate: the bootstraps,
the jackknife, jackknife+, rolling conformal, EnbPI and a bootstrap threshold.
They share one loop, `ReplicateRunner::fit_all` (`resampling/runner.rs`),
which returns `ReplicateFits`, and one check, `FitUnit::require`, whose message
says how many of how many fits of a replicate, block or origin succeeded and
how many the method needs. Every Python result of several fits has `failed`,
entries `"<Start|Replicate|Block|Origin> <index> failed: <reason>"` logged as
warnings, and, for fits of replicates, `not_converged`, at its top.

| Alternative | Why not |
|---|---|
| One loop per method, or `diagnostics.failed` | Counts and messages drift apart; three results have no diagnostics. |

**Guards:** `crates/commol-inference/tests/test_resampling.rs::every_replicate_method_counts_the_same_fits`,
`tests/uncertainty/test_resampling.py::test_every_method_lists_its_failed_replicates_in_failed`.

## A description is data, rendered apart

`Model.describe(layout)` returns a `ModelDescription`
(`context/description.py`): sections with a key, a title and blocks from a
closed set of Pydantic models (`Text`, `Table`, `Equations`, `StateGraph`,
`Matrix`). The sections every kind shares come from `Model`
(`context/_model/describe.py`); a kind adds its own through
`ModelStructure.describe` (`Aggregate`: `context/_model/aggregate_description.py`).
The renderers (`api/description.py`) read only blocks, so a new kind renders
in every format with no new renderer. The form of the dynamics and the
schedule come from the engine's list (`EngineType::dynamics_form`), because
one structure runs as equations or as events depending on its engine.

| Part | Rule |
|---|---|
| Equations | Computed by the code of the planner's flows (`transition_flows`), so a description shows what the engine simulates. The text rendering keeps the compact and expanded forms line by line. |
| Diagrams | Mermaid in the Markdown rendering, only for a graph that is not `expanded`; Graphviz DOT for every graph in a `.dot` file. Text, LaTeX and JSON draw none. |
| Layouts | `"default"`: overview, entities, dynamics, schedule, initialization, parameters, outputs. `"odd"`: the ODD protocol; an element commol cannot fill is marked `TO_BE_WRITTEN`. A section with no blocks is left out of every rendering. |

| Alternative | Why not |
|---|---|
| A printer per kind and per format (the former `print_equations`) | Every kind writes every format, and no caller can read or export the description. |
| A free-form dict | The renderers would guess its shape; typed blocks are checked by Pydantic and `ty`. |
| An image (PNG, SVG) of a graph | Needs Graphviz or a browser at run time, a dependency for a view. |
| TikZ in the LaTeX rendering | Needs node coordinates, so a layout algorithm; a PDF made from the DOT file does the same. |

**Guards:** `tests/description/` (`test_equations*.py` check the lines of the
equations through `equation_forms`; `test_rendering.py` checks the snapshots of
`description/snapshots/`, parses the DOT with `pydot` and compiles the LaTeX
with `latexmk` when they are installed), `tests/building/test_engine.py`
(every engine gives a form).

## Reports

A report is a method of the object of results whose data it describes:
`SimulationResults.report` (one run), `SimulationReplicates.report` (the
replicates of a stochastic engine), `Predictive.report` (the draws of a
sample), `Model.applied_parameters` (calibrated parameters, and the sections
of the model kind: `initial_amounts` and `category_splits` for an aggregate
model) and `IdentifiabilityReport.report`. The private functions of
`api/reporting.py` build the blocks, which are data, not documents. No module
function reports anything.

| Rule | Why, against the alternative |
|---|---|
| The report of a run is on the results, not on `Simulation` | `Simulation.report(results, ...)` puts the report on the object that produces the data and takes the data as an argument; `results.report(simulation, ...)` takes two objects that must agree. |
| `SimulationResults` and `SimulationReplicates` hold a reference to their `Model`, which `Simulation.run`, `run_at` and `replicates` set; `Predictive` reads it through `source.simulation` | A report reads the outputs of each bin and accumulator (`ModelStructure.outputs_by_source`); a copy of that map would be a second source of truth, and the figures also need `series_label`. A report reads only the structure, never the values of the parameters, so a later `update_parameters` does not change it. `Model` holds no Rust object, so results still pickle. Results built by hand have no model, and a report that needs one raises `ValueError` that names the bin or accumulator. |
| `Predictive.report` takes no point estimate | Each leaf of an ensemble comes from its members only; the mean of the draws is not a run of the model. |
| `Model.applied_parameters` is on the model | Most of the block is data of the model (one row per parameter, the sections of its structure); the values and the problem only add what calibration set. |

- A single run and an ensemble (replicates or a `Predictive`) differ only at
  the leaves (`ReportLeaf`); a predictive's members carry their weights.
- The key order of each entry is part of the contract, because it changes saved
  JSON files.
- Labels and summary figures belong to the caller, because they carry domain
  vocabulary.

**Guard:** `tests/reporting/` (`test_reports.py` checks that each method
gives the block of its definition, with its key order).

## Figures

A figure is read from the object of results whose data it draws, as a report
is: `SimulationResults.plot`, `SimulationReplicates.plot` and `Predictive.plot`
are accessors with `series(...)`, `cumulative(...)` and `observations(...)`,
and `PRCCResult`, `MorrisResult`, `SobolResult`, `ChaosResult` and
`ProfileResult` have `plot(...)`. The figures read the model of the results
(`results.model`, or `source.simulation` of a predictive), and only its
structure: `ModelStructure.outputs_by_source` and `series_label`.

`api/plotting/` has three layers:

| Module | What it holds |
|---|---|
| `api/plotting/accessors.py`, `api/plotting/compared.py` | The public layer: the accessors, and `plot_compared`, the figure of several results. Each method or function takes the arguments of its drawing function and passes them on. |
| `api/plotting/figures.py` | The drawing functions, private (`_run_series`, `_prcc`, …). Only this module imports matplotlib. |
| `api/plotting/panels.py` | The data of each panel (name, unit, line, band, observed values, prediction intervals; `ComparedPanel`, a line and a band per result; `ObservationPanel`, the model, band and window at each observation; `ComparedObservationPanel`, a model and a band per result at each observation), computed without matplotlib. |

| Rule | Why, against the alternative |
|---|---|
| Figures are methods of the results, over drawing functions that take `axes=` | `SimulationPlotter(simulation, results)` took two objects that must agree, and a third for a predictive; a facade built from one result has no place for several results on one figure. |
| The accessors and `plot()` are the only public way to draw a figure of one result | One public way per figure, as for reports. Public drawing functions too would double the documentation and the tests, and drift. |
| Several results on the same panels are one public function, `plot_compared({label: result}, …)`, exported from `commol` | A comparison describes no single result: a method of one result, or a mapping in `plot.series`, would give one of them a special role. Each result reads its own model, so the function checks that each has the outputs of the first and names the key of the one that does not. `axes=` alone draws on the same panels, but checks no outputs and gives no label or color per result. |
| A result of a comparison has the same color in every panel: `C0`, `C1`, … of the style, in the order of the keys, for its line and its band | The next color of the cycle of each `Axes` depends on what the panel already holds, so a result would change color between panels. |
| A figure returns a matplotlib `Figure`; `axes=` takes a `Figure` of commol, and finds the panel of each series by the label of its `Axes` (`ax.set_label` with the id of the series) | One return type. A `dict` of `Axes` loses the `Figure`, a tuple comes from every call, and a commol object locks the API in. A series with no panel raises `ValueError` that names it. |
| matplotlib is the optional extra `commol[plot]`, imported when a figure is drawn | The core needs no plotting library, and `import commol` loads none. Without it, a figure raises `ImportError` with the install line. seaborn drew only `lineplot` and `scatterplot`, which `Axes.plot` and `Axes.scatter` draw the same. |
| A panel per observed series is its own method, `plot.observations(observed_data)`, not a mode of `plot.series` | `plot.series` draws the model along the whole axis; `plot.observations` reads it only at the observations. A mode would change which panels and which axis a figure has with the value of one argument. |
| `plot.observations` reads the model only at the observations, each through its own operator (`run_observation_panels`, `replicate_observation_panels`, and `Predictive.at_observations(..., NoNoise())` in `predictive_observation_panels`) | The operator is defined for each observation in Rust (`ObservationOperator`), and a label can mix windows, outputs and factors, so the model away from the data would need a rule of its own in Python. A value with a window is a total over `(at - window, at]`, drawn as a step over its window with its band as a rectangle; a line between window ends would suggest values that do not exist. The run along the whole axis is `plot.series` on the same `axes=`. |
| `plot_compared(..., panels="observations")` draws one panel per observed series, each result read as `plot.observations` reads it (`compared_observation_panels` over `run_observation_panels`, `replicate_observation_panels` and `predictive_observation_panels`); the default `panels="series"` draws one panel per series | `plot_compared` is the one public function of a comparison, so a second function for the observed series would double its checks of the outputs, its colors and its `axes=`. The user chose this argument for #305, as #251 named it. `series` and `window` raise `ValueError` with `panels="observations"`: the labels set the panels, and each observation its own window. |
| The band of `plot.observations` says in its legend that it has no observation noise | It is the band of the model across replicates or draws, read with `NoNoise()`: not a prediction interval of the observations. |
| The style of a figure stays in that figure | `PlotConfig.style` takes matplotlib styles, applied in a copy of `matplotlib.rcParams`: every drawing function is under `_scoped_style` (`api/plotting/figures.py`, `matplotlib.rc_context`), so `rcParams` is as it was when the function returns or raises. seaborn set the style, palette and context of the whole process. A new drawing function takes the same decorator; `_apply_style` outside it changes the style of every later figure of the user. |

**Guards:** `tests/plotting/test_figures.py` (`test_a_figure_leaves_the_global_style`,
`test_import_commol_loads_no_plotting_library`,
`test_each_accessor_passes_on_the_arguments_of_its_function`,
`test_a_figure_draws_on_given_axes`), `tests/plotting/test_compared.py`
(`test_compared_runs_share_the_panels`, `test_compared_runs_need_the_same_outputs`,
`test_compared_results_draw_a_panel_per_observed_series`),
`tests/plotting/test_observed_series.py` (`test_each_observed_point_is_read_through_its_own_window`,
`test_a_windowed_value_is_drawn_as_a_step_over_its_window`) and
`tests/plotting/test_panels.py::test_the_panels_of_a_run_need_no_matplotlib`.

## Model files are written in full

`Model.to_json` writes every field, nulls included. Dropping nulls is not
round-trip safe: an uncalibrated `Parameter.value` or `BinFraction.fraction`
would lose a required key and fail to reload. `render_json`
(`context/_model/serialization.py`) keeps files readable by writing long numeric
arrays on one line. It is internal on purpose: an output format in the public
API becomes a compatibility promise.

## Public models forbid unknown fields

Every public Pydantic class sets `model_config = ConfigDict(extra="forbid")`.
An unknown field, such as a misspelt or an old name (`window_steps`), raises a
`ValidationError` that names it, instead of being dropped with no error.

**Guard:** `tests/building/test_unknown_fields.py`.
