# Results

[Back to Design decisions](../design-decisions.md)

How intervals, descriptions, reports and model files are made from a model and its runs.

## One confidence-interval convention

Every interval is a weighted quantile: the inverse of the weighted empirical
distribution, interpolated between the midpoints of the cumulative weights
(`statistics::quantiles` in `crates/commol-inference/src/statistics.rs`). With
equal weights it equals NumPy's `method="hazen"`; a weight of zero removes a
value. `ci_percentiles` (`api/inference/intervals.py`) turns a confidence level
into the two percentile points. `member_statistics` reduces each member to the
quantity of interest **first**, and then takes weighted quantiles across
members; taking quantiles first gives a band that describes no member. The
replicates of a stochastic run (`SimulationReplicates.quantiles`), a sample
and a predictive use the same quantile.

| Alternative | Why not |
|---|---|
| NumPy's default (`linear`) with equal weights and another rule with weights | Two conventions: a sample would change its interval when it gains weights. |
| Harrell–Davis | Smoother, but with no simple weighted form, and slower. |

**Rule:** every new interval goes through `weighted_quantiles` or
`member_statistics`.

**Guards:** `crates/commol-inference/src/statistics.rs` tests,
`tests/uncertainty/test_intervals.py`.

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
are accessors with `series(...)` and `cumulative(...)`, and `PRCCResult`,
`MorrisResult`, `SobolResult`, `ChaosResult` and `ProfileResult` have
`plot(...)`. The figures read the model of the results (`results.model`, or
`source.simulation` of a predictive), and only its structure:
`ModelStructure.outputs_by_source` and `series_label`.

`api/plotting/` has three layers:

| Module | What it holds |
|---|---|
| `api/plotting/accessors.py` | The accessors, the public layer. Each method takes the arguments of its drawing function and passes them on. |
| `api/plotting/figures.py` | The drawing functions, private (`_run_series`, `_prcc`, …). Only this module imports matplotlib. |
| `api/plotting/panels.py` | The data of each panel (name, unit, line, band, observed values, prediction intervals), computed without matplotlib. |

| Rule | Why, against the alternative |
|---|---|
| Figures are methods of the results, over drawing functions that take `axes=` | `SimulationPlotter(simulation, results)` took two objects that must agree, and a third for a predictive; a facade built from one result has no place for several results on one figure. |
| The accessors and `plot()` are the only public way to draw a figure | One public way per figure, as for reports. Public drawing functions too would double the documentation and the tests, and drift. |
| A figure returns a matplotlib `Figure`; `axes=` takes a `Figure` of commol, and finds the panel of each series by the label of its `Axes` (`ax.set_label` with the id of the series) | One return type. A `dict` of `Axes` loses the `Figure`, a tuple comes from every call, and a commol object locks the API in. A series with no panel raises `ValueError` that names it. |
| matplotlib is the optional extra `commol[plot]`, imported when a figure is drawn | The core needs no plotting library, and `import commol` loads none. Without it, a figure raises `ImportError` with the install line. seaborn drew only `lineplot` and `scatterplot`, which `Axes.plot` and `Axes.scatter` draw the same. |
| The style of a figure stays in that figure | `PlotConfig.style` takes matplotlib styles, applied in a copy of `matplotlib.rcParams`: every drawing function is under `_scoped_style` (`api/plotting/figures.py`, `matplotlib.rc_context`), so `rcParams` is as it was when the function returns or raises. seaborn set the style, palette and context of the whole process. A new drawing function takes the same decorator; `_apply_style` outside it changes the style of every later figure of the user. |

**Guards:** `tests/plotting/test_figures.py` (`test_a_figure_leaves_the_global_style`,
`test_import_commol_loads_no_plotting_library`,
`test_each_accessor_passes_on_the_arguments_of_its_function`,
`test_a_figure_draws_on_given_axes`) and
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
