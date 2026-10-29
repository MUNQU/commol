# Figures

[Back to Design decisions](../design-decisions.md)

How a figure is drawn from a result, and which module of `plotting/` holds
what. A report is read from its results in the same way
([Reports](results.md#reports)).

## Figures are read from their results

A figure is read from the object of results whose data it draws, as a report
is: `SimulationResults.plot`, `SimulationReplicates.plot` and `Predictive.plot`
are accessors with `series(...)`, `cumulative(...)` and `observations(...)`,
and `PRCCResult`, `MorrisResult`, `SobolResult`, `ChaosResult` and
`ProfileResult` have `plot(...)`. The figures read the model of the results
(`results.model`, or `source.simulation` of a predictive), and only its
structure: `ModelStructure.outputs_by_source` and `series_label`.

`plotting/` is the one drawing layer, and it knows no result: it sits above
`_model` and below `_simulation` in the layers of the package
([Python models](python-models.md#the-layers-of-the-package)). Each result
builds the data of its figures, and `plotting/` draws them:

| Module | What it holds |
|---|---|
| `_simulation/plots.py`, `uncertainty/plots.py` | The accessors of a run, of replicates and of a predictive, and their drawing functions (`run_series`, `predictive_observations`, …). Each method takes the arguments of its drawing function and passes them on; the drawing function builds the panels and imports `plotting/figures.py` when it draws. |
| `_simulation/panels.py`, `uncertainty/panels.py` | The data of the panels of a run, of replicates and of a predictive (`run_panels`, `replicate_panels`, `predictive_panels`, …), with the reductions they need (`spread` of `_statistics/members.py`, `NoNoise`, `_windowed_totals`). |
| `sensitivity/plots.py`, `profiles/plots.py` | The charts of a result with one figure (`prcc_charts`, `sobol_charts`, `profile_charts`, …) and the drawing function its `plot()` calls. |
| `plotting/compared.py` | `plot_compared`, the figure of several results. It reads each result through its `plot` accessor (`Plotted`, `PanelSource` of `plotting/panels.py`), so it imports no result. |
| `plotting/figures.py` | The drawing functions: `series_figure`, `cumulative_figure`, `observations_figure`, `chart_figure` and the figure of a comparison. Only this module imports matplotlib. |
| `plotting/panels.py`, `plotting/charts.py` | The data types of a panel (name, unit, line, band, observed values, prediction intervals; `ComparedPanel`, a line and a band per result; `ObservationPanel`, the model, band and window at each observation; `ComparedObservationPanel`, a model and a band per result at each observation) and the helpers that build one; the data types of a chart (`Line`, `Point`, `Bars`, `Rule`, `Chart`). Neither imports matplotlib. |

| Rule | Why, against the alternative |
|---|---|
| Figures are methods of the results, over drawing functions that take `axes=` | A facade built from a simulation and its results takes two objects that must agree, and a third for a predictive; a facade built from one result has no place for several results on one figure. |
| A result builds the data of its figure; `plotting/` draws data and imports no result | The dependencies point one way: `plotting/` sits below every result it draws, so a new result adds its panels or charts in its own package and changes no drawing code. A drawing layer that reads each result by its type imports every package above it. |
| The accessors and `plot()` are the only public way to draw a figure of one result | One public way per figure, as for reports. Public drawing functions too would double the documentation and the tests, and drift. |
| Several results on the same panels are one public function, `plot_compared({label: result}, …)`, exported from `commol` | A comparison describes no single result: a method of one result, or a mapping in `plot.series`, would give one of them a special role. Each result reads its own model, so the function checks that each has the outputs of the first and names the key of the one that does not. `axes=` alone draws on the same panels, but checks no outputs and gives no label or color per result. |
| A result of a comparison has the same color in every panel: `C0`, `C1`, … of the style, in the order of the keys, for its line and its band | The next color of the cycle of each `Axes` depends on what the panel already holds, so a result would change color between panels. |
| A figure returns a matplotlib `Figure`; `axes=` takes a `Figure` of commol, and finds the panel of each series by the label of its `Axes` (`ax.set_label` with the id of the series) | One return type. A `dict` of `Axes` loses the `Figure`, a tuple comes from every call, and a commol object locks the API in. A series with no panel raises `ValueError` that names it. |
| matplotlib is the optional extra `commol[plot]`, imported when a figure is drawn | The core needs no plotting library, and `import commol` loads none. Without it, a figure raises `ImportError` with the install line. `Axes.plot` and `Axes.scatter` draw what a figure needs, so a second plotting library adds nothing. |
| A panel per observed series is its own method, `plot.observations(observed_data)`, not a mode of `plot.series` | `plot.series` draws the model along the whole axis; `plot.observations` reads it only at the observations. A mode would change which panels and which axis a figure has with the value of one argument. |
| `plot.observations` reads the model only at the observations, each through its own operator (`run_observation_panels` and `replicate_observation_panels` of `_simulation/panels.py`, and `Predictive.at_observations(..., NoNoise())` in `predictive_observation_panels` of `uncertainty/panels.py`) | A predictive reads each observation through the operator of the engine (`ObservationOperator`). A run and its replicates have no calibration problem, so `_at_observations` reads each observation in Python with the rules of that operator: the window through the binding `window_increments` (`windowed_totals`), and scale times factor through `model_multiplier`. A label can mix windows, outputs, scales and factors, so the model away from the data would need a rule of its own. A value with a window is a total over `(at - window, at]`, drawn as a step over its window with its band as a rectangle; a line between window ends would suggest values that do not exist. The run along the whole axis is `plot.series` on the same `axes=`. |
| `plot_compared(..., panels="observations")` draws one panel per observed series, each result read as `plot.observations` reads it (`compared_observation_panels` over `PanelSource._observation_panels` of each accessor); the default `panels="series"` draws one panel per series | `plot_compared` is the one public function of a comparison, so a second function for the observed series would double its checks of the outputs, its colors and its `axes=`. `series` and `window` raise `ValueError` with `panels="observations"`: the labels set the panels, and each observation its own window. |
| The band of `plot.observations` says in its legend that it has no observation noise | It is the band of the model across replicates or draws, read with `NoNoise()`: not a prediction interval of the observations. |
| The style of a figure stays in that figure | `PlotConfig.style` takes matplotlib styles, applied in a copy of `matplotlib.rcParams`: every drawing function of `plotting/figures.py` is under `_scoped_style` (`matplotlib.rc_context`), so `rcParams` is as it was when the function returns or raises. A library that sets the style of the whole process changes every later figure. A new drawing function takes the same decorator; `_apply_style` outside it changes the style of every later figure of the user. |

**Guards:** `tests/plotting/test_figures.py` (`test_a_figure_leaves_the_global_style`,
`test_import_commol_loads_no_plotting_library`,
`test_each_accessor_passes_on_the_arguments_of_its_function`,
`test_a_figure_draws_on_given_axes`), `tests/plotting/test_compared.py`
(`test_compared_runs_share_the_panels`, `test_compared_runs_need_the_same_outputs`,
`test_compared_results_draw_a_panel_per_observed_series`),
`tests/plotting/test_observed_series.py` (`test_each_observed_point_is_read_through_its_own_window`,
`test_a_windowed_value_is_drawn_as_a_step_over_its_window`) and
`tests/plotting/test_panels.py::test_the_panels_of_a_run_need_no_matplotlib`.
