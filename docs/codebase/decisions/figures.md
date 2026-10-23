# Figures

[Back to Design decisions](../design-decisions.md)

How a figure is drawn from a result, and which module of `api/plotting/` holds
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

`api/plotting/` has three layers:

| Module | What it holds |
|---|---|
| `api/plotting/accessors.py`, `api/plotting/compared.py` | The public layer: the accessors, and `plot_compared`, the figure of several results. Each method or function takes the arguments of its drawing function and passes them on. |
| `api/plotting/figures.py` | The drawing functions, private (`_run_series`, `_prcc`, …). Only this module imports matplotlib. |
| `api/plotting/panels.py` | The data of each panel (name, unit, line, band, observed values, prediction intervals; `ComparedPanel`, a line and a band per result; `ObservationPanel`, the model, band and window at each observation; `ComparedObservationPanel`, a model and a band per result at each observation), computed without matplotlib. |

| Rule | Why, against the alternative |
|---|---|
| Figures are methods of the results, over drawing functions that take `axes=` | A facade built from a simulation and its results takes two objects that must agree, and a third for a predictive; a facade built from one result has no place for several results on one figure. |
| The accessors and `plot()` are the only public way to draw a figure of one result | One public way per figure, as for reports. Public drawing functions too would double the documentation and the tests, and drift. |
| Several results on the same panels are one public function, `plot_compared({label: result}, …)`, exported from `commol` | A comparison describes no single result: a method of one result, or a mapping in `plot.series`, would give one of them a special role. Each result reads its own model, so the function checks that each has the outputs of the first and names the key of the one that does not. `axes=` alone draws on the same panels, but checks no outputs and gives no label or color per result. |
| A result of a comparison has the same color in every panel: `C0`, `C1`, … of the style, in the order of the keys, for its line and its band | The next color of the cycle of each `Axes` depends on what the panel already holds, so a result would change color between panels. |
| A figure returns a matplotlib `Figure`; `axes=` takes a `Figure` of commol, and finds the panel of each series by the label of its `Axes` (`ax.set_label` with the id of the series) | One return type. A `dict` of `Axes` loses the `Figure`, a tuple comes from every call, and a commol object locks the API in. A series with no panel raises `ValueError` that names it. |
| matplotlib is the optional extra `commol[plot]`, imported when a figure is drawn | The core needs no plotting library, and `import commol` loads none. Without it, a figure raises `ImportError` with the install line. `Axes.plot` and `Axes.scatter` draw what a figure needs, so a second plotting library adds nothing. |
| A panel per observed series is its own method, `plot.observations(observed_data)`, not a mode of `plot.series` | `plot.series` draws the model along the whole axis; `plot.observations` reads it only at the observations. A mode would change which panels and which axis a figure has with the value of one argument. |
| `plot.observations` reads the model only at the observations, each through its own operator (`run_observation_panels`, `replicate_observation_panels`, and `Predictive.at_observations(..., NoNoise())` in `predictive_observation_panels`) | The operator is defined for each observation in Rust (`ObservationOperator`), and a label can mix windows, outputs and factors, so the model away from the data would need a rule of its own in Python. A value with a window is a total over `(at - window, at]`, drawn as a step over its window with its band as a rectangle; a line between window ends would suggest values that do not exist. The run along the whole axis is `plot.series` on the same `axes=`. |
| `plot_compared(..., panels="observations")` draws one panel per observed series, each result read as `plot.observations` reads it (`compared_observation_panels` over `run_observation_panels`, `replicate_observation_panels` and `predictive_observation_panels`); the default `panels="series"` draws one panel per series | `plot_compared` is the one public function of a comparison, so a second function for the observed series would double its checks of the outputs, its colors and its `axes=`. `series` and `window` raise `ValueError` with `panels="observations"`: the labels set the panels, and each observation its own window. |
| The band of `plot.observations` says in its legend that it has no observation noise | It is the band of the model across replicates or draws, read with `NoNoise()`: not a prediction interval of the observations. |
| The style of a figure stays in that figure | `PlotConfig.style` takes matplotlib styles, applied in a copy of `matplotlib.rcParams`: every drawing function is under `_scoped_style` (`api/plotting/figures.py`, `matplotlib.rc_context`), so `rcParams` is as it was when the function returns or raises. A library that sets the style of the whole process changes every later figure. A new drawing function takes the same decorator; `_apply_style` outside it changes the style of every later figure of the user. |

**Guards:** `tests/plotting/test_figures.py` (`test_a_figure_leaves_the_global_style`,
`test_import_commol_loads_no_plotting_library`,
`test_each_accessor_passes_on_the_arguments_of_its_function`,
`test_a_figure_draws_on_given_axes`), `tests/plotting/test_compared.py`
(`test_compared_runs_share_the_panels`, `test_compared_runs_need_the_same_outputs`,
`test_compared_results_draw_a_panel_per_observed_series`),
`tests/plotting/test_observed_series.py` (`test_each_observed_point_is_read_through_its_own_window`,
`test_a_windowed_value_is_drawn_as_a_step_over_its_window`) and
`tests/plotting/test_panels.py::test_the_panels_of_a_run_need_no_matplotlib`.
