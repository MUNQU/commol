# Plotting API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The figures are read from the results they draw, and return a matplotlib `Figure`. `results.plot`, `replicates.plot` and `predictive.plot` give the figures of a run, of replicates and of a predictive: `plot.series()`, `plot.cumulative()` and `plot.observations()`; each accessor is on the page of the object that gives it: [Simulation](https://munqu.github.io/commol/latest/api/simulation/index.md) and [Uncertainty](https://munqu.github.io/commol/latest/api/uncertainty/index.md). The results of PRCC, Morris, Sobol and polynomial chaos, and the result of a profile likelihood, draw their figure with `plot()`. `plot_compared` draws several runs, replicates or predictives on the same panels. The figures need the optional extra `commol[plot]`. Guide: [Plotting](https://munqu.github.io/commol/latest/guide/simulations/#plotting).

## plot_compared

```python
plot_compared(results: Mapping[str, SimulationResults | SimulationReplicates | Predictive], series: list[str] | None = None, window: float | None = None, observed_data: list[ObservedDataPoint] | None = None, panels: Literal['series', 'observations'] = 'series', axes: Figure | None = None, output_file: str | None = None, config: PlotConfig | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', confidence_level: float = 0.95, **kwargs: str | int | float | bool | None) -> Figure
```

Draw several results of the same model on the same panels, one panel per series or one per observed series.

Each panel has one line per result, labelled with its key and in the same color in every panel (`C0` for the first key, `C1` for the second, …). Replicates show their mean and the band of the replicates, and a predictive the weighted median of its draws and their band, in the color of their key. The observations are drawn once.

Parameters:

| Name               | Type                                | Description                                                                                                                                                                                                                                | Default                                                                                                                                                                                                                                                                                                                     |
| ------------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results`          | \`Mapping\[str, SimulationResults   | SimulationReplicates                                                                                                                                                                                                                       | Predictive\]\`                                                                                                                                                                                                                                                                                                              |
| `series`           | \`list[str]                         | None\`                                                                                                                                                                                                                                     | What to draw, one panel each, as in ResultsPlot.series. If None, every output. Only with panels="series".                                                                                                                                                                                                                   |
| `window`           | \`float                             | None\`                                                                                                                                                                                                                                     | The length of a window on the axis. Each line is then the gain of the series over each complete window, as Simulation.windowed_totals reads it, at the end of the window; a band is the band of these gains. If None, each line is the line of the plot.series of its result with observed_data. Only with panels="series". |
| `observed_data`    | \`list[ObservedDataPoint]           | None\`                                                                                                                                                                                                                                     | Observations, each drawn once as a point on the panel of its label. Without window, they also set how each line is read, as in ResultsPlot.series.                                                                                                                                                                          |
| `panels`           | `Literal['series', 'observations']` | "series": one panel per entry of series. "observations": one panel per label of observed_data, with each result read only at the observations, through the observation operator of each one, as ResultsPlot.observations reads one result. | `'series'`                                                                                                                                                                                                                                                                                                                  |
| `axes`             | \`Figure                            | None\`                                                                                                                                                                                                                                     | A figure of commol to draw on: each series is drawn on its panel, the Axes whose label is the id of the series (or the label of the observed series). If None, a new figure.                                                                                                                                                |
| `output_file`      | \`str                               | None\`                                                                                                                                                                                                                                     | Path to save the figure to. If None, the figure is only returned.                                                                                                                                                                                                                                                           |
| `config`           | \`PlotConfig                        | None\`                                                                                                                                                                                                                                     | Layout and style. If None, the defaults.                                                                                                                                                                                                                                                                                    |
| `show_legend`      | `bool`                              | Whether each panel shows its legend.                                                                                                                                                                                                       | `True`                                                                                                                                                                                                                                                                                                                      |
| `position_label`   | \`Callable\[[float], str\]          | None\`                                                                                                                                                                                                                                     | Turns a position into a tick label, as in ResultsPlot.series.                                                                                                                                                                                                                                                               |
| `tick_every`       | \`float                             | None\`                                                                                                                                                                                                                                     | The distance between labelled positions, as in ResultsPlot.series.                                                                                                                                                                                                                                                          |
| `x_label`          | `str`                               | The label of the x axis.                                                                                                                                                                                                                   | `'Step'`                                                                                                                                                                                                                                                                                                                    |
| `confidence_level` | `float`                             | The level of the bands of replicates and predictives, strictly between 0 and 1.                                                                                                                                                            | `0.95`                                                                                                                                                                                                                                                                                                                      |
| `**kwargs`         | \`str                               | int                                                                                                                                                                                                                                        | float                                                                                                                                                                                                                                                                                                                       |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`  | If there are no results; a result does not have the outputs of the first (the message names its key and the outputs that differ); window is not positive; the confidence level is out of range; axes has no panel of a series; panels is neither "series" nor "observations"; or, with panels="observations", series or window is given, there are no observations, or a result has no value at the position of an observation or at the start of its window. |
| `KeyError`    | If an entry of series is neither an output nor a bin or accumulator id of the model.                                                                                                                                                                                                                                                                                                                                                                          |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                                                                                                                                                                                                                                                                                                                                                                                  |

Examples:

```pycon
>>> figure = plot_compared({"k1 = 0.3": results, "k1 = 0.5": faster})
>>> checked = plot_compared(
...     {"k1 = 0.3": results, "k1 = 0.5": faster},
...     observed_data=observed,
...     panels="observations",
... )
```

Source code in `commol/api/plotting/compared.py`

```python
def plot_compared(
    results: "Mapping[str, SimulationResults | SimulationReplicates | Predictive]",
    series: list[str] | None = None,
    window: float | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    panels: Literal["series", "observations"] = "series",
    axes: "Figure | None" = None,
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    confidence_level: float = 0.95,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw several results of the same model on the same panels, one panel per
    series or one per observed series.

    Each panel has one line per result, labelled with its key and in the
    same color in every panel (``C0`` for the first key, ``C1`` for the
    second, …). Replicates show their mean and the band of the replicates,
    and a predictive the weighted median of its draws and their band, in the
    color of their key. The observations are drawn once.

    Parameters
    ----------
    results : Mapping[str, SimulationResults | SimulationReplicates | Predictive]
        The results, by the label of their lines. Each has the outputs of
        the first; the model of the first names the series.
    series : list[str] | None
        What to draw, one panel each, as in ``ResultsPlot.series``. If None,
        every output. Only with ``panels="series"``.
    window : float | None
        The length of a window on the axis. Each line is then the gain of
        the series over each complete window, as `Simulation.windowed_totals`
        reads it, at the end of the window; a band is the band of these
        gains. If None, each line is the line of the ``plot.series`` of its
        result with `observed_data`. Only with ``panels="series"``.
    observed_data : list[ObservedDataPoint] | None
        Observations, each drawn once as a point on the panel of its label.
        Without `window`, they also set how each line is read, as in
        ``ResultsPlot.series``.
    panels : Literal["series", "observations"]
        ``"series"``: one panel per entry of `series`. ``"observations"``:
        one panel per label of `observed_data`, with each result read only
        at the observations, through the observation operator of each one,
        as ``ResultsPlot.observations`` reads one result.
    axes : Figure | None
        A figure of commol to draw on: each series is drawn on its panel,
        the `Axes` whose label is the id of the series (or the label of the
        observed series). If None, a new figure.
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    show_legend : bool
        Whether each panel shows its legend.
    position_label : Callable[[float], str] | None
        Turns a position into a tick label, as in ``ResultsPlot.series``.
    tick_every : float | None
        The distance between labelled positions, as in
        ``ResultsPlot.series``.
    x_label : str
        The label of the x axis.
    confidence_level : float
        The level of the bands of replicates and predictives, strictly
        between 0 and 1.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of each result, such as
        ``linewidth``.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If there are no results; a result does not have the outputs of the
        first (the message names its key and the outputs that differ);
        `window` is not positive; the confidence level is out of range;
        ``axes`` has no panel of a series; `panels` is neither ``"series"``
        nor ``"observations"``; or, with ``panels="observations"``, `series`
        or `window` is given, there are no observations, or a result has no
        value at the position of an observation or at the start of its
        window.
    KeyError
        If an entry of ``series`` is neither an output nor a bin or
        accumulator id of the model.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).

    Examples
    --------
    >>> figure = plot_compared({"k1 = 0.3": results, "k1 = 0.5": faster})
    >>> checked = plot_compared(
    ...     {"k1 = 0.3": results, "k1 = 0.5": faster},
    ...     observed_data=observed,
    ...     panels="observations",
    ... )
    """
    from commol.api.plotting.figures import _compared_series

    return _compared_series(
        results,
        series=series,
        window=window,
        observed_data=observed_data,
        panels=panels,
        axes=axes,
        output_file=output_file,
        config=config,
        show_legend=show_legend,
        position_label=position_label,
        tick_every=tick_every,
        x_label=x_label,
        confidence_level=confidence_level,
        **kwargs,
    )
```

## plot

```python
plot(output_file: str | None = None, config: PlotConfig | None = None, x_label: str = 'Step', axes: Figure | None = None) -> Figure
```

Draw the PRCC of each input: one panel per series.

A series of a `Trajectory` draws one line per input over its positions; any other quantity draws one bar per input.

Parameters:

| Name          | Type         | Description                                         | Default                                                                                                             |
| ------------- | ------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str        | None\`                                              | Path to save the figure to. If None, the figure is only returned.                                                   |
| `config`      | \`PlotConfig | None\`                                              | Layout and style. If None, the defaults.                                                                            |
| `x_label`     | `str`        | The label of the x axis of a series over positions. | `'Step'`                                                                                                            |
| `axes`        | \`Figure     | None\`                                              | A figure of commol to draw on: each panel is the Axes whose label is the name of its series. If None, a new figure. |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | If axes has no panel of a series.                            |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/context/inference/global_sensitivity.py`

```python
def plot(
    self,
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    x_label: str = "Step",
    axes: "Figure | None" = None,
) -> "Figure":
    """
    Draw the PRCC of each input: one panel per series.

    A series of a `Trajectory` draws one line per input over its
    positions; any other quantity draws one bar per input.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    x_label : str
        The label of the x axis of a series over positions.
    axes : Figure | None
        A figure of commol to draw on: each panel is the `Axes` whose
        label is the name of its series. If None, a new figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``axes`` has no panel of a series.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _prcc

    return _prcc(
        self, output_file=output_file, config=config, x_label=x_label, axes=axes
    )
```

## plot

```python
plot(output_file: str | None = None, config: PlotConfig | None = None, axes: Figure | None = None) -> Figure
```

Draw `σ` against `μ*` of each input: one panel per quantity, one labelled point per input.

An input far to the right matters; one high up acts through interactions or a non-linear effect.

Parameters:

| Name          | Type         | Description | Default                                                                                                                |
| ------------- | ------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str        | None\`      | Path to save the figure to. If None, the figure is only returned.                                                      |
| `config`      | \`PlotConfig | None\`      | Layout and style. If None, the defaults.                                                                               |
| `axes`        | \`Figure     | None\`      | A figure of commol to draw on: each panel is the Axes whose label is the label of its quantity. If None, a new figure. |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | If axes has no panel of a quantity.                          |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/context/inference/global_sensitivity.py`

```python
def plot(
    self,
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    axes: "Figure | None" = None,
) -> "Figure":
    """
    Draw ``σ`` against ``μ*`` of each input: one panel per quantity,
    one labelled point per input.

    An input far to the right matters; one high up acts through
    interactions or a non-linear effect.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    axes : Figure | None
        A figure of commol to draw on: each panel is the `Axes` whose
        label is the label of its quantity. If None, a new figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``axes`` has no panel of a quantity.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _morris

    return _morris(self, output_file=output_file, config=config, axes=axes)
```

## plot

```python
plot(output_file: str | None = None, config: PlotConfig | None = None, axes: Figure | None = None) -> Figure
```

Draw the first-order and total Sobol indices: one panel per quantity, one pair of bars per input, with their intervals.

Parameters:

| Name          | Type         | Description | Default                                                                                                                |
| ------------- | ------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str        | None\`      | Path to save the figure to. If None, the figure is only returned.                                                      |
| `config`      | \`PlotConfig | None\`      | Layout and style. If None, the defaults.                                                                               |
| `axes`        | \`Figure     | None\`      | A figure of commol to draw on: each panel is the Axes whose label is the label of its quantity. If None, a new figure. |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | If axes has no panel of a quantity.                          |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/context/inference/global_sensitivity.py`

```python
def plot(
    self,
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    axes: "Figure | None" = None,
) -> "Figure":
    """
    Draw the first-order and total Sobol indices: one panel per
    quantity, one pair of bars per input, with their intervals.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    axes : Figure | None
        A figure of commol to draw on: each panel is the `Axes` whose
        label is the label of its quantity. If None, a new figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``axes`` has no panel of a quantity.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _sobol

    return _sobol(self, output_file=output_file, config=config, axes=axes)
```

## plot

```python
plot(output_file: str | None = None, config: PlotConfig | None = None, axes: Figure | None = None) -> Figure
```

Draw the first-order and total Sobol indices of the expansion: one panel per quantity, one pair of bars per input.

Parameters:

| Name          | Type         | Description | Default                                                                                                                |
| ------------- | ------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str        | None\`      | Path to save the figure to. If None, the figure is only returned.                                                      |
| `config`      | \`PlotConfig | None\`      | Layout and style. If None, the defaults.                                                                               |
| `axes`        | \`Figure     | None\`      | A figure of commol to draw on: each panel is the Axes whose label is the label of its quantity. If None, a new figure. |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | If axes has no panel of a quantity.                          |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/context/inference/global_sensitivity.py`

```python
def plot(
    self,
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    axes: "Figure | None" = None,
) -> "Figure":
    """
    Draw the first-order and total Sobol indices of the expansion: one
    panel per quantity, one pair of bars per input.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    axes : Figure | None
        A figure of commol to draw on: each panel is the `Axes` whose
        label is the label of its quantity. If None, a new figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``axes`` has no panel of a quantity.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _sobol

    return _sobol(self, output_file=output_file, config=config, axes=axes)
```

## plot

```python
plot(output_file: str | None = None, config: PlotConfig | None = None, axes: Figure | None = None) -> Figure
```

Draw each profile: one panel per parameter or prediction, with the statistic of each point, the threshold, and the ends of the interval.

A profile that stays under the threshold up to a bound is open on that side; a flat one stays near 0 over its whole range.

Parameters:

| Name          | Type         | Description | Default                                                                                                              |
| ------------- | ------------ | ----------- | -------------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str        | None\`      | Path to save the figure to. If None, the figure is only returned.                                                    |
| `config`      | \`PlotConfig | None\`      | Layout and style. If None, the defaults.                                                                             |
| `axes`        | \`Figure     | None\`      | A figure of commol to draw on: each panel is the Axes whose label is the name of its profile. If None, a new figure. |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | If axes has no panel of a profile.                           |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/context/inference/profiles.py`

```python
def plot(
    self,
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    axes: "Figure | None" = None,
) -> "Figure":
    """
    Draw each profile: one panel per parameter or prediction, with the
    statistic of each point, the threshold, and the ends of the interval.

    A profile that stays under the threshold up to a bound is open on that
    side; a flat one stays near 0 over its whole range.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    axes : Figure | None
        A figure of commol to draw on: each panel is the `Axes` whose
        label is the name of its profile. If None, a new figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``axes`` has no panel of a profile.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _profiles

    return _profiles(self, output_file=output_file, config=config, axes=axes)
```

## PlotConfig

Bases: `BaseModel`

Configuration for plot layout and styling.

Attributes:

| Name      | Type                  | Description                                                                                                                |
| --------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `figsize` | `tuple[float, float]` | Figure size in inches (width, height), of a figure of running totals; the other figures take their size from their panels. |
| `dpi`     | `int`                 | Dots per inch for figure resolution.                                                                                       |
| `layout`  | \`tuple[int, int]     | None\`                                                                                                                     |
| `style`   | \`str                 | list[str]                                                                                                                  |

Notes

The style applies to the figure of one call only: after the call, `matplotlib.rcParams` is as it was before it, so the figures that you draw with matplotlib keep your style. Given `axes=`, a figure draws on the panels of an earlier figure, and `figsize`, `dpi` and `layout` are those of that figure.

Examples:

```pycon
>>> # Direct configuration
>>> config = PlotConfig(style=["seaborn-v0_8-whitegrid", "seaborn-v0_8-talk"])
```

```pycon
>>> # Using presets
>>> config = PlotConfig.presentation()
>>> config = PlotConfig.publication()
```

Source code in `commol/context/visualization.py`

```python
class PlotConfig(BaseModel):
    """
    Configuration for plot layout and styling.

    Attributes
    ----------
    figsize : tuple[float, float]
        Figure size in inches (width, height), of a figure of running totals;
        the other figures take their size from their panels.
    dpi : int
        Dots per inch for figure resolution.
    layout : tuple[int, int] | None
        Panel layout as (rows, cols). If None, computed from the number of
        panels.
    style : str | list[str] | None
        A matplotlib style, or a list of styles applied in order: a name of
        ``matplotlib.style.available`` (such as ``"ggplot"`` or
        ``"seaborn-v0_8-whitegrid"``) or the path of a style file.
        See: https://matplotlib.org/stable/users/explain/customizing.html

    Notes
    -----
    The style applies to the figure of one call only: after the call,
    ``matplotlib.rcParams`` is as it was before it, so the figures that you
    draw with matplotlib keep your style. Given ``axes=``, a figure draws on
    the panels of an earlier figure, and `figsize`, `dpi` and `layout` are
    those of that figure.

    Examples
    --------
    >>> # Direct configuration
    >>> config = PlotConfig(style=["seaborn-v0_8-whitegrid", "seaborn-v0_8-talk"])

    >>> # Using presets
    >>> config = PlotConfig.presentation()
    >>> config = PlotConfig.publication()
    """

    model_config = ConfigDict(extra="forbid")

    figsize: tuple[float, float] = Field(
        default=(12, 8),
        description="Figure size in inches (width, height)",
    )
    dpi: int = Field(
        default=100,
        description="Dots per inch for figure resolution",
    )
    layout: tuple[int, int] | None = Field(
        default=None,
        description="Panel layout as (rows, cols). If None, computed.",
    )
    style: str | list[str] | None = Field(
        default=None,
        description="A matplotlib style, or styles applied in order",
    )

    @classmethod
    def presentation(cls) -> Self:
        """
        Create a configuration optimized for presentations.

        Large figure size, high DPI, a white grid, the large text of the
        "talk" style and a palette for colour-blind readers.

        Returns
        -------
        PlotConfig
            Configuration with presentation-friendly settings.
        """
        return cls(
            figsize=(16, 10),
            dpi=150,
            style=[
                "seaborn-v0_8-whitegrid",
                "seaborn-v0_8-talk",
                "seaborn-v0_8-colorblind",
            ],
        )

    @classmethod
    def publication(cls) -> Self:
        """
        Create a configuration optimized for academic publications.

        Smaller figure size suitable for journal columns, high DPI for print
        quality, ticks and the small text of the "paper" style.

        Returns
        -------
        PlotConfig
            Configuration with publication-friendly settings.
        """
        return cls(
            figsize=(8, 6),
            dpi=300,
            style=["seaborn-v0_8-ticks", "seaborn-v0_8-paper"],
        )

    @classmethod
    def notebook(cls) -> Self:
        """
        Create a configuration optimized for Jupyter notebooks.

        Balanced defaults for interactive exploration.

        Returns
        -------
        PlotConfig
            Configuration with notebook-friendly settings.
        """
        return cls(
            figsize=(12, 8),
            dpi=100,
            style=["seaborn-v0_8-whitegrid", "seaborn-v0_8-notebook"],
        )
```

### Methods:

#### presentation

```python
presentation() -> Self
```

Create a configuration optimized for presentations.

Large figure size, high DPI, a white grid, the large text of the "talk" style and a palette for colour-blind readers.

Returns:

| Type         | Description                                        |
| ------------ | -------------------------------------------------- |
| `PlotConfig` | Configuration with presentation-friendly settings. |

Source code in `commol/context/visualization.py`

```python
@classmethod
def presentation(cls) -> Self:
    """
    Create a configuration optimized for presentations.

    Large figure size, high DPI, a white grid, the large text of the
    "talk" style and a palette for colour-blind readers.

    Returns
    -------
    PlotConfig
        Configuration with presentation-friendly settings.
    """
    return cls(
        figsize=(16, 10),
        dpi=150,
        style=[
            "seaborn-v0_8-whitegrid",
            "seaborn-v0_8-talk",
            "seaborn-v0_8-colorblind",
        ],
    )
```

#### publication

```python
publication() -> Self
```

Create a configuration optimized for academic publications.

Smaller figure size suitable for journal columns, high DPI for print quality, ticks and the small text of the "paper" style.

Returns:

| Type         | Description                                       |
| ------------ | ------------------------------------------------- |
| `PlotConfig` | Configuration with publication-friendly settings. |

Source code in `commol/context/visualization.py`

```python
@classmethod
def publication(cls) -> Self:
    """
    Create a configuration optimized for academic publications.

    Smaller figure size suitable for journal columns, high DPI for print
    quality, ticks and the small text of the "paper" style.

    Returns
    -------
    PlotConfig
        Configuration with publication-friendly settings.
    """
    return cls(
        figsize=(8, 6),
        dpi=300,
        style=["seaborn-v0_8-ticks", "seaborn-v0_8-paper"],
    )
```

#### notebook

```python
notebook() -> Self
```

Create a configuration optimized for Jupyter notebooks.

Balanced defaults for interactive exploration.

Returns:

| Type         | Description                                    |
| ------------ | ---------------------------------------------- |
| `PlotConfig` | Configuration with notebook-friendly settings. |

Source code in `commol/context/visualization.py`

```python
@classmethod
def notebook(cls) -> Self:
    """
    Create a configuration optimized for Jupyter notebooks.

    Balanced defaults for interactive exploration.

    Returns
    -------
    PlotConfig
        Configuration with notebook-friendly settings.
    """
    return cls(
        figsize=(12, 8),
        dpi=100,
        style=["seaborn-v0_8-whitegrid", "seaborn-v0_8-notebook"],
    )
```
