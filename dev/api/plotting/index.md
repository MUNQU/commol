# Plotting API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The figures are read from the results they draw, and return a matplotlib `Figure`: `results.plot`, `replicates.plot` and `predictive.plot` give the figures of a run, of replicates and of a predictive, and the result of a sensitivity method or of a profile likelihood draws its figure with `plot()`. `plot_compared` draws several runs, replicates or predictives on the same panels. They need the optional extra `commol[plot]`. Guide: [Plotting](https://munqu.github.io/commol/latest/guide/simulations/#plotting).

## ResultsPlot

The figures of one run, read from `results.plot`.

Each figure has one panel per series, and returns the matplotlib `Figure`. Given `axes=`, the `Figure` of an earlier figure of commol, it draws each series on the panel of that series instead.

Source code in `commol/api/plotting/accessors.py`

```python
class ResultsPlot:
    """
    The figures of one run, read from ``results.plot``.

    Each figure has one panel per series, and returns the matplotlib
    `Figure`. Given ``axes=``, the `Figure` of an earlier figure of commol, it
    draws each series on the panel of that series instead.
    """

    def __init__(self, results: "SimulationResults"):
        """
        The figures of `results`.

        Parameters
        ----------
        results : SimulationResults
            The run.
        """
        self._results = results

    def series(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        prediction_intervals: "PredictionIntervals | None" = None,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw each series along the axis, one panel per series.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, each drawn as a point on the panel of its label.
            When the observations of a series share one sum of outputs and
            one factor and have no window, the line covers every position of
            the results. Otherwise the model of each observation is read
            through its own operator (outputs, window, factor) at its own
            position, as the loss reads it, and the line joins these values.
        calibration_result : CalibrationResult | None
            Its ``best_parameters`` give the values of the factors of the
            observations, which multiply the line; observed values stay in
            their own units.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        series : list[str] | None
            What to draw, one panel each: an output name (such as ``A_g1``),
            or a bin or accumulator id, which draws the sum of its outputs.
            An exact output name wins over an id. If None, every output.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position on the axis into a tick label. If None, a model
            with an origin (``ModelBuilder(origin=...)``) labels each tick
            with its date: ``2020-01-02``, with the time when it is not
            midnight. Otherwise the ticks show the position.
        tick_every : float | None
            Labels the plotted positions `tick_every` apart, counted from the
            first plotted position. If None, every plotted position, or the
            ticks matplotlib places on a dated axis.
        x_label : str
            The label of the x axis.
        prediction_intervals : PredictionIntervals | None
            Prediction intervals of a conformal method
            (``ConformalResult.intervals``). The panel of each of their
            series, by label, shows the band between their lower and upper
            ends and their point prediction, dashed; an unbounded interval
            is left out of the band.
        axes : Figure | None
            A figure of commol to draw on: each series is drawn on its panel,
            the `Axes` whose label is the id of the series. If None, a new
            figure.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the run, such as
            ``linewidth``, ``alpha`` or ``linestyle``.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id, or a bin or accumulator id of results with no
            model; the results have no value at the position of an
            observation read on its own or at the start of its window; or
            ``axes`` has no panel of a series.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _run_series

        return _run_series(
            self._results,
            output_file=output_file,
            observed_data=observed_data,
            calibration_result=calibration_result,
            config=config,
            series=series,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            prediction_intervals=prediction_intervals,
            axes=axes,
            **kwargs,
        )

    def cumulative(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw the running total of each series, one panel per series.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, drawn as their running total on the panel of their
            label. An observation with a factor is divided by the value of
            its factor in ``calibration_result``.
        calibration_result : CalibrationResult | None
            Its ``best_parameters`` give the values of the factors.
        config : PlotConfig | None
            Layout, size and style. If None, the defaults.
        series : list[str] | None
            What to draw, one panel each, as in `series`. If None, every
            output.
        axes : Figure | None
            A figure of commol to draw on, as in `series`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the run.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id, or ``axes`` has no panel of a series.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _run_cumulative

        return _run_cumulative(
            self._results,
            output_file=output_file,
            observed_data=observed_data,
            calibration_result=calibration_result,
            config=config,
            series=series,
            axes=axes,
            **kwargs,
        )
```

### Methods:

#### __init__

```python
__init__(results: SimulationResults)
```

The figures of `results`.

Parameters:

| Name      | Type                | Description | Default    |
| --------- | ------------------- | ----------- | ---------- |
| `results` | `SimulationResults` | The run.    | *required* |

Source code in `commol/api/plotting/accessors.py`

```python
def __init__(self, results: "SimulationResults"):
    """
    The figures of `results`.

    Parameters
    ----------
    results : SimulationResults
        The run.
    """
    self._results = results
```

#### series

```python
series(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, calibration_result: CalibrationResult | None = None, config: PlotConfig | None = None, series: list[str] | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', prediction_intervals: PredictionIntervals | None = None, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw each series along the axis, one panel per series.

Parameters:

| Name                   | Type                       | Description                          | Default                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------- | -------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `output_file`          | \`str                      | None\`                               | Path to save the figure to. If None, the figure is only returned.                                                                                                                                                                                                                                                                                                                          |
| `observed_data`        | \`list[ObservedDataPoint]  | None\`                               | Observations, each drawn as a point on the panel of its label. When the observations of a series share one sum of outputs and one factor and have no window, the line covers every position of the results. Otherwise the model of each observation is read through its own operator (outputs, window, factor) at its own position, as the loss reads it, and the line joins these values. |
| `calibration_result`   | \`CalibrationResult        | None\`                               | Its best_parameters give the values of the factors of the observations, which multiply the line; observed values stay in their own units.                                                                                                                                                                                                                                                  |
| `config`               | \`PlotConfig               | None\`                               | Layout and style. If None, the defaults.                                                                                                                                                                                                                                                                                                                                                   |
| `series`               | \`list[str]                | None\`                               | What to draw, one panel each: an output name (such as A_g1), or a bin or accumulator id, which draws the sum of its outputs. An exact output name wins over an id. If None, every output.                                                                                                                                                                                                  |
| `show_legend`          | `bool`                     | Whether each panel shows its legend. | `True`                                                                                                                                                                                                                                                                                                                                                                                     |
| `position_label`       | \`Callable\[[float], str\] | None\`                               | Turns a position on the axis into a tick label. If None, a model with an origin (ModelBuilder(origin=...)) labels each tick with its date: 2020-01-02, with the time when it is not midnight. Otherwise the ticks show the position.                                                                                                                                                       |
| `tick_every`           | \`float                    | None\`                               | Labels the plotted positions tick_every apart, counted from the first plotted position. If None, every plotted position, or the ticks matplotlib places on a dated axis.                                                                                                                                                                                                                   |
| `x_label`              | `str`                      | The label of the x axis.             | `'Step'`                                                                                                                                                                                                                                                                                                                                                                                   |
| `prediction_intervals` | \`PredictionIntervals      | None\`                               | Prediction intervals of a conformal method (ConformalResult.intervals). The panel of each of their series, by label, shows the band between their lower and upper ends and their point prediction, dashed; an unbounded interval is left out of the band.                                                                                                                                  |
| `axes`                 | \`Figure                   | None\`                               | A figure of commol to draw on: each series is drawn on its panel, the Axes whose label is the id of the series. If None, a new figure.                                                                                                                                                                                                                                                     |
| `**kwargs`             | \`str                      | int                                  | float                                                                                                                                                                                                                                                                                                                                                                                      |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                                                                                                                                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`  | If an entry of series is neither an output nor a bin or accumulator id, or a bin or accumulator id of results with no model; the results have no value at the position of an observation read on its own or at the start of its window; or axes has no panel of a series. |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                                                                                                                                                                                              |

Source code in `commol/api/plotting/accessors.py`

```python
def series(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    prediction_intervals: "PredictionIntervals | None" = None,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw each series along the axis, one panel per series.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, each drawn as a point on the panel of its label.
        When the observations of a series share one sum of outputs and
        one factor and have no window, the line covers every position of
        the results. Otherwise the model of each observation is read
        through its own operator (outputs, window, factor) at its own
        position, as the loss reads it, and the line joins these values.
    calibration_result : CalibrationResult | None
        Its ``best_parameters`` give the values of the factors of the
        observations, which multiply the line; observed values stay in
        their own units.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    series : list[str] | None
        What to draw, one panel each: an output name (such as ``A_g1``),
        or a bin or accumulator id, which draws the sum of its outputs.
        An exact output name wins over an id. If None, every output.
    show_legend : bool
        Whether each panel shows its legend.
    position_label : Callable[[float], str] | None
        Turns a position on the axis into a tick label. If None, a model
        with an origin (``ModelBuilder(origin=...)``) labels each tick
        with its date: ``2020-01-02``, with the time when it is not
        midnight. Otherwise the ticks show the position.
    tick_every : float | None
        Labels the plotted positions `tick_every` apart, counted from the
        first plotted position. If None, every plotted position, or the
        ticks matplotlib places on a dated axis.
    x_label : str
        The label of the x axis.
    prediction_intervals : PredictionIntervals | None
        Prediction intervals of a conformal method
        (``ConformalResult.intervals``). The panel of each of their
        series, by label, shows the band between their lower and upper
        ends and their point prediction, dashed; an unbounded interval
        is left out of the band.
    axes : Figure | None
        A figure of commol to draw on: each series is drawn on its panel,
        the `Axes` whose label is the id of the series. If None, a new
        figure.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the run, such as
        ``linewidth``, ``alpha`` or ``linestyle``.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If an entry of ``series`` is neither an output nor a bin or
        accumulator id, or a bin or accumulator id of results with no
        model; the results have no value at the position of an
        observation read on its own or at the start of its window; or
        ``axes`` has no panel of a series.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _run_series

    return _run_series(
        self._results,
        output_file=output_file,
        observed_data=observed_data,
        calibration_result=calibration_result,
        config=config,
        series=series,
        show_legend=show_legend,
        position_label=position_label,
        tick_every=tick_every,
        x_label=x_label,
        prediction_intervals=prediction_intervals,
        axes=axes,
        **kwargs,
    )
```

#### cumulative

```python
cumulative(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, calibration_result: CalibrationResult | None = None, config: PlotConfig | None = None, series: list[str] | None = None, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw the running total of each series, one panel per series.

Parameters:

| Name                 | Type                      | Description | Default                                                                                                                                                           |
| -------------------- | ------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_file`        | \`str                     | None\`      | Path to save the figure to. If None, the figure is only returned.                                                                                                 |
| `observed_data`      | \`list[ObservedDataPoint] | None\`      | Observations, drawn as their running total on the panel of their label. An observation with a factor is divided by the value of its factor in calibration_result. |
| `calibration_result` | \`CalibrationResult       | None\`      | Its best_parameters give the values of the factors.                                                                                                               |
| `config`             | \`PlotConfig              | None\`      | Layout, size and style. If None, the defaults.                                                                                                                    |
| `series`             | \`list[str]               | None\`      | What to draw, one panel each, as in series. If None, every output.                                                                                                |
| `axes`               | \`Figure                  | None\`      | A figure of commol to draw on, as in series.                                                                                                                      |
| `**kwargs`           | \`str                     | int         | float                                                                                                                                                             |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| `ValueError`  | If an entry of series is neither an output nor a bin or accumulator id, or axes has no panel of a series. |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                              |

Source code in `commol/api/plotting/accessors.py`

```python
def cumulative(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw the running total of each series, one panel per series.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, drawn as their running total on the panel of their
        label. An observation with a factor is divided by the value of
        its factor in ``calibration_result``.
    calibration_result : CalibrationResult | None
        Its ``best_parameters`` give the values of the factors.
    config : PlotConfig | None
        Layout, size and style. If None, the defaults.
    series : list[str] | None
        What to draw, one panel each, as in `series`. If None, every
        output.
    axes : Figure | None
        A figure of commol to draw on, as in `series`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the run.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If an entry of ``series`` is neither an output nor a bin or
        accumulator id, or ``axes`` has no panel of a series.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _run_cumulative

    return _run_cumulative(
        self._results,
        output_file=output_file,
        observed_data=observed_data,
        calibration_result=calibration_result,
        config=config,
        series=series,
        axes=axes,
        **kwargs,
    )
```

## ReplicatesPlot

The figures of the replicates of a stochastic engine, read from `replicates.plot`.

The figures are those of `ResultsPlot` for the mean of the replicates; `series` adds the band of the replicates.

Source code in `commol/api/plotting/accessors.py`

```python
class ReplicatesPlot:
    """
    The figures of the replicates of a stochastic engine, read from
    ``replicates.plot``.

    The figures are those of `ResultsPlot` for the mean of the replicates;
    `series` adds the band of the replicates.
    """

    def __init__(self, replicates: "SimulationReplicates"):
        """
        The figures of `replicates`.

        Parameters
        ----------
        replicates : SimulationReplicates
            The replicates.
        """
        self._replicates = replicates

    def series(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        prediction_intervals: "PredictionIntervals | None" = None,
        confidence_level: float = 0.95,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw the mean of each series and the band of the replicates, one
        panel per series.

        Each replicate is reduced to the series first, through the operator
        of its observations, then the band takes the interval of
        `confidence_level` across them (`member_statistics`).

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, as in `ResultsPlot.series`.
        calibration_result : CalibrationResult | None
            The values of the factors, as in `ResultsPlot.series`.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        series : list[str] | None
            What to draw, as in `ResultsPlot.series`.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position into a tick label, as in `ResultsPlot.series`.
        tick_every : float | None
            The distance between labelled positions, as in
            `ResultsPlot.series`.
        x_label : str
            The label of the x axis.
        prediction_intervals : PredictionIntervals | None
            Prediction intervals of a conformal method, as in
            `ResultsPlot.series`.
        confidence_level : float
            The level of the band, strictly between 0 and 1.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.series`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the mean.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            As `ResultsPlot.series`; or if there are no replicates or the
            confidence level is out of range.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _replicates_series

        return _replicates_series(
            self._replicates,
            output_file=output_file,
            observed_data=observed_data,
            calibration_result=calibration_result,
            config=config,
            series=series,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            prediction_intervals=prediction_intervals,
            confidence_level=confidence_level,
            axes=axes,
            **kwargs,
        )

    def cumulative(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw the running total of the mean of the replicates, as
        `ResultsPlot.cumulative` draws one run.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, as in `ResultsPlot.cumulative`.
        calibration_result : CalibrationResult | None
            The values of the factors, as in `ResultsPlot.cumulative`.
        config : PlotConfig | None
            Layout, size and style. If None, the defaults.
        series : list[str] | None
            What to draw, as in `ResultsPlot.series`.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.series`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the mean.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            As `ResultsPlot.cumulative`; or if there are no replicates.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _replicates_cumulative

        return _replicates_cumulative(
            self._replicates,
            output_file=output_file,
            observed_data=observed_data,
            calibration_result=calibration_result,
            config=config,
            series=series,
            axes=axes,
            **kwargs,
        )
```

### Methods:

#### __init__

```python
__init__(replicates: SimulationReplicates)
```

The figures of `replicates`.

Parameters:

| Name         | Type                   | Description     | Default    |
| ------------ | ---------------------- | --------------- | ---------- |
| `replicates` | `SimulationReplicates` | The replicates. | *required* |

Source code in `commol/api/plotting/accessors.py`

```python
def __init__(self, replicates: "SimulationReplicates"):
    """
    The figures of `replicates`.

    Parameters
    ----------
    replicates : SimulationReplicates
        The replicates.
    """
    self._replicates = replicates
```

#### series

```python
series(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, calibration_result: CalibrationResult | None = None, config: PlotConfig | None = None, series: list[str] | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', prediction_intervals: PredictionIntervals | None = None, confidence_level: float = 0.95, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw the mean of each series and the band of the replicates, one panel per series.

Each replicate is reduced to the series first, through the operator of its observations, then the band takes the interval of `confidence_level` across them (`member_statistics`).

Parameters:

| Name                   | Type                       | Description                                      | Default                                                               |
| ---------------------- | -------------------------- | ------------------------------------------------ | --------------------------------------------------------------------- |
| `output_file`          | \`str                      | None\`                                           | Path to save the figure to. If None, the figure is only returned.     |
| `observed_data`        | \`list[ObservedDataPoint]  | None\`                                           | Observations, as in ResultsPlot.series.                               |
| `calibration_result`   | \`CalibrationResult        | None\`                                           | The values of the factors, as in ResultsPlot.series.                  |
| `config`               | \`PlotConfig               | None\`                                           | Layout and style. If None, the defaults.                              |
| `series`               | \`list[str]                | None\`                                           | What to draw, as in ResultsPlot.series.                               |
| `show_legend`          | `bool`                     | Whether each panel shows its legend.             | `True`                                                                |
| `position_label`       | \`Callable\[[float], str\] | None\`                                           | Turns a position into a tick label, as in ResultsPlot.series.         |
| `tick_every`           | \`float                    | None\`                                           | The distance between labelled positions, as in ResultsPlot.series.    |
| `x_label`              | `str`                      | The label of the x axis.                         | `'Step'`                                                              |
| `prediction_intervals` | \`PredictionIntervals      | None\`                                           | Prediction intervals of a conformal method, as in ResultsPlot.series. |
| `confidence_level`     | `float`                    | The level of the band, strictly between 0 and 1. | `0.95`                                                                |
| `axes`                 | \`Figure                   | None\`                                           | A figure of commol to draw on, as in ResultsPlot.series.              |
| `**kwargs`             | \`str                      | int                                              | float                                                                 |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------- |
| `ValueError`  | As ResultsPlot.series; or if there are no replicates or the confidence level is out of range. |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                  |

Source code in `commol/api/plotting/accessors.py`

```python
def series(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    prediction_intervals: "PredictionIntervals | None" = None,
    confidence_level: float = 0.95,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw the mean of each series and the band of the replicates, one
    panel per series.

    Each replicate is reduced to the series first, through the operator
    of its observations, then the band takes the interval of
    `confidence_level` across them (`member_statistics`).

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, as in `ResultsPlot.series`.
    calibration_result : CalibrationResult | None
        The values of the factors, as in `ResultsPlot.series`.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    series : list[str] | None
        What to draw, as in `ResultsPlot.series`.
    show_legend : bool
        Whether each panel shows its legend.
    position_label : Callable[[float], str] | None
        Turns a position into a tick label, as in `ResultsPlot.series`.
    tick_every : float | None
        The distance between labelled positions, as in
        `ResultsPlot.series`.
    x_label : str
        The label of the x axis.
    prediction_intervals : PredictionIntervals | None
        Prediction intervals of a conformal method, as in
        `ResultsPlot.series`.
    confidence_level : float
        The level of the band, strictly between 0 and 1.
    axes : Figure | None
        A figure of commol to draw on, as in `ResultsPlot.series`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the mean.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        As `ResultsPlot.series`; or if there are no replicates or the
        confidence level is out of range.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _replicates_series

    return _replicates_series(
        self._replicates,
        output_file=output_file,
        observed_data=observed_data,
        calibration_result=calibration_result,
        config=config,
        series=series,
        show_legend=show_legend,
        position_label=position_label,
        tick_every=tick_every,
        x_label=x_label,
        prediction_intervals=prediction_intervals,
        confidence_level=confidence_level,
        axes=axes,
        **kwargs,
    )
```

#### cumulative

```python
cumulative(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, calibration_result: CalibrationResult | None = None, config: PlotConfig | None = None, series: list[str] | None = None, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw the running total of the mean of the replicates, as `ResultsPlot.cumulative` draws one run.

Parameters:

| Name                 | Type                      | Description | Default                                                           |
| -------------------- | ------------------------- | ----------- | ----------------------------------------------------------------- |
| `output_file`        | \`str                     | None\`      | Path to save the figure to. If None, the figure is only returned. |
| `observed_data`      | \`list[ObservedDataPoint] | None\`      | Observations, as in ResultsPlot.cumulative.                       |
| `calibration_result` | \`CalibrationResult       | None\`      | The values of the factors, as in ResultsPlot.cumulative.          |
| `config`             | \`PlotConfig              | None\`      | Layout, size and style. If None, the defaults.                    |
| `series`             | \`list[str]               | None\`      | What to draw, as in ResultsPlot.series.                           |
| `axes`               | \`Figure                  | None\`      | A figure of commol to draw on, as in ResultsPlot.series.          |
| `**kwargs`           | \`str                     | int         | float                                                             |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | As ResultsPlot.cumulative; or if there are no replicates.    |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/api/plotting/accessors.py`

```python
def cumulative(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw the running total of the mean of the replicates, as
    `ResultsPlot.cumulative` draws one run.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, as in `ResultsPlot.cumulative`.
    calibration_result : CalibrationResult | None
        The values of the factors, as in `ResultsPlot.cumulative`.
    config : PlotConfig | None
        Layout, size and style. If None, the defaults.
    series : list[str] | None
        What to draw, as in `ResultsPlot.series`.
    axes : Figure | None
        A figure of commol to draw on, as in `ResultsPlot.series`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the mean.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        As `ResultsPlot.cumulative`; or if there are no replicates.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _replicates_cumulative

    return _replicates_cumulative(
        self._replicates,
        output_file=output_file,
        observed_data=observed_data,
        calibration_result=calibration_result,
        config=config,
        series=series,
        axes=axes,
        **kwargs,
    )
```

## PredictivePlot

The figures of a predictive, read from `predictive.plot`.

Each panel shows the weighted median of the draws and their band, with no other run; the model of the source names the series.

Source code in `commol/api/plotting/accessors.py`

```python
class PredictivePlot:
    """
    The figures of a predictive, read from ``predictive.plot``.

    Each panel shows the weighted median of the draws and their band, with no
    other run; the model of the source names the series.
    """

    def __init__(self, predictive: "Predictive"):
        """
        The figures of `predictive`.

        Parameters
        ----------
        predictive : Predictive
            The predictive.
        """
        self._predictive = predictive

    def series(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        prediction_intervals: "PredictionIntervals | None" = None,
        confidence_level: float = 0.95,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw the weighted median of the draws of each series and their band,
        one panel per series.

        Each draw is read through the observation operator of the problem
        (the sums, windows and factors of the observations of the series,
        each at its own position when they do not share one operator with no
        window), then the draws are reduced with their weights.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, each drawn as a point on the panel of its label;
            they set the operator each draw is read through.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        series : list[str] | None
            What to draw, as in `ResultsPlot.series`.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position into a tick label, as in `ResultsPlot.series`.
        tick_every : float | None
            The distance between labelled positions, as in
            `ResultsPlot.series`.
        x_label : str
            The label of the x axis.
        prediction_intervals : PredictionIntervals | None
            Prediction intervals of a conformal method, as in
            `ResultsPlot.series`.
        confidence_level : float
            The level of the band, strictly between 0 and 1.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.series`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the median.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id, an observation names an unknown output or factor,
            the confidence level is out of range, or ``axes`` has no panel of
            a series.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _predictive_series

        return _predictive_series(
            self._predictive,
            output_file=output_file,
            observed_data=observed_data,
            config=config,
            series=series,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            prediction_intervals=prediction_intervals,
            confidence_level=confidence_level,
            axes=axes,
            **kwargs,
        )

    def cumulative(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        confidence_level: float = 0.95,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw the weighted median and the band of the running total of each
        series across the draws, one panel per series.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, drawn as their running total on the panel of their
            label. An observation with a factor is divided by the weighted
            median of that factor in the sample.
        config : PlotConfig | None
            Layout, size and style. If None, the defaults.
        series : list[str] | None
            What to draw, as in `ResultsPlot.series`.
        confidence_level : float
            The level of the band, strictly between 0 and 1.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.series`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the median.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id, the confidence level is out of range, or ``axes``
            has no panel of a series.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _predictive_cumulative

        return _predictive_cumulative(
            self._predictive,
            output_file=output_file,
            observed_data=observed_data,
            config=config,
            series=series,
            confidence_level=confidence_level,
            axes=axes,
            **kwargs,
        )
```

### Methods:

#### __init__

```python
__init__(predictive: Predictive)
```

The figures of `predictive`.

Parameters:

| Name         | Type         | Description     | Default    |
| ------------ | ------------ | --------------- | ---------- |
| `predictive` | `Predictive` | The predictive. | *required* |

Source code in `commol/api/plotting/accessors.py`

```python
def __init__(self, predictive: "Predictive"):
    """
    The figures of `predictive`.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    """
    self._predictive = predictive
```

#### series

```python
series(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, config: PlotConfig | None = None, series: list[str] | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', prediction_intervals: PredictionIntervals | None = None, confidence_level: float = 0.95, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw the weighted median of the draws of each series and their band, one panel per series.

Each draw is read through the observation operator of the problem (the sums, windows and factors of the observations of the series, each at its own position when they do not share one operator with no window), then the draws are reduced with their weights.

Parameters:

| Name                   | Type                       | Description                                      | Default                                                                                                         |
| ---------------------- | -------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `output_file`          | \`str                      | None\`                                           | Path to save the figure to. If None, the figure is only returned.                                               |
| `observed_data`        | \`list[ObservedDataPoint]  | None\`                                           | Observations, each drawn as a point on the panel of its label; they set the operator each draw is read through. |
| `config`               | \`PlotConfig               | None\`                                           | Layout and style. If None, the defaults.                                                                        |
| `series`               | \`list[str]                | None\`                                           | What to draw, as in ResultsPlot.series.                                                                         |
| `show_legend`          | `bool`                     | Whether each panel shows its legend.             | `True`                                                                                                          |
| `position_label`       | \`Callable\[[float], str\] | None\`                                           | Turns a position into a tick label, as in ResultsPlot.series.                                                   |
| `tick_every`           | \`float                    | None\`                                           | The distance between labelled positions, as in ResultsPlot.series.                                              |
| `x_label`              | `str`                      | The label of the x axis.                         | `'Step'`                                                                                                        |
| `prediction_intervals` | \`PredictionIntervals      | None\`                                           | Prediction intervals of a conformal method, as in ResultsPlot.series.                                           |
| `confidence_level`     | `float`                    | The level of the band, strictly between 0 and 1. | `0.95`                                                                                                          |
| `axes`                 | \`Figure                   | None\`                                           | A figure of commol to draw on, as in ResultsPlot.series.                                                        |
| `**kwargs`             | \`str                      | int                                              | float                                                                                                           |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                                                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`  | If an entry of series is neither an output nor a bin or accumulator id, an observation names an unknown output or factor, the confidence level is out of range, or axes has no panel of a series. |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                                                                                                                      |

Source code in `commol/api/plotting/accessors.py`

```python
def series(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    prediction_intervals: "PredictionIntervals | None" = None,
    confidence_level: float = 0.95,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw the weighted median of the draws of each series and their band,
    one panel per series.

    Each draw is read through the observation operator of the problem
    (the sums, windows and factors of the observations of the series,
    each at its own position when they do not share one operator with no
    window), then the draws are reduced with their weights.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, each drawn as a point on the panel of its label;
        they set the operator each draw is read through.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    series : list[str] | None
        What to draw, as in `ResultsPlot.series`.
    show_legend : bool
        Whether each panel shows its legend.
    position_label : Callable[[float], str] | None
        Turns a position into a tick label, as in `ResultsPlot.series`.
    tick_every : float | None
        The distance between labelled positions, as in
        `ResultsPlot.series`.
    x_label : str
        The label of the x axis.
    prediction_intervals : PredictionIntervals | None
        Prediction intervals of a conformal method, as in
        `ResultsPlot.series`.
    confidence_level : float
        The level of the band, strictly between 0 and 1.
    axes : Figure | None
        A figure of commol to draw on, as in `ResultsPlot.series`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the median.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If an entry of ``series`` is neither an output nor a bin or
        accumulator id, an observation names an unknown output or factor,
        the confidence level is out of range, or ``axes`` has no panel of
        a series.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _predictive_series

    return _predictive_series(
        self._predictive,
        output_file=output_file,
        observed_data=observed_data,
        config=config,
        series=series,
        show_legend=show_legend,
        position_label=position_label,
        tick_every=tick_every,
        x_label=x_label,
        prediction_intervals=prediction_intervals,
        confidence_level=confidence_level,
        axes=axes,
        **kwargs,
    )
```

#### cumulative

```python
cumulative(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, config: PlotConfig | None = None, series: list[str] | None = None, confidence_level: float = 0.95, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw the weighted median and the band of the running total of each series across the draws, one panel per series.

Parameters:

| Name               | Type                      | Description                                      | Default                                                                                                                                                              |
| ------------------ | ------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_file`      | \`str                     | None\`                                           | Path to save the figure to. If None, the figure is only returned.                                                                                                    |
| `observed_data`    | \`list[ObservedDataPoint] | None\`                                           | Observations, drawn as their running total on the panel of their label. An observation with a factor is divided by the weighted median of that factor in the sample. |
| `config`           | \`PlotConfig              | None\`                                           | Layout, size and style. If None, the defaults.                                                                                                                       |
| `series`           | \`list[str]               | None\`                                           | What to draw, as in ResultsPlot.series.                                                                                                                              |
| `confidence_level` | `float`                   | The level of the band, strictly between 0 and 1. | `0.95`                                                                                                                                                               |
| `axes`             | \`Figure                  | None\`                                           | A figure of commol to draw on, as in ResultsPlot.series.                                                                                                             |
| `**kwargs`         | \`str                     | int                                              | float                                                                                                                                                                |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                                                                     |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`  | If an entry of series is neither an output nor a bin or accumulator id, the confidence level is out of range, or axes has no panel of a series. |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                                                                    |

Source code in `commol/api/plotting/accessors.py`

```python
def cumulative(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    confidence_level: float = 0.95,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw the weighted median and the band of the running total of each
    series across the draws, one panel per series.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, drawn as their running total on the panel of their
        label. An observation with a factor is divided by the weighted
        median of that factor in the sample.
    config : PlotConfig | None
        Layout, size and style. If None, the defaults.
    series : list[str] | None
        What to draw, as in `ResultsPlot.series`.
    confidence_level : float
        The level of the band, strictly between 0 and 1.
    axes : Figure | None
        A figure of commol to draw on, as in `ResultsPlot.series`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the median.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If an entry of ``series`` is neither an output nor a bin or
        accumulator id, the confidence level is out of range, or ``axes``
        has no panel of a series.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.api.plotting.figures import _predictive_cumulative

    return _predictive_cumulative(
        self._predictive,
        output_file=output_file,
        observed_data=observed_data,
        config=config,
        series=series,
        confidence_level=confidence_level,
        axes=axes,
        **kwargs,
    )
```

## plot_compared

```python
plot_compared(results: Mapping[str, SimulationResults | SimulationReplicates | Predictive], series: list[str] | None = None, window: float | None = None, observed_data: list[ObservedDataPoint] | None = None, axes: Figure | None = None, output_file: str | None = None, config: PlotConfig | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', confidence_level: float = 0.95, **kwargs: str | int | float | bool | None) -> Figure
```

Draw several results of the same model on the same panels, one panel per series.

Each panel has one line per result, labelled with its key and in the same color in every panel (`C0` for the first key, `C1` for the second, …). Replicates show their mean and the band of the replicates, and a predictive the weighted median of its draws and their band, in the color of their key. The observations are drawn once.

Parameters:

| Name               | Type                              | Description                                                                     | Default                                                                                                                                                                                                                                                                                          |
| ------------------ | --------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `results`          | \`Mapping\[str, SimulationResults | SimulationReplicates                                                            | Predictive\]\`                                                                                                                                                                                                                                                                                   |
| `series`           | \`list[str]                       | None\`                                                                          | What to draw, one panel each, as in ResultsPlot.series. If None, every output.                                                                                                                                                                                                                   |
| `window`           | \`float                           | None\`                                                                          | The length of a window on the axis. Each line is then the gain of the series over each complete window, as Simulation.windowed_totals reads it, at the end of the window; a band is the band of these gains. If None, each line is the line of the plot.series of its result with observed_data. |
| `observed_data`    | \`list[ObservedDataPoint]         | None\`                                                                          | Observations, each drawn once as a point on the panel of its label. Without window, they also set how each line is read, as in ResultsPlot.series.                                                                                                                                               |
| `axes`             | \`Figure                          | None\`                                                                          | A figure of commol to draw on: each series is drawn on its panel, the Axes whose label is the id of the series. If None, a new figure.                                                                                                                                                           |
| `output_file`      | \`str                             | None\`                                                                          | Path to save the figure to. If None, the figure is only returned.                                                                                                                                                                                                                                |
| `config`           | \`PlotConfig                      | None\`                                                                          | Layout and style. If None, the defaults.                                                                                                                                                                                                                                                         |
| `show_legend`      | `bool`                            | Whether each panel shows its legend.                                            | `True`                                                                                                                                                                                                                                                                                           |
| `position_label`   | \`Callable\[[float], str\]        | None\`                                                                          | Turns a position into a tick label, as in ResultsPlot.series.                                                                                                                                                                                                                                    |
| `tick_every`       | \`float                           | None\`                                                                          | The distance between labelled positions, as in ResultsPlot.series.                                                                                                                                                                                                                               |
| `x_label`          | `str`                             | The label of the x axis.                                                        | `'Step'`                                                                                                                                                                                                                                                                                         |
| `confidence_level` | `float`                           | The level of the bands of replicates and predictives, strictly between 0 and 1. | `0.95`                                                                                                                                                                                                                                                                                           |
| `**kwargs`         | \`str                             | int                                                                             | float                                                                                                                                                                                                                                                                                            |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                                                                                                                                                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ValueError`  | If there are no results; a result does not have the outputs of the first (the message names its key and the outputs that differ); an entry of series is neither an output nor a bin or accumulator id; window is not positive; the confidence level is out of range; or axes has no panel of a series. |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                                                                                                                                                                                                                           |

Examples:

```pycon
>>> figure = plot_compared({"k1 = 0.3": results, "k1 = 0.5": faster})
```

Source code in `commol/api/plotting/compared.py`

```python
def plot_compared(
    results: "Mapping[str, SimulationResults | SimulationReplicates | Predictive]",
    series: list[str] | None = None,
    window: float | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
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
    series.

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
        every output.
    window : float | None
        The length of a window on the axis. Each line is then the gain of
        the series over each complete window, as `Simulation.windowed_totals`
        reads it, at the end of the window; a band is the band of these
        gains. If None, each line is the line of the ``plot.series`` of its
        result with `observed_data`.
    observed_data : list[ObservedDataPoint] | None
        Observations, each drawn once as a point on the panel of its label.
        Without `window`, they also set how each line is read, as in
        ``ResultsPlot.series``.
    axes : Figure | None
        A figure of commol to draw on: each series is drawn on its panel,
        the `Axes` whose label is the id of the series. If None, a new
        figure.
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
        first (the message names its key and the outputs that differ); an
        entry of ``series`` is neither an output nor a bin or accumulator
        id; `window` is not positive; the confidence level is out of range;
        or ``axes`` has no panel of a series.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).

    Examples
    --------
    >>> figure = plot_compared({"k1 = 0.3": results, "k1 = 0.5": faster})
    """
    from commol.api.plotting.figures import _compared_series

    return _compared_series(
        results,
        series=series,
        window=window,
        observed_data=observed_data,
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
