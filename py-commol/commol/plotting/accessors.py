"""The `plot` accessors of a run, of replicates and of a predictive.

``results.plot.series()``, ``results.plot.cumulative()`` and
``results.plot.observations()`` draw the figures of the results they are read
from, with the model of those results. Each method takes the arguments of its
drawing function (`commol.plotting.figures`) and passes them on;
matplotlib is imported when a figure is drawn.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from commol.uncertainty.predictive import Predictive
    from commol._simulation.results import SimulationReplicates, SimulationResults
    from commol._fit.result import CalibrationResult
    from commol._problem.problem import ObservedDataPoint
    from commol.conformal.methods import PredictionIntervals
    from commol.plotting.config import PlotConfig


class ResultsPlot:
    """
    The figures of one run, read from ``results.plot``.

    Each figure has one panel per series, or per observed series for
    `observations`, and returns the matplotlib `Figure`. Given ``axes=``, the
    `Figure` of an earlier figure of commol, it draws each series on the
    panel of that series instead.
    """

    def __init__(self, results: "SimulationResults") -> None:
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
            If an entry of ``series`` is a bin or accumulator id of results
            with no model; the results have no value at the position of an
            observation read on its own or at the start of its window; or
            ``axes`` has no panel of a series.
        KeyError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id of the model.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.plotting.figures import _run_series

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
            If ``axes`` has no panel of a series.
        KeyError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id of the model.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.plotting.figures import _run_cumulative

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

    def observations(
        self,
        observed_data: "list[ObservedDataPoint]",
        output_file: str | None = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw each observed series with the model read through its
        observation operator, one panel per label.

        The model is read only at the observations: the value of each
        observation through its own operator (the sum of its outputs, its
        window and its factor), as the loss reads it. A value with a window is
        a total over ``(at - window, at]``, drawn as a step over its window
        with a marker at ``at``. A value with no window is a marker, joined
        to the next only as a guide. The run along the whole axis is
        `series`, on the same figure with ``axes=``.

        Parameters
        ----------
        observed_data : list[ObservedDataPoint]
            The observations. Each label is one panel, in the order of its
            first observation, with the `Axes` label of the label.
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        calibration_result : CalibrationResult | None
            Its ``best_parameters`` give the values of the factors of the
            observations, which multiply the model; observed values stay in
            their own units.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position into a tick label, as in `series`.
        tick_every : float | None
            The distance between labelled observation positions, as in
            `series`.
        x_label : str
            The label of the x axis.
        axes : Figure | None
            A figure of commol to draw on: each observed series is drawn on
            the `Axes` whose label is its label. If None, a new figure.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the steps and the guide of the model,
            such as ``linewidth`` or ``alpha``.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If there are no observations; the results have no value at the
            position of an observation or at the start of its window; or
            ``axes`` has no panel of a label.
        KeyError
            If an observation names an output the results do not have.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.plotting.figures import _run_observations

        return _run_observations(
            self._results,
            observed_data,
            output_file=output_file,
            calibration_result=calibration_result,
            config=config,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            axes=axes,
            **kwargs,
        )


class ReplicatesPlot:
    """
    The figures of the replicates of a stochastic engine, read from
    ``replicates.plot``.

    The figures are those of `ResultsPlot` for the mean of the replicates;
    `series` adds the band of the replicates.
    """

    def __init__(self, replicates: "SimulationReplicates") -> None:
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
        KeyError
            As `ResultsPlot.series`.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.plotting.figures import _replicates_series

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
        KeyError
            As `ResultsPlot.cumulative`.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.plotting.figures import _replicates_cumulative

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

    def observations(
        self,
        observed_data: "list[ObservedDataPoint]",
        output_file: str | None = None,
        calibration_result: "CalibrationResult | None" = None,
        config: "PlotConfig | None" = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        confidence_level: float = 0.95,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw each observed series with the mean of the replicates and their
        band, read through its observation operator, one panel per label.

        Each replicate is read through the operator of each observation
        first, as `ResultsPlot.observations` reads a run, then the band takes
        the interval of `confidence_level` across them
        (`member_statistics`). A band over a window is a rectangle over that
        window. The band is a band of the model, with no observation noise,
        not a prediction interval, and its legend says so.

        Parameters
        ----------
        observed_data : list[ObservedDataPoint]
            The observations, as in `ResultsPlot.observations`.
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        calibration_result : CalibrationResult | None
            The values of the factors, as in `ResultsPlot.observations`.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position into a tick label, as in `ResultsPlot.series`.
        tick_every : float | None
            The distance between labelled positions, as in
            `ResultsPlot.series`.
        x_label : str
            The label of the x axis.
        confidence_level : float
            The level of the band, strictly between 0 and 1.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.observations`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the steps and the guide of the mean.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            As `ResultsPlot.observations`; or if there are no replicates or
            the confidence level is out of range.
        KeyError
            If an observation names an output the replicates do not have.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.plotting.figures import _replicates_observations

        return _replicates_observations(
            self._replicates,
            observed_data,
            output_file=output_file,
            calibration_result=calibration_result,
            config=config,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            confidence_level=confidence_level,
            axes=axes,
            **kwargs,
        )


class PredictivePlot:
    """
    The figures of a predictive, read from ``predictive.plot``.

    Each panel shows the weighted median of the draws and their band, with no
    other run; the model of the source names the series.
    """

    def __init__(self, predictive: "Predictive") -> None:
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
            If an observation names an unknown output or factor, the
            confidence level is out of range, or ``axes`` has no panel of a
            series.
        KeyError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id of the model.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.plotting.figures import _predictive_series

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
            If the confidence level is out of range, or ``axes`` has no panel
            of a series.
        KeyError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id of the model.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.plotting.figures import _predictive_cumulative

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

    def observations(
        self,
        observed_data: "list[ObservedDataPoint]",
        output_file: str | None = None,
        config: "PlotConfig | None" = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        confidence_level: float = 0.95,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw each observed series with the weighted median of the draws and
        their band, read through its observation operator, one panel per
        label.

        The values are those of ``at_observations(observed_data, NoNoise())``:
        each draw's value of each observation through its own operator (the
        sum of its outputs, its window and the draw's value of its factor),
        the values the loss compares. A value with a window is drawn as a
        step over its window with a marker at ``at``, and its band as a
        rectangle over the same window; a value with no window is a marker,
        joined only as a guide. The band is a band of the model, with no
        observation noise, not a prediction interval, and its legend says so.

        Parameters
        ----------
        observed_data : list[ObservedDataPoint]
            The observations, as in `ResultsPlot.observations`.
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position into a tick label, as in `ResultsPlot.series`.
        tick_every : float | None
            The distance between labelled positions, as in
            `ResultsPlot.series`.
        x_label : str
            The label of the x axis.
        confidence_level : float
            The level of the band, strictly between 0 and 1.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.observations`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the steps and the guide of the median.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If there are no observations, an observation names an unknown
            output or factor, the confidence level is out of range, or
            ``axes`` has no panel of a label.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.plotting.figures import _predictive_observations

        return _predictive_observations(
            self._predictive,
            observed_data,
            output_file=output_file,
            config=config,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            confidence_level=confidence_level,
            axes=axes,
            **kwargs,
        )
