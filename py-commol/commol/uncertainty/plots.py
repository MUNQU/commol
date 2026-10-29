"""The `plot` accessor of a predictive, and its figures.

``predictive.plot.series()``, ``predictive.plot.cumulative()`` and
``predictive.plot.observations()`` draw the figures of the predictive. Each
method passes its arguments to its drawing function, which builds the panels
(`commol.uncertainty.panels`) and draws them (`commol.plotting.figures`);
matplotlib is imported when a figure is drawn. The accessor is also what
`plot_compared` reads of a predictive (`commol.plotting.panels.PanelSource`).
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from commol.plotting.panels import Band, ObservationPanel, Panel, PanelOutputs, Trace
from commol.uncertainty.panels import (
    predictive_cumulative_panels,
    predictive_observation_panels,
    predictive_panels,
    predictive_window,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from commol._data.intervals import PredictionIntervalsData
    from commol._model.model import Model
    from commol._problem.problem import ObservedDataPoint
    from commol.plotting.config import PlotConfig
    from commol.uncertainty.predictive import Predictive


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
        prediction_intervals: "PredictionIntervalsData | None" = None,
        confidence_level: float = 0.95,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw the weighted median of the draws of each series and their band,
        one panel per series.

        Each draw is read through the observation operator of the problem
        (the sums, windows, scales and factors of the observations of the
        series, each at its own position when they do not share one operator
        with no window), then the draws are reduced with their weights.

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
        prediction_intervals : PredictionIntervalsData | None
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
        return predictive_series(
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
            label. Each observation is divided by its scale and by the
            weighted median of its factor in the sample.
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
            If an observation with a factor names a factor the problem does
            not have, a parameter that is not a factor, an unknown output, or
            a position or a window start the engine does not accept; the
            confidence level is out of range; or ``axes`` has no panel of a
            series.
        KeyError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id of the model.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        return predictive_cumulative(
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
        sum of its outputs, its window, its scale and the draw's value of its
        factor),
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
        return predictive_observations(
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

    def _panel_model(self) -> "Model | None":
        """The model of the source of the predictive."""
        return self._predictive.source.simulation.model_definition

    def _panel_outputs(self) -> PanelOutputs:
        """The outputs of the predictive."""
        return PanelOutputs(self._predictive.output_names, self._panel_model())

    def _series_panels(
        self,
        confidence_level: float,
        observed_data: "list[ObservedDataPoint]",
        series: list[str],
    ) -> list[Panel]:
        """The panels of `series`, as `series` draws them."""
        return predictive_panels(
            self._predictive, confidence_level, observed_data, series
        )

    def _window_trace(
        self,
        outputs: PanelOutputs,
        series_id: str,
        confidence_level: float,
        window: float,
    ) -> tuple[Trace, Band | None]:
        """The median gain of a series over each complete window, and its band."""
        return predictive_window(
            self._predictive, outputs, series_id, confidence_level, window
        )

    def _observation_panels(
        self, confidence_level: float, observed_data: "list[ObservedDataPoint]"
    ) -> list[ObservationPanel]:
        """The panels of the observed series, as `observations` draws them."""
        return predictive_observation_panels(
            self._predictive, confidence_level, observed_data
        )


def predictive_series(
    predictive: "Predictive",
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    prediction_intervals: "PredictionIntervalsData | None" = None,
    confidence_level: float = 0.95,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """The figure of `PredictivePlot.series`."""
    from commol.plotting.figures import series_figure

    return series_figure(
        predictive_panels(predictive, confidence_level, observed_data, series),
        predictive.source.simulation.model_definition,
        output_file,
        config,
        show_legend,
        position_label,
        tick_every,
        x_label,
        prediction_intervals,
        axes,
        kwargs,
    )


def predictive_cumulative(
    predictive: "Predictive",
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    confidence_level: float = 0.95,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """The figure of `PredictivePlot.cumulative`."""
    from commol.plotting.figures import cumulative_figure

    return cumulative_figure(
        predictive_cumulative_panels(
            predictive, confidence_level, observed_data, series
        ),
        output_file,
        config,
        axes,
        kwargs,
    )


def predictive_observations(
    predictive: "Predictive",
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
    """The figure of `PredictivePlot.observations`."""
    from commol.plotting.figures import observations_figure

    return observations_figure(
        predictive_observation_panels(predictive, confidence_level, observed_data),
        predictive.source.simulation.model_definition,
        output_file,
        config,
        show_legend,
        position_label,
        tick_every,
        x_label,
        axes,
        kwargs,
    )
