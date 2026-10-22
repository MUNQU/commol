"""The figure that draws several results on the same panels.

`plot_compared` is a function, not a method: a comparison describes no single
result, and a method of one of them would give that result a special role. It
passes its arguments to its drawing function (`commol.api.plotting.figures`);
matplotlib is imported when the figure is drawn.
"""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from commol.api.inference.predictive import Predictive
    from commol.api.results import SimulationReplicates, SimulationResults
    from commol.context.calibration import ObservedDataPoint
    from commol.context.visualization import PlotConfig


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
