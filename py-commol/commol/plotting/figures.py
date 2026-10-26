"""The drawing functions of the figures, private to commol.

Each result builds the data of its figure and passes it here: the panels of
a run, of replicates and of a predictive (`series_figure`,
`cumulative_figure`, `observations_figure`), and the charts of a result with
one figure (`chart_figure`). `plot_compared` reads each result through its
`plot` accessor (`commol.plotting.panels.PanelSource`). Each figure is drawn
in a copy of `matplotlib.rcParams` (`_scoped_style`) and is returned as the
matplotlib `Figure`. Each panel holds the id of its series as the
label of its `Axes`, so a later figure given that `Figure` as `axes` draws on
the same panels.

Importing this module imports matplotlib, the optional extra ``commol[plot]``.
"""

import functools
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import time
from typing import TYPE_CHECKING, Literal

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter
    from matplotlib.typing import ColorType
except ModuleNotFoundError as error:
    raise ImportError(
        "The figures of commol need matplotlib. Install it with: "
        "pip install 'commol[plot]'"
    ) from error

from commol.commol_rs import _commol_rs as commol_rs
from commol.plotting import panels as _panels
from commol.plotting.charts import Chart, Line, Point
from commol.plotting.config import PlotConfig
from commol.plotting.panels import (
    Plotted,
    ComparedObservationPanel,
    ObservationPanel,
    Panel,
)

if TYPE_CHECKING:
    from commol._model.axis_units import AxisUnits
    from commol._model.model import Model
    from commol._problem.problem import ObservedDataPoint
    from commol._data.intervals import PredictionIntervalsData

logger = logging.getLogger("commol.plotting")

PositionLabel = Callable[[float], str]
LineKeywords = str | int | float | bool | None


def _scoped_style[**P, R](draw: Callable[P, R]) -> Callable[P, R]:
    """
    Run `draw` in a copy of `matplotlib.rcParams`.

    The style of a `PlotConfig` applies to the figure that `draw` makes, and
    `matplotlib.rcParams` is as it was before the call when `draw` returns or
    raises.
    """

    @functools.wraps(draw)
    def scoped(*args: P.args, **kwargs: P.kwargs) -> R:
        with mpl.rc_context():
            return draw(*args, **kwargs)

    return scoped


@_scoped_style
def _compared_series(
    results: Mapping[str, Plotted],
    series: list[str] | None = None,
    window: float | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    panels: Literal["series", "observations"] = "series",
    axes: Figure | None = None,
    output_file: str | None = None,
    config: PlotConfig | None = None,
    show_legend: bool = True,
    position_label: PositionLabel | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    confidence_level: float = 0.95,
    **kwargs: LineKeywords,
) -> Figure:
    """The figure of `plot_compared`."""
    config = config or PlotConfig()
    _apply_style(config)
    if panels == "observations":
        if series is not None or window is not None:
            raise ValueError(
                "The panels of observed series are the labels of `observed_data`, "
                "read through their observation operators: give no `series` "
                "and no `window`."
            )
        figure = _compared_observations_figure(
            _panels.compared_observation_panels(
                results, confidence_level, observed_data or []
            ),
            next(iter(results.values())).plot._panel_model(),
            config,
            show_legend,
            position_label,
            tick_every,
            x_label,
            axes,
            kwargs,
        )
    elif panels == "series":
        figure = _compared_series_figure(
            _panels.compared_panels(
                results, confidence_level, observed_data, series, window
            ),
            next(iter(results.values())).plot._panel_model(),
            config,
            show_legend,
            position_label,
            tick_every,
            x_label,
            axes,
            kwargs,
        )
    else:
        raise ValueError(f"`panels` is 'series' or 'observations', not {panels!r}.")
    _finish(figure, output_file, config, "Compared")
    return figure


@_scoped_style
def series_figure(
    panels: list[Panel],
    model: "Model | None",
    output_file: str | None,
    config: PlotConfig | None,
    show_legend: bool,
    position_label: PositionLabel | None,
    tick_every: float | None,
    x_label: str,
    prediction_intervals: "PredictionIntervalsData | None",
    axes: Figure | None,
    keywords: dict[str, LineKeywords],
) -> Figure:
    """Draw the panels of series, with their prediction intervals."""
    config = config or PlotConfig()
    _apply_style(config)
    panels = _panels.with_intervals(panels, prediction_intervals)
    figure, grid = _axes_of([panel.series for panel in panels], config, axes)
    for ax, panel in zip(grid, panels, strict=True):
        _draw_panel(ax, panel, keywords, x_label)
        _label_positions(ax, panel.line.positions, position_label, tick_every, model)
        if show_legend:
            ax.legend()
    _finish(figure, output_file, config, "Series")
    return figure


@_scoped_style
def cumulative_figure(
    panels: list[Panel],
    output_file: str | None,
    config: PlotConfig | None,
    axes: Figure | None,
    keywords: dict[str, LineKeywords],
) -> Figure:
    """Draw the panels of running totals, of the size of the configuration."""
    config = config or PlotConfig()
    _apply_style(config)
    figure, grid = _axes_of(
        [panel.series for panel in panels], config, axes, config.figsize
    )
    for ax, panel in zip(grid, panels, strict=True):
        _draw_panel(ax, panel, keywords, "Step")
        ax.legend()
    _finish(figure, output_file, config, "Cumulative")
    return figure


def _compared_series_figure(
    panels: list[_panels.ComparedPanel],
    model: "Model | None",
    config: PlotConfig,
    show_legend: bool,
    position_label: PositionLabel | None,
    tick_every: float | None,
    x_label: str,
    axes: Figure | None,
    keywords: dict[str, LineKeywords],
) -> Figure:
    """Draw the panels of series of a comparison."""
    figure, grid = _axes_of([panel.series for panel in panels], config, axes)
    for ax, panel in zip(grid, panels, strict=True):
        _draw_compared_panel(ax, panel, keywords, x_label)
        positions = sorted({at for line in panel.lines for at in line.positions})
        _label_positions(ax, positions, position_label, tick_every, model)
        if show_legend:
            ax.legend()
    return figure


def _compared_observations_figure(
    panels: list[ComparedObservationPanel],
    model: "Model | None",
    config: PlotConfig,
    show_legend: bool,
    position_label: PositionLabel | None,
    tick_every: float | None,
    x_label: str,
    axes: Figure | None,
    keywords: dict[str, LineKeywords],
) -> Figure:
    """Draw the panels of observed series of a comparison."""
    figure, grid = _axes_of([panel.series for panel in panels], config, axes)
    for ax, panel in zip(grid, panels, strict=True):
        for index, (trace, band) in enumerate(
            zip(panel.models, panel.bands, strict=True)
        ):
            _draw_observation_model(
                ax, trace, panel.windows, band, keywords, f"C{index}"
            )
        _draw_observed(ax, panel.observed)
        _label_panel(ax, panel, x_label)
        _label_positions(
            ax, panel.observed.positions, position_label, tick_every, model
        )
        if show_legend:
            ax.legend()
    return figure


@_scoped_style
def observations_figure(
    panels: list[ObservationPanel],
    model: "Model | None",
    output_file: str | None,
    config: PlotConfig | None,
    show_legend: bool,
    position_label: PositionLabel | None,
    tick_every: float | None,
    x_label: str,
    axes: Figure | None,
    keywords: dict[str, LineKeywords],
) -> Figure:
    """Draw the panels of observed series."""
    config = config or PlotConfig()
    _apply_style(config)
    figure, grid = _axes_of([panel.series for panel in panels], config, axes)
    for ax, panel in zip(grid, panels, strict=True):
        _draw_observation_panel(ax, panel, keywords, x_label)
        _label_positions(ax, panel.model.positions, position_label, tick_every, model)
        if show_legend:
            ax.legend()
    _finish(figure, output_file, config, "Observations")
    return figure


@_scoped_style
def chart_figure(
    charts: list[Chart],
    output_file: str | None,
    config: PlotConfig | None,
    axes: Figure | None,
    name: str,
) -> Figure:
    """Draw the charts of a result, one panel each, as the figure `name`."""
    config = config or PlotConfig()
    _apply_style(config)
    figure, grid = _axes_of([chart.name for chart in charts], config, axes)
    for ax, chart in zip(grid, charts, strict=True):
        _draw_chart(ax, chart)
    _finish(figure, output_file, config, name)
    return figure


def _draw_chart(ax: Axes, chart: Chart) -> None:
    """Draw the marks, the rules and the labels of one chart."""
    for mark in chart.marks:
        if isinstance(mark, Line):
            ax.plot(
                mark.positions,
                mark.values,
                marker="o",
                label=mark.label,
                markersize=mark.size,
            )
        elif isinstance(mark, Point):
            ax.scatter([mark.position], [mark.value], label=mark.label)
            ax.annotate(mark.label, (mark.position, mark.value))
        else:
            ax.bar(
                mark.places,
                mark.heights,
                mark.width,
                yerr=mark.errors,
                capsize=3,
                label=mark.label,
            )
    for rule in chart.rules:
        if rule.vertical:
            ax.axvline(
                rule.at, color="grey", linestyle=rule.style, linewidth=rule.width
            )
        else:
            ax.axhline(
                rule.at, color="grey", linestyle=rule.style, linewidth=rule.width
            )
    if chart.ticks is not None:
        ax.set_xticks(range(len(chart.ticks)))
        ax.set_xticklabels(chart.ticks)
    if chart.x_label is not None:
        ax.set_xlabel(chart.x_label)
    ax.set_ylabel(chart.y_label)
    if chart.y_limits is not None:
        ax.set_ylim(*chart.y_limits)
    ax.set_title(chart.name)
    if chart.legend:
        ax.legend()


def _draw_observation_panel(
    ax: Axes,
    panel: ObservationPanel,
    keywords: dict[str, LineKeywords],
    x_label: str,
) -> None:
    """
    Draw one observed series: the model value of each observation, its band,
    the observed values and the labels.
    """
    _draw_observation_model(ax, panel.model, panel.windows, panel.band, keywords)
    _draw_observed(ax, panel.observed)
    _label_panel(ax, panel, x_label)


def _draw_observation_model(
    ax: Axes,
    model: _panels.Trace,
    windows: list[float | None],
    band: _panels.Band | None,
    keywords: dict[str, LineKeywords],
    color: ColorType | None = None,
) -> None:
    """
    Draw the model value of each observation and its band, in `color` or in
    the next color of the cycle.

    A value with a window is a total over ``(at - window, at]``: a step over
    its window with a marker at ``at``, and its band a rectangle over the same
    window. A value with no window is a marker; a dotted line joins these
    markers only as a guide, and their band is filled between them.
    `keywords` go to the steps and to the guide; the markers and the band
    take their colour.
    """
    points = list(zip(model.positions, windows, model.values, range(len(model.values))))
    windowed = [(at, window, value, i) for at, window, value, i in points if window]
    single = [(at, value, i) for at, window, value, i in points if not window]
    lines = []
    if windowed:
        xs: list[float] = []
        ys: list[float] = []
        for at, window, value, _ in windowed:
            xs += [commol_rs.calibration.window_start(at, window), at, math.nan]
            ys += [value, value, math.nan]
        (steps,) = ax.plot(xs, ys, label=model.label, color=color)
        lines.append(steps)
    if single:
        (guide,) = ax.plot(
            [at for at, _, _ in single],
            [value for _, value, _ in single],
            linestyle=":",
            linewidth=1,
            color=lines[0].get_color() if lines else color,
            label="_nolegend_" if lines else model.label,
        )
        lines.append(guide)
    for line in lines:
        line.set(**keywords)
    color = lines[0].get_color()
    ax.plot(
        model.positions,
        model.values,
        marker="o",
        linestyle="none",
        color=color,
        label="_nolegend_",
    )
    if band is not None:
        if windowed:
            ax.bar(
                [
                    commol_rs.calibration.window_start(at, window)
                    for at, window, _, _ in windowed
                ],
                [band.upper[i] - band.lower[i] for _, _, _, i in windowed],
                width=[window for _, window, _, _ in windowed],
                bottom=[band.lower[i] for _, _, _, i in windowed],
                align="edge",
                color=color,
                alpha=0.3,
                label=band.label,
            )
        if single:
            _fill(
                ax,
                _panels.Band(
                    [at for at, _, _ in single],
                    [band.lower[i] for _, _, i in single],
                    [band.upper[i] for _, _, i in single],
                    "_nolegend_" if windowed else band.label,
                ),
                color,
            )


def _draw_panel(
    ax: Axes, panel: Panel, keywords: dict[str, LineKeywords], x_label: str
) -> None:
    """
    Draw one panel: its line, which takes `keywords`, its band, its observed
    points, its prediction intervals, and its labels.
    """
    line = panel.line
    (drawn,) = ax.plot(line.positions, line.values, label=line.label)
    drawn.set(**keywords)
    if panel.band is not None:
        _fill(ax, panel.band)
    if panel.observed is not None:
        _draw_observed(ax, panel.observed)
    if panel.intervals is not None and panel.prediction is not None:
        _fill(ax, panel.intervals)
        ax.plot(
            panel.prediction.positions,
            panel.prediction.values,
            linestyle="--",
            label=panel.prediction.label,
        )
    _label_panel(ax, panel, x_label)


def _draw_compared_panel(
    ax: Axes,
    panel: _panels.ComparedPanel,
    keywords: dict[str, LineKeywords],
    x_label: str,
) -> None:
    """
    Draw one panel of a comparison: the line and the band of each result in
    its own color of the cycle (``C0``, ``C1``, …), which takes `keywords`,
    the observed points and the labels.
    """
    for index, (line, band) in enumerate(zip(panel.lines, panel.bands, strict=True)):
        color = f"C{index}"
        (drawn,) = ax.plot(line.positions, line.values, label=line.label, color=color)
        drawn.set(**keywords)
        if band is not None:
            _fill(ax, band, color)
    if panel.observed is not None:
        _draw_observed(ax, panel.observed)
    _label_panel(ax, panel, x_label)


def _draw_observed(ax: Axes, observed: _panels.Trace) -> None:
    """Draw the observed values as points."""
    ax.scatter(
        observed.positions,
        observed.values,
        label=observed.label,
        color="red",
        s=30,
        alpha=0.7,
        zorder=5,
    )


def _label_panel(
    ax: Axes,
    panel: Panel | _panels.ComparedPanel | ObservationPanel | ComparedObservationPanel,
    x_label: str,
) -> None:
    """
    Label a panel: the id of its series as the label of its `Axes`, the
    axes, the title and the grid.
    """
    ax.set_label(panel.series)
    ax.set_xlabel(x_label)
    ax.set_ylabel(panel.unit)
    ax.set_title(panel.name)
    ax.grid(True, alpha=0.3)


def _fill(ax: Axes, band: _panels.Band, color: ColorType | None = None) -> None:
    """Fill a band, in `color` or in the next color of the cycle."""
    ax.fill_between(
        band.positions,
        band.lower,
        band.upper,
        alpha=0.3,
        color=color,
        label=band.label,
    )


def _apply_style(config: PlotConfig) -> None:
    """
    Apply the matplotlib style of `config`.

    It sets `matplotlib.rcParams`, so only a function under `_scoped_style`
    calls it: the style stays in the figure of that function.
    """
    if config.style:
        plt.style.use(config.style)


def _axes_of(
    ids: Sequence[str],
    config: PlotConfig,
    axes: Figure | None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, list[Axes]]:
    """
    The figure and the panel of each id.

    With `axes`, the panel of an id is the `Axes` of that figure whose label
    is the id. Otherwise a new figure of the layout of `config`, of `figsize`
    or of the size of its panels, whose panels beyond the ids are hidden.

    Raises
    ------
    ValueError
        If `axes` has no panel of an id.
    """
    if axes is not None:
        by_label = {ax.get_label(): ax for ax in axes.axes}
        missing = [series_id for series_id in ids if series_id not in by_label]
        if missing:
            raise ValueError(
                f"The figure has no panel of {missing}. Its panels: "
                f"{[label for label in by_label if label]}."
            )
        return axes, [by_label[series_id] for series_id in ids]
    rows, cols = config.layout or _layout(len(ids))
    figure, grid = plt.subplots(
        rows,
        cols,
        figsize=figsize or (cols * 4.5, rows * 3.5 + 1.0),
        dpi=config.dpi,
        squeeze=False,
    )
    panels = list(grid.flat)
    for ax, series_id in zip(panels, ids, strict=False):
        ax.set_label(series_id)
    for ax in panels[len(ids) :]:
        ax.set_visible(False)
    return figure, panels[: len(ids)]


def _layout(count: int) -> tuple[int, int]:
    """The rows and columns of a figure of `count` panels."""
    if count == 1:
        return (1, 1)
    if count == 2:
        return (1, 2)
    if count <= 4:
        return (2, 2)
    cols = math.ceil(math.sqrt(count))
    return (math.ceil(count / cols), cols)


def _finish(
    figure: Figure, output_file: str | None, config: PlotConfig, name: str
) -> None:
    """Lay the figure out, and save it to `output_file` when there is one."""
    figure.tight_layout()
    if output_file:
        figure.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"{name} figure saved to {output_file}")


def _label_positions(
    ax: Axes,
    positions: Sequence[float],
    position_label: PositionLabel | None,
    tick_every: float | None,
    model: "Model | None",
) -> None:
    """
    Label the x axis with `position_label`, or with dates on a dated axis.

    With no `position_label` and a model with an origin, a tick shows the
    date of its position: at the plotted positions `tick_every` apart, or at
    the ticks matplotlib places when `tick_every` is None. With neither, the
    ticks show the position.
    """
    if position_label is None:
        if model is None or model.origin is None:
            return
        axis = model.axis_units()
        if tick_every is None:
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda at, _: _date_label(axis, at))
            )
            ax.tick_params(axis="x", labelrotation=45)
            return
        position_label = functools.partial(_date_label, axis)
    _apply_x_labels(ax, positions, position_label, tick_every)


def _apply_x_labels(
    ax: Axes,
    positions: Sequence[float],
    position_label: PositionLabel,
    tick_every: float | None,
) -> None:
    """
    Label the x axis at the plotted positions `tick_every` apart.

    The tick grid is phased to the first plotted position rather than to
    position 0: a series read at its observations is plotted at their
    positions, so anchoring at 0 would put ticks on positions that carry no
    plotted point.
    """
    if tick_every is not None and positions:
        anchor = positions[0]
        windows_in = commol_rs.calibration.windows_in
        ticks = [
            position
            for position in positions
            if windows_in(position - anchor, tick_every).is_integer()
        ]
    else:
        ticks = list(positions)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [position_label(position) for position in ticks], rotation=45, ha="right"
    )


def _date_label(axis: "AxisUnits", position: float) -> str:
    """The date of a position, with its time when it is not midnight."""
    moment = axis.moment_of(position)
    if moment.time() == time():
        return moment.date().isoformat()
    return moment.isoformat(sep=" ")
