"""The drawing functions of the figures, private to commol.

The `plot` accessors of the results (`commol.api.plotting.accessors`) and the
`plot` method of a result with one figure call them, with the arguments they
take. Each draws in a copy of `matplotlib.rcParams` (`_scoped_style`) and
returns the matplotlib `Figure`. Each panel holds the id of its series as the
label of its `Axes`, so a later figure given that `Figure` as `axes` draws on
the same panels.

Importing this module imports matplotlib, the optional extra ``commol[plot]``.
"""

import functools
import logging
import math
from collections import defaultdict
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

from commol.api.plotting import panels as _panels
from commol.api.plotting.panels import (
    Compared,
    ComparedObservationPanel,
    ObservationPanel,
    Panel,
)
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.inference.global_sensitivity import SobolResult
from commol.context.visualization import PlotConfig

if TYPE_CHECKING:
    from commol.api.inference.predictive import Predictive
    from commol.api.results import SimulationReplicates, SimulationResults
    from commol.context.axis_units import AxisUnits
    from commol.context.calibration import CalibrationResult, ObservedDataPoint
    from commol.context.inference.conformal import PredictionIntervals
    from commol.context.inference.global_sensitivity import (
        ChaosResult,
        MorrisResult,
        PRCCResult,
    )
    from commol.context.inference.profiles import ProfileResult
    from commol.context.model import Model

logger = logging.getLogger(__name__)

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
def _run_series(
    results: "SimulationResults",
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: PlotConfig | None = None,
    series: list[str] | None = None,
    show_legend: bool = True,
    position_label: PositionLabel | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    prediction_intervals: "PredictionIntervals | None" = None,
    axes: Figure | None = None,
    **kwargs: LineKeywords,
) -> Figure:
    """The figure of `ResultsPlot.series`."""
    return _series_figure(
        _panels.run_panels(results, observed_data, calibration_result, series),
        results.model,
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


@_scoped_style
def _replicates_series(
    replicates: "SimulationReplicates",
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: PlotConfig | None = None,
    series: list[str] | None = None,
    show_legend: bool = True,
    position_label: PositionLabel | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    prediction_intervals: "PredictionIntervals | None" = None,
    confidence_level: float = 0.95,
    axes: Figure | None = None,
    **kwargs: LineKeywords,
) -> Figure:
    """The figure of `ReplicatesPlot.series`."""
    return _series_figure(
        _panels.replicate_panels(
            replicates, confidence_level, observed_data, calibration_result, series
        ),
        replicates.model,
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


@_scoped_style
def _predictive_series(
    predictive: "Predictive",
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    config: PlotConfig | None = None,
    series: list[str] | None = None,
    show_legend: bool = True,
    position_label: PositionLabel | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    prediction_intervals: "PredictionIntervals | None" = None,
    confidence_level: float = 0.95,
    axes: Figure | None = None,
    **kwargs: LineKeywords,
) -> Figure:
    """The figure of `PredictivePlot.series`."""
    return _series_figure(
        _panels.predictive_panels(predictive, confidence_level, observed_data, series),
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


@_scoped_style
def _run_cumulative(
    results: "SimulationResults",
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: PlotConfig | None = None,
    series: list[str] | None = None,
    axes: Figure | None = None,
    **kwargs: LineKeywords,
) -> Figure:
    """The figure of `ResultsPlot.cumulative`."""
    return _cumulative_figure(
        _panels.cumulative_panels(results, observed_data, calibration_result, series),
        output_file,
        config,
        axes,
        kwargs,
    )


@_scoped_style
def _replicates_cumulative(
    replicates: "SimulationReplicates",
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    calibration_result: "CalibrationResult | None" = None,
    config: PlotConfig | None = None,
    series: list[str] | None = None,
    axes: Figure | None = None,
    **kwargs: LineKeywords,
) -> Figure:
    """The figure of `ReplicatesPlot.cumulative`."""
    return _cumulative_figure(
        _panels.cumulative_panels(
            replicates.mean(), observed_data, calibration_result, series
        ),
        output_file,
        config,
        axes,
        kwargs,
    )


@_scoped_style
def _predictive_cumulative(
    predictive: "Predictive",
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    config: PlotConfig | None = None,
    series: list[str] | None = None,
    confidence_level: float = 0.95,
    axes: Figure | None = None,
    **kwargs: LineKeywords,
) -> Figure:
    """The figure of `PredictivePlot.cumulative`."""
    return _cumulative_figure(
        _panels.predictive_cumulative_panels(
            predictive, confidence_level, observed_data, series
        ),
        output_file,
        config,
        axes,
        kwargs,
    )


@_scoped_style
def _compared_series(
    results: Mapping[str, Compared],
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
            _panels.model_of(next(iter(results.values()))),
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
            _panels.model_of(next(iter(results.values()))),
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
def _run_observations(
    results: "SimulationResults",
    observed_data: "list[ObservedDataPoint]",
    output_file: str | None = None,
    calibration_result: "CalibrationResult | None" = None,
    config: PlotConfig | None = None,
    show_legend: bool = True,
    position_label: PositionLabel | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    axes: Figure | None = None,
    **kwargs: LineKeywords,
) -> Figure:
    """The figure of `ResultsPlot.observations`."""
    return _observations_figure(
        _panels.run_observation_panels(results, observed_data, calibration_result),
        results.model,
        output_file,
        config,
        show_legend,
        position_label,
        tick_every,
        x_label,
        axes,
        kwargs,
    )


@_scoped_style
def _replicates_observations(
    replicates: "SimulationReplicates",
    observed_data: "list[ObservedDataPoint]",
    output_file: str | None = None,
    calibration_result: "CalibrationResult | None" = None,
    config: PlotConfig | None = None,
    show_legend: bool = True,
    position_label: PositionLabel | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    confidence_level: float = 0.95,
    axes: Figure | None = None,
    **kwargs: LineKeywords,
) -> Figure:
    """The figure of `ReplicatesPlot.observations`."""
    return _observations_figure(
        _panels.replicate_observation_panels(
            replicates, confidence_level, observed_data, calibration_result
        ),
        replicates.model,
        output_file,
        config,
        show_legend,
        position_label,
        tick_every,
        x_label,
        axes,
        kwargs,
    )


@_scoped_style
def _predictive_observations(
    predictive: "Predictive",
    observed_data: "list[ObservedDataPoint]",
    output_file: str | None = None,
    config: PlotConfig | None = None,
    show_legend: bool = True,
    position_label: PositionLabel | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    confidence_level: float = 0.95,
    axes: Figure | None = None,
    **kwargs: LineKeywords,
) -> Figure:
    """The figure of `PredictivePlot.observations`."""
    return _observations_figure(
        _panels.predictive_observation_panels(
            predictive, confidence_level, observed_data
        ),
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


@_scoped_style
def _prcc(
    result: "PRCCResult",
    output_file: str | None = None,
    config: PlotConfig | None = None,
    x_label: str = "Step",
    axes: Figure | None = None,
) -> Figure:
    """The figure of `PRCCResult.plot`."""
    config = config or PlotConfig()
    _apply_style(config)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, series in enumerate(result.series):
        groups[series].append(index)
    names = list(groups)
    figure, panels = _axes_of(names, config, axes)
    for ax, name in zip(panels, names, strict=True):
        indices = groups[name]
        positions = [
            position
            for position in (result.positions[i] for i in indices)
            if position is not None
        ]
        if len(positions) == len(indices):
            for j, input_name in enumerate(result.inputs):
                ax.plot(
                    positions,
                    [result.coefficients[i][j] for i in indices],
                    marker="o",
                    label=input_name,
                )
            ax.set_xlabel(x_label)
            ax.legend()
        else:
            ax.bar(result.inputs, result.coefficients[indices[0]])
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("PRCC")
        ax.set_title(name)
    _finish(figure, output_file, config, "PRCC")
    return figure


@_scoped_style
def _morris(
    result: "MorrisResult",
    output_file: str | None = None,
    config: PlotConfig | None = None,
    axes: Figure | None = None,
) -> Figure:
    """The figure of `MorrisResult.plot`."""
    config = config or PlotConfig()
    _apply_style(config)
    figure, panels = _axes_of(result.labels, config, axes)
    for q, ax in enumerate(panels):
        for j, input_name in enumerate(result.inputs):
            mu_star, sigma = result.mu_star[q][j], result.sigma[q][j]
            ax.scatter([mu_star], [sigma], label=input_name)
            ax.annotate(input_name, (mu_star, sigma))
        ax.set_xlabel("μ*")
        ax.set_ylabel("σ")
        ax.set_title(result.labels[q])
        ax.legend()
    _finish(figure, output_file, config, "Morris")
    return figure


@_scoped_style
def _sobol(
    result: "SobolResult | ChaosResult",
    output_file: str | None = None,
    config: PlotConfig | None = None,
    axes: Figure | None = None,
) -> Figure:
    """The figure of `SobolResult.plot` and `ChaosResult.plot`."""
    config = config or PlotConfig()
    _apply_style(config)
    inputs = result.inputs
    figure, panels = _axes_of(result.labels, config, axes)
    width = 0.4
    places = list(range(len(inputs)))
    for q, ax in enumerate(panels):
        for offset, name, values in (
            (-width / 2, "First order", result.first[q]),
            (width / 2, "Total", result.total[q]),
        ):
            errors = None
            if isinstance(result, SobolResult):
                lower, upper = (
                    (result.first_lower[q], result.first_upper[q])
                    if name == "First order"
                    else (result.total_lower[q], result.total_upper[q])
                )
                errors = [
                    [max(v - low, 0.0) for v, low in zip(values, lower, strict=True)],
                    [max(high - v, 0.0) for v, high in zip(values, upper, strict=True)],
                ]
            ax.bar(
                [place + offset for place in places],
                values,
                width,
                yerr=errors,
                capsize=3,
                label=name,
            )
        ax.set_xticks(places)
        ax.set_xticklabels(inputs)
        ax.set_ylabel("Sobol index")
        ax.set_title(result.labels[q])
        ax.legend()
    _finish(figure, output_file, config, "Sobol")
    return figure


@_scoped_style
def _profiles(
    result: "ProfileResult",
    output_file: str | None = None,
    config: PlotConfig | None = None,
    axes: Figure | None = None,
) -> Figure:
    """The figure of `ProfileResult.plot`."""
    config = config or PlotConfig()
    _apply_style(config)
    names = list(result.profiles)
    figure, panels = _axes_of(names, config, axes)
    for ax, name in zip(panels, names, strict=True):
        profile = result.profiles[name]
        ax.plot(profile.values, profile.statistics, marker="o", markersize=3)
        ax.axhline(profile.threshold, color="grey", linestyle="--", linewidth=1)
        interval = result.intervals[name]
        for end in (interval.lower, interval.upper):
            if end is not None:
                ax.axvline(end, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel(name)
        ax.set_ylabel("Profile statistic")
        ax.set_title(name)
    _finish(figure, output_file, config, "Profiles")
    return figure


def _series_figure(
    panels: list[Panel],
    model: "Model | None",
    output_file: str | None,
    config: PlotConfig | None,
    show_legend: bool,
    position_label: PositionLabel | None,
    tick_every: float | None,
    x_label: str,
    prediction_intervals: "PredictionIntervals | None",
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


def _cumulative_figure(
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


def _observations_figure(
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
