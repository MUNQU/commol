"""The figures of the results of the sensitivity methods.

The `plot` method of a result calls its drawing function here, which builds
one chart per series or quantity (`commol.plotting.charts`) and draws them
(`commol.plotting.figures.chart_figure`); matplotlib is imported when a
figure is drawn.
"""

from collections import defaultdict
from typing import TYPE_CHECKING

from commol.plotting.charts import Bars, Chart, Line, Point, Rule
from commol.sensitivity.results import SobolResult

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from commol.plotting.config import PlotConfig
    from commol.sensitivity.results import ChaosResult, MorrisResult, PRCCResult


def prcc_figure(
    result: "PRCCResult",
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    x_label: str = "Step",
    axes: "Figure | None" = None,
) -> "Figure":
    """The figure of `PRCCResult.plot`."""
    from commol.plotting.figures import chart_figure

    return chart_figure(prcc_charts(result, x_label), output_file, config, axes, "PRCC")


def morris_figure(
    result: "MorrisResult",
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    axes: "Figure | None" = None,
) -> "Figure":
    """The figure of `MorrisResult.plot`."""
    from commol.plotting.figures import chart_figure

    return chart_figure(morris_charts(result), output_file, config, axes, "Morris")


def sobol_figure(
    result: "SobolResult | ChaosResult",
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    axes: "Figure | None" = None,
) -> "Figure":
    """The figure of `SobolResult.plot` and `ChaosResult.plot`."""
    from commol.plotting.figures import chart_figure

    return chart_figure(sobol_charts(result), output_file, config, axes, "Sobol")


def prcc_charts(result: "PRCCResult", x_label: str) -> list[Chart]:
    """
    One chart per series: a line per input over the positions of a series
    over positions, or a bar per input.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for index, series in enumerate(result.series):
        groups[series].append(index)
    charts = []
    for name, indices in groups.items():
        positions = [
            position
            for position in (result.positions[i] for i in indices)
            if position is not None
        ]
        over_positions = len(positions) == len(indices)
        marks: list[Line | Point | Bars] = (
            [
                Line(
                    positions,
                    [result.coefficients[i][j] for i in indices],
                    label=input_name,
                )
                for j, input_name in enumerate(result.inputs)
            ]
            if over_positions
            else [Bars(list(result.inputs), list(result.coefficients[indices[0]]))]
        )
        charts.append(
            Chart(
                name=name,
                y_label="PRCC",
                marks=marks,
                rules=[Rule(0.0, vertical=False, width=0.8)],
                x_label=x_label if over_positions else None,
                y_limits=(-1.05, 1.05),
                legend=over_positions,
            )
        )
    return charts


def morris_charts(result: "MorrisResult") -> list[Chart]:
    """One chart per quantity: the point (μ*, σ) of each input."""
    return [
        Chart(
            name=label,
            y_label="σ",
            marks=[
                Point(result.mu_star[q][j], result.sigma[q][j], input_name)
                for j, input_name in enumerate(result.inputs)
            ],
            x_label="μ*",
            legend=True,
        )
        for q, label in enumerate(result.labels)
    ]


def sobol_charts(result: "SobolResult | ChaosResult") -> list[Chart]:
    """
    One chart per quantity: the first-order and the total index of each
    input, side by side, with the bounds of a `SobolResult` as error bars.
    """
    width = 0.4
    places = list(range(len(result.inputs)))
    charts = []
    for q, label in enumerate(result.labels):
        marks: list[Line | Point | Bars] = []
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
            marks.append(
                Bars(
                    [place + offset for place in places],
                    list(values),
                    width,
                    errors=errors,
                    label=name,
                )
            )
        charts.append(
            Chart(
                name=label,
                y_label="Sobol index",
                marks=marks,
                ticks=list(result.inputs),
                legend=True,
            )
        )
    return charts
