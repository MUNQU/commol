"""The data of the charts of a result with one figure, computed without
matplotlib.

A chart is one panel: its marks (lines, points, bars), its grey rules and its
labels. The `plot` method of a sensitivity or a profile result builds its
charts (`commol.sensitivity.plots`, `commol.profiles.plots`), and
`commol.plotting.figures.chart_figure` draws them.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Line:
    """
    Values at positions, joined by a line with a round marker at each.

    Attributes
    ----------
    positions : list[float]
        Positions on the x axis.
    values : list[float]
        One value per position.
    label : str | None
        The label of the legend, or None for none.
    size : float | None
        The size of the markers, or None for the size of the style.
    """

    positions: list[float]
    values: list[float]
    label: str | None = None
    size: float | None = None


@dataclass(frozen=True)
class Point:
    """
    One point, annotated with its label.

    Attributes
    ----------
    position : float
        Its place on the x axis.
    value : float
        Its value.
    label : str
        The label of the legend and of the annotation.
    """

    position: float
    value: float
    label: str


@dataclass(frozen=True)
class Bars:
    """
    One bar per place.

    Attributes
    ----------
    places : list[float] | list[str]
        The place of each bar: a position, or a name.
    heights : list[float]
        The height of each bar.
    width : float
        The width of each bar.
    errors : list[list[float]] | None
        The lengths of the error bars below and above each bar, or None for
        none.
    label : str | None
        The label of the legend, or None for none.
    """

    places: list[float] | list[str]
    heights: list[float]
    width: float = 0.8
    errors: list[list[float]] | None = None
    label: str | None = None


@dataclass(frozen=True)
class Rule:
    """
    A grey line across the panel.

    Attributes
    ----------
    at : float
        Where it crosses the axis.
    vertical : bool
        Whether it crosses the x axis; otherwise it crosses the y axis.
    width : float
        The width of the line.
    style : str | None
        The matplotlib line style, or None for a solid line.
    """

    at: float
    vertical: bool
    width: float
    style: str | None = None


@dataclass(frozen=True)
class Chart:
    """
    One panel of a figure of a result.

    Attributes
    ----------
    name : str
        The title of the panel, and the label of its `Axes`.
    y_label : str
        The label of the y axis.
    marks : list[Line | Point | Bars]
        What the panel draws, in order.
    rules : list[Rule]
        The rules, drawn after the marks.
    x_label : str | None
        The label of the x axis, or None for none.
    ticks : list[str] | None
        The labels of the x ticks at 0, 1, 2, …, or None for the ticks of
        matplotlib.
    y_limits : tuple[float, float] | None
        The limits of the y axis, or None for those of matplotlib.
    legend : bool
        Whether the panel shows its legend.
    """

    name: str
    y_label: str
    marks: list["Line | Point | Bars"]
    rules: list[Rule] = field(default_factory=list)
    x_label: str | None = None
    ticks: list[str] | None = None
    y_limits: tuple[float, float] | None = None
    legend: bool = False
