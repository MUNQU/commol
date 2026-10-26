"""The figure of a `ProfileResult`.

`ProfileResult.plot` calls `profiles_figure`, which builds one chart per
profile (`commol.plotting.charts`) and draws them
(`commol.plotting.figures.chart_figure`); matplotlib is imported when the
figure is drawn.
"""

from typing import TYPE_CHECKING

from commol.plotting.charts import Chart, Line, Rule

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from commol.plotting.config import PlotConfig
    from commol.profiles.results import ProfileResult


def profiles_figure(
    result: "ProfileResult",
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    axes: "Figure | None" = None,
) -> "Figure":
    """The figure of `ProfileResult.plot`."""
    from commol.plotting.figures import chart_figure

    return chart_figure(profile_charts(result), output_file, config, axes, "Profiles")


def profile_charts(result: "ProfileResult") -> list[Chart]:
    """
    One chart per profile: its statistic over the values of its quantity, a
    dashed rule at its threshold and a dotted rule at each finite end of its
    interval.
    """
    charts = []
    for name, profile in result.profiles.items():
        interval = result.intervals[name]
        charts.append(
            Chart(
                name=name,
                y_label="Profile statistic",
                marks=[Line(list(profile.values), list(profile.statistics), size=3)],
                rules=[
                    Rule(profile.threshold, vertical=False, width=1, style="--"),
                    *(
                        Rule(end, vertical=True, width=1, style=":")
                        for end in (interval.lower, interval.upper)
                        if end is not None
                    ),
                ],
                x_label=name,
            )
        )
    return charts
