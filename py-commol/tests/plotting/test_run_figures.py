"""The figures of a run: ``results.plot.series()`` and
``results.plot.cumulative()``, their panels, sizes, ticks and dates."""

from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from commol import ObservedDataPoint, Simulation
from commol._simulation.results import SimulationResults
from commol.plotting import PlotConfig, figures
from commol.plotting.panels import _group_observed
from tests.support import (
    abc_builder,
    build,
)
from tests.support.plotted import (
    BINS,
    TITLES,
    dated_run,
    fit,
    scatter,
    tick_labels,
    visible,
)


def _assert_draws(fig: Figure, expected: dict[str, list[float]]) -> None:
    """One visible panel per series, titled by its bin, with its values."""
    axes = visible(fig)
    assert [ax.get_title() for ax in axes] == [TITLES[name] for name in expected]
    for ax, values in zip(axes, expected.values(), strict=True):
        np.testing.assert_allclose(np.asarray(ax.lines[0].get_ydata()), values)


def _observed_b(results: dict[str, list[float]]) -> list[ObservedDataPoint]:
    return [
        ObservedDataPoint(at=step, output="B", value=results["B"][step])
        for step in range(0, 100, 10)
    ]


class TestRunFigures:
    def test_plot_series_draws_every_bin(self, run: SimulationResults) -> None:
        fig = run.plot.series()

        _assert_draws(fig, {name: run[name] for name in BINS})

    def test_plot_series_draws_the_observations(self, run: SimulationResults) -> None:
        observed = _observed_b(run)

        fig = run.plot.series(observed_data=observed)

        ax = visible(fig)[1]
        markers = scatter(ax)
        np.testing.assert_allclose(markers[:, 0], [p.at for p in observed])
        np.testing.assert_allclose(markers[:, 1], [p.value for p in observed])
        (points,) = ax.collections
        np.testing.assert_allclose(points.get_facecolor(), [[1.0, 0.0, 0.0, 0.7]])
        assert (list(points.get_sizes()), points.get_zorder()) == ([30.0], 5)

    def test_plot_series_with_calibration_result(self, run: SimulationResults) -> None:
        scale_factor = 0.5
        observed = [
            ObservedDataPoint(
                at=i,
                output="B",
                value=run["B"][i] * scale_factor,
                factor_id="detection_rate",
            )
            for i in range(0, 100, 10)
        ]

        fig = run.plot.series(
            observed_data=observed,
            calibration_result=fit(detection_rate=scale_factor),
            series=["B"],
        )

        ax = fig.axes[0]
        plotted_y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        assert plotted_y[10] == pytest.approx(run["B"][10] * scale_factor)
        assert scatter(ax)[1, 1] == pytest.approx(run["B"][10] * scale_factor)

    def test_plot_series_subset_of_bins(self, run: SimulationResults) -> None:
        fig = run.plot.series(series=["B", "C"])

        _assert_draws(fig, {name: run[name] for name in ("B", "C")})

    def test_plot_series_with_custom_config(self, run: SimulationResults) -> None:
        config = PlotConfig(
            figsize=(12, 8),
            dpi=150,
            layout=(1, 3),
            style=["seaborn-v0_8-whitegrid", "seaborn-v0_8-notebook"],
        )

        fig = run.plot.series(config=config)

        assert len(fig.axes) == 3
        assert fig.dpi == 150
        _assert_draws(fig, {name: run[name] for name in BINS})

    def test_plot_cumulative_draws_the_running_sums(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.cumulative()

        _assert_draws(fig, {name: list(np.cumsum(run[name])) for name in BINS})

    def test_plot_cumulative_draws_the_running_sum_of_observations(
        self, run: SimulationResults
    ) -> None:
        observed = _observed_b(run)

        fig = run.plot.cumulative(observed_data=observed)

        markers = scatter(visible(fig)[1])
        np.testing.assert_allclose(
            markers[:, 1], np.cumsum([p.value for p in observed])
        )

    def test_a_cumulative_plot_has_the_size_of_its_configuration(
        self, run: SimulationResults
    ) -> None:
        config = PlotConfig(figsize=(10, 6), dpi=80, layout=(1, 4))

        fig = run.plot.cumulative(config=config)

        assert tuple(fig.get_size_inches()) == (10, 6)
        assert fig.dpi == 80
        assert (len(fig.axes), len(visible(fig))) == (4, 3)

    def test_a_series_plot_has_the_size_of_its_panels(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.series(config=PlotConfig(layout=(2, 2)))

        assert tuple(fig.get_size_inches()) == (9.0, 8.0)
        assert (len(fig.axes), len(visible(fig))) == (4, 3)

    @pytest.mark.parametrize("method", ["series", "cumulative"])
    def test_a_plot_of_one_series_has_one_panel(
        self, run: SimulationResults, method: str
    ) -> None:
        fig = getattr(run.plot, method)(series=["B"])

        assert [ax.get_title() for ax in fig.axes] == ["State B"]

    @pytest.mark.parametrize(
        ("method", "name"),
        [("series", "Series"), ("cumulative", "Cumulative")],
    )
    def test_a_saved_plot_is_cropped_and_logged(
        self,
        run: SimulationResults,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        method: str,
        name: str,
    ) -> None:
        path = tmp_path / "plot.png"
        uncropped = tmp_path / "uncropped.png"

        with caplog.at_level("INFO", logger="commol"):
            fig = getattr(run.plot, method)(output_file=str(path))
        fig.savefig(uncropped, dpi=fig.dpi)

        assert f"{name} figure saved to {path}" in caplog.messages
        assert plt.imread(path).shape != plt.imread(uncropped).shape

    def test_a_cumulative_plot_labels_its_line_and_its_observations(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.cumulative(observed_data=_observed_b(run))

        ax = visible(fig)[1]
        (markers,) = [c for c in ax.collections if len(c.get_offsets()) == 10]
        np.testing.assert_allclose(scatter(ax)[:, 0], range(0, 100, 10))
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == (
            "Step",
            "",
            "State B",
        )
        grid = ax.xaxis.get_gridlines()[0]
        assert (grid.get_visible(), grid.get_alpha()) == (True, 0.3)
        assert ax.lines[0].get_label() == "Simulation (Cumulative)"
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "Simulation (Cumulative)",
            "Observed (Cumulative)",
        ]
        np.testing.assert_allclose(markers.get_facecolor(), [[1.0, 0.0, 0.0, 0.7]])
        assert (list(markers.get_sizes()), markers.get_zorder()) == ([30.0], 5)

    def test_a_cumulative_plot_labels_the_unit_of_its_series(self) -> None:
        run = Simulation(build(abc_builder(bin_unit="item"))).run(5)

        fig = run.plot.cumulative()

        assert [ax.get_ylabel() for ax in visible(fig)] == ["item"] * 3

    def test_cumulative_observations_of_a_factor_are_divided_by_it(
        self, run: SimulationResults
    ) -> None:
        observed = [
            ObservedDataPoint(at=step, output="B", value=10.0, factor_id="rate")
            for step in (10, 20)
        ] + [
            ObservedDataPoint(at=30, output="B", value=10.0),
            ObservedDataPoint(at=40, output="B", value=10.0, factor_id="other"),
        ]

        fig = run.plot.cumulative(
            observed_data=observed, calibration_result=fit(rate=0.5), series=["B"]
        )

        np.testing.assert_allclose(scatter(fig.axes[0])[:, 1], [20.0, 40.0, 50.0, 60.0])

    def test_layout(self) -> None:
        assert figures._layout(1) == (1, 1)
        assert figures._layout(2) == (1, 2)
        assert figures._layout(3) == (2, 2)
        assert figures._layout(4) == (2, 2)
        assert figures._layout(5) == (2, 3)
        assert figures._layout(9) == (3, 3)

    def test_group_observed_data(self) -> None:
        observed = [
            ObservedDataPoint(at=20, output="B", value=60.0),
            ObservedDataPoint(at=10, output="B", value=50.0),
            ObservedDataPoint(at=10, output="C", value=10.0),
        ]

        grouped = _group_observed(observed)

        assert {name: [p.at for p in points] for name, points in grouped.items()} == {
            "B": [10, 20],
            "C": [10],
        }

    def test_apply_x_labels_anchors_ticks_to_first_plotted_step(self) -> None:
        """Ticks follow the series' own grid, not the absolute step 0."""
        _fig, ax = plt.subplots()

        # A series read at its observations: the plotted steps are their
        # positions, which start at 7 rather than at 0.
        window = 7
        positions = list(range(window, 8 * window + 1, window))
        figures._apply_x_labels(ax, positions, str, tick_every=2 * window)

        ticks = list(ax.get_xticks())
        assert ticks == [window, 3 * window, 5 * window, 7 * window]

    def test_apply_x_labels_without_tick_every_labels_all_steps(self) -> None:
        _fig, ax = plt.subplots()

        figures._apply_x_labels(ax, [5, 10, 15], lambda p: f"day {p}", tick_every=None)

        assert list(ax.get_xticks()) == [5, 10, 15]
        labels = ax.get_xticklabels()
        assert [label.get_text() for label in labels] == ["day 5", "day 10", "day 15"]
        assert {(label.get_rotation(), label.get_ha()) for label in labels} == {
            (45.0, "right")
        }

    def test_apply_x_labels_with_empty_series(self) -> None:
        """An empty series produces no ticks instead of an index error."""
        _fig, ax = plt.subplots()

        figures._apply_x_labels(ax, [], str, tick_every=4)

        assert list(ax.get_xticks()) == []

    def test_plot_series_passes_keywords_to_the_lines(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.series(linewidth=3, alpha=0.7, linestyle="--")

        for ax in visible(fig):
            line = ax.lines[0]
            assert (line.get_linewidth(), line.get_alpha(), line.get_linestyle()) == (
                3.0,
                0.7,
                "--",
            )

    def test_plot_cumulative_passes_keywords_to_the_lines(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.cumulative(linewidth=2, alpha=0.8)

        for ax in visible(fig):
            assert (ax.lines[0].get_linewidth(), ax.lines[0].get_alpha()) == (2.0, 0.8)

    def test_a_series_figure_without_legend_has_none(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.series(show_legend=False)

        assert {ax.get_legend() for ax in visible(fig)} == {None}

    @pytest.mark.parametrize(
        ("preset", "expected"),
        [
            (
                PlotConfig.presentation,
                PlotConfig(
                    figsize=(16, 10),
                    dpi=150,
                    style=[
                        "seaborn-v0_8-whitegrid",
                        "seaborn-v0_8-talk",
                        "seaborn-v0_8-colorblind",
                    ],
                ),
            ),
            (
                PlotConfig.publication,
                PlotConfig(
                    figsize=(8, 6),
                    dpi=300,
                    style=["seaborn-v0_8-ticks", "seaborn-v0_8-paper"],
                ),
            ),
            (
                PlotConfig.notebook,
                PlotConfig(
                    figsize=(12, 8),
                    dpi=100,
                    style=["seaborn-v0_8-whitegrid", "seaborn-v0_8-notebook"],
                ),
            ),
        ],
    )
    def test_plot_config_presets(
        self, run: SimulationResults, preset, expected: PlotConfig
    ) -> None:
        config = preset()

        fig = run.plot.series(config=config)

        assert config == expected
        assert fig.dpi == expected.dpi


def test_an_axis_with_origin_labels_ticks_with_dates() -> None:
    fig = dated_run("day", 40).plot.series(series=["B"])

    labels = tick_labels(fig)
    assert labels
    for tick, label in labels.items():
        assert label == (date(2020, 1, 1) + timedelta(days=tick)).isoformat()


def test_dates_label_the_plotted_positions_tick_every_apart() -> None:
    fig = dated_run("day", 20).plot.series(series=["B"], tick_every=7)

    assert tick_labels(fig) == {
        0.0: "2020-01-01",
        7.0: "2020-01-08",
        14.0: "2020-01-15",
    }


def test_a_date_label_shows_the_time_away_from_midnight() -> None:
    fig = dated_run("hour", 30).plot.series(series=["B"], tick_every=6)

    assert tick_labels(fig) == {
        0.0: "2020-01-01",
        6.0: "2020-01-01 06:00:00",
        12.0: "2020-01-01 12:00:00",
        18.0: "2020-01-01 18:00:00",
        24.0: "2020-01-02",
        30.0: "2020-01-02 06:00:00",
    }


def test_a_position_label_wins_over_the_dates() -> None:
    fig = dated_run("day", 20).plot.series(
        series=["B"], position_label=lambda at: f"p{at:g}", tick_every=10
    )

    assert tick_labels(fig) == {0.0: "p0", 10.0: "p10", 20.0: "p20"}
