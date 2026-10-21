"""SimulationPlotter draws the series, observations and bands it is given."""

from datetime import date, timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParameterSample,
    PlotConfig,
    Predictive,
    SampleMethod,
    Simulation,
)
from commol.api.plotter import SimulationPlotter
from commol.context.calibration import CalibrationResult
from tests.support import (
    a_to_b_builder,
    abc_builder,
    bounded,
    build,
    build_ode,
    build_stochastic,
)

BINS = ["A", "B", "C"]
TITLES = {"A": "State A", "B": "State B", "C": "State C"}


@pytest.fixture
def plotter(abc_simulation: Simulation) -> SimulationPlotter:
    """A plotter of 100 steps of the default model of ``abc_builder``."""
    return SimulationPlotter(abc_simulation, abc_simulation.run(100))


def _visible(fig: Figure):
    return [ax for ax in fig.axes if ax.get_visible()]


def _assert_draws(fig: Figure, expected: dict[str, list[float]]) -> None:
    """One visible subplot per series, titled by its bin, with its values."""
    axes = _visible(fig)
    assert [ax.get_title() for ax in axes] == [TITLES[name] for name in expected]
    for ax, values in zip(axes, expected.values(), strict=True):
        np.testing.assert_allclose(np.asarray(ax.lines[0].get_ydata()), values)


def _scatter(ax) -> np.ndarray:
    """The offsets of the observation markers of a subplot."""
    offsets = [np.asarray(c.get_offsets(), dtype=float) for c in ax.collections]
    return offsets[int(np.argmax([len(o) for o in offsets]))]


def _observed_b(results: dict[str, list[float]]) -> list[ObservedDataPoint]:
    return [
        ObservedDataPoint(at=step, output="B", value=results["B"][step])
        for step in range(0, 100, 10)
    ]


class TestSimulationPlotter:
    def test_plot_series_draws_every_bin(self, plotter: SimulationPlotter) -> None:
        fig = plotter.plot_series()

        _assert_draws(fig, {name: plotter.results[name] for name in BINS})

    def test_plot_series_saves_to_file(
        self, plotter: SimulationPlotter, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "series_plot.png"

        plotter.plot_series(output_file=str(output_path))

        assert output_path.stat().st_size > 0

    def test_plot_series_draws_the_observations(
        self, plotter: SimulationPlotter
    ) -> None:
        observed = _observed_b(plotter.results)

        fig = plotter.plot_series(observed_data=observed)

        markers = _scatter(_visible(fig)[1])
        np.testing.assert_allclose(markers[:, 0], [p.at for p in observed])
        np.testing.assert_allclose(markers[:, 1], [p.value for p in observed])

    def test_plot_series_with_calibration_result(
        self, plotter: SimulationPlotter
    ) -> None:
        results = plotter.results
        scale_factor = 0.5
        observed = [
            ObservedDataPoint(
                at=i,
                output="B",
                value=results["B"][i] * scale_factor,
                factor_id="detection_rate",
            )
            for i in range(0, 100, 10)
        ]
        calibration_result = CalibrationResult(
            best_parameters={"detection_rate": scale_factor},
            final_loss=0.0,
            iterations=1,
            converged=True,
            termination_reason="Test",
        )

        fig = plotter.plot_series(
            observed_data=observed,
            calibration_result=calibration_result,
            series=["B"],
        )

        ax = fig.axes[0]
        plotted_y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        assert plotted_y[10] == pytest.approx(results["B"][10] * scale_factor)
        assert _scatter(ax)[1, 1] == pytest.approx(results["B"][10] * scale_factor)

    def test_plot_series_subset_of_bins(self, plotter: SimulationPlotter) -> None:
        fig = plotter.plot_series(series=["B", "C"])

        _assert_draws(fig, {name: plotter.results[name] for name in ("B", "C")})

    def test_plot_series_with_custom_config(self, plotter: SimulationPlotter) -> None:
        config = PlotConfig(
            figsize=(12, 8),
            dpi=150,
            layout=(1, 3),
            style="whitegrid",
            palette="Set2",
            context="notebook",
        )

        fig = plotter.plot_series(config=config)

        assert len(fig.axes) == 3
        assert fig.dpi == 150
        _assert_draws(fig, {name: plotter.results[name] for name in BINS})

    def test_plot_cumulative_draws_the_running_sums(
        self, plotter: SimulationPlotter
    ) -> None:
        fig = plotter.plot_cumulative()

        _assert_draws(
            fig, {name: list(np.cumsum(plotter.results[name])) for name in BINS}
        )

    def test_plot_cumulative_saves_to_file(
        self, plotter: SimulationPlotter, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "cumulative_plot.png"

        plotter.plot_cumulative(output_file=str(output_path))

        assert output_path.stat().st_size > 0

    def test_plot_cumulative_draws_the_running_sum_of_observations(
        self, plotter: SimulationPlotter
    ) -> None:
        observed = _observed_b(plotter.results)

        fig = plotter.plot_cumulative(observed_data=observed)

        markers = _scatter(_visible(fig)[1])
        np.testing.assert_allclose(
            markers[:, 1], np.cumsum([p.value for p in observed])
        )

    def test_a_cumulative_plot_has_the_size_of_its_configuration(
        self, plotter: SimulationPlotter
    ) -> None:
        config = PlotConfig(figsize=(10, 6), dpi=80, layout=(1, 4))

        fig = plotter.plot_cumulative(config=config)

        assert tuple(fig.get_size_inches()) == (10, 6)
        assert fig.dpi == 80
        assert (len(fig.axes), len(_visible(fig))) == (4, 3)

    def test_a_series_plot_has_the_size_of_its_subplots(
        self, plotter: SimulationPlotter
    ) -> None:
        fig = plotter.plot_series(config=PlotConfig(layout=(2, 2)))

        assert tuple(fig.get_size_inches()) == (9.0, 8.0)
        assert (len(fig.axes), len(_visible(fig))) == (4, 3)

    @pytest.mark.parametrize("method", ["plot_series", "plot_cumulative"])
    def test_a_plot_of_one_series_has_one_subplot(
        self, plotter: SimulationPlotter, method: str
    ) -> None:
        fig = getattr(plotter, method)(series=["B"])

        assert [ax.get_title() for ax in fig.axes] == ["State B"]

    @pytest.mark.parametrize(
        ("method", "name"),
        [("plot_series", "Series"), ("plot_cumulative", "Cumulative")],
    )
    def test_a_saved_plot_is_cropped_and_logged(
        self,
        plotter: SimulationPlotter,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        method: str,
        name: str,
    ) -> None:
        path = tmp_path / "plot.png"
        uncropped = tmp_path / "uncropped.png"

        with caplog.at_level("INFO", logger="commol"):
            fig = getattr(plotter, method)(output_file=str(path))
        fig.savefig(uncropped, dpi=fig.dpi)

        assert caplog.messages[0] == f"Starting {method}"
        assert f"{name} plot saved to {path}" in caplog.messages
        assert plt.imread(path).shape != plt.imread(uncropped).shape

    def test_a_cumulative_plot_labels_its_line_and_its_observations(
        self, plotter: SimulationPlotter
    ) -> None:
        fig = plotter.plot_cumulative(observed_data=_observed_b(plotter.results))

        ax = _visible(fig)[1]
        (markers,) = [c for c in ax.collections if len(c.get_offsets()) == 10]
        np.testing.assert_allclose(_scatter(ax)[:, 0], range(0, 100, 10))
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
        simulation = Simulation(build(abc_builder(bin_unit="item")))

        fig = SimulationPlotter(simulation, simulation.run(5)).plot_cumulative()

        assert [ax.get_ylabel() for ax in _visible(fig)] == ["item"] * 3

    def test_cumulative_observations_of_a_factor_are_divided_by_it(
        self, plotter: SimulationPlotter
    ) -> None:
        observed = [
            ObservedDataPoint(at=step, output="B", value=10.0, factor_id="rate")
            for step in (10, 20)
        ] + [
            ObservedDataPoint(at=30, output="B", value=10.0),
            ObservedDataPoint(at=40, output="B", value=10.0, factor_id="other"),
        ]
        result = CalibrationResult(
            best_parameters={"rate": 0.5},
            final_loss=0.0,
            iterations=1,
            converged=True,
            termination_reason="Test",
        )

        fig = plotter.plot_cumulative(
            observed_data=observed, calibration_result=result, series=["B"]
        )

        np.testing.assert_allclose(
            _scatter(fig.axes[0])[:, 1], [20.0, 40.0, 50.0, 60.0]
        )

    def test_calculate_layout(self, plotter: SimulationPlotter) -> None:
        assert plotter._calculate_layout(1) == (1, 1)
        assert plotter._calculate_layout(2) == (1, 2)
        assert plotter._calculate_layout(3) == (2, 2)
        assert plotter._calculate_layout(4) == (2, 2)
        assert plotter._calculate_layout(5) == (2, 3)
        assert plotter._calculate_layout(9) == (3, 3)

    def test_group_observed_data(self, plotter: SimulationPlotter) -> None:
        observed = [
            ObservedDataPoint(at=10, output="B", value=50.0),
            ObservedDataPoint(at=20, output="B", value=60.0),
            ObservedDataPoint(at=10, output="C", value=10.0),
        ]

        grouped = plotter._group_observed_data(observed)

        assert {name: [p.at for p in points] for name, points in grouped.items()} == {
            "B": [10, 20],
            "C": [10],
        }

    def test_apply_x_labels_anchors_ticks_to_first_plotted_step(
        self, plotter: SimulationPlotter
    ) -> None:
        """Ticks follow the series' own grid, not the absolute step 0."""
        _fig, ax = plt.subplots()

        # A series read at its observations: the plotted steps are their
        # positions, which start at 7 rather than at 0.
        window = 7
        positions = list(range(window, 8 * window + 1, window))
        plotter._apply_x_labels(ax, positions, str, tick_every=2 * window)

        ticks = list(ax.get_xticks())
        assert ticks == [window, 3 * window, 5 * window, 7 * window]

    def test_apply_x_labels_without_tick_every_labels_all_steps(
        self, plotter: SimulationPlotter
    ) -> None:
        _fig, ax = plt.subplots()

        plotter._apply_x_labels(ax, [5, 10, 15], lambda p: f"day {p}", tick_every=None)

        assert list(ax.get_xticks()) == [5, 10, 15]
        labels = ax.get_xticklabels()
        assert [label.get_text() for label in labels] == ["day 5", "day 10", "day 15"]
        assert {(label.get_rotation(), label.get_ha()) for label in labels} == {
            (45.0, "right")
        }

    def test_apply_x_labels_with_empty_series(self, plotter: SimulationPlotter) -> None:
        """An empty series produces no ticks instead of an index error."""
        _fig, ax = plt.subplots()

        plotter._apply_x_labels(ax, [], str, tick_every=4)

        assert list(ax.get_xticks()) == []

    def test_plot_series_passes_keywords_to_the_lines(
        self, plotter: SimulationPlotter
    ) -> None:
        fig = plotter.plot_series(linewidth=3, alpha=0.7, linestyle="--")

        for ax in _visible(fig):
            line = ax.lines[0]
            assert (line.get_linewidth(), line.get_alpha(), line.get_linestyle()) == (
                3.0,
                0.7,
                "--",
            )

    def test_plot_cumulative_passes_keywords_to_the_lines(
        self, plotter: SimulationPlotter
    ) -> None:
        fig = plotter.plot_cumulative(linewidth=2, alpha=0.8)

        for ax in _visible(fig):
            assert (ax.lines[0].get_linewidth(), ax.lines[0].get_alpha()) == (2.0, 0.8)

    @pytest.mark.parametrize(
        ("preset", "expected"),
        [
            (
                PlotConfig.presentation,
                PlotConfig(
                    figsize=(16, 10),
                    dpi=150,
                    style="whitegrid",
                    context="talk",
                    palette="colorblind",
                ),
            ),
            (
                PlotConfig.publication,
                PlotConfig(figsize=(8, 6), dpi=300, style="ticks", context="paper"),
            ),
            (
                PlotConfig.notebook,
                PlotConfig(
                    figsize=(12, 8), dpi=100, style="whitegrid", context="notebook"
                ),
            ),
        ],
    )
    def test_plot_config_presets(
        self, plotter: SimulationPlotter, preset, expected: PlotConfig
    ) -> None:
        config = preset()

        fig = plotter.plot_series(config=config)

        assert config == expected
        assert fig.dpi == expected.dpi


@pytest.mark.parametrize("draw", ["plot_series", "plot_cumulative"])
def test_a_figure_leaves_the_global_style(
    plotter: SimulationPlotter, default_style: dict[str, object], draw: str
) -> None:
    """
    The style, palette and context of the configuration are in the figure,
    and ``matplotlib.rcParams`` is as it was before the call.
    """
    config = PlotConfig(style="darkgrid", palette="Set2", context="poster")

    fig = getattr(plotter, draw)(config=config)

    assert {k for k, v in default_style.items() if mpl.rcParams[k] != v} == set()
    ax = _visible(fig)[0]
    assert ax.get_facecolor() == to_rgba(sns.axes_style("darkgrid")["axes.facecolor"])
    assert to_rgba(ax.lines[0].get_color()) == to_rgba(sns.color_palette("Set2")[0])
    poster = sns.plotting_context("poster")["axes.titlesize"]
    assert ax.title.get_fontsize() == poster


def _dated_plotter(step_unit: str, steps: int) -> SimulationPlotter:
    """A plotter of `steps` steps of a model whose axis starts on 2020-01-01."""
    model = build(abc_builder(step_unit=step_unit, origin=date(2020, 1, 1)))
    simulation = Simulation(model)
    return SimulationPlotter(simulation, simulation.run(steps))


def _tick_labels(fig: Figure) -> dict[float, str]:
    """The labelled ticks of the first subplot, after the figure is drawn."""
    fig.canvas.draw()
    ax = fig.axes[0]
    return {
        float(tick): label.get_text()
        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=True)
        if label.get_text()
    }


def test_an_axis_with_origin_labels_ticks_with_dates() -> None:
    fig = _dated_plotter("day", 40).plot_series(series=["B"])

    labels = _tick_labels(fig)
    assert labels
    for tick, label in labels.items():
        assert label == (date(2020, 1, 1) + timedelta(days=tick)).isoformat()


def test_dates_label_the_plotted_positions_tick_every_apart() -> None:
    fig = _dated_plotter("day", 20).plot_series(series=["B"], tick_every=7)

    assert _tick_labels(fig) == {
        0.0: "2020-01-01",
        7.0: "2020-01-08",
        14.0: "2020-01-15",
    }


def test_a_date_label_shows_the_time_away_from_midnight() -> None:
    fig = _dated_plotter("hour", 30).plot_series(series=["B"], tick_every=6)

    assert _tick_labels(fig) == {
        0.0: "2020-01-01",
        6.0: "2020-01-01 06:00:00",
        12.0: "2020-01-01 12:00:00",
        18.0: "2020-01-01 18:00:00",
        24.0: "2020-01-02",
        30.0: "2020-01-02 06:00:00",
    }


def test_a_position_label_wins_over_the_dates() -> None:
    fig = _dated_plotter("day", 20).plot_series(
        series=["B"], position_label=lambda at: f"p{at:g}", tick_every=10
    )

    assert _tick_labels(fig) == {0.0: "p0", 10.0: "p10", 20.0: "p20"}


class TestPredictiveSeries:
    """
    A predictive plots the weighted median and band of its draws, each read
    through the observation operator of the problem.
    """

    window = 5

    def _calibrator(self) -> Calibrator:
        model = build(
            a_to_b_builder(parameters={"k1": None}, rate="k1", accumulator="events")
        )
        problem = CalibrationProblem(
            observed_data=[
                ObservedDataPoint(at=10, output="events", value=1.0, window=5)
            ],
            parameters=[bounded("k1", 0.0, 0.5)],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )
        return Calibrator(Simulation(model), problem)

    def _predictive(self, values: list[float]) -> Predictive:
        sample = ParameterSample.equally_weighted(
            ["k1"], [(0.0, 0.5)], [[value] for value in values], SampleMethod(name="t")
        )
        return Predictive(self._calibrator(), sample, list(range(41)))

    def test_a_windowed_band_of_one_draw_is_its_windowed_run(self) -> None:
        predictive = self._predictive([0.05])
        run = predictive.runs()[0]
        observed = [
            ObservedDataPoint(at=at, output="events", value=1.0, window=self.window)
            for at in (10, 25, 40)
        ]
        plotter = SimulationPlotter(predictive.source.simulation, run)

        figure = plotter.plot_series(
            observed_data=observed, calibration_result=predictive, series=["events"]
        )

        ends = [10, 25, 40]
        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == ends
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [run["events"][end] - run["events"][end - 5] for end in ends]
        )

    def test_a_band_reads_each_point_through_its_own_window(self) -> None:
        predictive = self._predictive([0.05])
        run = predictive.runs()[0]
        observed = [
            ObservedDataPoint(at=at, output="events", value=1.0, window=window)
            for at, window in ((10, 5), (30, 20))
        ]
        plotter = SimulationPlotter(predictive.source.simulation, run)

        figure = plotter.plot_series(
            observed_data=observed, calibration_result=predictive, series=["events"]
        )

        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == [10, 30]
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [
                run["events"][10] - run["events"][5],
                run["events"][30] - run["events"][10],
            ]
        )

    def test_the_band_of_two_draws_spans_their_runs(self) -> None:
        predictive = self._predictive([0.02, 0.1])
        low, high = predictive.runs()
        plotter = SimulationPlotter(predictive.source.simulation, low)

        figure = plotter.plot_series(calibration_result=predictive, series=["B"])

        filled = [c for c in figure.axes[0].collections if c.get_paths()]
        band = filled[0].get_paths()[0].vertices[:, 1]
        assert band.min() == pytest.approx(min(low["B"]))
        assert band.max() == pytest.approx(max(high["B"]))

    @staticmethod
    def _factor_predictive(observed: list[ObservedDataPoint]) -> Predictive:
        """A predictive of one draw, ``k1 = 0.05`` and the factor ``f1 = 0.5``."""
        model = build(a_to_b_builder(parameters={"k1": None}, rate="k1"))
        problem = CalibrationProblem(
            observed_data=observed,
            parameters=[
                bounded("k1", 0.0, 0.5),
                bounded("f1", 0.1, 1.0, kind="factor"),
            ],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )
        sample = ParameterSample.equally_weighted(
            ["k1", "f1"],
            [(0.0, 0.5), (0.1, 1.0)],
            [[0.05, 0.5]],
            SampleMethod(name="t"),
        )
        return Predictive(
            Calibrator(Simulation(model), problem), sample, list(range(41))
        )

    def test_a_band_reads_each_point_through_its_own_factor(self) -> None:
        observed = [
            ObservedDataPoint(at=10, output="B", value=1.0, factor_id="f1"),
            ObservedDataPoint(at=20, output="B", value=1.0),
        ]
        predictive = self._factor_predictive(observed)
        run = predictive.runs()[0]
        plotter = SimulationPlotter(predictive.source.simulation, run)

        figure = plotter.plot_series(
            observed_data=observed, calibration_result=predictive, series=["B"]
        )

        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == [10, 20]
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [0.5 * run["B"][10], run["B"][20]]
        )

    def test_a_band_of_one_factor_covers_every_position(self) -> None:
        observed = [
            ObservedDataPoint(at=at, output="B", value=1.0, factor_id="f1")
            for at in (10, 20)
        ]
        predictive = self._factor_predictive(observed)
        run = predictive.runs()[0]
        plotter = SimulationPlotter(predictive.source.simulation, run)

        figure = plotter.plot_series(
            observed_data=observed, calibration_result=predictive, series=["B"]
        )

        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == run.positions
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [0.5 * value for value in run["B"]]
        )

    def test_a_band_labels_its_positions(self) -> None:
        predictive = self._predictive([0.05])
        plotter = SimulationPlotter(predictive.source.simulation, predictive.runs()[0])

        figure = plotter.plot_series(
            calibration_result=predictive,
            series=["B"],
            position_label=lambda at: f"day {at:g}",
            tick_every=20,
        )

        labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
        assert labels == ["day 0", "day 20", "day 40"]

    def test_a_cumulative_band_spans_the_running_totals_of_its_draws(self) -> None:
        predictive = self._predictive([0.02, 0.1])
        low, high = predictive.runs()
        plotter = SimulationPlotter(predictive.source.simulation, low)
        observed = [ObservedDataPoint(at=at, output="B", value=1.0) for at in (5, 9)]

        figure = plotter.plot_cumulative(
            observed_data=observed, calibration_result=predictive, series=["B"]
        )

        ax = figure.axes[0]
        band = [c for c in ax.collections if c.get_paths()][0]
        vertices = band.get_paths()[0].vertices[:, 1]
        assert vertices.max() == pytest.approx(float(np.cumsum(high["B"])[-1]))
        edges = band.get_paths()[0].vertices
        # The closed path repeats a vertex.
        at_the_end = sorted({round(float(y), 6) for x, y in edges if x == 40})
        assert at_the_end == pytest.approx(
            [float(np.cumsum(low["B"])[-1]), float(np.cumsum(high["B"])[-1])]
        )
        assert band.get_alpha() == 0.3
        assert "95% of the draws" in [t.get_text() for t in ax.get_legend().get_texts()]
        assert ax.lines[0].get_label() == "Median of the draws (cumulative)"
        np.testing.assert_allclose(_scatter(ax), [[5.0, 1.0], [9.0, 2.0]])
        assert ax.get_title() == "B"


class TestObservationOperator:
    """The plotted line is the series as the observations read it."""

    def test_a_sum_of_outputs_plots_their_sum(self, plotter: SimulationPlotter) -> None:
        observed = [ObservedDataPoint(at=10, outputs=["B", "C"], label="B", value=1.0)]

        fig = plotter.plot_series(observed_data=observed, series=["B"])

        expected = np.add(plotter.results["B"], plotter.results["C"])
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), expected)

    def test_a_sum_of_an_unknown_output_is_an_error(
        self, plotter: SimulationPlotter
    ) -> None:
        observed = [ObservedDataPoint(at=10, outputs=["B", "Z"], label="B", value=1.0)]

        with pytest.raises(KeyError, match=r"aggregate components: \['Z'\]"):
            plotter.plot_series(observed_data=observed, series=["B"])

    def test_a_factor_it_cannot_read_leaves_the_line_unscaled(
        self,
        plotter: SimulationPlotter,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        observed = [ObservedDataPoint(at=10, output="B", value=1.0, factor_id="f3")]
        result = CalibrationResult(
            best_parameters={"f1": 0.5},
            final_loss=0.0,
            iterations=1,
            converged=True,
            termination_reason="Test",
        )

        fig = plotter.plot_series(
            observed_data=observed, calibration_result=result, series=["B"]
        )

        np.testing.assert_allclose(
            fig.axes[0].lines[0].get_ydata(), plotter.results["B"]
        )
        assert "Factor parameter 'f3' not found" in caplog.text

    def test_points_of_different_factors_are_each_read_through_their_own(
        self, plotter: SimulationPlotter
    ) -> None:
        observed = [
            ObservedDataPoint(at=10, output="B", value=1.0, factor_id="f1"),
            ObservedDataPoint(at=20, output="B", value=1.0, factor_id="f2"),
            ObservedDataPoint(at=30, outputs=["B", "C"], label="B", value=1.0),
        ]
        result = CalibrationResult(
            best_parameters={"f1": 0.5, "f2": 2.0},
            final_loss=0.0,
            iterations=1,
            converged=True,
            termination_reason="Test",
        )

        fig = plotter.plot_series(
            observed_data=observed, calibration_result=result, series=["B"]
        )

        line = fig.axes[0].lines[0]
        results = plotter.results
        np.testing.assert_allclose(line.get_xdata(), [10, 20, 30])
        np.testing.assert_allclose(
            line.get_ydata(),
            [
                0.5 * results["B"][10],
                2.0 * results["B"][20],
                results["B"][30] + results["C"][30],
            ],
        )

    def test_the_band_of_windowed_replicates_spans_their_window_totals(self) -> None:
        simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))
        replicates = simulation.replicates(5, seed=2).run(20)
        observed = [
            ObservedDataPoint(at=at, output="B", value=1.0, window=5) for at in (10, 20)
        ]

        fig = SimulationPlotter(simulation, replicates).plot_series(
            observed_data=observed, series=["B"]
        )

        totals = [
            [run["B"][end] - run["B"][end - 5] for end in (10, 20)]
            for run in replicates
        ]
        band = [c for c in fig.axes[0].collections if c.get_paths()][0]
        vertices = band.get_paths()[0].vertices
        assert sorted({float(x) for x, _ in vertices}) == [10.0, 20.0]
        assert vertices[:, 1].min() >= np.min(totals) - 1e-9
        assert vertices[:, 1].max() <= np.max(totals) + 1e-9


class TestSeriesSelection:
    @pytest.fixture
    def stratified(self) -> tuple[Simulation, dict[str, list[float]]]:
        model = build(a_to_b_builder(stratifications={"group": {"g1": 0.3, "g2": 0.7}}))
        simulation = Simulation(model)
        return simulation, simulation.run(10)

    def test_a_bin_id_plots_the_sum_of_its_outputs(self, stratified) -> None:
        simulation, results = stratified

        fig = SimulationPlotter(simulation, results).plot_series(series=["B"])

        plotted = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)
        expected = np.add(results["B_g1"], results["B_g2"])
        np.testing.assert_allclose(plotted, expected)

    def test_an_output_name_plots_that_output(self, stratified) -> None:
        simulation, results = stratified

        fig = SimulationPlotter(simulation, results).plot_series(series=["B_g1"])

        plotted = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)
        np.testing.assert_allclose(plotted, results["B_g1"])

    def test_a_bin_id_in_a_cumulative_plot(self, stratified) -> None:
        simulation, results = stratified

        fig = SimulationPlotter(simulation, results).plot_cumulative(series=["B"])

        plotted = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)
        expected = np.cumsum(np.add(results["B_g1"], results["B_g2"]))
        np.testing.assert_allclose(plotted, expected)

    @pytest.mark.parametrize("method", ["plot_series", "plot_cumulative"])
    def test_an_unknown_id_is_a_clear_error(self, stratified, method: str) -> None:
        simulation, results = stratified
        plotter = SimulationPlotter(simulation, results)

        with pytest.raises(ValueError, match="'X' is neither an output"):
            getattr(plotter, method)(series=["X"])


class TestDifferentialModel:
    def test_observations_at_non_integer_steps_are_drawn_where_they_are(
        self,
    ) -> None:
        simulation = Simulation(build_ode(abc_builder()))
        steps = [0.5, 10.25, 40.75]
        at = simulation.run_at(steps)
        observed = [
            ObservedDataPoint(at=step, output="B", value=value)
            for step, value in zip(steps, at["B"], strict=True)
        ]

        fig = SimulationPlotter(simulation, simulation.run(50)).plot_series(
            observed_data=observed, series=["B"]
        )

        np.testing.assert_allclose(_scatter(fig.axes[0])[:, 0], steps)

    def _events(self) -> Simulation:
        return Simulation(
            build_ode(
                a_to_b_builder(parameters={"k1": 0.05}, rate="k1", accumulator="events")
            )
        )

    def test_a_window_of_non_whole_steps_draws_its_increments(self) -> None:
        simulation = self._events()
        ends = [5.0, 12.5, 20.0]
        observed = [
            ObservedDataPoint(at=end, output="events", value=1.0, window=2.5)
            for end in ends
        ]
        results = simulation.run_at([2.5 * j for j in range(9)])

        fig = SimulationPlotter(simulation, results).plot_series(
            observed_data=observed, series=["events"]
        )

        line = fig.axes[0].lines[0]
        np.testing.assert_allclose(line.get_xdata(), ends)
        np.testing.assert_allclose(
            line.get_ydata(),
            simulation.windowed_totals(results, ["events"], 2.5, ends),
        )

    def test_a_window_the_results_do_not_hold_is_an_error(self) -> None:
        simulation = self._events()
        observed = [ObservedDataPoint(at=5, output="events", value=1.0, window=2.5)]

        with pytest.raises(ValueError, match="no value at 2.5"):
            SimulationPlotter(simulation, simulation.run(20)).plot_series(
                observed_data=observed
            )


def test_each_observed_point_is_read_through_its_own_window_and_position() -> None:
    """
    The model line of a series sits on each observation: its value there is
    the observation read through its own window, as the loss reads it.
    """
    simulation = Simulation(
        build_ode(
            a_to_b_builder(parameters={"k1": 0.05}, rate="k1", accumulator="events")
        )
    )
    run_at_half_steps = simulation.run_at([0.5 * j for j in range(41)])
    observed = [
        ObservedDataPoint(at=7.5, output="events", value=10.0, window=7),
        ObservedDataPoint(at=17.5, output="events", value=12.0, window=14),
    ]

    fig = SimulationPlotter(simulation, run_at_half_steps).plot_series(
        observed_data=observed, series=["events"]
    )

    line = fig.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), [7.5, 17.5])
    np.testing.assert_allclose(
        line.get_ydata(),
        [
            simulation.windowed_totals(run_at_half_steps, ["events"], 7, [7.5])[0],
            simulation.windowed_totals(run_at_half_steps, ["events"], 14, [17.5])[0],
        ],
    )
