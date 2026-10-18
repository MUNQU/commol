"""SimulationPlotter draws the series, observations and bands it is given."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
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

        # A windowed series: plotted steps are window ends, starting at the
        # window width rather than at 0.
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
        plotter = SimulationPlotter(predictive.calibrator.simulation, run)

        figure = plotter.plot_series(
            observed_data=observed, calibration_result=predictive, series=["events"]
        )

        ends = list(range(5, 41, 5))
        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == ends
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [run["events"][end] - run["events"][end - 5] for end in ends]
        )

    def test_the_band_of_two_draws_spans_their_runs(self) -> None:
        predictive = self._predictive([0.02, 0.1])
        low, high = predictive.runs()
        plotter = SimulationPlotter(predictive.calibrator.simulation, low)

        figure = plotter.plot_series(calibration_result=predictive, series=["B"])

        filled = [c for c in figure.axes[0].collections if c.get_paths()]
        band = filled[0].get_paths()[0].vertices[:, 1]
        assert band.min() == pytest.approx(min(low["B"]))
        assert band.max() == pytest.approx(max(high["B"]))

    def test_a_band_reads_one_factor(self) -> None:
        predictive = self._predictive([0.05])
        observed = [
            ObservedDataPoint(at=at, output="B", value=1.0, factor_id=factor_id)
            for at, factor_id in ((10, "f1"), (20, "f2"))
        ]
        plotter = SimulationPlotter(
            predictive.calibrator.simulation, predictive.runs()[0]
        )

        with pytest.raises(ValueError, match=r"several factors \['f1', 'f2'\]"):
            plotter.plot_series(
                observed_data=observed, calibration_result=predictive, series=["B"]
            )

    def test_a_band_labels_its_positions(self) -> None:
        predictive = self._predictive([0.05])
        plotter = SimulationPlotter(
            predictive.calibrator.simulation, predictive.runs()[0]
        )

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
        plotter = SimulationPlotter(predictive.calibrator.simulation, low)
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

    @pytest.mark.parametrize(
        ("factor_ids", "message"),
        [
            (["f1", "f2"], "Multiple factor_id values"),
            (["f3"], "Factor parameter 'f3' not found"),
        ],
    )
    def test_a_factor_it_cannot_read_leaves_the_line_unscaled(
        self,
        plotter: SimulationPlotter,
        caplog: pytest.LogCaptureFixture,
        factor_ids: list[str],
        message: str,
    ) -> None:
        observed = [
            ObservedDataPoint(at=10 * i, output="B", value=1.0, factor_id=factor_id)
            for i, factor_id in enumerate(factor_ids, start=1)
        ]
        result = CalibrationResult(
            best_parameters={"f1": 0.5, "f2": 0.5},
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
        assert message in caplog.text

    def test_the_band_of_windowed_replicates_spans_their_window_totals(self) -> None:
        simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))
        replicates = simulation.replicates(5, seed=2).run(20)
        observed = [ObservedDataPoint(at=10, output="B", value=1.0, window=5)]

        fig = SimulationPlotter(simulation, replicates).plot_series(
            observed_data=observed, series=["B"]
        )

        totals = [
            [run["B"][end] - run["B"][end - 5] for end in (5, 10, 15, 20)]
            for run in replicates
        ]
        band = [c for c in fig.axes[0].collections if c.get_paths()][0]
        vertices = band.get_paths()[0].vertices
        assert sorted({float(x) for x, _ in vertices}) == [5.0, 10.0, 15.0, 20.0]
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
        observed = [ObservedDataPoint(at=5, output="events", value=1.0, window=2.5)]
        results = simulation.run_at([2.5 * j for j in range(9)])

        fig = SimulationPlotter(simulation, results).plot_series(
            observed_data=observed, series=["events"]
        )

        line = fig.axes[0].lines[0]
        ends = [2.5 * j for j in range(1, 9)]
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
