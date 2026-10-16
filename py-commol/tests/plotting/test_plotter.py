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
from tests.support import a_to_b_builder, abc_builder, bounded, build, build_ode

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
    def test_plotter_initialization(self, abc_simulation: Simulation) -> None:
        results = abc_simulation.run(100)

        plotter = SimulationPlotter(abc_simulation, results)

        assert plotter.simulation == abc_simulation
        assert plotter.results == results
        assert len(plotter.bins) == 3
        assert plotter.results.positions == [float(step) for step in range(101)]

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

    def test_plot_series_with_seaborn_config(self, plotter: SimulationPlotter) -> None:
        config = PlotConfig(style="dark", palette="husl", context="talk")

        fig = plotter.plot_series(config=config)

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

        plotter._apply_x_labels(ax, [5, 10, 15], str, tick_every=None)

        assert list(ax.get_xticks()) == [5, 10, 15]

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
        ("preset", "figsize", "dpi", "context"),
        [
            (PlotConfig.presentation, (16, 10), 150, "talk"),
            (PlotConfig.publication, (8, 6), 300, "paper"),
            (PlotConfig.notebook, (12, 8), 100, "notebook"),
        ],
    )
    def test_plot_config_presets(
        self, plotter: SimulationPlotter, preset, figsize, dpi, context
    ) -> None:
        config = preset()

        fig = plotter.plot_series(config=config)

        assert (config.figsize, config.dpi, config.context) == (figsize, dpi, context)
        assert fig.dpi == dpi


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
