"""SimulationPlotter draws the series, observations and bands it is given."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from commol import ObservedDataPoint, PlotConfig, Simulation
from commol.api.plotter import SimulationPlotter
from commol.context.calibration import CalibrationResult
from tests.support import a_to_b_builder, build

BINS = ["S", "I", "R"]
TITLES = {"S": "Susceptible", "I": "Infected", "R": "Recovered"}


@pytest.fixture
def plotter(sir_simulation: Simulation) -> SimulationPlotter:
    """A plotter of 100 steps of the default SIR model."""
    return SimulationPlotter(sir_simulation, sir_simulation.run(100))


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


def _observed_infected(results: dict[str, list[float]]) -> list[ObservedDataPoint]:
    return [
        ObservedDataPoint(step=step, compartment="I", value=results["I"][step])
        for step in range(0, 100, 10)
    ]


class TestSimulationPlotter:
    def test_plotter_initialization(self, sir_simulation: Simulation) -> None:
        results = sir_simulation.run(100)

        plotter = SimulationPlotter(sir_simulation, results)

        assert plotter.simulation == sir_simulation
        assert plotter.results == results
        assert len(plotter.bins) == 3
        assert plotter.num_steps == 100

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
        observed = _observed_infected(plotter.results)

        fig = plotter.plot_series(observed_data=observed)

        markers = _scatter(_visible(fig)[1])
        np.testing.assert_allclose(markers[:, 0], [p.step for p in observed])
        np.testing.assert_allclose(markers[:, 1], [p.value for p in observed])

    def test_plot_series_with_calibration_result(
        self, plotter: SimulationPlotter
    ) -> None:
        results = plotter.results
        scale_factor = 0.5
        observed = [
            ObservedDataPoint(
                step=i,
                compartment="I",
                value=results["I"][i] * scale_factor,
                scale_id="detection_rate",
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
            series=["I"],
        )

        ax = fig.axes[0]
        plotted_y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        assert plotted_y[10] == pytest.approx(results["I"][10] * scale_factor)
        assert _scatter(ax)[1, 1] == pytest.approx(results["I"][10] * scale_factor)

    def test_plot_series_subset_of_bins(self, plotter: SimulationPlotter) -> None:
        fig = plotter.plot_series(series=["I", "R"])

        _assert_draws(fig, {name: plotter.results[name] for name in ("I", "R")})

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
        observed = _observed_infected(plotter.results)

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
            ObservedDataPoint(step=10, compartment="I", value=50.0),
            ObservedDataPoint(step=20, compartment="I", value=60.0),
            ObservedDataPoint(step=10, compartment="R", value=10.0),
        ]

        grouped = plotter._group_observed_data(observed)

        assert {name: [p.step for p in points] for name, points in grouped.items()} == {
            "I": [10, 20],
            "R": [10],
        }

    def test_apply_x_labels_anchors_ticks_to_first_plotted_step(
        self, plotter: SimulationPlotter
    ) -> None:
        """Ticks follow the series' own grid, not the absolute step 0."""
        _fig, ax = plt.subplots()

        # A windowed series: plotted steps are window ends, starting at the
        # window width rather than at 0.
        window = 7
        time_steps = list(range(window, 8 * window + 1, window))
        plotter._apply_x_labels(ax, time_steps, str, tick_every=2 * window)

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


class TestWindowedProbabilisticSeries:
    """
    Windowed ensemble bands are plotted against the steps they were taken at.

    `windowed_prediction_*` is anchored at the observation steps, which need
    not be every multiple of the window, so pairing it with a grid of multiples
    would mismatch the axis.
    """

    def _model(self):
        return build(
            a_to_b_builder(parameters={"k1": 0.05}, rate="k1", accumulator="events")
        )

    def test_plots_when_observations_are_sparser_than_the_window_grid(self) -> None:
        from commol.context.probabilistic_calibration import (
            EnsembleSolution,
            ProbabilisticCalibrationResult,
        )

        window = 5
        simulation = Simulation(self._model())
        results = simulation.run(40)
        # Only three windows are observed, while the grid of multiples has eight.
        steps = [10, 25, 40]
        observed = [
            ObservedDataPoint(
                step=step,
                compartment="events",
                value=results["events"][step] - results["events"][step - window],
                window_steps=window,
            )
            for step in steps
        ]
        windowed = [
            results["events"][step] - results["events"][step - window] for step in steps
        ]
        ensemble = EnsembleSolution(
            ensemble_size=1,
            selected_indices=[0],
            ensemble_parameters=[{"k1": 0.05}],
            parameter_statistics={},
            prediction_median={k: list(v) for k, v in results.items()},
            prediction_ci_lower={k: list(v) for k, v in results.items()},
            prediction_ci_upper={k: list(v) for k, v in results.items()},
            windowed_prediction_steps={"events": steps},
            windowed_prediction_median={"events": windowed},
            windowed_prediction_ci_lower={"events": windowed},
            windowed_prediction_ci_upper={"events": windowed},
            coverage_percentage=100.0,
            average_ci_width=0.0,
            ci_width=0.0,
            coverage=1.0,
            point_parameters={"k1": 0.05},
            point_loss=0.0,
            central_loss=0.0,
            observation_diagnostics={},
            selection_diagnostics={},
        )
        result = ProbabilisticCalibrationResult(
            selected_ensemble=ensemble,
            selection_algorithm="nsga2",
            n_runs_performed=1,
            n_unique_evaluations=1,
            n_clusters_used=1,
            confidence_level=0.95,
        )

        plotter = SimulationPlotter(simulation, results)
        figure = plotter.plot_series(
            observed_data=observed, calibration_result=result, series=["events"]
        )

        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == steps
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            windowed
        )


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
