"""The figures of a predictive and of replicates: the band of the draws or of
the replicates, each read through the observation operator."""

import numpy as np
import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.uncertainty import ParameterSample, Predictive, SampleMethod
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    build_stochastic,
)
from tests.support.plotted import (
    predictive_of,
    scatter,
)


class TestPredictiveSeries:
    """
    A predictive draws the weighted median and band of its draws, each read
    through the observation operator of the problem.
    """

    window = 5

    def test_a_windowed_band_of_one_draw_is_its_windowed_run(self) -> None:
        predictive = predictive_of([0.05])
        run = predictive.runs()[0]
        observed = [
            ObservedDataPoint(at=at, output="events", value=1.0, window=self.window)
            for at in (10, 25, 40)
        ]

        figure = predictive.plot.series(observed_data=observed, series=["events"])

        ends = [10, 25, 40]
        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == ends
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [run["events"][end] - run["events"][end - 5] for end in ends]
        )

    def test_a_band_reads_each_point_through_its_own_window(self) -> None:
        predictive = predictive_of([0.05])
        run = predictive.runs()[0]
        observed = [
            ObservedDataPoint(at=at, output="events", value=1.0, window=window)
            for at, window in ((10, 5), (30, 20))
        ]

        figure = predictive.plot.series(observed_data=observed, series=["events"])

        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == [10, 30]
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [
                run["events"][10] - run["events"][5],
                run["events"][30] - run["events"][10],
            ]
        )

    def test_the_band_of_two_draws_spans_their_runs(self) -> None:
        predictive = predictive_of([0.02, 0.1])
        low, high = predictive.runs()

        figure = predictive.plot.series(series=["B"])

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

        figure = predictive.plot.series(observed_data=observed, series=["B"])

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

        figure = predictive.plot.series(observed_data=observed, series=["B"])

        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == run.positions
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [0.5 * value for value in run["B"]]
        )

    def test_a_band_labels_its_positions(self) -> None:
        predictive = predictive_of([0.05])

        figure = predictive.plot.series(
            series=["B"],
            position_label=lambda at: f"day {at:g}",
            tick_every=20,
        )

        labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
        assert labels == ["day 0", "day 20", "day 40"]

    def test_a_cumulative_band_spans_the_running_totals_of_its_draws(self) -> None:
        predictive = predictive_of([0.02, 0.1])
        low, high = predictive.runs()
        observed = [ObservedDataPoint(at=at, output="B", value=1.0) for at in (5, 9)]

        figure = predictive.plot.cumulative(observed_data=observed, series=["B"])

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
        np.testing.assert_allclose(scatter(ax), [[5.0, 1.0], [9.0, 2.0]])
        assert ax.get_title() == "B"


class TestReplicateFigures:
    @pytest.fixture
    def replicates(self):
        simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))
        return simulation.replicates(5, seed=2).run(20)

    def test_the_band_of_windowed_replicates_spans_their_window_totals(
        self, replicates
    ) -> None:
        observed = [
            ObservedDataPoint(at=at, output="B", value=1.0, window=5) for at in (10, 20)
        ]

        fig = replicates.plot.series(observed_data=observed, series=["B"])

        totals = [
            [run["B"][end] - run["B"][end - 5] for end in (10, 20)]
            for run in replicates
        ]
        band = [c for c in fig.axes[0].collections if c.get_paths()][0]
        vertices = band.get_paths()[0].vertices
        assert sorted({float(x) for x, _ in vertices}) == [10.0, 20.0]
        assert vertices[:, 1].min() >= np.min(totals) - 1e-9
        assert vertices[:, 1].max() <= np.max(totals) + 1e-9

    def test_the_cumulative_figure_of_replicates_is_that_of_their_mean(
        self, replicates
    ) -> None:
        fig = replicates.plot.cumulative(series=["B"])

        np.testing.assert_allclose(
            fig.axes[0].lines[0].get_ydata(), np.cumsum(replicates.mean()["B"])
        )
