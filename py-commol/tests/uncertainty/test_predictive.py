"""The predictive of a sample: runs, bands, observations, plots and reports."""

import math

import numpy as np
import pytest

from commol import (
    InfeasibleStateError,
    CalibrationProblem,
    NelderMeadConfig,
    NoNoise,
    ObservedDataPoint,
    Simulation,
    Calibrator,
    ParameterSample,
    Predictive,
    SampleMethod,
    SimulationPlotter,
    member_statistics,
    simulation_report,
)
from tests.support import a_to_b_builder, bounded, build, build_stochastic

POSITIONS = list(range(51))


def _sample(values: list[list[float]], weights: list[float] | None = None):
    return ParameterSample(
        names=["k1", "k2"],
        bounds=[(0.0, 0.5), (0.0, 0.3)],
        values=values,
        log_weights=[math.log(w) for w in weights or [1.0] * len(values)],
        method=SampleMethod(name="test"),
    )


class TestBands:
    @pytest.mark.parametrize(
        ("inflow", "value", "error", "message"),
        [
            ("0", 1.5, InfeasibleStateError, "Parameter set 0 is not feasible"),
            (
                "exp(10000 * k)",
                0.5,
                ValueError,
                "Parameter set 0 gives a value that is not finite",
            ),
        ],
        ids=["infeasible", "not finite"],
    )
    def test_a_draw_without_a_run_names_its_set(
        self, inflow: str, value: float, error: type[Exception], message: str
    ) -> None:
        builder = a_to_b_builder(parameters={"k": None}).add_transition(
            id="t_in", source=[], target=["B"], rate=inflow
        )
        model = build(builder)
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=1, output="A", value=900.0)],
            parameters=[bounded("k", 0.0, 2.0)],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )
        calibrator = Calibrator(Simulation(model), problem)
        sample = ParameterSample.equally_weighted(
            ["k"], [(0.0, 2.0)], [[value]], SampleMethod(name="t")
        )

        with pytest.raises(error, match=message):
            Predictive(calibrator, sample, [0, 1, 2])

    def test_a_band_of_an_unknown_output_names_the_outputs(
        self, calibrator: Calibrator
    ) -> None:
        predictive = Predictive(calibrator, _sample([[0.1, 0.05]]), POSITIONS)

        with pytest.raises(ValueError, match=r"Unknown outputs \['Z'\]. Outputs: "):
            predictive.series_band(["Z"], 0.9)

    def test_the_band_of_one_draw_is_that_draw(
        self, calibrator: Calibrator, truth_run: dict[str, list[float]]
    ) -> None:
        predictive = Predictive(calibrator, _sample([[0.1, 0.05]]), POSITIONS)
        run = truth_run

        band = predictive.band(0.9)

        for name in ("A", "B", "C"):
            assert band.lower[name] == pytest.approx(run[name])
            assert band.median[name] == pytest.approx(run[name])
            assert band.upper[name] == pytest.approx(run[name])

    def test_the_weighted_band_is_member_statistics_of_the_runs(
        self, calibrator: Calibrator
    ) -> None:
        weights = [0.2, 0.5, 0.3]
        sample = _sample([[0.08, 0.04], [0.1, 0.05], [0.13, 0.07]], weights)
        predictive = Predictive(calibrator, sample, POSITIONS)

        band = predictive.band(0.8)
        expected = member_statistics(
            [run["B"] for run in predictive.runs()], 0.8, predictive.weights
        )

        assert predictive.weights == pytest.approx(weights)
        assert band.lower["B"] == pytest.approx(expected["ci_lower"])
        assert band.median["B"] == pytest.approx(expected["median"])
        assert band.upper["B"] == pytest.approx(expected["ci_upper"])
        assert band.mean["B"] == pytest.approx(expected["mean"])

    def test_a_cumulative_band_reduces_each_draws_running_total(
        self, calibrator: Calibrator
    ) -> None:
        predictive = Predictive(
            calibrator, _sample([[0.08, 0.04], [0.13, 0.07]]), POSITIONS
        )

        band = predictive.series_band(["B", "C"], 0.9, cumulative=True)
        totals = [
            np.cumsum(np.add(run["B"], run["C"])).tolist() for run in predictive.runs()
        ]

        assert band.median == pytest.approx(member_statistics(totals, 0.9)["median"])

    def test_a_sample_of_other_parameters_is_rejected(
        self, calibrator: Calibrator
    ) -> None:
        sample = ParameterSample.equally_weighted(
            ["k2", "k1"],
            [(0.0, 0.3), (0.0, 0.5)],
            [[0.05, 0.1]],
            SampleMethod(name="t"),
        )

        with pytest.raises(ValueError, match="in this order"):
            Predictive(calibrator, sample, POSITIONS)


class TestObservations:
    def test_the_replicates_of_a_draw_share_its_weight(self) -> None:
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=5, output="A", value=600.0)],
            parameters=[bounded("k", 0.01, 0.5)],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
            replicates=4,
            replicate_summary="mean",
            seed=7,
        )
        model = build_stochastic(
            a_to_b_builder(parameters={"k": None}), "StochasticSteps"
        )
        calibrator = Calibrator(Simulation(model), problem)
        sample = ParameterSample(
            names=["k"],
            bounds=[(0.01, 0.5)],
            values=[[0.1], [0.2]],
            log_weights=[math.log(0.25), math.log(0.75)],
            method=SampleMethod(name="test"),
        )

        draws = Predictive(calibrator, sample, [5.0]).at_observations(
            problem.observed_data, NoNoise()
        )

        assert draws.weights == pytest.approx([0.0625] * 4 + [0.1875] * 4)

    def test_draws_read_observations_as_the_loss_does(
        self, calibrator: Calibrator
    ) -> None:
        sample = _sample([[0.1, 0.05], [0.12, 0.06]])
        predictive = Predictive(calibrator, sample, POSITIONS)

        draws = predictive.at_observations(calibrator.problem.observed_data, NoNoise())
        expected = calibrator.rust_problem.predict_observations(sample.values)

        np.testing.assert_allclose(draws.values, [per_set[0] for per_set in expected])

    def test_the_truth_predicts_the_data(
        self, calibrator: Calibrator, truth_run: dict[str, list[float]]
    ) -> None:
        predictive = Predictive(calibrator, _sample([[0.1, 0.05]]), POSITIONS)

        draws = predictive.at_observations(calibrator.problem.observed_data, NoNoise())

        assert draws.values[0] == pytest.approx(draws.observed)
        # The window of C from 20 to 40 is read as an increment.
        run = truth_run
        assert draws.values[0][-1] == pytest.approx(run["C"][40] - run["C"][20])


class TestReaders:
    def test_a_plot_draws_the_band_of_a_predictive(
        self, calibrator: Calibrator
    ) -> None:
        predictive = Predictive(
            calibrator, _sample([[0.08, 0.04], [0.13, 0.07]]), POSITIONS
        )
        plotter = SimulationPlotter(
            calibrator.simulation, predictive.runs()[0], confidence_level=0.9
        )

        figure = plotter.plot_series(calibration_result=predictive, series=["B"])

        median = figure.axes[0].lines[0].get_ydata()
        assert np.asarray(median) == pytest.approx(
            predictive.series_band(["B"], 0.9).median
        )

    def test_a_report_reduces_the_draws_with_their_weights(
        self, calibrator: Calibrator
    ) -> None:
        weights = [0.25, 0.75]
        predictive = Predictive(
            calibrator, _sample([[0.08, 0.04], [0.13, 0.07]], weights), POSITIONS
        )
        runs = predictive.runs()

        report = simulation_report(
            calibrator.simulation,
            runs[0],
            at=[10, 20, 30, 40, 50],
            window=10,
            num_windows=5,
            ensemble_runs=predictive,
            confidence_level=0.9,
        )

        leaf = report["series"]["B"]
        assert isinstance(leaf, dict)
        expected = member_statistics(
            [[run["B"][at] for at in (10, 20, 30, 40, 50)] for run in runs],
            0.9,
            weights,
        )
        assert leaf["mean"] == pytest.approx(expected["mean"])
