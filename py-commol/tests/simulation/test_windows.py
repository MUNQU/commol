"""Tests for reducing cumulative series to per-window increments, and for
reading series at positions."""

import math

import pytest

from collections.abc import Iterable, Sequence

from commol import (
    AxisUnits,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.api.plotting.panels import compared_panels
from commol.api.results import SimulationResults
from commol.commol_rs import _commol_rs as commol_rs
from tests.support import a_to_b_builder, bounded, build

SEED = 42
WINDOW = 5
NUM_STEPS = 40


def windowed_totals(
    series: Sequence[float],
    positions: Iterable[float],
    window: float,
    at: Iterable[float] | None = None,
) -> list[float]:
    """The windowed totals of one series, read by results without a model."""
    return SimulationResults(positions, {"s": series}).windowed_totals(
        ["s"], window, at
    )


def _model(k1: float | None = 0.05) -> Simulation:
    builder = a_to_b_builder(parameters={"k1": k1}, rate="k1", accumulator="events")
    return Simulation(build(builder))


def test_windows_come_from_the_engine() -> None:
    """Results, reports and figures read the windows that the engine computes
    (`crates/commol-calibration/src/observation.rs`), and a window that is
    not a finite, positive length gets the message of the engine."""
    engine = commol_rs.calibration
    results = _model().run(23)
    events = results["events"]
    ends = engine.window_ends(WINDOW, 23)

    report = results.report([23], WINDOW, len(ends), accumulators=["events"])
    windowed = report["accumulators"]["events"]["windowed"]["total"]
    [panel] = compared_panels({"run": results}, 0.9, series=["events"], window=WINDOW)

    assert ends == [5, 10, 15, 20]
    assert AxisUnits().window_ends(WINDOW, "step", 23) == ends
    assert [engine.window_bounds(i, WINDOW)[1] for i in range(4)] == ends
    assert windowed_totals(events, results.positions, WINDOW) == (
        windowed_totals(events, results.positions, WINDOW, ends)
    )
    assert windowed == windowed_totals(events, results.positions, WINDOW, ends)
    assert panel.lines[0].positions == ends
    readers = [
        lambda window: windowed_totals(events, results.positions, window),
        lambda window: windowed_totals(events, results.positions, window, [5]),
        lambda window: results.report([23], window, 1, accumulators=["events"]),
        lambda window: compared_panels({"run": results}, 0.9, window=window),
    ]
    for window in [0, -1, math.nan, math.inf]:
        with pytest.raises(ValueError) as message:
            engine.check_window(window)
        for read in readers:
            with pytest.raises(ValueError) as error:
                read(window)
            assert str(error.value) == str(message.value)


class TestWindowedTotals:
    def test_increments_of_a_cumulative_series(self) -> None:
        series = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]

        assert windowed_totals(series, range(6), 2) == [3.0, 7.0]

    def test_default_grid_drops_a_trailing_partial_window(self) -> None:
        series = [float(value) for value in range(12)]

        assert windowed_totals(series, range(12), 5) == [5.0, 5.0]

    def test_explicit_ends_need_not_be_multiples_of_the_window(self) -> None:
        series = [float(value) for value in range(12)]

        assert windowed_totals(series, range(12), 4, [7, 11]) == [4.0, 4.0]

    def test_a_series_is_read_at_its_positions(self) -> None:
        positions = [0.0, 0.5, 1.0, 1.5]
        series = [0.0, 2.0, 3.0, 7.0]

        assert windowed_totals(series, positions, 0.5) == [2.0, 1.0, 4.0]

    def test_end_without_a_complete_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no complete window"):
            windowed_totals([0.0, 1.0, 2.0], range(3), 5, [2])

    def test_a_window_may_start_before_zero_by_the_tolerance_of_its_end(
        self,
    ) -> None:
        series = [float(value) for value in range(1001)]

        assert windowed_totals(series, range(1001), 1000 + 1e-7, [1000]) == [1000.0]
        assert windowed_totals(series[:2], range(2), 1 + 0.5e-9, [1]) == [1.0]
        with pytest.raises(ValueError) as error:
            windowed_totals(series[:2], range(2), 1 + 1.5e-9, [1])
        assert str(error.value) == (
            f"Position 1 has no complete window of {1 + 1.5e-9} before it."
        )

    def test_a_series_has_one_value_per_position(self) -> None:
        with pytest.raises(ValueError, match="2 values for 3 positions"):
            windowed_totals([0.0, 1.0], range(3), 1)

    def test_an_empty_series_has_no_windows(self) -> None:
        assert windowed_totals([], [], 1) == []

    def test_end_beyond_the_series_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no value at 9"):
            windowed_totals([0.0, 1.0, 2.0], range(3), 1, [9])

    def test_results_window_the_total_of_their_accumulators(self) -> None:
        results = _model().run(NUM_STEPS)

        by_hand = windowed_totals(results["events"], results.positions, WINDOW)
        at = [7, 12]

        assert results.windowed_totals(["events"], WINDOW) == by_hand
        assert results.windowed_totals(["events"], WINDOW, at) == (
            windowed_totals(results["events"], results.positions, WINDOW, at)
        )


class TestAgreementWithTheCalibrationLoss:
    """
    The reduction must return the values a calibration compares against.

    Observations built from `windowed_totals` of a known run are reproduced
    exactly when the model is simulated with the parameters that produced them,
    so the loss of that parameter set is zero.
    """

    def _observations(self, simulation: Simulation) -> list[ObservedDataPoint]:
        results = simulation.run(NUM_STEPS)
        steps = AxisUnits().window_ends(WINDOW, "step", NUM_STEPS)
        values = results.windowed_totals(["events"], WINDOW, steps)
        return [
            ObservedDataPoint(
                at=step,
                output="events",
                value=value,
                window=WINDOW,
            )
            for step, value in zip(steps, values, strict=True)
        ]

    def test_loss_is_zero_at_the_parameters_that_produced_the_observations(
        self,
    ) -> None:
        truth = 0.05
        observations = self._observations(_model(truth))
        problem = CalibrationProblem(
            observed_data=observations,
            parameters=[bounded("k1", truth, truth + 1e-12, initial_guess=truth)],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=5),
            seed=SEED,
        )

        result = Calibrator(_model(None), problem).run()

        assert result.final_loss == pytest.approx(0.0, abs=1e-12)

    def test_a_shifted_window_would_not_reproduce_the_loss(self) -> None:
        """A window of the wrong length gives observations the loss rejects."""
        truth = 0.05
        simulation = _model(truth)
        results = simulation.run(NUM_STEPS)
        steps = AxisUnits().window_ends(WINDOW, "step", NUM_STEPS)
        wrong = windowed_totals(results["events"], results.positions, WINDOW - 1, steps)

        observations = [
            ObservedDataPoint(
                at=step,
                output="events",
                value=value,
                window=WINDOW,
            )
            for step, value in zip(steps, wrong, strict=True)
        ]
        problem = CalibrationProblem(
            observed_data=observations,
            parameters=[bounded("k1", truth, truth + 1e-12, initial_guess=truth)],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=5),
            seed=SEED,
        )

        result = Calibrator(_model(None), problem).run()

        assert result.final_loss > 1.0


def test_windowed_totals_equal_the_prediction_of_a_windowed_observation() -> None:
    truth = 0.05
    results = _model(truth).run(14)
    observations = [
        ObservedDataPoint(at=at, output="events", value=0.0, window=7) for at in (7, 14)
    ]
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=[bounded("k1", 0.01, 0.1)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(),
    )
    calibrator = Calibrator(_model(None), problem)

    [[predictions]] = calibrator._rust_problem.predict_observations([[truth]])

    assert windowed_totals(results["events"], results.positions, 7, [7, 14]) == (
        predictions
    )


def test_windowed_totals_take_output_names() -> None:
    """
    An output name stands for itself, so some strata of an accumulator sum.

    The three strata start equal and move at one rate, so two of them gain
    two thirds of what the whole accumulator gains.
    """
    builder = a_to_b_builder(
        parameters={"k1": 0.1},
        rate="k1",
        per_compartment=True,
        accumulator="moved",
        amounts={"A": 90, "B": 0},
        stratifications={"g": {"g1": 1 / 3, "g2": 1 / 3, "g3": 1 / 3}},
    )
    results = Simulation(build(builder)).run(4)

    some = results.windowed_totals(["moved_g2", "moved_g3"], 2)
    whole = results.windowed_totals(["moved"], 2)
    by_hand = windowed_totals(
        [
            g2 + g3
            for g2, g3 in zip(results["moved_g2"], results["moved_g3"], strict=True)
        ],
        results.positions,
        2,
    )

    assert some == pytest.approx(by_hand)
    assert some == pytest.approx([2 / 3 * value for value in whole])
    assert all(value > 0 for value in some)


def test_results_read_bins_and_accumulators_at_positions() -> None:
    """
    `SimulationResults.sample` reads bins and accumulators by id at
    positions, next to `windowed_totals`; results without a model read output
    names.

    A bin id stands for all its strata, so A at the end of each window is the
    sum of its strata there; an accumulator read at the two ends of a window
    gives the gain `windowed_totals` reports over it.
    """
    builder = a_to_b_builder(
        parameters={"k1": 0.1},
        rate="k1",
        per_compartment=True,
        accumulator="moved",
        amounts={"A": 90, "B": 0},
        stratifications={"g": {"g1": 1 / 3, "g2": 1 / 3, "g3": 1 / 3}},
    )
    results = Simulation(build(builder)).run(6)
    ends = [2, 4, 6]
    strata = [f"A_{group}" for group in ("g1", "g2", "g3")]
    by_hand = SimulationResults(results.positions, {"A_g1": results["A_g1"]})

    a_at_ends = results.sample(["A"], ends)
    expected = [sum(results[name][int(end)] for name in strata) for end in ends]
    moved_at = results.sample(["moved"], [0, *ends])
    gained = results.windowed_totals(["moved"], 2, ends)

    assert by_hand.sample(["A_g1"], ends) == [results["A_g1"][end] for end in ends]
    assert a_at_ends == pytest.approx(expected)
    assert results.sample(strata, ends) == pytest.approx(expected)
    assert [end - start for start, end in zip(moved_at, moved_at[1:])] == (
        pytest.approx(gained)
    )
    with pytest.raises(ValueError, match="no value at 2.5"):
        results.sample(["A"], [2.5])
    with pytest.raises(KeyError, match="Missing output series in results: \\['A'\\]"):
        by_hand.sample(["A"], ends)
