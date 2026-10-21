"""Tests for reducing cumulative series to per-window increments."""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
    windowed_totals,
)
from commol.context._windows import window_end, window_ends, window_index, window_start
from tests.support import a_to_b_builder, bounded, build

SEED = 42
WINDOW = 5
NUM_STEPS = 40


def _model(k1: float | None = 0.05) -> Simulation:
    builder = a_to_b_builder(parameters={"k1": k1}, rate="k1", accumulator="events")
    return Simulation(build(builder))


class TestWindowEnds:
    def test_lists_every_complete_window(self) -> None:
        assert window_ends(5, 20) == [5, 10, 15, 20]

    def test_trailing_partial_window_is_dropped(self) -> None:
        assert window_ends(5, 23) == [5, 10, 15, 20]

    def test_run_shorter_than_one_window_gives_nothing(self) -> None:
        assert window_ends(5, 3) == []
        assert window_ends(5, 0) == []
        assert window_ends(5, -5) == []

    def test_a_window_need_not_be_a_whole_number(self) -> None:
        assert window_ends(0.1, 0.3) == pytest.approx([0.1, 0.2, 0.3])

    @pytest.mark.parametrize("window", [0, -1])
    def test_non_positive_window_is_rejected(self, window: int) -> None:
        with pytest.raises(ValueError, match="positive length"):
            window_ends(window, 20)


class TestWindowGrid:
    """The window grid is one mapping, usable in both directions."""

    PERIOD = 168
    INDICES = range(33)

    def test_window_index_inverts_window_end(self) -> None:
        for index in self.INDICES:
            assert window_index(window_end(index, self.PERIOD), self.PERIOD) == index
            assert window_index(window_end(index, 2.5), 2.5) == index

    def test_every_step_maps_to_its_containing_window(self) -> None:
        for index in self.INDICES:
            start = int(window_start(index, self.PERIOD))
            end = int(window_end(index, self.PERIOD))
            for step in range(start + 1, end + 1):
                assert window_index(step, self.PERIOD) == index

    def test_a_position_between_steps_belongs_to_the_window_that_holds_it(
        self,
    ) -> None:
        # Window 1 of length 7 covers (7, 14].
        assert [window_index(at, 7) for at in [0.5, 7, 7.5, 14, 14.5]] == [
            0,
            0,
            1,
            1,
            2,
        ]

    def test_a_computed_end_belongs_to_its_window(self) -> None:
        # 0.1 + 0.2 is 0.30000000000000004, the end of window 2 of 0.1.
        assert window_index(0.1 + 0.2, 0.1) == 2

    def test_position_zero_belongs_to_the_first_window(self) -> None:
        assert window_index(0, self.PERIOD) == 0

    def test_windows_span_the_period_and_meet_end_to_start(self) -> None:
        for index in self.INDICES:
            start = window_start(index, self.PERIOD)
            end = window_end(index, self.PERIOD)
            assert end - start == self.PERIOD
            assert end == window_start(index + 1, self.PERIOD)

    def test_negative_index_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="index must not be negative"):
            window_start(-1, self.PERIOD)

    def test_negative_position_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at must not be negative"):
            window_index(-1, self.PERIOD)

    @pytest.mark.parametrize("window", [0, -1])
    def test_non_positive_window_is_rejected(self, window: int) -> None:
        with pytest.raises(ValueError, match="positive length"):
            window_end(0, window)


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

    def test_an_empty_series_has_no_windows(self) -> None:
        assert windowed_totals([], [], 1) == []

    def test_end_beyond_the_series_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no value at 9"):
            windowed_totals([0.0, 1.0, 2.0], range(3), 1, [9])

    @pytest.mark.parametrize("window", [0, -1])
    def test_non_positive_window_is_rejected(self, window: int) -> None:
        with pytest.raises(ValueError, match="positive length"):
            windowed_totals([0.0, 1.0], range(2), window)

    def test_simulation_windows_the_total_of_its_accumulators(self) -> None:
        simulation = _model()
        results = simulation.run(NUM_STEPS)

        by_hand = windowed_totals(results["events"], results.positions, WINDOW)
        at = [7, 12]

        assert simulation.windowed_totals(results, ["events"], WINDOW) == by_hand
        assert simulation.windowed_totals(results, ["events"], WINDOW, at) == (
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
        steps = window_ends(WINDOW, NUM_STEPS)
        values = simulation.windowed_totals(results, ["events"], WINDOW, steps)
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
        steps = window_ends(WINDOW, NUM_STEPS)
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
    simulation = Simulation(build(builder))
    results = simulation.run(4)

    some = simulation.windowed_totals(results, ["moved_g2", "moved_g3"], 2)
    whole = simulation.windowed_totals(results, ["moved"], 2)
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
