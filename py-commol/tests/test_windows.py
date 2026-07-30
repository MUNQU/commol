"""Tests for reducing cumulative series to per-window increments."""

import pytest

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
    window_end_steps,
    windowed_totals,
)
from commol.api.model_builder import BinFractionDict
from commol.constants import ModelTypes

SEED = 42
WINDOW = 5
NUM_STEPS = 40


def _model(k1: float | None = 0.05) -> Simulation:
    bin_fractions: list[BinFractionDict] = [
        {"bin": "A", "fraction": 1.0},
        {"bin": "B", "fraction": 0.0},
    ]
    model = (
        ModelBuilder(name="Windows", version="1.0")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_accumulator(id="events", name="Events")
        .add_parameter(id="k1", value=k1)
        .add_transition(
            id="flow",
            source=["A"],
            target=["B"],
            rate="k1",
            accumulators=["events"],
        )
        .set_initial_conditions(population_size=1000, bin_fractions=bin_fractions)
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )
    return Simulation(model)


class TestWindowEndSteps:
    def test_lists_every_complete_window(self) -> None:
        assert window_end_steps(5, 20) == [5, 10, 15, 20]

    def test_trailing_partial_window_is_dropped(self) -> None:
        assert window_end_steps(5, 23) == [5, 10, 15, 20]

    def test_run_shorter_than_one_window_gives_nothing(self) -> None:
        assert window_end_steps(5, 3) == []

    @pytest.mark.parametrize("window_steps", [0, -1])
    def test_non_positive_window_is_rejected(self, window_steps: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            window_end_steps(window_steps, 20)


class TestWindowedTotals:
    def test_increments_of_a_cumulative_series(self) -> None:
        series = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]

        assert windowed_totals(series, 2) == [3.0, 7.0]

    def test_default_grid_drops_a_trailing_partial_window(self) -> None:
        series = [float(value) for value in range(12)]

        assert windowed_totals(series, 5) == [5.0, 5.0]

    def test_explicit_steps_need_not_be_multiples_of_the_window(self) -> None:
        series = [float(value) for value in range(12)]

        assert windowed_totals(series, 4, [7, 11]) == [4.0, 4.0]

    def test_step_without_a_complete_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no complete window"):
            windowed_totals([0.0, 1.0, 2.0], 5, [2])

    def test_step_beyond_the_series_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside a series"):
            windowed_totals([0.0, 1.0, 2.0], 1, [9])

    @pytest.mark.parametrize("window_steps", [0, -1])
    def test_non_positive_window_is_rejected(self, window_steps: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            windowed_totals([0.0, 1.0], window_steps)

    def test_simulation_windows_the_total_of_its_accumulators(self) -> None:
        simulation = _model()
        results = simulation.run(NUM_STEPS)

        by_hand = windowed_totals(results["events"], WINDOW)

        assert simulation.windowed_totals(results, ["events"], WINDOW) == by_hand


class TestAgreementWithTheCalibrationLoss:
    """
    The reduction must return the values a calibration compares against.

    Observations built from `windowed_totals` of a known run are reproduced
    exactly when the model is simulated with the parameters that produced them,
    so the loss of that parameter set is zero.
    """

    def _observations(self, simulation: Simulation) -> list[ObservedDataPoint]:
        results = simulation.run(NUM_STEPS)
        steps = window_end_steps(WINDOW, NUM_STEPS)
        values = simulation.windowed_totals(results, ["events"], WINDOW, steps)
        return [
            ObservedDataPoint(
                step=step,
                compartment="events",
                value=value,
                window_steps=WINDOW,
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
            parameters=[
                CalibrationParameter(
                    id="k1",
                    parameter_type="parameter",
                    min_bound=truth,
                    max_bound=truth + 1e-12,
                    initial_guess=truth,
                )
            ],
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
        steps = window_end_steps(WINDOW, NUM_STEPS)
        wrong = windowed_totals(results["events"], WINDOW - 1, steps)

        observations = [
            ObservedDataPoint(
                step=step,
                compartment="events",
                value=value,
                window_steps=WINDOW,
            )
            for step, value in zip(steps, wrong, strict=True)
        ]
        problem = CalibrationProblem(
            observed_data=observations,
            parameters=[
                CalibrationParameter(
                    id="k1",
                    parameter_type="parameter",
                    min_bound=truth,
                    max_bound=truth + 1e-12,
                    initial_guess=truth,
                )
            ],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=5),
            seed=SEED,
        )

        result = Calibrator(_model(None), problem).run()

        assert result.final_loss > 1.0
