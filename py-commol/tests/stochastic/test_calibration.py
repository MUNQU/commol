"""Calibration of a stochastic engine against data of its own runs.

Each evaluation runs the same replicates (common random numbers, seeded by
`CalibrationProblem.seed`), so the loss is a deterministic function of the
parameters and Nelder-Mead converges.
"""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.context.engine import EngineName
from tests.support import STOCHASTIC_ENGINES, a_to_b_builder, bounded, build_stochastic

RATE = 0.1
POSITIONS = [2.0, 5.0, 10.0]


def _data(engine: EngineName) -> list[ObservedDataPoint]:
    """The mean of A over 200 runs at the rate `RATE`."""
    simulation = Simulation(build_stochastic(a_to_b_builder(), engine))
    mean = simulation.replicates(200, seed=21).run_at(POSITIONS).mean()
    return [
        ObservedDataPoint(at=at, output="A", value=value)
        for at, value in zip(POSITIONS, mean["A"], strict=True)
    ]


def _calibrate(engine: EngineName, summary: str):
    builder = a_to_b_builder(parameters={"k": None})
    problem = CalibrationProblem(
        observed_data=_data(engine),
        parameters=[bounded("k", 0.01, 0.5, initial_guess=0.3)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=200, sd_tolerance=1e-10),
        replicates=40,
        replicate_summary=summary,  # ty: ignore[invalid-argument-type]
        seed=7,
    )
    simulation = Simulation(build_stochastic(builder, engine))
    return Calibrator(simulation, problem).run()


@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
class TestCalibration:
    @pytest.mark.parametrize("summary", ["mean", "crps"])
    def test_a_rate_is_recovered_from_the_runs(
        self, engine: EngineName, summary: str
    ) -> None:
        result = _calibrate(engine, summary)

        # 200 runs of 1000 units against 40 replicates: within a few percent.
        assert result.best_parameters["k"] == pytest.approx(RATE, rel=0.03)

    def test_one_seed_gives_the_same_calibration(self, engine: EngineName) -> None:
        first = _calibrate(engine, "mean")
        second = _calibrate(engine, "mean")

        assert first.best_parameters == second.best_parameters
        assert first.final_loss == second.final_loss
