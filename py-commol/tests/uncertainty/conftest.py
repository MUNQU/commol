"""A calibrator of the ``abc_builder`` model, for the methods and readers, and
a simulation of it, for the sensitivity methods."""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from tests.support import abc_builder, bounded, build, observed


@pytest.fixture
def truth_run_50() -> dict[str, list[float]]:
    """50 steps of ``abc_builder`` with ``k1 = 0.1`` and ``k2 = 0.05``."""
    return Simulation(build(abc_builder(parameters={"k1": 0.1, "k2": 0.05}))).run(50)


@pytest.fixture
def calibrator(truth_run_50: dict[str, list[float]]) -> Calibrator:
    """``k1`` and ``k2`` of ``abc_builder``, fitted to B at 10, 20, …, 50 and
    to C over the window from 20 to 40 (the truth is 0.1 and 0.05)."""
    run = truth_run_50
    window = ObservedDataPoint(
        at=40, output="C", value=run["C"][40] - run["C"][20], window=20
    )
    problem = CalibrationProblem(
        observed_data=observed(run, "B", range(10, 51, 10)) + [window],
        parameters=[bounded("k1", 0.0, 0.5), bounded("k2", 0.0, 0.3)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=500, sd_tolerance=1e-10),
        seed=5,
    )
    model = build(abc_builder(parameters={"k1": None, "k2": None}))
    return Calibrator(Simulation(model), problem)


@pytest.fixture
def simulation() -> Simulation:
    """``abc_builder`` with ``k1 = 0.3`` and ``k2 = 0.1``."""
    return Simulation(build(abc_builder()))
