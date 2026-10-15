"""A run below zero is a point the calibration avoids, not a failure."""

import pytest

from commol import CalibrationProblem, Calibrator, NelderMeadConfig, Simulation
from tests.support import a_to_b_builder, bounded, build, observed

TRUE_K = 0.2


def test_bounds_that_include_infeasible_rates_still_find_the_rate_of_the_data() -> None:
    truth = build(a_to_b_builder(parameters={"k": TRUE_K}))
    problem = CalibrationProblem(
        observed_data=observed(Simulation(truth).run(10), "A", range(1, 11)),
        # Above k = 1 a step moves more than A holds; the simplex starts at 2.
        parameters=[bounded("k", 0.0, 3.0, initial_guess=0.5)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(
            initial_simplex_size=0.5, sd_tolerance=1e-12
        ),
    )

    result = Calibrator(Simulation(truth), problem).run()

    assert result.best_parameters["k"] == pytest.approx(TRUE_K, abs=1e-4)
