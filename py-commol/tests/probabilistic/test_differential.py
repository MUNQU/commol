"""Probabilistic calibration of a DifferentialEquations model."""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.context.probabilistic_calibration import (
    ProbabilisticCalibrationConfig,
    ProbClusteringConfig,
    ProbRepresentativeConfig,
)
from tests.support import bounded, build_ode, sir_builder

STEPS = [0.5 + 1.5 * j for j in range(20)]
"""Observation steps, none of them an integer."""


def test_observations_at_non_integer_steps_give_an_ensemble() -> None:
    truth = Simulation(build_ode(sir_builder(accumulator="cases"))).run_at(STEPS)
    observed = [
        ObservedDataPoint(at=step, output="I", value=value)
        for step, value in zip(STEPS, truth["I"], strict=True)
    ] + [
        ObservedDataPoint(
            at=STEPS[i],
            output="cases",
            value=truth["cases"][i] - truth["cases"][i - 2],
            window=3.0,
        )
        for i in range(2, 20, 2)
    ]
    problem = CalibrationProblem(
        observed_data=observed,
        parameters=[bounded("beta"), bounded("gamma", high=0.5)],
        loss_function="sse",
        optimization_config=ParticleSwarmConfig(
            num_particles=20, max_iterations=50, verbose=False
        ),
        seed=1,
    )
    problem.probabilistic_config = ProbabilisticCalibrationConfig(
        n_runs=4,
        clustering=ProbClusteringConfig(n_clusters=2),
        representative_selection=ProbRepresentativeConfig(max_representatives=80),
    )
    model = build_ode(
        sir_builder(parameters={"beta": None, "gamma": None}, accumulator="cases")
    )

    result = Calibrator(Simulation(model), problem).run_probabilistic()

    ensemble = result.selected_ensemble
    assert ensemble.point_parameters == pytest.approx(
        {"beta": 0.3, "gamma": 0.1}, rel=1e-2
    )
    assert ensemble.windowed_prediction_positions["cases"] == STEPS[2:20:2]
    assert ensemble.coverage_percentage > 0
