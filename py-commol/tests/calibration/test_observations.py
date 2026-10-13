"""Scale factors fit observations that see only a share of an output."""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ParticleSwarmConfig,
    Simulation,
)
from commol.context.model import Model
from tests.support import bounded, observed

SEED = 42


def _fit(
    truth: Model,
    observations,
    parameters,
    config: NelderMeadConfig | ParticleSwarmConfig,
):
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=parameters,
        loss_function="sse",
        optimization_config=config,
        seed=SEED,
    )
    return Calibrator(Simulation(truth), problem).run()


class TestScaleFactors:
    def test_a_scale_alone_is_recovered(self, truth: Model, truth_run) -> None:
        observations = observed(
            truth_run, "I", range(0, 100, 10), scale=0.5, scale_id="detection_rate"
        )

        result = _fit(
            truth,
            observations,
            [bounded("detection_rate", 0.1, 1.0, kind="scale", initial_guess=0.7)],
            NelderMeadConfig(max_iterations=500, verbose=False),
        )

        assert result.best_parameters["detection_rate"] == pytest.approx(0.5, abs=1e-4)
        assert result.converged

    def test_a_scale_and_a_parameter_are_recovered_together(
        self, truth: Model, truth_run
    ) -> None:
        observations = observed(
            truth_run, "I", range(0, 100, 5), scale=0.3, scale_id="detection_rate"
        )

        result = _fit(
            truth,
            observations,
            [bounded("beta"), bounded("detection_rate", 0.1, 1.0, kind="scale")],
            ParticleSwarmConfig(max_iterations=300, num_particles=25, verbose=False),
        )

        assert result.best_parameters == pytest.approx(
            {"beta": 0.1, "detection_rate": 0.3}, abs=1e-3
        )

    def test_each_output_has_its_own_scale(self, truth: Model, truth_run) -> None:
        observations = observed(
            truth_run, "I", range(0, 100, 10), scale=0.6, scale_id="i_detection_rate"
        ) + observed(
            truth_run, "R", range(0, 100, 10), scale=0.9, scale_id="r_detection_rate"
        )

        result = _fit(
            truth,
            observations,
            [
                bounded("i_detection_rate", 0.1, 1.0, kind="scale"),
                bounded("r_detection_rate", 0.1, 1.0, kind="scale"),
            ],
            NelderMeadConfig(max_iterations=500, verbose=False),
        )

        assert result.best_parameters == pytest.approx(
            {"i_detection_rate": 0.6, "r_detection_rate": 0.9}, abs=1e-3
        )
