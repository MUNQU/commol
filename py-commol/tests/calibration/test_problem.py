"""The calibration problem: losses, normalization, and the whole workflow."""

import math

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.context.model import Model
from tests.support import bounded, build, observed, sir_builder

SEED = 42


class TestLosses:
    @pytest.mark.parametrize("loss_function", ["sse", "rmse", "mae"])
    @pytest.mark.parametrize("normalize", [False, True])
    def test_loss_config_carries_normalization_flag(
        self, truth: Model, loss_function: str, normalize: bool
    ) -> None:
        """The problem's normalize_observations must reach the Rust LossConfig."""
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(step=0, compartment="I", value=10.0)],
            parameters=[bounded("beta")],
            loss_function=loss_function,
            normalize_observations=normalize,
            optimization_config=NelderMeadConfig(max_iterations=1, verbose=False),
        )

        loss_config = Calibrator(Simulation(truth), problem)._build_loss_config()

        assert loss_config.normalize_observations is normalize

    def test_normalization_changes_deterministic_loss(
        self, truth: Model, truth_run
    ) -> None:
        """Normalization divides each residual by its series' RMS."""
        # Biased targets: no parameter choice drives the residuals to zero,
        # otherwise both losses collapse to 0 and cannot be told apart.
        observations = [
            ObservedDataPoint(step=i, compartment="I", value=truth_run["I"][i] + 25.0)
            for i in range(100)
        ]

        def final_loss(normalize: bool) -> float:
            problem = CalibrationProblem(
                observed_data=observations,
                parameters=[bounded("beta"), bounded("gamma")],
                loss_function="sse",
                normalize_observations=normalize,
                optimization_config=ParticleSwarmConfig(
                    max_iterations=50, verbose=False
                ),
                seed=SEED,
            )
            return Calibrator(Simulation(truth), problem).run().final_loss

        rms = math.sqrt(
            sum(point.value**2 for point in observations) / len(observations)
        )
        assert rms > 1.0
        assert final_loss(normalize=True) < final_loss(normalize=False)


class TestWorkflow:
    def test_calibrate_write_back_and_simulate(self) -> None:
        """
        The quick path of the guide: a model whose unknowns are None cannot
        run; calibration finds them; written back, the model runs again.
        """
        truth = Simulation(
            build(sir_builder(parameters={"beta": 0.1, "gamma": 0.05}))
        ).run(100)
        model = build(sir_builder(parameters={"beta": None, "gamma": None}))
        with pytest.raises(ValueError, match="Cannot run Simulation"):
            Simulation(model).run(100)
        problem = CalibrationProblem(
            observed_data=observed(truth, "I", range(100)),
            parameters=[bounded("beta"), bounded("gamma")],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1000, verbose=False),
        )

        result = Calibrator(Simulation(model), problem).run()
        model.update_parameters(result.best_parameters)
        rerun = Simulation(model).run(100)

        assert result.best_parameters == pytest.approx(
            {"beta": 0.1, "gamma": 0.05}, abs=1e-5
        )
        assert rerun["I"] == pytest.approx(truth["I"], rel=1e-3)
