import math

import pytest

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    LossConfig,
    LossFunction,
    Model,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    OptimizationAlgorithm,
    OptimizationConfig,
    ParticleSwarmConfig,
    Simulation,
)
from commol.constants import ModelTypes


class TestCalibrator:
    @pytest.fixture(scope="class")
    def model(self) -> Model:
        builder = (
            ModelBuilder(name="Test SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.1)
            .add_parameter(id="gamma", value=0.05)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )
        return builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

    def test_model_calibration_nelder_mead(self, model: Model):
        """
        Test calibration of SIR model parameters using Nelder-Mead algorithm.
        """
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(100)
        ]
        parameters = [
            CalibrationParameter(id="beta", min_bound=0.0, max_bound=1.0),
            CalibrationParameter(id="gamma", min_bound=0.0, max_bound=1.0),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_config=LossConfig(function=LossFunction.SSE),
            optimization_config=OptimizationConfig(
                algorithm=OptimizationAlgorithm.NELDER_MEAD,
                config=NelderMeadConfig(max_iterations=1000, verbose=False),
            ),
        )

        result = Calibrator(simulation, problem).run()

        assert math.isclose(result.best_parameters["beta"], 0.1, abs_tol=1e-5)
        assert math.isclose(result.best_parameters["gamma"], 0.05, abs_tol=1e-5)

    def test_model_calibration_particle_swarm(self, model: Model):
        """
        Test calibration of SIR model parameters using Particle Swarm Optimization.
        """
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(100)
        ]
        parameters = [
            CalibrationParameter(id="beta", min_bound=0.0, max_bound=1.0),
            CalibrationParameter(id="gamma", min_bound=0.0, max_bound=1.0),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_config=LossConfig(function=LossFunction.SSE),
            optimization_config=OptimizationConfig(
                algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
                config=ParticleSwarmConfig(max_iterations=200, verbose=False),
            ),
        )

        # Retry up to 3 times due to stochastic nature of PSO
        max_attempts = 3
        last_result = None

        for attempt in range(max_attempts):
            result = Calibrator(simulation, problem).run()
            last_result = result

            # Check if calibration succeeded
            beta_ok = math.isclose(result.best_parameters["beta"], 0.1, abs_tol=1e-5)
            gamma_ok = math.isclose(result.best_parameters["gamma"], 0.05, abs_tol=1e-5)

            if beta_ok and gamma_ok:
                # Success!
                return

            if attempt < max_attempts - 1:
                # Not the last attempt, will retry
                print(
                    (
                        f"\nAttempt {attempt + 1} failed. "
                        f"beta={result.best_parameters['beta']:.6f} (expected 0.1), "
                        f"gamma={result.best_parameters['gamma']:.6f} (expected 0.05). "
                        f"Retrying..."
                    )
                )

        # All attempts failed, show final values and fail
        assert last_result is not None
        pytest.fail(
            (
                f"Calibration failed after {max_attempts} attempts. "
                f"Final values: beta={last_result.best_parameters['beta']:.6f} "
                f"(expected 0.1), gamma={last_result.best_parameters['gamma']:.6f} "
                f"(expected 0.05)"
            )
        )
