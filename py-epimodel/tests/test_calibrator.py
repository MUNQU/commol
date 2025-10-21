import math

import pytest

from epimodel import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    LossConfig,
    LossFunction,
    Model,
    ModelBuilder,
    ObservedDataPoint,
    OptimizationAlgorithm,
    OptimizationConfig,
    ParticleSwarmConfig,
    Simulation,
)
from epimodel.constants import ModelTypes


class TestCalibrator:
    @pytest.fixture(scope="class")
    def model(self) -> Model:
        builder = (
            ModelBuilder(name="Test SIR", version="1.0")
            .add_disease_state(id="S", name="Susceptible")
            .add_disease_state(id="I", name="Infected")
            .add_disease_state(id="R", name="Recovered")
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
                disease_state_fractions=[
                    {"disease_state": "S", "fraction": 0.99},
                    {"disease_state": "I", "fraction": 0.01},
                    {"disease_state": "R", "fraction": 0.0},
                ],
            )
        )
        return builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

    def test_model_calibration(self, model: Model):
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

        calibrator = Calibrator(simulation, problem).run()

        assert math.isclose(calibrator.best_parameters["beta"], 0.1, abs_tol=1e-5)
        assert math.isclose(calibrator.best_parameters["gamma"], 0.05, abs_tol=1e-5)
