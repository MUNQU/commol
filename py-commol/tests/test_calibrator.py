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
    Parameter,
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

    def test_parameter_with_none_value(self):
        """Test that Parameter can be created with None value."""
        param = Parameter(id="beta", value=None)
        assert param.id == "beta"
        assert param.value is None
        assert not param.is_calibrated()

    def test_parameter_is_calibrated_method(self):
        """Test the is_calibrated() method."""
        uncalibrated_param = Parameter(id="beta", value=None)
        calibrated_param = Parameter(id="gamma", value=0.05)

        assert not uncalibrated_param.is_calibrated()
        assert calibrated_param.is_calibrated()

    def test_simulation_fails_with_uncalibrated_parameters(self):
        """Test that Simulation raises ValueError with uncalibrated parameters."""
        builder = (
            ModelBuilder(name="Test SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=None)  # Uncalibrated
            .add_parameter(id="gamma", value=0.05)  # Calibrated
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
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

        # Attempting to create a Simulation should fail
        with pytest.raises(ValueError) as exc_info:
            _ = Simulation(model)

        error_message = str(exc_info.value)
        assert (
            "Cannot run Simulation" in error_message
            or "Cannot create Simulation" in error_message
        )
        assert "beta" in error_message
        assert "calibration" in error_message.lower() or "None" in error_message

    def test_get_uncalibrated_parameters(self):
        """Test Model.get_uncalibrated_parameters() method."""
        builder = (
            ModelBuilder(name="Test SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=None)  # Uncalibrated
            .add_parameter(id="gamma", value=0.05)  # Calibrated
            .add_parameter(id="delta", value=None)  # Uncalibrated
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
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

        uncalibrated = model.get_uncalibrated_parameters()
        assert len(uncalibrated) == 2
        assert "beta" in uncalibrated
        assert "delta" in uncalibrated
        assert "gamma" not in uncalibrated

    def test_update_parameters(self):
        """Test Model.update_parameters() method."""
        builder = (
            ModelBuilder(name="Test SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=None)  # Uncalibrated
            .add_parameter(id="gamma", value=None)  # Uncalibrated
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
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

        # Update parameters
        model.update_parameters({"beta": 0.1, "gamma": 0.05})

        # Check that parameters are updated
        assert model.parameters[0].value == 0.1
        assert model.parameters[1].value == 0.05
        assert len(model.get_uncalibrated_parameters()) == 0

    def test_update_parameters_with_invalid_id(self):
        """Test that update_parameters raises error for invalid parameter ID."""
        builder = (
            ModelBuilder(name="Test SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=None)
            .add_parameter(id="gamma", value=None)
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
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

        # Attempt to update with invalid parameter ID
        with pytest.raises(ValueError) as exc_info:
            model.update_parameters({"invalid_param": 0.1})

        error_message = str(exc_info.value)
        assert "invalid_param" in error_message
        assert "not found" in error_message.lower()

    def test_full_calibration_workflow(self):
        """
        Test the complete workflow:
        1. Create model with None parameter values
        2. Create temporary model with values for generating observed data
        3. Calibrate the uncalibrated model
        4. Update parameters
        5. Run simulation
        """
        # Step 1: Create model with None parameters (to be calibrated)
        builder_uncalibrated = (
            ModelBuilder(name="Test SIR Uncalibrated", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=None)  # To be calibrated
            .add_parameter(id="gamma", value=None)  # To be calibrated
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
        model_uncalibrated = builder_uncalibrated.build(
            typology=ModelTypes.DIFFERENCE_EQUATIONS
        )

        # Verify we cannot create a simulation yet
        with pytest.raises(ValueError):
            _ = Simulation(model_uncalibrated)

        # Step 2: Create a temporary model with known values to generate observed data
        builder_known = (
            ModelBuilder(name="Test SIR Known", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.1)  # Known value
            .add_parameter(id="gamma", value=0.05)  # Known value
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
        model_known = builder_known.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
        simulation_known = Simulation(model_known)
        results = simulation_known.run(100, output_format="dict_of_lists")

        # Step 3: Prepare calibration using the uncalibrated model
        # First, update with initial guesses for calibration
        model_uncalibrated.update_parameters({"beta": 0.2, "gamma": 0.1})

        # Now we can create a simulation for calibration
        simulation_for_calibration = Simulation(model_uncalibrated)

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

        calibrator = Calibrator(simulation_for_calibration, problem)
        result = calibrator.run()

        # Step 4: Update the model with calibrated values
        model_uncalibrated.update_parameters(result.best_parameters)

        # Step 5: Now we can run a new simulation with the calibrated model
        final_simulation = Simulation(model_uncalibrated)
        final_results = final_simulation.run(100, output_format="dict_of_lists")

        # Verify calibration was successful
        assert math.isclose(result.best_parameters["beta"], 0.1, abs_tol=1e-5)
        assert math.isclose(result.best_parameters["gamma"], 0.05, abs_tol=1e-5)

        # Verify the simulation runs successfully
        assert "I" in final_results
        assert len(final_results["I"]) == 101  # 100 steps + initial state
