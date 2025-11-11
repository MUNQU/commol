import math

import pytest

from commol import (
    CalibrationParameter,
    CalibrationParameterType,
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
        return builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

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
            CalibrationParameter(
                id="beta",
                parameter_type=CalibrationParameterType.PARAMETER,
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type=CalibrationParameterType.PARAMETER,
                min_bound=0.0,
                max_bound=1.0,
            ),
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
            CalibrationParameter(
                id="beta",
                parameter_type=CalibrationParameterType.PARAMETER,
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type=CalibrationParameterType.PARAMETER,
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_config=LossConfig(function=LossFunction.SSE),
            optimization_config=OptimizationConfig(
                algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
                config=ParticleSwarmConfig.create(max_iterations=200, verbose=False),
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
        """Test that Simulation.run() raises ValueError with uncalibrated parameters."""
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
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Creating a Simulation should succeed
        simulation = Simulation(model)

        # But attempting to run the simulation should fail
        with pytest.raises(ValueError) as exc_info:
            _ = simulation.run(100)

        error_message = str(exc_info.value)
        assert "Cannot run Simulation" in error_message
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
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

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
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

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
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

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
            typology=ModelTypes.DIFFERENCE_EQUATIONS.value
        )

        # Verify we can create a simulation, but cannot run it yet
        simulation_uncalibrated = Simulation(model_uncalibrated)
        with pytest.raises(ValueError):
            _ = simulation_uncalibrated.run(100)

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
        model_known = builder_known.build(
            typology=ModelTypes.DIFFERENCE_EQUATIONS.value
        )
        simulation_known = Simulation(model_known)
        results = simulation_known.run(100, output_format="dict_of_lists")

        # Step 3: Prepare calibration using the uncalibrated model
        # First, update with None value for calibration
        model_uncalibrated.update_parameters({"beta": None, "gamma": None})

        # Now we can create a simulation for calibration
        simulation_for_calibration = Simulation(model_uncalibrated)

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(100)
        ]

        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type=CalibrationParameterType.PARAMETER,
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type=CalibrationParameterType.PARAMETER,
                min_bound=0.0,
                max_bound=1.0,
            ),
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

    def test_calibrate_initial_condition(self):
        """
        Test calibrating initial conditions while keeping parameters fixed.
        """
        # Create a model with known parameters and known initial conditions
        true_model = (
            ModelBuilder(name="SIR True", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3, unit="1/day")
            .add_parameter(id="gamma", value=0.1, unit="1/day")
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
                    {"bin": "S", "fraction": 0.98},
                    {"bin": "I", "fraction": 0.02},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        ).build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Generate observed data
        true_simulation = Simulation(true_model)
        true_results = true_simulation.run(50, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=true_results["I"][i])
            for i in range(0, 50, 5)
        ]

        # Create test model with wrong initial I value
        test_model = (
            ModelBuilder(name="SIR Test", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3, unit="1/day")
            .add_parameter(id="gamma", value=0.1, unit="1/day")
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
                    {"bin": "S", "fraction": 0.98},
                    {"bin": "I", "fraction": None},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        ).build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Create simulation - None values are allowed for calibration
        simulation = Simulation(test_model)

        # Calibrate initial I population
        parameters = [
            CalibrationParameter(
                id="I",
                parameter_type=CalibrationParameterType.INITIAL_CONDITION,
                min_bound=0.0,
                max_bound=0.1,
                initial_guess=0.01,  # Starting point for optimization
            )
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_config=LossConfig(function=LossFunction.SSE),
            optimization_config=OptimizationConfig(
                algorithm=OptimizationAlgorithm.NELDER_MEAD,
                config=NelderMeadConfig(
                    max_iterations=5000,
                    sd_tolerance=1e-9,  # Stricter convergence criterion
                    verbose=False,
                ),
            ),
        )

        calibrator = Calibrator(simulation, problem)
        result = calibrator.run()

        # After calibration, update model with calibrated values
        test_model.update_initial_conditions(result.best_parameters)

        assert result.converged
        assert math.isclose(result.best_parameters["I"], 0.02)

    def test_calibrate_parameter_and_initial_condition_together(self):
        """
        Test calibrating both a parameter and an initial condition simultaneously
        using Particle Swarm Optimization with advanced features to avoid stagnation:
        - Latin Hypercube Sampling initialization
        - Time-Varying Acceleration Coefficients (TVAC)
        - Velocity clamping
        - Mutation for escaping local optima
        """
        # Generate observed data
        true_model = (
            ModelBuilder(name="SIR True", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3, unit="1/day")
            .add_parameter(id="gamma", value=0.1, unit="1/day")
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
                    {"bin": "S", "fraction": 0.98},
                    {"bin": "I", "fraction": 0.02},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        ).build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        true_simulation = Simulation(true_model)
        true_results = true_simulation.run(50, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=true_results["I"][i])
            for i in range(0, 50, 5)
        ]

        # Create test model with wrong beta and initial I
        test_model = (
            ModelBuilder(name="SIR Test", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=None, unit="1/day")
            .add_parameter(id="gamma", value=0.1, unit="1/day")
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
                    {"bin": "S", "fraction": 0.98},
                    {"bin": "I", "fraction": None},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        ).build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Create simulation - None values are allowed for calibration
        simulation = Simulation(test_model)

        # Calibrate both beta and initial I
        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type=CalibrationParameterType.PARAMETER,
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="I",
                parameter_type=CalibrationParameterType.INITIAL_CONDITION,
                min_bound=0.0,
                max_bound=0.1,  # Fraction range
            ),
        ]

        # Create PSO config with advanced features to avoid stagnation
        pso_config = (
            ParticleSwarmConfig.create(
                num_particles=40, max_iterations=1000, verbose=False
            )
            # Enable Latin Hypercube Sampling for better initial particle distribution
            .with_initialization_strategy("latin_hypercube")
            # Enable Time-Varying Acceleration Coefficients (TVAC)
            # Cognitive factor decreases from 2.5 to 0.5 (exploration to exploitation)
            # Social factor increases from 0.5 to 2.5 (individual to swarm guidance)
            .with_tvac(c1_initial=2.5, c1_final=0.5, c2_initial=0.5, c2_final=2.5)
            # Enable velocity clamping to prevent particles from moving too fast
            .with_velocity_clamping(0.2)
            # Enable Gaussian mutation on global best to escape local optima
            .with_mutation(
                strategy="gaussian",
                scale=0.1,
                probability=0.05,
                application="global_best",
            )
        )

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_config=LossConfig(function=LossFunction.SSE),
            optimization_config=OptimizationConfig(
                algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
                config=pso_config,
            ),
        )

        # Retry up to 3 times due to stochastic nature of PSO
        max_attempts = 3
        last_result = None

        for attempt in range(max_attempts):
            result = Calibrator(simulation, problem).run()
            last_result = result

            # Check if calibration succeeded
            beta_ok = math.isclose(result.best_parameters["beta"], 0.3, abs_tol=0.001)
            I_ok = math.isclose(result.best_parameters["I"], 0.02, abs_tol=0.001)

            if beta_ok and I_ok:
                test_model.update_parameters({"beta": result.best_parameters["beta"]})
                test_model.update_initial_conditions({"I": result.best_parameters["I"]})
                return

            if attempt < max_attempts - 1:
                # Not the last attempt, will retry
                print(
                    (
                        f"\nAdvanced PSO attempt {attempt + 1} failed. "
                        f"beta={result.best_parameters['beta']:.6f} (expected 0.3), "
                        f"I={result.best_parameters['I']:.6f} (expected 0.02). "
                        f"Retrying..."
                    )
                )

        # All attempts failed, show final values and fail
        assert last_result is not None
        pytest.fail(
            (
                f"Advanced PSO calibration failed after {max_attempts} attempts. "
                f"Final values: beta={last_result.best_parameters['beta']:.6f} "
                f"(expected 0.3), I={last_result.best_parameters['I']:.6f} "
                f"(expected 0.02)"
            )
        )

    def test_invalid_bin_id_for_initial_condition_raises_error(self):
        """Test that using an invalid bin ID for initial condition raises an error."""
        model = (
            ModelBuilder(name="SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection", source=["S"], target=["I"], rate="beta * S * I / N"
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
        ).build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        observed_data = [ObservedDataPoint(step=10, compartment="I", value=50.0)]

        # Try to calibrate a bin that doesn't exist
        parameters = [
            CalibrationParameter(
                id="X",  # Invalid bin ID
                parameter_type=CalibrationParameterType.INITIAL_CONDITION,
                min_bound=0.0,
                max_bound=100.0,
            )
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_config=LossConfig(function=LossFunction.SSE),
            optimization_config=OptimizationConfig(
                algorithm=OptimizationAlgorithm.NELDER_MEAD,
                config=NelderMeadConfig(max_iterations=100),
            ),
        )

        simulation = Simulation(model)

        # Should raise ValueError during calibrator initialization
        with pytest.raises(ValueError, match="not found in model bins"):
            _ = Calibrator(simulation, problem)

    def test_particle_swarm_with_advanced_features(self, model: Model):
        """
        Test Particle Swarm Optimization with advanced features:
        - Latin Hypercube Sampling initialization
        - Time-Varying Acceleration Coefficients (TVAC)
        - Velocity clamping
        - Mutation for escaping local optima
        """
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(100)
        ]
        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type=CalibrationParameterType.PARAMETER,
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type=CalibrationParameterType.PARAMETER,
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        # Create PSO config with advanced features using builder pattern
        pso_config = (
            ParticleSwarmConfig.create(
                num_particles=30, max_iterations=200, verbose=False
            )
            # Enable Latin Hypercube Sampling for better initial particle distribution
            .with_initialization_strategy("latin_hypercube")
            # Enable Time-Varying Acceleration Coefficients (TVAC)
            # Cognitive factor decreases from 2.5 to 0.5 (exploration to exploitation)
            # Social factor increases from 0.5 to 2.5 (individual to swarm guidance)
            .with_tvac(c1_initial=2.5, c1_final=0.5, c2_initial=0.5, c2_final=2.5)
            # Enable velocity clamping to prevent particles from moving too fast
            .with_velocity_clamping(0.2)
            # Enable Gaussian mutation on global best to escape local optima
            .with_mutation(
                strategy="gaussian",
                scale=0.1,
                probability=0.05,
                application="global_best",
            )
        )

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_config=LossConfig(function=LossFunction.SSE),
            optimization_config=OptimizationConfig(
                algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
                config=pso_config,
            ),
        )

        # Retry up to 3 times due to stochastic nature of PSO
        max_attempts = 3
        last_result = None

        for attempt in range(max_attempts):
            result = Calibrator(simulation, problem).run()
            last_result = result

            # Check if calibration succeeded
            beta_ok = math.isclose(result.best_parameters["beta"], 0.1, abs_tol=1e-4)
            gamma_ok = math.isclose(result.best_parameters["gamma"], 0.05, abs_tol=1e-4)

            if beta_ok and gamma_ok:
                # Success!
                return

            if attempt < max_attempts - 1:
                # Not the last attempt, will retry
                print(
                    (
                        f"\nAdvanced PSO attempt {attempt + 1} failed. "
                        f"beta={result.best_parameters['beta']:.6f} (expected 0.1), "
                        f"gamma={result.best_parameters['gamma']:.6f} (expected 0.05). "
                        f"Retrying..."
                    )
                )

        # All attempts failed, show final values and fail
        assert last_result is not None
        pytest.fail(
            (
                f"Advanced PSO calibration failed after {max_attempts} attempts. "
                f"Final values: beta={last_result.best_parameters['beta']:.6f} "
                f"(expected 0.1), gamma={last_result.best_parameters['gamma']:.6f} "
                f"(expected 0.05)"
            )
        )

    def test_particle_swarm_with_chaotic_inertia(self, model: Model):
        """
        Test Particle Swarm Optimization with chaotic inertia weight.
        Chaotic inertia uses a logistic map to generate non-linear dynamics,
        helping particles escape local optima.
        """
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(100)
        ]
        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type=CalibrationParameterType.PARAMETER,
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type=CalibrationParameterType.PARAMETER,
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        # Create PSO config with chaotic inertia using builder pattern
        pso_config = (
            ParticleSwarmConfig.create(
                num_particles=25, max_iterations=200, verbose=False
            )
            # Enable chaotic inertia weight
            # (varies between 0.4 and 0.9 using logistic map)
            .with_chaotic_inertia(w_min=0.4, w_max=0.9)
            # Use opposition-based initialization for diverse starting positions
            .with_initialization_strategy("opposition_based")
        )

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_config=LossConfig(function=LossFunction.SSE),
            optimization_config=OptimizationConfig(
                algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
                config=pso_config,
            ),
        )

        result = Calibrator(simulation, problem).run()

        # With advanced features, calibration should be successful
        # Allow slightly larger tolerance due to stochastic nature
        assert math.isclose(result.best_parameters["beta"], 0.1, abs_tol=2e-4)
        assert math.isclose(result.best_parameters["gamma"], 0.05, abs_tol=2e-4)
