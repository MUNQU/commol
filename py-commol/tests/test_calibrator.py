import math

import pytest

from commol import (
    CalibrationConstraint,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    Model,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.constants import ModelTypes
from commol.context.parameter import Parameter

SEED = 42


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
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1000, verbose=False),
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
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=ParticleSwarmConfig(max_iterations=200, verbose=False),
            seed=SEED,
        )

        result = Calibrator(simulation, problem).run()

        # With seed, calibration is now deterministic and reproducible
        assert math.isclose(result.best_parameters["beta"], 0.1, abs_tol=1e-5)
        assert math.isclose(result.best_parameters["gamma"], 0.05, abs_tol=1e-5)

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
        # Create model with None parameters (to be calibrated)
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

        # Create a temporary model with known values to generate observed data
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

        # Prepare calibration using the uncalibrated model
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
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1000, verbose=False),
        )

        calibrator = Calibrator(simulation_for_calibration, problem)
        result = calibrator.run()

        # Update the model with calibrated values
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
                parameter_type="initial_condition",
                min_bound=0.0,
                max_bound=0.1,
                initial_guess=0.01,  # Starting point for optimization
            )
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=NelderMeadConfig(
                max_iterations=5000,
                sd_tolerance=1e-9,  # Stricter convergence criterion
                verbose=False,
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
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="I",
                parameter_type="initial_condition",
                min_bound=0.0,
                max_bound=0.1,  # Fraction range
            ),
        ]

        # Create PSO config with advanced features to avoid stagnation
        # Using fluent API for configuration
        pso_config = (
            ParticleSwarmConfig(num_particles=40, max_iterations=1000, verbose=False)
            # Time-Varying Acceleration Coefficients (TVAC)
            # Cognitive factor decreases from 2.5 to 0.5 (exploration to exploitation)
            # Social factor increases from 0.5 to 2.5 (individual to swarm guidance)
            .acceleration(
                "time_varying",
                c1_initial=2.5,
                c1_final=0.5,
                c2_initial=0.5,
                c2_final=2.5,
            )
            # Velocity control
            .velocity(clamp_factor=0.2)
            # Gaussian mutation on global best to escape local optima
            .mutation(
                "gaussian", scale=0.1, probability=0.05, application="global_best"
            )
        )

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=pso_config,
            seed=SEED,
        )

        result = Calibrator(simulation, problem).run()

        # With seed, calibration is now deterministic and reproducible
        assert math.isclose(result.best_parameters["beta"], 0.3, abs_tol=0.001)
        assert math.isclose(result.best_parameters["I"], 0.02, abs_tol=0.001)

        # Update model with calibrated values
        test_model.update_parameters({"beta": result.best_parameters["beta"]})
        test_model.update_initial_conditions({"I": result.best_parameters["I"]})

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
                parameter_type="initial_condition",
                min_bound=0.0,
                max_bound=100.0,
            )
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=100),
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
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        # Create PSO config with advanced features using fluent API
        pso_config = (
            ParticleSwarmConfig(num_particles=30, max_iterations=200, verbose=False)
            # Time-Varying Acceleration Coefficients (TVAC)
            # Cognitive factor decreases from 2.5 to 0.5 (exploration to exploitation)
            # Social factor increases from 0.5 to 2.5 (individual to swarm guidance)
            .acceleration(
                "time_varying",
                c1_initial=2.5,
                c1_final=0.5,
                c2_initial=0.5,
                c2_final=2.5,
            )
            # Velocity clamping to prevent particles from moving too fast
            .velocity(clamp_factor=0.2)
            # Gaussian mutation on global best to escape local optima
            .mutation(
                "gaussian", scale=0.1, probability=0.05, application="global_best"
            )
        )

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=pso_config,
            seed=SEED,
        )

        result = Calibrator(simulation, problem).run()

        # With seed, advanced PSO calibration is now deterministic and reproducible
        assert math.isclose(result.best_parameters["beta"], 0.1, abs_tol=1e-4)
        assert math.isclose(result.best_parameters["gamma"], 0.05, abs_tol=1e-4)

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
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        # Create PSO config with chaotic inertia using fluent API
        pso_config = (
            ParticleSwarmConfig(
                num_particles=25,
                max_iterations=200,
                verbose=False,
                # Opposition-based initialization for better initial population
                initialization="opposition_based",
            )
            # Chaotic inertia weight (varies between 0.4 and 0.9 using logistic map)
            .inertia("chaotic", w_min=0.4, w_max=0.9)
        )

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=pso_config,
            seed=SEED,
        )

        result = Calibrator(simulation, problem).run()

        # With advanced features, calibration should be successful
        # Allow slightly larger tolerance due to stochastic nature
        assert math.isclose(result.best_parameters["beta"], 0.1, abs_tol=2e-4)
        assert math.isclose(result.best_parameters["gamma"], 0.05, abs_tol=2e-4)

    def test_scale_parameter_calibration(self, model: Model):
        """
        Test calibration with scale parameter.
        Simulates the case where observed data is scaled by an unknown detection rate.
        """
        # Run simulation with known parameters
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        # Create "observed" data that's scaled down by 0.5
        # (simulating 50% detection rate)
        true_scale = 0.5
        observed_data = [
            ObservedDataPoint(
                step=i,
                compartment="I",
                value=results["I"][i] * true_scale,
                scale_id="detection_rate",
            )
            for i in range(0, 100, 10)  # Sample every 10 steps
        ]

        # Define calibration parameters including the scale
        parameters = [
            CalibrationParameter(
                id="detection_rate",
                parameter_type="scale",
                min_bound=0.1,
                max_bound=1.0,
                initial_guess=0.7,
            ),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=500, verbose=False),
        )

        result = Calibrator(simulation, problem).run()

        # Should recover the true scale parameter
        assert math.isclose(
            result.best_parameters["detection_rate"], true_scale, abs_tol=1e-4
        )
        assert result.converged

    def test_scale_and_parameter_calibration_combined(self, model: Model):
        """
        Test calibration with both model parameters and scale parameters.
        """
        # Run simulation with known parameters
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        # Create "observed" data that's scaled and noisy
        true_scale = 0.3
        observed_data = [
            ObservedDataPoint(
                step=i,
                compartment="I",
                value=results["I"][i] * true_scale,
                scale_id="detection_rate",
            )
            for i in range(0, 100, 5)
        ]

        # Calibrate both the model parameter beta and the scale
        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="detection_rate",
                parameter_type="scale",
                min_bound=0.1,
                max_bound=1.0,
            ),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=ParticleSwarmConfig(
                max_iterations=300, num_particles=25, verbose=False
            ),
            seed=SEED,
        )

        result = Calibrator(simulation, problem).run()

        # Should recover both the model parameter and scale
        assert math.isclose(result.best_parameters["beta"], 0.1, abs_tol=1e-3)
        assert math.isclose(
            result.best_parameters["detection_rate"], true_scale, abs_tol=1e-3
        )

    def test_multiple_scales_for_different_compartments(self, model: Model):
        """
        Test calibration with different scale parameters for different compartments.
        """
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        # Different detection rates for I and R compartments
        i_scale = 0.6
        r_scale = 0.9

        observed_data = []
        # Infected observations with one scale
        for i in range(0, 100, 10):
            observed_data.append(
                ObservedDataPoint(
                    step=i,
                    compartment="I",
                    value=results["I"][i] * i_scale,
                    scale_id="i_detection_rate",
                )
            )
        # Recovered observations with different scale
        for i in range(0, 100, 10):
            observed_data.append(
                ObservedDataPoint(
                    step=i,
                    compartment="R",
                    value=results["R"][i] * r_scale,
                    scale_id="r_detection_rate",
                )
            )

        parameters = [
            CalibrationParameter(
                id="i_detection_rate",
                parameter_type="scale",
                min_bound=0.1,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="r_detection_rate",
                parameter_type="scale",
                min_bound=0.1,
                max_bound=1.0,
            ),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=500, verbose=False),
        )

        result = Calibrator(simulation, problem).run()

        # Should recover both scale parameters
        assert math.isclose(
            result.best_parameters["i_detection_rate"], i_scale, abs_tol=1e-3
        )
        assert math.isclose(
            result.best_parameters["r_detection_rate"], r_scale, abs_tol=1e-3
        )

    def test_calibration_with_constraint(self, model: Model):
        """Test calibration with constraint"""
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(0, 100, 10)
        ]

        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=0.5,
            ),
        ]

        # Constraint: beta/gamma must be <= 5
        constraints = [
            CalibrationConstraint(
                id="r0_bound",
                expression="5.0 - beta/gamma",
                description="beta/gamma <= 5",
                weight=1.0,
            )
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            constraints=constraints,
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=200, verbose=False),
        )

        result = Calibrator(simulation, problem).run()

        # Verify constraint is satisfied
        op = result.best_parameters["beta"] / result.best_parameters["gamma"]
        assert op <= 5.0, f"constraint violated: {op}"

    def test_calibration_with_linear_constraint(self, model: Model):
        """Test calibration with linear sum constraint"""
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(0, 100, 10)
        ]

        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        # Constraint: beta + gamma <= 0.2
        constraints = [
            CalibrationConstraint(
                id="sum_bound",
                expression="0.2 - (beta + gamma)",
                description="beta + gamma <= 0.2",
            )
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            constraints=constraints,
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=200, verbose=False),
        )

        result = Calibrator(simulation, problem).run()

        # Verify constraint is satisfied
        param_sum = result.best_parameters["beta"] + result.best_parameters["gamma"]
        assert param_sum <= 0.21, f"Sum constraint violated: {param_sum}"

    def test_calibration_with_ordering_constraint(self, model: Model):
        """Test calibration with ordering constraint (beta >= gamma)"""
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(0, 100, 10)
        ]

        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        # Constraint: beta >= gamma
        constraints = [
            CalibrationConstraint(
                id="beta_ge_gamma",
                expression="beta - gamma",
                description="beta >= gamma",
            )
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            constraints=constraints,
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=200, verbose=False),
        )

        result = Calibrator(simulation, problem).run()

        # Verify constraint is satisfied
        assert (
            result.best_parameters["beta"] >= result.best_parameters["gamma"] - 1e-6
        ), "Ordering constraint violated"

    def test_calibration_with_time_dependent_constraint(self, model: Model):
        """Test calibration with time-dependent constraint on compartment values"""
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(0, 100, 10)
        ]

        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        # Time-dependent constraint: I <= 100 at specific time steps
        constraints = [
            CalibrationConstraint(
                id="peak_infected",
                expression="100.0 - I",
                description="Infected never exceeds 100",
                time_steps=[10, 20, 30, 40, 50],
            )
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            constraints=constraints,
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=200, verbose=False),
        )

        result = Calibrator(simulation, problem).run()

        # Run simulation with calibrated parameters to check constraint
        calibrated_model = model.model_copy(deep=True)
        for param in calibrated_model.parameters:
            if param.id == "beta":
                param.value = result.best_parameters["beta"]

        calibrated_sim = Simulation(calibrated_model)
        calibrated_results = calibrated_sim.run(100, output_format="dict_of_lists")

        # Verify constraint at specified time steps
        for ts in [10, 20, 30, 40, 50]:
            assert calibrated_results["I"][ts] <= 105.0, (
                f"Time constraint violated at step {ts}: "
                f"I={calibrated_results['I'][ts]}"
            )

    def test_calibration_with_compartment_sum_constraint(self, model: Model):
        """Test time-dependent constraint on sum of two compartments."""
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        # Create observed data for infected compartment
        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(0, 100, 10)
        ]

        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=0.5,
            ),
        ]

        # Time-dependent constraint: S + I <= 950 at specific time steps
        constraints = [
            CalibrationConstraint(
                id="min_recovered",
                expression="950.0 - (S + I)",
                description="S + I <= 950",
                time_steps=[30, 50, 70],
            )
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            constraints=constraints,
            loss_function="sse",
            optimization_config=ParticleSwarmConfig(
                num_particles=20,
                max_iterations=200,
                verbose=False,
                initialization="latin_hypercube",
            ),
            seed=SEED,
        )

        result = Calibrator(simulation, problem).run()

        # Run simulation with calibrated parameters to verify constraint
        calibrated_model = model.model_copy(deep=True)
        for param in calibrated_model.parameters:
            if param.id in result.best_parameters:
                param.value = result.best_parameters[param.id]

        calibrated_sim = Simulation(calibrated_model)
        calibrated_results = calibrated_sim.run(100, output_format="dict_of_lists")

        # Verify constraint at specified time steps
        for ts in [30, 50, 70]:
            s_plus_i = calibrated_results["S"][ts] + calibrated_results["I"][ts]
            assert s_plus_i <= 955.0, (
                f"Compartment sum constraint violated at step {ts}: S+I={s_plus_i:.2f}"
            )

            # Also verify R >= 45 (allowing some tolerance)
            r = calibrated_results["R"][ts]
            assert r >= 45.0, f"Recovered below threshold at step {ts}: R={r:.2f}"

    def test_calibration_with_multiple_constraints(self, model: Model):
        """Test calibration with multiple simultaneous constraints"""
        simulation = Simulation(model)
        results = simulation.run(100, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=results["I"][i])
            for i in range(0, 100, 10)
        ]

        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=0.5,
            ),
        ]

        # Multiple constraints
        constraints = [
            CalibrationConstraint(
                id="r0_bound",
                expression="5.0 - beta/gamma",
                description="R0 <= 5",
            ),
            CalibrationConstraint(
                id="ordering",
                expression="beta - gamma",
                description="beta >= gamma",
            ),
            CalibrationConstraint(
                id="sum_bound",
                expression="0.6 - (beta + gamma)",
                description="Sum <= 0.6",
            ),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            constraints=constraints,
            loss_function="sse",
            optimization_config=ParticleSwarmConfig(
                num_particles=20, max_iterations=200, verbose=False
            ),
            seed=SEED,
        )

        result = Calibrator(simulation, problem).run()

        # Verify all constraints are satisfied
        beta = result.best_parameters["beta"]
        gamma = result.best_parameters["gamma"]

        r0 = beta / gamma
        assert r0 <= 5.1, f"R0 constraint violated: {r0}"

        assert beta >= gamma - 1e-6, "Ordering constraint violated"

        param_sum = beta + gamma
        assert param_sum <= 0.61, f"Sum constraint violated: {param_sum}"

    def test_constraint_expression_security_validation(self):
        """Test that constraint expressions are validated for security threats."""
        from pydantic import ValidationError

        # Valid expressions should work
        valid_constraint = CalibrationConstraint(
            id="valid",
            expression="5.0 - beta/gamma",
            description="Valid R0 constraint",
        )
        assert valid_constraint.expression == "5.0 - beta/gamma"

        # Dangerous Python patterns should be blocked
        with pytest.raises(ValidationError) as exc_info:
            CalibrationConstraint(
                id="evil_eval",
                expression="eval(beta)",
            )
        assert "Security validation failed" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CalibrationConstraint(
                id="evil_import",
                expression="import os",
            )
        assert "Security validation failed" in str(exc_info.value)

        # Dangerous Rust patterns should be blocked
        with pytest.raises(ValidationError) as exc_info:
            CalibrationConstraint(
                id="evil_unsafe",
                expression="unsafe { beta }",
            )
        assert "Security validation failed" in str(exc_info.value)

        # Encoding attacks should be blocked
        with pytest.raises(ValidationError) as exc_info:
            CalibrationConstraint(
                id="evil_hex",
                expression=r"beta + \x41",
            )
        assert "Security validation failed" in str(exc_info.value)

        # Time-dependent constraints should also be validated
        with pytest.raises(ValidationError) as exc_info:
            CalibrationConstraint(
                id="evil_time_dependent",
                expression="eval(I)",
                time_steps=[10, 20, 30],
            )
        assert "Security validation failed" in str(exc_info.value)

        # Valid time-dependent constraint should work
        valid_time_constraint = CalibrationConstraint(
            id="peak_infected",
            expression="500.0 - I",
            time_steps=[10, 20, 30],
        )
        assert valid_time_constraint.expression == "500.0 - I"
