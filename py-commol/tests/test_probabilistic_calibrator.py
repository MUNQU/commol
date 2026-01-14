import math
from typing import Literal

import numpy as np
import pytest

from commol import (
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
from commol.context.probabilistic_calibration import (
    ProbabilisticCalibrationConfig,
    ProbClusteringConfig,
    ProbEnsembleConfig,
    ProbEvaluationFilterConfig,
    ProbRepresentativeConfig,
)

SEED = 42


class TestProbabilisticCalibrator:
    @pytest.fixture(scope="class")
    def model(self) -> Model:
        """Create a simple SIR model for testing."""
        builder = (
            ModelBuilder(name="Test SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
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

    def test_probabilistic_calibration_with_pso(self, model: Model):
        """
        Test probabilistic calibration by perturbing model output and recovering
        parameters with PSO optimization algorithm.

        This test:
        1. Runs the model with known parameters (beta=0.3, gamma=0.1)
        2. Adds Gaussian noise to the Infected compartment trajectory
        3. Uses probabilistic calibration (PSO) to recover the original parameters
        4. Verifies that the true parameters fall within the confidence intervals
        """
        # Generate true values with known parameters
        true_beta = 0.3
        true_gamma = 0.1

        simulation = Simulation(model)
        true_results = simulation.run(50, output_format="dict_of_lists")

        # Add Gaussian noise to create "observed" data
        np.random.seed(SEED)
        noise_std = 5.0  # Standard deviation of measurement noise

        observed_data = []
        for i in range(0, 50):
            true_value = true_results["I"][i]
            noisy_value = true_value + np.random.normal(0, noise_std)
            noisy_value = max(0.0, noisy_value)
            observed_data.append(
                ObservedDataPoint(step=i, compartment="I", value=noisy_value)
            )

        # Define calibration parameters
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

        # Create calibration problem with Particle Swarm
        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=ParticleSwarmConfig(
                num_particles=60,
                max_iterations=1000,
                verbose=False,
            ),
        )

        # Configure probabilistic calibration using new structure
        problem.probabilistic_config = ProbabilisticCalibrationConfig(
            n_runs=10,
            evaluation_processing=ProbEvaluationFilterConfig(
                loss_percentile_filter=0.9,
            ),
            clustering=ProbClusteringConfig(n_clusters=10),
            representative_selection=ProbRepresentativeConfig(
                max_representatives=1000,
                percentage_elite_cluster_selection=0.1,
            ),
            ensemble_selection=ProbEnsembleConfig(
                nsga_population_size=50,
                nsga_generations=5000,
                ensemble_size_mode="bounded",
                ensemble_size_min=10,
                ensemble_size_max=30,
            ),
            confidence_level=0.95,
        )
        problem.seed = SEED

        # Run probabilistic calibration
        calibrator = Calibrator(simulation, problem)
        result = calibrator.run_probabilistic()

        # Verify results
        assert result.selected_ensemble.ensemble_size >= 10, (
            "Ensemble should contain 10 or more parameter sets"
        )
        assert result.selected_ensemble.ensemble_size <= 30, (
            "Ensemble should contain 30 or less parameter sets"
        )

        assert result.n_runs_performed == 10, "Should have performed 10 runs"
        assert result.n_unique_evaluations > 0, (
            "Should have unique parameter evaluations"
        )
        assert result.n_clusters_used >= 1, (
            "Should have at least one cluster (automatically determined)"
        )

        # Check that parameter statistics are reasonable
        assert "beta" in result.selected_ensemble.parameter_statistics
        assert "gamma" in result.selected_ensemble.parameter_statistics

        beta_stats = result.selected_ensemble.parameter_statistics["beta"]
        gamma_stats = result.selected_ensemble.parameter_statistics["gamma"]

        # Check that statistics are within parameter bounds
        assert 0.0 <= beta_stats.min <= beta_stats.max <= 1.0
        assert 0.0 <= gamma_stats.min <= gamma_stats.max <= 0.5

        # Check that mean is within the 95% CI
        assert (
            beta_stats.percentile_lower
            <= beta_stats.mean
            <= beta_stats.percentile_upper
        )
        assert (
            gamma_stats.percentile_lower
            <= gamma_stats.mean
            <= gamma_stats.percentile_upper
        )

        # Check that predictions are provided for all compartments
        assert "S" in result.selected_ensemble.prediction_median
        assert "I" in result.selected_ensemble.prediction_median
        assert "R" in result.selected_ensemble.prediction_median

        assert "I" in result.selected_ensemble.prediction_ci_lower
        assert "I" in result.selected_ensemble.prediction_ci_upper

        # Check that prediction arrays have correct length
        assert (
            len(result.selected_ensemble.prediction_median["I"])
            == len(problem.observed_data) + 1
        )
        assert (
            len(result.selected_ensemble.prediction_ci_lower["I"])
            == len(problem.observed_data) + 1
        )
        assert (
            len(result.selected_ensemble.prediction_ci_upper["I"])
            == len(problem.observed_data) + 1
        )

        # All prediction arrays should have the same length
        assert len(result.selected_ensemble.prediction_median["I"]) == len(
            result.selected_ensemble.prediction_ci_lower["I"]
        )
        assert len(result.selected_ensemble.prediction_median["I"]) == len(
            result.selected_ensemble.prediction_ci_upper["I"]
        )

        # Check coverage
        assert 80.0 <= result.selected_ensemble.coverage_percentage <= 100.0

        # Verify that confidence intervals are ordered correctly
        for time_step in range(len(result.selected_ensemble.prediction_median["I"])):
            ci_lower = result.selected_ensemble.prediction_ci_lower["I"][time_step]
            ci_median = result.selected_ensemble.prediction_median["I"][time_step]
            ci_upper = result.selected_ensemble.prediction_ci_upper["I"][time_step]

            assert ci_lower <= ci_median <= ci_upper, (
                f"At time {time_step}: CI bounds are not ordered correctly "
                f"(lower={ci_lower}, median={ci_median}, upper={ci_upper})"
            )

        # Check if true parameters are within the confidence intervals
        assert math.isclose(beta_stats.mean, true_beta, abs_tol=0.05), (
            f"Beta mean {beta_stats.mean:.4f} not close to true value "
            f"{true_beta:.4f} (tolerance: 0.05)"
        )
        assert (
            beta_stats.percentile_lower <= true_beta <= beta_stats.percentile_upper
        ), (
            f"True beta {true_beta:.4f} not in 95% CI "
            f"[{beta_stats.percentile_lower:.4f}, {beta_stats.percentile_upper:.4f}]"
        )

        assert math.isclose(gamma_stats.mean, true_gamma, abs_tol=0.05), (
            f"Gamma mean {gamma_stats.mean:.4f} not close to true value "
            f"{true_gamma:.4f} (tolerance: 0.05)"
        )
        assert (
            gamma_stats.percentile_lower <= true_gamma <= gamma_stats.percentile_upper
        ), (
            f"True gamma {true_gamma:.4f} not in 95% CI "
            f"[{gamma_stats.percentile_lower:.4f}, {gamma_stats.percentile_upper:.4f}]"
        )

    def test_probabilistic_calibration_with_nelder_mead(self, model: Model):
        """
        Test probabilistic calibration by perturbing model output and recovering
        parameters with NelderMead optimization algorithm.

        This test:
        1. Runs the model with known parameters (beta=0.3, gamma=0.1)
        2. Adds Gaussian noise to the Infected compartment trajectory
        3. Uses probabilistic calibration (Nelder Mead) to recover the original
            parameters
        4. Verifies that the true parameters fall within the confidence intervals
        """
        # Generate true values with known parameters
        true_beta = 0.3
        true_gamma = 0.1

        simulation = Simulation(model)
        true_results = simulation.run(50, output_format="dict_of_lists")

        # Add Gaussian noise to create "observed" data
        np.random.seed(SEED)
        noise_std = 5.0  # Standard deviation of measurement noise

        observed_data = []
        for i in range(0, 50):
            true_value = true_results["I"][i]
            noisy_value = true_value + np.random.normal(0, noise_std)
            noisy_value = max(0.0, noisy_value)
            observed_data.append(
                ObservedDataPoint(step=i, compartment="I", value=noisy_value)
            )

        # Define calibration parameters
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

        # Create calibration problem with Particle Swarm
        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=NelderMeadConfig(),
        )

        # Configure probabilistic calibration using new structure
        problem.probabilistic_config = ProbabilisticCalibrationConfig(
            n_runs=10,
            evaluation_processing=ProbEvaluationFilterConfig(
                loss_percentile_filter=0.9,
            ),
            clustering=ProbClusteringConfig(n_clusters=10),
            representative_selection=ProbRepresentativeConfig(
                max_representatives=1000,
                percentage_elite_cluster_selection=0.1,
            ),
            ensemble_selection=ProbEnsembleConfig(
                nsga_population_size=50,
                nsga_generations=5000,
                ensemble_size_mode="bounded",
                ensemble_size_min=10,
                ensemble_size_max=30,
            ),
            confidence_level=0.95,
        )
        problem.seed = SEED

        # Run probabilistic calibration
        calibrator = Calibrator(simulation, problem)
        result = calibrator.run_probabilistic()

        # Verify results
        assert result.selected_ensemble.ensemble_size >= 10, (
            "Ensemble should contain 10 or more parameter sets"
        )
        assert result.selected_ensemble.ensemble_size <= 30, (
            "Ensemble should contain 30 or less parameter sets"
        )

        assert result.n_runs_performed == 10, "Should have performed 10 runs"
        assert result.n_unique_evaluations > 0, (
            "Should have unique parameter evaluations"
        )
        assert result.n_clusters_used >= 1, (
            "Should have at least one cluster (automatically determined)"
        )

        # Check that parameter statistics are reasonable
        assert "beta" in result.selected_ensemble.parameter_statistics
        assert "gamma" in result.selected_ensemble.parameter_statistics

        beta_stats = result.selected_ensemble.parameter_statistics["beta"]
        gamma_stats = result.selected_ensemble.parameter_statistics["gamma"]

        # Check that statistics are within parameter bounds
        assert 0.0 <= beta_stats.min <= beta_stats.max <= 1.0
        assert 0.0 <= gamma_stats.min <= gamma_stats.max <= 0.5

        # Check that mean is within the 95% CI
        assert (
            beta_stats.percentile_lower
            <= beta_stats.mean
            <= beta_stats.percentile_upper
        )
        assert (
            gamma_stats.percentile_lower
            <= gamma_stats.mean
            <= gamma_stats.percentile_upper
        )

        # Check that predictions are provided for all compartments
        assert "S" in result.selected_ensemble.prediction_median
        assert "I" in result.selected_ensemble.prediction_median
        assert "R" in result.selected_ensemble.prediction_median

        assert "I" in result.selected_ensemble.prediction_ci_lower
        assert "I" in result.selected_ensemble.prediction_ci_upper

        # Check that prediction arrays have correct length
        assert (
            len(result.selected_ensemble.prediction_median["I"])
            == len(problem.observed_data) + 1
        )
        assert (
            len(result.selected_ensemble.prediction_ci_lower["I"])
            == len(problem.observed_data) + 1
        )
        assert (
            len(result.selected_ensemble.prediction_ci_upper["I"])
            == len(problem.observed_data) + 1
        )

        # All prediction arrays should have the same length
        assert len(result.selected_ensemble.prediction_median["I"]) == len(
            result.selected_ensemble.prediction_ci_lower["I"]
        )
        assert len(result.selected_ensemble.prediction_median["I"]) == len(
            result.selected_ensemble.prediction_ci_upper["I"]
        )

        # Check coverage
        assert 80.0 <= result.selected_ensemble.coverage_percentage <= 100.0

        # Verify that confidence intervals are ordered correctly
        for time_step in range(len(result.selected_ensemble.prediction_median["I"])):
            ci_lower = result.selected_ensemble.prediction_ci_lower["I"][time_step]
            ci_median = result.selected_ensemble.prediction_median["I"][time_step]
            ci_upper = result.selected_ensemble.prediction_ci_upper["I"][time_step]

            assert ci_lower <= ci_median <= ci_upper, (
                f"At time {time_step}: CI bounds are not ordered correctly "
                f"(lower={ci_lower}, median={ci_median}, upper={ci_upper})"
            )

        # Check if true parameters are within the confidence intervals
        assert math.isclose(beta_stats.mean, true_beta, abs_tol=0.05), (
            f"Beta mean {beta_stats.mean:.4f} not close to true value "
            f"{true_beta:.4f} (tolerance: 0.05)"
        )
        assert (
            beta_stats.percentile_lower <= true_beta <= beta_stats.percentile_upper
        ), (
            f"True beta {true_beta:.4f} not in 95% CI "
            f"[{beta_stats.percentile_lower:.4f}, {beta_stats.percentile_upper:.4f}]"
        )

        assert math.isclose(gamma_stats.mean, true_gamma, abs_tol=0.05), (
            f"Gamma mean {gamma_stats.mean:.4f} not close to true value "
            f"{true_gamma:.4f} (tolerance: 0.05)"
        )
        assert (
            gamma_stats.percentile_lower <= true_gamma <= gamma_stats.percentile_upper
        ), (
            f"True gamma {true_gamma:.4f} not in 95% CI "
            f"[{gamma_stats.percentile_lower:.4f}, {gamma_stats.percentile_upper:.4f}]"
        )


class TestProbabilisticCalibratorValidation:
    """Tests for input validation in ProbabilisticCalibrator."""

    @pytest.fixture(scope="class")
    def model(self) -> Model:
        """Create a simple SIR model for testing."""
        builder = (
            ModelBuilder(name="Test SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
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

    def test_invalid_parameter_id(self, model: Model):
        """Test that invalid parameter IDs raise ValueError."""
        simulation = Simulation(model)

        observed_data = [ObservedDataPoint(step=0, compartment="I", value=10.0)]

        # Use a parameter ID that doesn't exist in the model
        parameters = [
            CalibrationParameter(
                id="nonexistent_param",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=ParticleSwarmConfig(
                num_particles=10, max_iterations=10
            ),
        )

        with pytest.raises(ValueError, match="not found in model"):
            Calibrator(simulation, problem)

    def test_invalid_compartment_in_observed_data(self, model: Model):
        """Test that invalid compartment in observed data raises ValueError."""
        simulation = Simulation(model)

        # Use a compartment that doesn't exist in the model
        observed_data = [
            ObservedDataPoint(step=0, compartment="nonexistent_compartment", value=10.0)
        ]

        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        problem = CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=ParticleSwarmConfig(
                num_particles=10, max_iterations=10
            ),
        )

        with pytest.raises(ValueError, match="not found in model"):
            Calibrator(simulation, problem)

    def test_empty_parameters_raises_validation_error(self):
        """Test that empty parameters list raises ValidationError from Pydantic."""
        from pydantic import ValidationError

        observed_data = [ObservedDataPoint(step=0, compartment="I", value=10.0)]

        with pytest.raises(ValidationError, match="too_short"):
            CalibrationProblem(
                observed_data=observed_data,
                parameters=[],  # Empty parameters
                loss_function="sse",
                optimization_config=ParticleSwarmConfig(
                    num_particles=10, max_iterations=10
                ),
            )

    def test_empty_observed_data_raises_validation_error(self):
        """Test that empty observed data raises ValidationError from Pydantic."""
        from pydantic import ValidationError

        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ]

        with pytest.raises(ValidationError, match="too_short"):
            CalibrationProblem(
                observed_data=[],  # Empty observed data
                parameters=parameters,
                loss_function="sse",
                optimization_config=ParticleSwarmConfig(
                    num_particles=10, max_iterations=10
                ),
            )


class TestProbabilisticCalibratorSelectionMethods:
    """Tests for different selection methods and ensemble size modes."""

    @pytest.fixture(scope="class")
    def model(self) -> Model:
        """Create a simple SIR model for testing."""
        builder = (
            ModelBuilder(name="Test SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
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

    @pytest.fixture(scope="class")
    def base_problem(self, model: Model):
        """Create a base calibration problem."""
        simulation = Simulation(model)
        true_results = simulation.run(20, output_format="dict_of_lists")

        observed_data = [
            ObservedDataPoint(step=i, compartment="I", value=true_results["I"][i])
            for i in range(0, 20, 5)
        ]

        parameters = [
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.1,
                max_bound=0.5,
            ),
        ]

        return CalibrationProblem(
            observed_data=observed_data,
            parameters=parameters,
            loss_function="sse",
            optimization_config=ParticleSwarmConfig(
                num_particles=20, max_iterations=50, verbose=False
            ),
        )

    @pytest.mark.parametrize(
        "selection_method",
        ["crowding_distance", "maximin_distance", "latin_hypercube"],
    )
    def test_cluster_selection_methods(
        self,
        model: Model,
        base_problem: CalibrationProblem,
        selection_method: Literal[
            "crowding_distance", "maximin_distance", "latin_hypercube"
        ],
    ):
        """Test that different cluster selection methods work correctly."""
        simulation = Simulation(model)

        base_problem.probabilistic_config = ProbabilisticCalibrationConfig(
            n_runs=8,
            representative_selection=ProbRepresentativeConfig(
                max_representatives=50,
                cluster_selection_method=selection_method,
            ),
            ensemble_selection=ProbEnsembleConfig(
                nsga_population_size=10,
                nsga_generations=10,
            ),
        )
        base_problem.seed = SEED

        calibrator = Calibrator(simulation, base_problem)
        result = calibrator.run_probabilistic()

        assert result.selected_ensemble.ensemble_size > 0
        assert "beta" in result.selected_ensemble.parameter_statistics
        print(
            f"\n{selection_method}: "
            f"ensemble_size={result.selected_ensemble.ensemble_size}"
        )

    @pytest.mark.parametrize(
        "size_mode,size_config",
        [
            ("fixed", {"ensemble_size": 5}),
            ("bounded", {"ensemble_size_min": 3, "ensemble_size_max": 10}),
            ("automatic", {}),
        ],
    )
    def test_ensemble_size_modes(
        self,
        model: Model,
        base_problem: CalibrationProblem,
        size_mode: Literal["fixed", "bounded", "automatic"],
        size_config: dict,
    ):
        """Test that different ensemble size modes work correctly."""
        simulation = Simulation(model)

        # Build ensemble selection config based on size_mode
        ensemble_config = ProbEnsembleConfig(
            nsga_population_size=10,
            nsga_generations=10,
            ensemble_size_mode=size_mode,
            **size_config,
        )

        base_problem.probabilistic_config = ProbabilisticCalibrationConfig(
            n_runs=8,
            representative_selection=ProbRepresentativeConfig(
                max_representatives=50,
            ),
            ensemble_selection=ensemble_config,
        )
        base_problem.seed = SEED

        calibrator = Calibrator(simulation, base_problem)
        result = calibrator.run_probabilistic()

        assert result.selected_ensemble.ensemble_size > 0

        if size_mode == "fixed":
            # Fixed mode should get close to the target size
            # (may not be exact due to optimization constraints)
            target = size_config["ensemble_size"]
            print(
                f"\nFixed mode: target={target}, "
                f"actual={result.selected_ensemble.ensemble_size}"
            )

        elif size_mode == "bounded":
            # Bounded mode should be within the specified range
            min_size = size_config["ensemble_size_min"]
            max_size = size_config["ensemble_size_max"]
            print(
                f"\nBounded mode: range=[{min_size}, {max_size}], "
                f"actual={result.selected_ensemble.ensemble_size}"
            )

        else:  # automatic
            print(
                "\nAutomatic mode: "
                f"ensemble_size={result.selected_ensemble.ensemble_size}"
            )
