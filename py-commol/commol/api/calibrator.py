import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from commol.commol_rs.commol_rs import (
        CalibrationParameterTypeProtocol,
        DifferenceEquationsProtocol,
        LossConfigProtocol,
        OptimizationConfigProtocol,
    )

try:
    from commol.commol_rs import commol_rs

    rust_core = commol_rs.core
    rust_difference = commol_rs.difference
    rust_calibration = commol_rs.calibration
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e

from commol.api.simulation import Simulation
from commol.context.calibration import (
    CalibrationParameterType,
    CalibrationProblem,
    CalibrationResult,
    LossFunction,
    NelderMeadConfig,
    OptimizationAlgorithm,
    ParticleSwarmConfig,
)

logger = logging.getLogger(__name__)


class Calibrator:
    """
    A Facade for running parameter calibration from a defined CalibrationProblem.
    """

    def __init__(
        self,
        simulation: Simulation,
        problem: CalibrationProblem,
    ):
        """
        Initializes the calibration from a Simulation and CalibrationProblem.

        Parameters
        ----------
        simulation : Simulation
            A fully initialized Simulation object with the model to calibrate.
        problem : CalibrationProblem
            A fully constructed and validated calibration problem definition.
        """
        logger.info(
            f"Initializing Calibration for model: '{simulation.model_definition.name}'"
        )
        self.simulation: Simulation = simulation
        self.problem: CalibrationProblem = problem
        self._engine: "DifferenceEquationsProtocol" = simulation.engine

        logger.info(
            (
                f"Calibration initialized with {len(problem.parameters)} parameters "
                f"and {len(problem.observed_data)} observed data points."
            )
        )

        # Validate calibration parameters against model
        self._validate_calibration_parameters()

    def run(self) -> CalibrationResult:
        """
        Runs the calibration optimization.

        Returns
        -------
        CalibrationResult
            Object containing the optimized parameter values, final loss,
            convergence status, and other optimization statistics.

        Raises
        ------
        ImportError
            If Rust extension is not available.
        ValueError
            If calibration problem setup is invalid.
        RuntimeError
            If optimization fails.
        """
        logger.info(
            (
                f"Starting calibration with "
                f"{self.problem.optimization_config.algorithm.value} algorithm and "
                f"{self.problem.loss_config.function.value} loss function."
            )
        )

        # Convert observed data to Rust types
        rust_observed_data = [
            rust_calibration.ObservedDataPoint(
                step=point.step,
                compartment=point.compartment,
                value=point.value,
                weight=point.weight,
                scale_id=point.scale_id,
            )
            for point in self.problem.observed_data
        ]

        # Convert parameters to Rust types
        rust_parameters = [
            rust_calibration.CalibrationParameter(
                id=param.id,
                parameter_type=self._to_rust_parameter_type(param.parameter_type),
                min_bound=param.min_bound,
                max_bound=param.max_bound,
                initial_guess=param.initial_guess,
            )
            for param in self.problem.parameters
        ]

        # Convert constraints to Rust types
        rust_constraints = [
            rust_calibration.CalibrationConstraint(
                id=constraint.id,
                expression=constraint.expression,
                description=constraint.description,
                weight=constraint.weight,
                time_steps=constraint.time_steps,
            )
            for constraint in self.problem.constraints
        ]

        # Convert loss config to Rust type
        rust_loss_config = self._build_loss_config()

        # Convert optimization config to Rust type
        rust_optimization_config = self._build_optimization_config()

        logger.info("Converted problem definition to Rust types.")
        logger.info("Running optimization...")

        # Get initial population size for initial condition fraction conversion
        initial_population_size = self._get_initial_population_size()

        # Call the Rust calibrate function
        rust_result = rust_calibration.calibrate(
            self._engine,
            rust_observed_data,
            rust_parameters,
            rust_constraints,
            rust_loss_config,
            rust_optimization_config,
            initial_population_size,
        )

        # Convert result back to Python CalibrationResult
        result = CalibrationResult(
            best_parameters=rust_result.best_parameters,
            parameter_names=rust_result.parameter_names,
            best_parameters_list=rust_result.best_parameters_list,
            final_loss=rust_result.final_loss,
            iterations=rust_result.iterations,
            converged=rust_result.converged,
            termination_reason=rust_result.termination_reason,
        )

        logger.info(
            (
                f"Calibration finished after {result.iterations} iterations. "
                f"Final loss: {result.final_loss:.6f}"
            )
        )

        return result

    def run_with_history(self):
        """
        Runs the calibration optimization and returns evaluation history.

        Returns all objective function evaluations that occurred during
        optimization, not just the final best result. This is useful for
        probabilistic calibration where we want to explore the parameter space.

        Returns
        -------
        CalibrationResultWithHistory
            Object containing optimized parameters, final loss, and all evaluations.

        Raises
        ------
        ImportError
            If Rust extension is not available.
        ValueError
            If calibration problem setup is invalid.
        RuntimeError
            If optimization fails.
        """
        logger.info("Starting calibration with history tracking")

        # Convert observed data to Rust types
        rust_observed_data = [
            rust_calibration.ObservedDataPoint(
                step=point.step,
                compartment=point.compartment,
                value=point.value,
                weight=point.weight,
                scale_id=point.scale_id,
            )
            for point in self.problem.observed_data
        ]

        # Convert parameters to Rust types
        rust_parameters = [
            rust_calibration.CalibrationParameter(
                id=param.id,
                parameter_type=self._to_rust_parameter_type(param.parameter_type),
                min_bound=param.min_bound,
                max_bound=param.max_bound,
                initial_guess=param.initial_guess,
            )
            for param in self.problem.parameters
        ]

        # Convert constraints to Rust types
        rust_constraints = [
            rust_calibration.CalibrationConstraint(
                id=constraint.id,
                expression=constraint.expression,
                description=constraint.description,
                weight=constraint.weight,
                time_steps=constraint.time_steps,
            )
            for constraint in self.problem.constraints
        ]

        # Convert loss config to Rust type
        rust_loss_config = self._build_loss_config()

        # Convert optimization config to Rust type
        rust_optimization_config = self._build_optimization_config()

        # Get initial population size for initial condition fraction conversion
        initial_population_size = self._get_initial_population_size()

        # Call the Rust calibrate_with_history function
        rust_result = rust_calibration.calibrate_with_history(
            self._engine,
            rust_observed_data,
            rust_parameters,
            rust_constraints,
            rust_loss_config,
            rust_optimization_config,
            initial_population_size,
        )

        logger.info(
            f"Calibration finished after {rust_result.iterations} iterations. "
            f"Final loss: {rust_result.final_loss:.6f}. "
            f"Collected {len(rust_result.evaluations)} evaluations."
        )

        return rust_result

    def _to_rust_parameter_type(
        self, param_type: CalibrationParameterType
    ) -> "CalibrationParameterTypeProtocol":
        """Convert Python CalibrationParameterType to Rust type."""
        if param_type == CalibrationParameterType.PARAMETER:
            return rust_calibration.CalibrationParameterType.Parameter
        elif param_type == CalibrationParameterType.INITIAL_CONDITION:
            return rust_calibration.CalibrationParameterType.InitialCondition
        elif param_type == CalibrationParameterType.SCALE:
            return rust_calibration.CalibrationParameterType.Scale
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _build_loss_config(self) -> "LossConfigProtocol":
        """Convert Python LossConfig to Rust LossConfig."""
        loss_func = self.problem.loss_config.function

        if loss_func == LossFunction.SSE:
            return rust_calibration.LossConfig.sum_squared_error()
        elif loss_func == LossFunction.RMSE:
            return rust_calibration.LossConfig.root_mean_squared_error()
        elif loss_func == LossFunction.MAE:
            return rust_calibration.LossConfig.mean_absolute_error()
        elif loss_func == LossFunction.WEIGHTED_SSE:
            return rust_calibration.LossConfig.weighted_sse()
        else:
            raise ValueError(f"Unsupported loss function: {loss_func}.")

    def _build_optimization_config(self) -> "OptimizationConfigProtocol":
        """Convert Python OptimizationConfig to Rust OptimizationConfig."""
        opt_config = self.problem.optimization_config

        if opt_config.algorithm == OptimizationAlgorithm.NELDER_MEAD:
            if not isinstance(opt_config.config, NelderMeadConfig):
                raise ValueError(
                    (
                        f"Expected NelderMeadConfig for Nelder-Mead algorithm, "
                        f"got {type(opt_config.config).__name__}"
                    )
                )

            nm_config = rust_calibration.NelderMeadConfig(
                max_iterations=opt_config.config.max_iterations,
                sd_tolerance=opt_config.config.sd_tolerance,
                simplex_perturbation=opt_config.config.simplex_perturbation,
                alpha=opt_config.config.alpha,
                gamma=opt_config.config.gamma,
                rho=opt_config.config.rho,
                sigma=opt_config.config.sigma,
                verbose=opt_config.config.verbose,
                header_interval=opt_config.config.header_interval,
            )
            return rust_calibration.OptimizationConfig.nelder_mead(nm_config)

        elif opt_config.algorithm == OptimizationAlgorithm.PARTICLE_SWARM:
            if not isinstance(opt_config.config, ParticleSwarmConfig):
                raise ValueError(
                    (
                        f"Expected ParticleSwarmConfig for Particle Swarm algorithm, "
                        f"got {type(opt_config.config).__name__}"
                    )
                )

            ps_config = rust_calibration.ParticleSwarmConfig(
                num_particles=opt_config.config.num_particles,
                max_iterations=opt_config.config.max_iterations,
                target_cost=opt_config.config.target_cost,
                inertia_factor=opt_config.config.inertia_factor,
                cognitive_factor=opt_config.config.cognitive_factor,
                social_factor=opt_config.config.social_factor,
                default_acceleration_coefficient=opt_config.config.default_acceleration_coefficient,
                seed=self.problem.seed,  # Use seed from CalibrationProblem
                verbose=opt_config.config.verbose,
                header_interval=opt_config.config.header_interval,
            )
            return rust_calibration.OptimizationConfig.particle_swarm(ps_config)

        else:
            raise ValueError(
                f"Unsupported optimization algorithm: {opt_config.algorithm}"
            )

    @property
    def num_parameters(self) -> int:
        """Number of parameters being calibrated."""
        return len(self.problem.parameters)

    @property
    def num_observations(self) -> int:
        """Number of observed data points."""
        return len(self.problem.observed_data)

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters being calibrated."""
        return [param.id for param in self.problem.parameters]

    def _validate_calibration_parameters(self) -> None:
        """
        Validate that all calibration parameters exist in the model.

        Raises
        ------
        ValueError
            If a parameter ID doesn't exist in the model or if a bin ID is invalid.
        """
        model = self.simulation.model_definition
        model_param_ids = {p.id for p in model.parameters}
        model_bin_ids = {b.id for b in model.population.bins}

        for param in self.problem.parameters:
            if param.parameter_type == CalibrationParameterType.PARAMETER:
                if param.id not in model_param_ids:
                    raise ValueError(
                        f"Calibration parameter '{param.id}' not found in model "
                        f"parameters. Available parameters: "
                        f"{sorted(model_param_ids)}"
                    )
            elif param.parameter_type == CalibrationParameterType.INITIAL_CONDITION:
                if param.id not in model_bin_ids:
                    raise ValueError(
                        f"Calibration initial condition '{param.id}' not found in "
                        f"model bins. Available bins: {sorted(model_bin_ids)}"
                    )
            elif param.parameter_type == CalibrationParameterType.SCALE:
                if param.min_bound <= 0 or param.max_bound <= 0:
                    raise ValueError(
                        f"Scale parameter '{param.id}' must have positive bounds "
                        f"(got min={param.min_bound}, max={param.max_bound})"
                    )
            else:
                raise ValueError(
                    f"Unknown calibration parameter type: {param.parameter_type}"
                )

    def _get_compartment_index(self, bin_id: str) -> int:
        """
        Get the index of a bin/compartment by its ID.

        Parameters
        ----------
        bin_id : str
            The bin identifier

        Returns
        -------
        int
            The index of the compartment in the population vector

        Raises
        ------
        ValueError
            If the bin_id is not found
        """
        bins = self.simulation.model_definition.population.bins
        for idx, bin_obj in enumerate(bins):
            if bin_obj.id == bin_id:
                return idx
        raise ValueError(f"Bin '{bin_id}' not found in model")

    def _get_initial_population_size(self) -> int:
        """Get the initial population size from the model."""
        initial_conditions = (
            self.simulation.model_definition.population.initial_conditions
        )
        return initial_conditions.population_size
