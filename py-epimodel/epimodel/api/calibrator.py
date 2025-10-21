import logging
from typing import TYPE_CHECKING

from epimodel.api.simulation import Simulation
from epimodel.context.calibration import (
    CalibrationProblem,
    CalibrationResult,
    LossFunction,
    NelderMeadConfig,
    OptimizationAlgorithm,
    ParticleSwarmConfig,
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from epimodel.epimodel_rs.epimodel_rs import (
        CalibrationModule,
        DifferenceEquationsProtocol,
        LossConfigProtocol,
        OptimizationConfigProtocol,
    )


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

        try:
            from epimodel.epimodel_rs.epimodel_rs import calibration as rust_calibration
        except ImportError as e:
            raise ImportError(f"Error importing Rust extension: {e}") from e

        # Convert observed data to Rust types
        rust_observed_data = [
            rust_calibration.ObservedDataPoint(
                step=point.step,
                compartment=point.compartment,
                value=point.value,
                weight=point.weight,
            )
            for point in self.problem.observed_data
        ]

        # Convert parameters to Rust types
        rust_parameters = [
            rust_calibration.CalibrationParameter(
                id=param.id,
                min_bound=param.min_bound,
                max_bound=param.max_bound,
                initial_guess=param.initial_guess,
            )
            for param in self.problem.parameters
        ]

        # Convert loss config to Rust type
        rust_loss_config = self._build_loss_config(rust_calibration)

        # Convert optimization config to Rust type
        rust_optimization_config = self._build_optimization_config(rust_calibration)

        logger.info("Converted problem definition to Rust types.")
        logger.info("Running optimization...")

        # Call the Rust calibrate function
        rust_result = rust_calibration.calibrate(
            self._engine,
            rust_observed_data,
            rust_parameters,
            rust_loss_config,
            rust_optimization_config,
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

    def _build_loss_config(
        self, rust_calibration: "CalibrationModule"
    ) -> "LossConfigProtocol":
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

    def _build_optimization_config(
        self, rust_calibration: "CalibrationModule"
    ) -> "OptimizationConfigProtocol":
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
