import logging
from typing import TYPE_CHECKING

from commol.commol_rs import commol_rs
from commol.context.constants import (
    LOSS_MAE,
    LOSS_RMSE,
    LOSS_SSE,
    LOSS_WEIGHTED_SSE,
    OPT_ALG_NELDER_MEAD,
    OPT_ALG_PARTICLE_SWARM,
    PARAM_TYPE_INITIAL_CONDITION,
    PARAM_TYPE_PARAMETER,
    PARAM_TYPE_SCALE,
)

if TYPE_CHECKING:
    from commol.api.simulation import Simulation
    from commol.commol_rs.commol_rs import (
        CalibrationParameterTypeProtocol,
        CalibrationResultWithHistoryProtocol,
        LossConfigProtocol,
        OptimizationConfigProtocol,
    )
    from commol.context.calibration import CalibrationProblem


logger = logging.getLogger(__name__)


class CalibrationRunner:
    """Handles running multiple calibrations in parallel.

    This class is responsible for:
    - Converting Python types to Rust types for the calibration problem
    - Executing parallel calibration runs via Rust/Rayon
    - Returning calibration results with evaluation history

    Parameters
    ----------
    simulation : Simulation
        A fully initialized Simulation object with the model to calibrate
    problem : CalibrationProblem
        A fully constructed and validated calibration problem definition
    seed : int
        Random seed for reproducibility. Each calibration run gets a derived seed
        (seed + run_index).
    """

    def __init__(
        self,
        simulation: "Simulation",
        problem: "CalibrationProblem",
        seed: int,
    ):
        self.simulation = simulation
        self.problem = problem
        self.seed = seed

    def run_multiple(
        self,
        n_runs: int,
    ) -> list["CalibrationResultWithHistoryProtocol"]:
        """Run multiple calibration attempts in parallel using Rust.

        Parameters
        ----------
        n_runs : int
            Number of calibration runs to perform

        Returns
        -------
        list[CalibrationResultWithHistoryProtocol]
            List of calibration results with evaluation history

        Raises
        ------
        RuntimeError
            If all calibration runs fail
        """
        logger.info(f"Running {n_runs} calibrations in parallel")

        # Convert observed data to Rust types
        rust_observed_data = [
            commol_rs.calibration.ObservedDataPoint(
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
            commol_rs.calibration.CalibrationParameter(
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
            commol_rs.calibration.CalibrationConstraint(
                id=constraint.id,
                expression=constraint.expression,
                description=constraint.description,
                weight=constraint.weight,
                time_steps=constraint.time_steps,
            )
            for constraint in self.problem.constraints
        ]

        # Convert loss and optimization configs
        rust_loss_config = self._build_loss_config()
        rust_optimization_config = self._build_optimization_config()

        # Get initial population size
        initial_population_size = self._get_initial_population_size()

        # Call Rust function for parallel execution
        try:
            results = commol_rs.calibration.run_multiple_calibrations(
                self.simulation.engine,
                rust_observed_data,
                rust_parameters,
                rust_constraints,
                rust_loss_config,
                rust_optimization_config,
                initial_population_size,
                n_runs,
                self.seed,
            )
            logger.info(f"Completed {len(results)}/{n_runs} calibrations successfully")
            return results
        except Exception as e:
            raise RuntimeError(f"Parallel calibrations failed: {e}") from e

    def _to_rust_parameter_type(
        self, param_type: str
    ) -> "CalibrationParameterTypeProtocol":
        """Convert Python CalibrationParameterType to Rust type."""
        if param_type == PARAM_TYPE_PARAMETER:
            return commol_rs.calibration.CalibrationParameterType.Parameter
        elif param_type == PARAM_TYPE_INITIAL_CONDITION:
            return commol_rs.calibration.CalibrationParameterType.InitialCondition
        elif param_type == PARAM_TYPE_SCALE:
            return commol_rs.calibration.CalibrationParameterType.Scale
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _build_loss_config(self) -> "LossConfigProtocol":
        """Convert Python loss function to Rust LossConfig."""
        loss_func = self.problem.loss_function

        if loss_func == LOSS_SSE:
            return commol_rs.calibration.LossConfig.sse()
        elif loss_func == LOSS_RMSE:
            return commol_rs.calibration.LossConfig.rmse()
        elif loss_func == LOSS_MAE:
            return commol_rs.calibration.LossConfig.mae()
        elif loss_func == LOSS_WEIGHTED_SSE:
            return commol_rs.calibration.LossConfig.weighted_sse()
        else:
            raise ValueError(f"Unsupported loss function: {loss_func}")

    def _build_optimization_config(self) -> "OptimizationConfigProtocol":
        """Convert Python OptimizationConfig to Rust OptimizationConfig."""
        from commol.context.calibration import (
            NelderMeadConfig,
            ParticleSwarmConfig,
        )

        opt_config = self.problem.optimization_config

        if opt_config.algorithm == OPT_ALG_NELDER_MEAD:
            if not isinstance(opt_config.config, NelderMeadConfig):
                raise ValueError(
                    f"Expected NelderMeadConfig for Nelder-Mead algorithm, "
                    f"got {type(opt_config.config).__name__}"
                )

            nm_config = commol_rs.calibration.NelderMeadConfig(
                max_iterations=opt_config.config.max_iterations,
                sd_tolerance=opt_config.config.sd_tolerance,
                alpha=opt_config.config.alpha,
                gamma=opt_config.config.gamma,
                rho=opt_config.config.rho,
                sigma=opt_config.config.sigma,
                verbose=opt_config.config.verbose,
                header_interval=opt_config.config.header_interval,
            )
            return commol_rs.calibration.OptimizationConfig.nelder_mead(nm_config)

        elif opt_config.algorithm == OPT_ALG_PARTICLE_SWARM:
            if not isinstance(opt_config.config, ParticleSwarmConfig):
                raise ValueError(
                    f"Expected ParticleSwarmConfig for Particle Swarm algorithm, "
                    f"got {type(opt_config.config).__name__}"
                )

            ps_config = commol_rs.calibration.ParticleSwarmConfig(
                num_particles=opt_config.config.num_particles,
                max_iterations=opt_config.config.max_iterations,
                target_cost=opt_config.config.target_cost,
                inertia_factor=opt_config.config.inertia_factor,
                cognitive_factor=opt_config.config.cognitive_factor,
                social_factor=opt_config.config.social_factor,
                seed=self.problem.seed,
                verbose=opt_config.config.verbose,
                header_interval=opt_config.config.header_interval,
            )
            return commol_rs.calibration.OptimizationConfig.particle_swarm(ps_config)

        else:
            raise ValueError(
                f"Unsupported optimization algorithm: {opt_config.algorithm}"
            )

    def _get_initial_population_size(self) -> int:
        """Get the initial population size from the model."""
        initial_conditions = (
            self.simulation.model_definition.population.initial_conditions
        )
        return initial_conditions.population_size
