import logging
from typing import TYPE_CHECKING

from commol.commol_rs import commol_rs
from commol.context.calibration import (
    PSOChaoticInertia,
    PSOConstantAcceleration,
    PSOConstantInertia,
    PSOTimeVaryingAcceleration,
)
from commol.context.constants import (
    CalibrationParameterType,
    LossFunction,
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
        A fully initialized Simulation object with the model to calibrate.
    problem : CalibrationProblem
        A fully constructed and validated calibration problem definition.
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
            Number of calibration runs to perform.

        Returns
        -------
        list[CalibrationResultWithHistoryProtocol]
            List of calibration results with evaluation history.

        Raises
        ------
        RuntimeError
            If all calibration runs fail.
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
        if param_type == CalibrationParameterType.PARAMETER:
            return commol_rs.calibration.CalibrationParameterType.Parameter
        elif param_type == CalibrationParameterType.INITIAL_CONDITION:
            return commol_rs.calibration.CalibrationParameterType.InitialCondition
        elif param_type == CalibrationParameterType.SCALE:
            return commol_rs.calibration.CalibrationParameterType.Scale
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _build_loss_config(self) -> "LossConfigProtocol":
        """Convert Python loss function to Rust LossConfig."""
        loss_func = self.problem.loss_function

        if loss_func == LossFunction.SSE:
            return commol_rs.calibration.LossConfig.sse()
        elif loss_func == LossFunction.RMSE:
            return commol_rs.calibration.LossConfig.rmse()
        elif loss_func == LossFunction.MAE:
            return commol_rs.calibration.LossConfig.mae()
        elif loss_func == LossFunction.WEIGHTED_SSE:
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

        if isinstance(opt_config, NelderMeadConfig):
            nm_config = commol_rs.calibration.NelderMeadConfig(
                max_iterations=opt_config.max_iterations,
                sd_tolerance=opt_config.sd_tolerance,
                alpha=opt_config.alpha,
                gamma=opt_config.gamma,
                rho=opt_config.rho,
                sigma=opt_config.sigma,
                verbose=opt_config.verbose,
                header_interval=opt_config.header_interval,
            )
            return commol_rs.calibration.OptimizationConfig.nelder_mead(nm_config)

        elif isinstance(opt_config, ParticleSwarmConfig):
            return self._build_pso_config(opt_config)

        else:
            raise ValueError(
                f"Unsupported optimization config type: {type(opt_config).__name__}"
            )

    def _build_pso_config(self, opt_config) -> "OptimizationConfigProtocol":
        """Convert ParticleSwarmConfig to Rust OptimizationConfig."""
        rust_inertia = self._build_rust_inertia(opt_config.inertia_config)
        rust_acceleration = self._build_rust_acceleration(
            opt_config.acceleration_config
        )

        mutation = opt_config.mutation_config
        rust_mutation = None
        if mutation is not None:
            rust_mutation = commol_rs.calibration.PSOMutation(
                strategy=mutation.strategy,
                scale=mutation.scale,
                probability=mutation.probability,
                application=mutation.application,
            )

        velocity = opt_config.velocity_config
        rust_velocity = None
        if velocity is not None:
            rust_velocity = commol_rs.calibration.PSOVelocity(
                clamp_factor=velocity.clamp_factor,
                mutation_threshold=velocity.mutation_threshold,
            )

        ps_config = commol_rs.calibration.ParticleSwarmConfig(
            num_particles=opt_config.num_particles,
            max_iterations=opt_config.max_iterations,
            verbose=opt_config.verbose,
            inertia=rust_inertia,
            acceleration=rust_acceleration,
            mutation=rust_mutation,
            velocity=rust_velocity,
            initialization=opt_config.initialization,
            seed=self.problem.seed,
        )
        return commol_rs.calibration.OptimizationConfig.particle_swarm(ps_config)

    def _build_rust_inertia(
        self, inertia: PSOConstantInertia | PSOChaoticInertia | None
    ):
        """Convert Python inertia config to Rust type."""
        if inertia is None:
            return None
        if isinstance(inertia, PSOConstantInertia):
            return commol_rs.calibration.PSOInertiaConstant(factor=inertia.factor)
        if isinstance(inertia, PSOChaoticInertia):
            return commol_rs.calibration.PSOInertiaChaotic(
                w_min=inertia.w_min, w_max=inertia.w_max
            )
        return None

    def _build_rust_acceleration(
        self, acceleration: PSOConstantAcceleration | PSOTimeVaryingAcceleration | None
    ):
        """Convert Python acceleration config to Rust type."""
        if acceleration is None:
            return None
        if isinstance(acceleration, PSOConstantAcceleration):
            return commol_rs.calibration.PSOAccelerationConstant(
                cognitive=acceleration.cognitive,
                social=acceleration.social,
            )
        if isinstance(acceleration, PSOTimeVaryingAcceleration):
            return commol_rs.calibration.PSOAccelerationTimeVarying(
                c1_initial=acceleration.c1_initial,
                c1_final=acceleration.c1_final,
                c2_initial=acceleration.c2_initial,
                c2_final=acceleration.c2_final,
            )
        return None

    def _get_initial_population_size(self) -> int:
        """Get the initial population size from the model."""
        initial_conditions = (
            self.simulation.model_definition.population.initial_conditions
        )
        return initial_conditions.population_size
