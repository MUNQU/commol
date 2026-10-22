"""Calibration of a model against observed data."""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import (
        CalibrationProblemProtocol,
        CalibrationResultProtocol,
        EngineProtocol,
        LossConfigProtocol,
        ObservationDrawsProtocol,
        OptimizationConfigProtocol,
        PredictiveProtocol,
    )
    from commol.context.model import Model

from commol.api.simulation import Simulation, check_seed, draw_seed
from commol.commol_rs import _commol_rs as commol_rs
from commol.context._model import engine_state
from commol.context.calibration import (
    CalibrationEvaluation,
    CalibrationParameter,
    CalibrationHistory,
    CalibrationProblem,
    CalibrationResult,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    PSOChaoticInertia,
    PSOConstantAcceleration,
    PSOConstantInertia,
    PSOTimeVaryingAcceleration,
    StartSummary,
)
from commol.context.constants import (
    CalibrationParameterType,
    OptimizationAlgorithm,
)
from commol.context.inference.local import DeltaBand, LocalResult
from commol.context.inference.method import IdentifiabilityMethod, UncertaintyMethod
from commol.context.inference.noise import (
    DistributionNoise,
    ObservationNoise,
    ResidualNoise,
)
from commol.context.inference.sample import ParameterSample, SampleMethod

rust_calibration = commol_rs.calibration

logger = logging.getLogger(__name__)

ResultT = TypeVar("ResultT", bound=BaseModel)


class Calibrator:
    """
    Calibrate the parameters of a simulation against observed data.

    The calibration problem is built in Rust once, when the calibrator is
    created, and every optimization, prediction and uncertainty method reads
    that one problem.
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

        Raises
        ------
        ValueError
            If a parameter, observation, position or constraint does not fit
            the model.
        """
        logger.info(
            f"Initializing Calibration for model: '{simulation.model_definition.name}'"
        )
        self.simulation: Simulation = simulation
        self.problem: CalibrationProblem = problem
        self._engine: "EngineProtocol" = simulation.engine

        logger.info(
            (
                f"Calibration initialized with {len(problem.parameters)} parameters "
                f"and {len(problem.observed_data)} observed data points."
            )
        )

        loss_config = self._build_loss_config()
        self._validate_calibration_parameters()
        self._validate_observed_data()
        self._warn_constraint_positions_beyond_observations()
        models = problem.observation_models
        self._rust_problem: "CalibrationProblemProtocol" = (
            commol_rs.inference.CalibrationProblem(
                self._engine,
                engine_state.rust_observed_data(problem.observed_data),
                engine_state.rust_parameters(problem.parameters),
                engine_state.rust_constraints(problem.constraints),
                loss_config=loss_config,
                initial_condition_remainder=problem.initial_condition_remainder,
                observation_models=(
                    None
                    if models is None
                    else engine_state.rust_observation_models(models)
                ),
            )
        )

    @property
    def rust_problem(self) -> "CalibrationProblemProtocol":
        """
        The calibration problem built in Rust, for the runners of methods.

        Returns
        -------
        CalibrationProblemProtocol
            The problem every optimization, prediction and method reads.
        """
        return self._rust_problem

    def rust_predictive(
        self, sample: ParameterSample, positions: list[float]
    ) -> "PredictiveProtocol":
        """
        The run of every draw of `sample` at `positions`, for a `Predictive`:
        each replicate of a stochastic engine has a seed of its own, derived
        from the calibrator's seed.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the problem's order.
        positions : list[float]
            Positions to report at, sorted and unique.

        Returns
        -------
        PredictiveProtocol
            The runs, computed in Rust.
        """
        return self._rust_problem.predictive(sample.rust_sample, positions, self.seed)

    def with_model(self, model: "Model") -> "Calibrator":
        """
        The same calibration problem on another model, with the same seed.

        The draws of a predictive of the copy take the seeds they take here,
        so `Scenarios` runs every scenario on common random numbers.

        Parameters
        ----------
        model : Model
            The model, such as a copy of this one with other parameter values.

        Returns
        -------
        Calibrator
            A calibrator of `model`, with `problem` and its seed set to
            `seed`.

        Raises
        ------
        ValueError
            If a parameter, observation, position or constraint of the problem
            does not fit `model`.
        """
        return Calibrator(
            Simulation(model), self.problem.model_copy(update={"seed": self.seed})
        )

    def point_sample(self, result: CalibrationResult) -> ParameterSample:
        """
        The point estimate of a fit as a sample of one draw.

        A `Predictive` or `Scenarios` of it runs the model at the fit, so a
        fit is read as any sample is.

        Parameters
        ----------
        result : CalibrationResult
            A fit of this calibration problem.

        Returns
        -------
        ParameterSample
            One draw at ``result.best_parameters``, with the names and bounds
            of the parameters of the problem and
            ``SampleMethod(name="point", settings={"seed": result.seed})``.

        Raises
        ------
        ValueError
            If the names of `result` are not those of the problem; the message
            names the names that differ.
        """
        names = self.parameter_names
        missing = [name for name in names if name not in result.best_parameters]
        extra = [name for name in result.best_parameters if name not in names]
        if missing or extra:
            raise ValueError(
                f"The result is not a fit of this problem: it lacks {missing} and "
                f"has {extra}, which the problem does not have."
            )
        return ParameterSample.equally_weighted(
            names,
            [(p.min_bound, p.max_bound) for p in self.problem.parameters],
            [[result.best_parameters[name] for name in names]],
            SampleMethod(name="point", settings={"seed": result.seed}),
        )

    def delta_band(
        self, result: LocalResult, positions: Sequence[float], confidence_level: float
    ) -> DeltaBand:
        """
        The band of every output of a local approximation by the delta method.

        The model is linearized at the fit of `result`: the variance of an
        output is ``J Σ Jᵀ``, with ``J`` its derivatives on the unbounded
        scale and ``Σ`` the covariance of `result`. The band is the
        prediction at the fit plus and minus the quantile of a normal, or of
        the Student t of a `SandwichResult`, times its standard deviation. It
        runs the model a few times per parameter, not once per draw, and reads
        only the fields of `result`, so a result read back from JSON works.

        Parameters
        ----------
        result : LaplaceResult | SandwichResult
            A local approximation of this calibration problem.
        positions : Sequence[float]
            Positions to report at.
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        DeltaBand
            The prediction at the fit and the ends of the interval of each
            output at each position.

        Raises
        ------
        ValueError
            If the result's parameters are not the problem's, the level is not
            strictly between 0 and 1, or the engine rejects a position.
        """
        if result.sample.names != self.parameter_names:
            raise ValueError(
                f"The result has the parameters {result.sample.names}; the problem "
                f"has {self.parameter_names}."
            )
        points = [float(position) for position in positions]
        outputs, center, lower, upper = commol_rs.inference.delta_band(
            self._rust_problem,
            [result.fit.best_parameters[name] for name in self.parameter_names],
            result.covariance,
            result.degrees_of_freedom(),
            points,
            confidence_level,
        )
        return DeltaBand(
            confidence_level=confidence_level,
            positions=points,
            center=dict(zip(outputs, center, strict=True)),
            lower=dict(zip(outputs, lower, strict=True)),
            upper=dict(zip(outputs, upper, strict=True)),
        )

    def rust_observation_draws(
        self,
        sample: ParameterSample,
        observed_data: list[ObservedDataPoint],
        noise: ObservationNoise,
    ) -> "ObservationDrawsProtocol":
        """
        Every draw's value of `observed_data` with `noise`, for a `Predictive`.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the problem's order.
        observed_data : list[ObservedDataPoint]
            The observations.
        noise : NoNoise | ResidualNoise | DistributionNoise
            The observation noise, drawn from the calibrator's seed.

        Returns
        -------
        ObservationDrawsProtocol
            The draws, computed in Rust.
        """
        residual_noise = None
        if isinstance(noise, ResidualNoise):
            residual_noise = (
                [noise.fit.best_parameters[name] for name in self.parameter_names],
                noise.block_length,
                noise.residual_scale,
            )
        return self._rust_problem.observation_draws(
            sample.rust_sample,
            engine_state.rust_observed_data(observed_data),
            self.seed,
            residual_noise=residual_noise,
            distribution_noise=isinstance(noise, DistributionNoise),
        )

    def rust_log_score(
        self, sample: ParameterSample, observed_data: list[ObservedDataPoint]
    ) -> list[float]:
        """
        The log score of each observation under the predictive of `sample`,
        for `commol.scores.log_score`.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the problem's order.
        observed_data : list[ObservedDataPoint]
            The observations.

        Returns
        -------
        list[float]
            One log score per observation, computed in Rust.
        """
        return self._rust_problem.log_score(
            sample.rust_sample,
            engine_state.rust_observed_data(observed_data),
            self.seed,
        )

    @property
    def rust_optimizer(self) -> "OptimizationConfigProtocol":
        """
        The optimizer built in Rust, for the runners of methods.

        Returns
        -------
        OptimizationConfigProtocol
            The optimizer of ``problem.optimization_config``.
        """
        return self._build_optimization_config()

    @property
    def seed(self) -> int:
        """
        The seed of the calibration's random numbers.

        Returns
        -------
        int
            ``problem.seed``, or a seed drawn from the operating system once
            per calibrator and logged.
        """
        if self.problem.seed is not None:
            return check_seed(self.problem.seed)
        if not hasattr(self, "_drawn_seed"):
            self._drawn_seed = draw_seed()
            logger.info(
                "No seed set; drew the seed %d, which repeats this calibration.",
                self._drawn_seed,
            )
        return self._drawn_seed

    def run(self, starts: int = 1) -> CalibrationResult:
        """
        Runs the calibration optimization from one or more starts.

        Start 0 is the initial guess of each parameter. Each further start is a
        row of a Latin hypercube over the bounds, drawn from `seed`, so the
        starts cover the parameter space. They run in parallel, on the
        threads of `commol.threads`; the result is the best of them, and lists
        every start. Distinct ends with equal losses are a sign that the data
        do not identify the parameters.

        Parameters
        ----------
        starts : int, optional
            Number of starts, at least 1 (default 1). A verbose optimizer
            shows one start, so it needs ``starts=1``.

        Returns
        -------
        CalibrationResult
            The best start: the optimized parameter values, in the order of
            the problem's parameters, its final loss and convergence, every
            start (``starts``), the reason of each start that failed
            (``failed_runs``, also logged as warnings) and the seed
            (``seed``).

        Raises
        ------
        ValueError
            If `starts` is below 1, or a verbose optimizer has more than one
            start.
        RuntimeError
            If every start fails.
        """
        algorithm_name = (
            OptimizationAlgorithm.NELDER_MEAD
            if isinstance(self.problem.optimization_config, NelderMeadConfig)
            else OptimizationAlgorithm.PARTICLE_SWARM
        )
        measure = (
            f"{self.problem.loss_function} loss function"
            if self.problem.loss_function is not None
            else "the negative log-likelihood of its observation models"
        )
        logger.info(
            f"Starting calibration with {algorithm_name} algorithm, "
            f"{measure} and {starts} start(s)."
        )
        if starts < 1:
            raise ValueError(f"A calibration needs at least one start, got {starts}.")
        seed = self.seed if starts > 1 else 0
        outcomes = self._rust_problem.optimize(
            self._build_optimization_config(), starts, seed
        )
        names = self.parameter_names
        summaries: list[tuple[StartSummary, "CalibrationResultProtocol"]] = []
        failed_runs: list[str] = []
        for index, outcome in enumerate(outcomes):
            if outcome.result is None:
                reason = f"Start {index} failed: {outcome.error}"
                logger.warning(reason)
                failed_runs.append(reason)
                continue
            summaries.append(
                (
                    StartSummary(
                        start=dict(zip(names, outcome.start, strict=True)),
                        end=dict(outcome.result.best_parameters),
                        loss=outcome.result.final_loss,
                        converged=outcome.result.converged,
                    ),
                    outcome.result,
                )
            )
        if not summaries:
            raise RuntimeError("Every start failed: " + "; ".join(failed_runs))
        best = min(summaries, key=lambda summary: summary[0].loss)[1]
        result = CalibrationResult(
            best_parameters=dict(best.best_parameters),
            final_loss=best.final_loss,
            iterations=best.iterations,
            converged=best.converged,
            termination_reason=best.termination_reason,
            starts=[summary for summary, _ in summaries],
            failed_runs=failed_runs,
            seed=self.seed,
        )
        logger.info(
            (
                f"Calibration finished after {result.iterations} iterations. "
                f"Final loss: {result.final_loss:.6f}"
            )
        )
        return result

    def run_with_history(self) -> CalibrationHistory:
        """
        Runs the calibration from the initial guess and keeps every evaluation.

        Returns
        -------
        CalibrationHistory
            The result, and every evaluation of a feasible point the optimizer
            made, in order. Infeasible points, which the objective penalizes,
            are not kept.

        Raises
        ------
        RuntimeError
            If optimization fails.
        """
        logger.info("Starting calibration with history tracking")
        history = self._rust_problem.optimize_with_history(
            self._build_optimization_config()
        )
        names = self.parameter_names
        rust_result = history.result
        result = CalibrationResult(
            best_parameters=dict(rust_result.best_parameters),
            final_loss=rust_result.final_loss,
            iterations=rust_result.iterations,
            converged=rust_result.converged,
            termination_reason=rust_result.termination_reason,
            seed=self.seed,
        )
        evaluations = [
            CalibrationEvaluation(
                parameters=dict(zip(names, evaluation.parameters, strict=True)),
                loss=evaluation.loss,
            )
            for evaluation in history.evaluations
        ]
        logger.info(
            f"Calibration finished after {result.iterations} iterations. "
            f"Final loss: {result.final_loss:.6f}. "
            f"Collected {len(evaluations)} evaluations."
        )
        return CalibrationHistory(result=result, evaluations=evaluations)

    def uncertainty(self, method: UncertaintyMethod[ResultT]) -> ResultT:
        """
        Run an uncertainty method on this calibration problem.

        Each method is a configuration (a subclass of `UncertaintyMethod`)
        whose type sets the result type; its runner is registered in
        `commol.api.inference.registry`. The guide's table "Which method
        answers which question" lists them.

        Parameters
        ----------
        method : UncertaintyMethod[ResultT]
            The configuration of the method.

        Returns
        -------
        ResultT
            The result of the method.

        Raises
        ------
        TypeError
            If `method` is an identifiability method; the message names
            ``Calibrator.identifiability``.
        ValueError
            If no runner is registered for the configuration's type; the
            message names the registered ones.
        """
        from commol.api.inference.registry import run_method

        return run_method(self, method, UncertaintyMethod)

    def identifiability(self, method: IdentifiabilityMethod[ResultT]) -> ResultT:
        """
        Run an identifiability method on this calibration problem.

        Each method answers which parameters the model and the data
        determine, from another kind of evidence: `StructuralRank`,
        `Collinearity`, `SampleIdentifiability` or `StartsIdentifiability`.
        Each returns an `IdentifiabilityReport`; `IdentifiabilityReport.merge`
        joins them.

        Parameters
        ----------
        method : IdentifiabilityMethod[ResultT]
            The configuration of the method.

        Returns
        -------
        ResultT
            The result of the method.

        Raises
        ------
        TypeError
            If `method` is an uncertainty method; the message names
            ``Calibrator.uncertainty``.
        ValueError
            If no runner is registered for the configuration's type, or the
            method fails (see the method).
        """
        from commol.api.inference.registry import run_method

        return run_method(self, method, IdentifiabilityMethod)

    def _build_loss_config(self) -> "LossConfigProtocol | None":
        """The Rust loss, with the replicates of a stochastic engine; None with
        observation models."""
        if self.problem.loss_function is None:
            return None
        return engine_state.rust_loss_config(
            self.problem,
            self.problem.loss_function,
            self.simulation.model_definition.engine.type,
        )

    def _build_optimization_config(self) -> "OptimizationConfigProtocol":
        """Convert Python OptimizationConfig to Rust OptimizationConfig."""
        opt_config = self.problem.optimization_config

        if isinstance(opt_config, NelderMeadConfig):
            nm_config = rust_calibration.NelderMeadConfig(
                max_iterations=opt_config.max_iterations,
                sd_tolerance=opt_config.sd_tolerance,
                initial_simplex_size=opt_config.initial_simplex_size,
                alpha=opt_config.alpha,
                gamma=opt_config.gamma,
                rho=opt_config.rho,
                sigma=opt_config.sigma,
                verbose=opt_config.verbose,
                header_interval=opt_config.header_interval,
            )
            return rust_calibration.OptimizationConfig.nelder_mead(nm_config)

        return self._build_pso_config(opt_config)

    def _build_pso_config(
        self, opt_config: ParticleSwarmConfig
    ) -> "OptimizationConfigProtocol":
        """Convert ParticleSwarmConfig to Rust OptimizationConfig."""
        rust_inertia = self._build_rust_inertia(opt_config.inertia_config)
        rust_acceleration = self._build_rust_acceleration(
            opt_config.acceleration_config
        )

        mutation = opt_config.mutation_config
        rust_mutation = None
        if mutation is not None:
            rust_mutation = rust_calibration.PSOMutation(
                strategy=mutation.strategy,
                scale=mutation.scale,
                probability=mutation.probability,
                application=mutation.application,
            )

        velocity = opt_config.velocity_config
        rust_velocity = None
        if velocity is not None:
            rust_velocity = rust_calibration.PSOVelocity(
                clamp_factor=velocity.clamp_factor,
                mutation_threshold=velocity.mutation_threshold,
            )

        improvement = opt_config.improvement_stop_config
        radius = opt_config.radius_stop_config
        ps_config = rust_calibration.ParticleSwarmConfig(
            num_particles=opt_config.num_particles,
            max_iterations=opt_config.max_iterations,
            verbose=opt_config.verbose,
            inertia=rust_inertia,
            acceleration=rust_acceleration,
            mutation=rust_mutation,
            velocity=rust_velocity,
            initialization=opt_config.initialization,
            seed=self.seed,
            improvement_stop=(
                None
                if improvement is None
                else (improvement.window, improvement.tol_abs, improvement.tol_rel)
            ),
            radius_stop=None if radius is None else radius.tol,
        )
        return rust_calibration.OptimizationConfig.particle_swarm(ps_config)

    def _build_rust_inertia(
        self, inertia: PSOConstantInertia | PSOChaoticInertia | None
    ):
        """Convert Python inertia config to Rust type."""
        if inertia is None:
            return None
        if isinstance(inertia, PSOConstantInertia):
            return rust_calibration.PSOInertiaConstant(factor=inertia.factor)
        return rust_calibration.PSOInertiaChaotic(
            w_min=inertia.w_min, w_max=inertia.w_max
        )

    def _build_rust_acceleration(
        self, acceleration: PSOConstantAcceleration | PSOTimeVaryingAcceleration | None
    ):
        """Convert Python acceleration config to Rust type."""
        if acceleration is None:
            return None
        if isinstance(acceleration, PSOConstantAcceleration):
            return rust_calibration.PSOAccelerationConstant(
                cognitive=acceleration.cognitive,
                social=acceleration.social,
            )
        return rust_calibration.PSOAccelerationTimeVarying(
            c1_initial=acceleration.c1_initial,
            c1_final=acceleration.c1_final,
            c2_initial=acceleration.c2_initial,
            c2_final=acceleration.c2_final,
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
        check_values(
            self.simulation.model_definition,
            self._engine,
            self.problem.parameters,
            self.problem.initial_condition_remainder,
        )

    def _validate_observed_data(self) -> None:
        """Validate that observed data compartments exist and have valid steps.

        Raises
        ------
        ValueError
            If observed data contains invalid compartments or negative positions.
        """
        simulation_output_names = set(self._engine.output_names)

        for obs in self.problem.observed_data:
            observed_outputs = obs.observed_outputs
            missing_outputs = [
                output
                for output in observed_outputs
                if output not in simulation_output_names
            ]
            if missing_outputs:
                raise ValueError(
                    f"Observed data output(s) {missing_outputs} for observation "
                    f"'{obs.label}' not found in model simulation outputs. "
                    f"Available outputs: {sorted(simulation_output_names)}"
                )

        engine_state.check_positions(
            self._engine, self.problem.observed_data, self.problem.constraints
        )

    def _warn_constraint_positions_beyond_observations(self) -> None:
        """Warn when constraints at positions extend beyond observed data."""
        max_observed_position = max(obs.at for obs in self.problem.observed_data)
        for constraint in self.problem.constraints:
            if constraint.at and max(constraint.at) > max_observed_position:
                logger.warning(
                    "Constraint '%s' has positions beyond the maximum observed "
                    "position (%s); simulation will be extended to evaluate them.",
                    constraint.id,
                    max_observed_position,
                )


def check_values(
    model: "Model",
    engine: "EngineProtocol",
    parameters: Sequence[CalibrationParameter],
    remainder: str | None,
) -> None:
    """
    Check that a model has every value a calibration or an analysis sets.

    A parameter must be one of the model's, a factor must have positive
    bounds, and the structure and the engine check the values of the model
    kind, such as initial conditions, and the remainder.

    Parameters
    ----------
    model : Model
        The model the values are set on.
    engine : EngineProtocol
        The engine built from it.
    parameters : Sequence[CalibrationParameter]
        The values set, with their types and bounds.
    remainder : str | None
        The id that takes what the initial conditions leave, if any.

    Raises
    ------
    ValueError
        If a parameter is not one of the model's, a factor has a bound at or
        below 0, or the structure or the engine rejects a value.
    """
    model_param_ids = {p.id for p in model.parameters}
    for param in parameters:
        if param.parameter_type == CalibrationParameterType.PARAMETER:
            if param.id not in model_param_ids:
                raise ValueError(
                    f"Calibration parameter '{param.id}' not found in model "
                    f"parameters. Available parameters: "
                    f"{sorted(model_param_ids)}"
                )
        elif param.parameter_type == CalibrationParameterType.FACTOR:
            if param.min_bound <= 0 or param.max_bound <= 0:
                raise ValueError(
                    f"Factor parameter '{param.id}' must have positive bounds "
                    f"(got min={param.min_bound}, max={param.max_bound})"
                )
    engine_state.check_model_values(model, engine, parameters, remainder)
    model.structure.check_calibration_values(model, engine, parameters, remainder)
