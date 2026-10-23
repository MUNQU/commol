# Calibration API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Fits a model to observations. Guide: [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md).

## Calibrator

Calibrate the parameters of a simulation against observed data.

The calibration problem is built in Rust once, when the calibrator is created, and every optimization, prediction and uncertainty method reads that one problem.

Source code in `commol/api/calibrator.py`

```python
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
            strictly between 0 and 1, the engine is stochastic, or the engine
            rejects a position.
        RuntimeError
            If the derivatives fail at the fit.
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
            start (``starts``), each start that failed (``failed``, also
            logged as warnings) and the seed (``seed``).

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
        failed: list[str] = []
        for index, outcome in enumerate(outcomes):
            if outcome.result is None:
                reason = f"Start {index} failed: {outcome.error}"
                logger.warning(reason)
                failed.append(reason)
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
            raise RuntimeError("Every start failed: " + "; ".join(failed))
        best = min(summaries, key=lambda summary: summary[0].loss)[1]
        result = CalibrationResult(
            best_parameters=dict(best.best_parameters),
            final_loss=best.final_loss,
            iterations=best.iterations,
            converged=best.converged,
            termination_reason=best.termination_reason,
            starts=[summary for summary, _ in summaries],
            failed=failed,
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
            If no runner is registered for the configuration's type (the
            message names the registered ones), or
            a setting, an input, the data or the model does not fit the
            method; the message says why.
        RuntimeError
            If a computation of the method that could start fails: an engine
            run, an optimizer, every replicate, or a sampler that does not
            reach its target.
        InfeasibleStateError
            If a draw of a prediction leaves the values its model allows.
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
            If no runner is registered for the configuration's type, or
            a setting, an input, the data or the model does not fit the
            method; the message says why.
        RuntimeError
            If a computation of the method that could start fails: an engine
            run or an optimizer.
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
            particles=opt_config.particles,
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
```

### Attributes

#### rust_problem

```python
rust_problem: CalibrationProblemProtocol
```

The calibration problem built in Rust, for the runners of methods.

Returns:

| Type                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `CalibrationProblemProtocol` | The problem every optimization, prediction and method reads. |

#### rust_optimizer

```python
rust_optimizer: OptimizationConfigProtocol
```

The optimizer built in Rust, for the runners of methods.

Returns:

| Type                         | Description                                   |
| ---------------------------- | --------------------------------------------- |
| `OptimizationConfigProtocol` | The optimizer of problem.optimization_config. |

#### seed

```python
seed: int
```

The seed of the calibration's random numbers.

Returns:

| Type  | Description                                                                             |
| ----- | --------------------------------------------------------------------------------------- |
| `int` | problem.seed, or a seed drawn from the operating system once per calibrator and logged. |

#### num_parameters

```python
num_parameters: int
```

Number of parameters being calibrated.

#### num_observations

```python
num_observations: int
```

Number of observed data points.

#### parameter_names

```python
parameter_names: list[str]
```

Names of parameters being calibrated.

### Methods:

#### __init__

```python
__init__(simulation: Simulation, problem: CalibrationProblem)
```

Initializes the calibration from a Simulation and CalibrationProblem.

Parameters:

| Name         | Type                 | Description                                                        | Default    |
| ------------ | -------------------- | ------------------------------------------------------------------ | ---------- |
| `simulation` | `Simulation`         | A fully initialized Simulation object with the model to calibrate. | *required* |
| `problem`    | `CalibrationProblem` | A fully constructed and validated calibration problem definition.  | *required* |

Raises:

| Type         | Description                                                                 |
| ------------ | --------------------------------------------------------------------------- |
| `ValueError` | If a parameter, observation, position or constraint does not fit the model. |

Source code in `commol/api/calibrator.py`

```python
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
```

#### rust_predictive

```python
rust_predictive(sample: ParameterSample, positions: list[float]) -> PredictiveProtocol
```

The run of every draw of `sample` at `positions`, for a `Predictive`: each replicate of a stochastic engine has a seed of its own, derived from the calibrator's seed.

Parameters:

| Name        | Type              | Description                                | Default    |
| ----------- | ----------------- | ------------------------------------------ | ---------- |
| `sample`    | `ParameterSample` | The draws, in the problem's order.         | *required* |
| `positions` | `list[float]`     | Positions to report at, sorted and unique. | *required* |

Returns:

| Type                 | Description                 |
| -------------------- | --------------------------- |
| `PredictiveProtocol` | The runs, computed in Rust. |

Source code in `commol/api/calibrator.py`

```python
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
```

#### with_model

```python
with_model(model: Model) -> Calibrator
```

The same calibration problem on another model, with the same seed.

The draws of a predictive of the copy take the seeds they take here, so `Scenarios` runs every scenario on common random numbers.

Parameters:

| Name    | Type    | Description                                                        | Default    |
| ------- | ------- | ------------------------------------------------------------------ | ---------- |
| `model` | `Model` | The model, such as a copy of this one with other parameter values. | *required* |

Returns:

| Type         | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| `Calibrator` | A calibrator of model, with problem and its seed set to seed. |

Raises:

| Type         | Description                                                                            |
| ------------ | -------------------------------------------------------------------------------------- |
| `ValueError` | If a parameter, observation, position or constraint of the problem does not fit model. |

Source code in `commol/api/calibrator.py`

```python
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
```

#### point_sample

```python
point_sample(result: CalibrationResult) -> ParameterSample
```

The point estimate of a fit as a sample of one draw.

A `Predictive` or `Scenarios` of it runs the model at the fit, so a fit is read as any sample is.

Parameters:

| Name     | Type                | Description                        | Default    |
| -------- | ------------------- | ---------------------------------- | ---------- |
| `result` | `CalibrationResult` | A fit of this calibration problem. | *required* |

Returns:

| Type              | Description                                                                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ParameterSample` | One draw at result.best_parameters, with the names and bounds of the parameters of the problem and SampleMethod(name="point", settings={"seed": result.seed}). |

Raises:

| Type         | Description                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------- |
| `ValueError` | If the names of result are not those of the problem; the message names the names that differ. |

Source code in `commol/api/calibrator.py`

```python
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
```

#### delta_band

```python
delta_band(result: LocalResult, positions: Sequence[float], confidence_level: float) -> DeltaBand
```

The band of every output of a local approximation by the delta method.

The model is linearized at the fit of `result`: the variance of an output is `J Σ Jᵀ`, with `J` its derivatives on the unbounded scale and `Σ` the covariance of `result`. The band is the prediction at the fit plus and minus the quantile of a normal, or of the Student t of a `SandwichResult`, times its standard deviation. It runs the model a few times per parameter, not once per draw, and reads only the fields of `result`, so a result read back from JSON works.

Parameters:

| Name               | Type              | Description               | Default                                            |
| ------------------ | ----------------- | ------------------------- | -------------------------------------------------- |
| `result`           | \`LaplaceResult   | SandwichResult\`          | A local approximation of this calibration problem. |
| `positions`        | `Sequence[float]` | Positions to report at.   | *required*                                         |
| `confidence_level` | `float`           | Strictly between 0 and 1. | *required*                                         |

Returns:

| Type        | Description                                                                             |
| ----------- | --------------------------------------------------------------------------------------- |
| `DeltaBand` | The prediction at the fit and the ends of the interval of each output at each position. |

Raises:

| Type           | Description                                                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`   | If the result's parameters are not the problem's, the level is not strictly between 0 and 1, the engine is stochastic, or the engine rejects a position. |
| `RuntimeError` | If the derivatives fail at the fit.                                                                                                                      |

Source code in `commol/api/calibrator.py`

```python
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
        strictly between 0 and 1, the engine is stochastic, or the engine
        rejects a position.
    RuntimeError
        If the derivatives fail at the fit.
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
```

#### rust_observation_draws

```python
rust_observation_draws(sample: ParameterSample, observed_data: list[ObservedDataPoint], noise: ObservationNoise) -> ObservationDrawsProtocol
```

Every draw's value of `observed_data` with `noise`, for a `Predictive`.

Parameters:

| Name            | Type                      | Description                        | Default             |
| --------------- | ------------------------- | ---------------------------------- | ------------------- |
| `sample`        | `ParameterSample`         | The draws, in the problem's order. | *required*          |
| `observed_data` | `list[ObservedDataPoint]` | The observations.                  | *required*          |
| `noise`         | \`NoNoise                 | ResidualNoise                      | DistributionNoise\` |

Returns:

| Type                       | Description                  |
| -------------------------- | ---------------------------- |
| `ObservationDrawsProtocol` | The draws, computed in Rust. |

Source code in `commol/api/calibrator.py`

```python
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
```

#### rust_log_score

```python
rust_log_score(sample: ParameterSample, observed_data: list[ObservedDataPoint]) -> list[float]
```

The log score of each observation under the predictive of `sample`, for `commol.scores.log_score`.

Parameters:

| Name            | Type                      | Description                        | Default    |
| --------------- | ------------------------- | ---------------------------------- | ---------- |
| `sample`        | `ParameterSample`         | The draws, in the problem's order. | *required* |
| `observed_data` | `list[ObservedDataPoint]` | The observations.                  | *required* |

Returns:

| Type          | Description                                      |
| ------------- | ------------------------------------------------ |
| `list[float]` | One log score per observation, computed in Rust. |

Source code in `commol/api/calibrator.py`

```python
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
```

#### run

```python
run(starts: int = 1) -> CalibrationResult
```

Runs the calibration optimization from one or more starts.

Start 0 is the initial guess of each parameter. Each further start is a row of a Latin hypercube over the bounds, drawn from `seed`, so the starts cover the parameter space. They run in parallel, on the threads of `commol.threads`; the result is the best of them, and lists every start. Distinct ends with equal losses are a sign that the data do not identify the parameters.

Parameters:

| Name     | Type  | Description                                                                                          | Default |
| -------- | ----- | ---------------------------------------------------------------------------------------------------- | ------- |
| `starts` | `int` | Number of starts, at least 1 (default 1). A verbose optimizer shows one start, so it needs starts=1. | `1`     |

Returns:

| Type                | Description                                                                                                                                                                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CalibrationResult` | The best start: the optimized parameter values, in the order of the problem's parameters, its final loss and convergence, every start (starts), each start that failed (failed, also logged as warnings) and the seed (seed). |

Raises:

| Type           | Description                                                           |
| -------------- | --------------------------------------------------------------------- |
| `ValueError`   | If starts is below 1, or a verbose optimizer has more than one start. |
| `RuntimeError` | If every start fails.                                                 |

Source code in `commol/api/calibrator.py`

```python
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
        start (``starts``), each start that failed (``failed``, also
        logged as warnings) and the seed (``seed``).

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
    failed: list[str] = []
    for index, outcome in enumerate(outcomes):
        if outcome.result is None:
            reason = f"Start {index} failed: {outcome.error}"
            logger.warning(reason)
            failed.append(reason)
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
        raise RuntimeError("Every start failed: " + "; ".join(failed))
    best = min(summaries, key=lambda summary: summary[0].loss)[1]
    result = CalibrationResult(
        best_parameters=dict(best.best_parameters),
        final_loss=best.final_loss,
        iterations=best.iterations,
        converged=best.converged,
        termination_reason=best.termination_reason,
        starts=[summary for summary, _ in summaries],
        failed=failed,
        seed=self.seed,
    )
    logger.info(
        (
            f"Calibration finished after {result.iterations} iterations. "
            f"Final loss: {result.final_loss:.6f}"
        )
    )
    return result
```

#### run_with_history

```python
run_with_history() -> CalibrationHistory
```

Runs the calibration from the initial guess and keeps every evaluation.

Returns:

| Type                 | Description                                                                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CalibrationHistory` | The result, and every evaluation of a feasible point the optimizer made, in order. Infeasible points, which the objective penalizes, are not kept. |

Raises:

| Type           | Description            |
| -------------- | ---------------------- |
| `RuntimeError` | If optimization fails. |

Source code in `commol/api/calibrator.py`

```python
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
```

#### uncertainty

```python
uncertainty(method: UncertaintyMethod[ResultT]) -> ResultT
```

Run an uncertainty method on this calibration problem.

Each method is a configuration (a subclass of `UncertaintyMethod`) whose type sets the result type; its runner is registered in `commol.api.inference.registry`. The guide's table "Which method answers which question" lists them.

Parameters:

| Name     | Type                         | Description                      | Default    |
| -------- | ---------------------------- | -------------------------------- | ---------- |
| `method` | `UncertaintyMethod[ResultT]` | The configuration of the method. | *required* |

Returns:

| Type      | Description               |
| --------- | ------------------------- |
| `ResultT` | The result of the method. |

Raises:

| Type                   | Description                                                                                                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TypeError`            | If method is an identifiability method; the message names Calibrator.identifiability.                                                                                                         |
| `ValueError`           | If no runner is registered for the configuration's type (the message names the registered ones), or a setting, an input, the data or the model does not fit the method; the message says why. |
| `RuntimeError`         | If a computation of the method that could start fails: an engine run, an optimizer, every replicate, or a sampler that does not reach its target.                                             |
| `InfeasibleStateError` | If a draw of a prediction leaves the values its model allows.                                                                                                                                 |

Source code in `commol/api/calibrator.py`

```python
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
        If no runner is registered for the configuration's type (the
        message names the registered ones), or
        a setting, an input, the data or the model does not fit the
        method; the message says why.
    RuntimeError
        If a computation of the method that could start fails: an engine
        run, an optimizer, every replicate, or a sampler that does not
        reach its target.
    InfeasibleStateError
        If a draw of a prediction leaves the values its model allows.
    """
    from commol.api.inference.registry import run_method

    return run_method(self, method, UncertaintyMethod)
```

#### identifiability

```python
identifiability(method: IdentifiabilityMethod[ResultT]) -> ResultT
```

Run an identifiability method on this calibration problem.

Each method answers which parameters the model and the data determine, from another kind of evidence: `StructuralRank`, `Collinearity`, `SampleIdentifiability` or `StartsIdentifiability`. Each returns an `IdentifiabilityReport`; `IdentifiabilityReport.merge` joins them.

Parameters:

| Name     | Type                             | Description                      | Default    |
| -------- | -------------------------------- | -------------------------------- | ---------- |
| `method` | `IdentifiabilityMethod[ResultT]` | The configuration of the method. | *required* |

Returns:

| Type      | Description               |
| --------- | ------------------------- |
| `ResultT` | The result of the method. |

Raises:

| Type           | Description                                                                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TypeError`    | If method is an uncertainty method; the message names Calibrator.uncertainty.                                                                         |
| `ValueError`   | If no runner is registered for the configuration's type, or a setting, an input, the data or the model does not fit the method; the message says why. |
| `RuntimeError` | If a computation of the method that could start fails: an engine run or an optimizer.                                                                 |

Source code in `commol/api/calibrator.py`

```python
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
        If no runner is registered for the configuration's type, or
        a setting, an input, the data or the model does not fit the
        method; the message says why.
    RuntimeError
        If a computation of the method that could start fails: an engine
        run or an optimizer.
    """
    from commol.api.inference.registry import run_method

    return run_method(self, method, IdentifiabilityMethod)
```

## CalibrationProblem

Bases: `BaseModel`

Defines a complete calibration problem.

This class encapsulates all the information needed to calibrate model parameters against observed data. It provides validation of the calibration setup but delegates the actual optimization to the Rust backend.

Attributes:

| Name                          | Type                          | Description                                                                                                                                                                                               |
| ----------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `observed_data`               | `list[ObservedDataPoint]`     | List of observed data points to fit against                                                                                                                                                               |
| `parameters`                  | `list[CalibrationParameter]`  | List of parameters to calibrate with their bounds                                                                                                                                                         |
| `constraints`                 | `list[CalibrationConstraint]` | List of constraints on calibration parameters (optional, default: empty list)                                                                                                                             |
| `loss_function`               | \`{sse, rmse, mae}            | None\`                                                                                                                                                                                                    |
| `observation_models`          | \`dict[str, ObservationModel] | None\`                                                                                                                                                                                                    |
| `normalize_observations`      | `bool`                        | When True, divide each residual by the RMS of its observation series before applying the loss metric (default: False). It needs a loss: observation models set the scale of each series with their noise. |
| `optimization_config`         | `OptimizationConfig`          | Configuration for the optimization algorithm                                                                                                                                                              |
| `initial_condition_remainder` | \`str                         | None\`                                                                                                                                                                                                    |
| `replicates`                  | \`int                         | None\`                                                                                                                                                                                                    |
| `replicate_summary`           | \`{mean, median, crps}        | Quantile                                                                                                                                                                                                  |
| `seed`                        | \`int                         | None\`                                                                                                                                                                                                    |

Source code in `commol/context/calibration.py`

```python
class CalibrationProblem(BaseModel):
    """
    Defines a complete calibration problem.

    This class encapsulates all the information needed to calibrate model
    parameters against observed data. It provides validation of the calibration
    setup but delegates the actual optimization to the Rust backend.

    Attributes
    ----------
    observed_data : list[ObservedDataPoint]
        List of observed data points to fit against
    parameters : list[CalibrationParameter]
        List of parameters to calibrate with their bounds
    constraints : list[CalibrationConstraint]
        List of constraints on calibration parameters (optional, default: empty list)
    loss_function : {"sse", "rmse", "mae"} | None
        The loss that measures the fit, with the observation weights. A
        problem has a loss or ``observation_models``, exactly one, with no
        default: what to fit is a modelling choice.
    observation_models : dict[str, ObservationModel] | None
        One observation model per series label (`NormalObservations`,
        `LogNormalObservations`, `StudentTObservations`,
        `PoissonObservations`, `NegativeBinomialObservations`,
        `BinomialObservations`): the problem fits by maximum likelihood, and
        its loss is the negative log-likelihood. Every observed series needs
        one, and every parameter of type ``"noise"`` must be an argument of
        one. It needs a deterministic engine.
    normalize_observations : bool
        When True, divide each residual by the RMS of its observation series
        before applying the loss metric (default: False). It needs a loss:
        observation models set the scale of each series with their noise.
    optimization_config : OptimizationConfig
        Configuration for the optimization algorithm
    initial_condition_remainder : str | None
        Expanded compartment or bin that absorbs the difference to the
        total once the calibrated initial conditions are applied
        (default: None). It is not calibrated itself. With a remainder, the
        sum of the initial amounts always equals ``total``; without one,
        calibrated initial conditions are applied as given and only a sum
        above ``total`` is penalized.
    replicates : int | None
        For a stochastic engine: how many replicates each evaluation runs,
        with common random numbers (the same seeds, derived from ``seed``, at
        every evaluation), so the loss is a deterministic function of the
        parameters. Required with a stochastic engine; a deterministic engine
        does not use it.
    replicate_summary : {"mean", "median", "crps"} | Quantile | None
        For a stochastic engine: what each observation is compared with. The
        mean, the median or a `Quantile` of the replicates, with
        ``loss_function``; or ``"crps"``, the continuous ranked probability
        score of their distribution, whose loss is the weighted mean score.
        Required with a stochastic engine, with no default: what to fit is a
        modelling choice.
    seed : int | None
        Seed of every random number of a calibration (default: None, a seed
        drawn from the operating system, which `Calibrator.run` logs):
        the replicates of a stochastic engine, the starts of
        ``Calibrator.run(starts=n)``, a particle swarm, and the uncertainty
        methods. Required with a stochastic engine.
    """

    model_config = ConfigDict(extra="forbid")

    observed_data: list[ObservedDataPoint] = Field(
        default=..., min_length=1, description="Observed data points"
    )
    parameters: list[CalibrationParameter] = Field(
        default=..., min_length=1, description="Parameters to calibrate"
    )
    constraints: list[CalibrationConstraint] = Field(
        default_factory=list,
        description="Constraints on calibration parameters",
    )
    loss_function: str | None = Field(
        default=None,
        description="Loss function to use for measuring fit quality",
    )
    observation_models: dict[str, ObservationModel] | None = Field(
        default=None,
        description="One observation model per series label",
    )
    normalize_observations: bool = Field(
        default=False,
        description=(
            "Normalize each residual by the RMS of its observation series before "
            "applying the loss metric. Keeps series measured on very different "
            "scales comparable in the loss."
        ),
    )
    optimization_config: NelderMeadConfig | ParticleSwarmConfig = Field(
        default=..., description="Optimization algorithm configuration"
    )
    initial_condition_remainder: str | None = Field(
        default=None,
        description=(
            "Expanded compartment or bin that absorbs the difference to the "
            "total once the calibrated initial conditions are applied."
        ),
    )
    replicates: int | None = Field(
        default=None,
        ge=1,
        description="Replicates per evaluation of a stochastic engine.",
    )
    replicate_summary: ReplicateSummary | None = Field(
        default=None,
        description="What the observations are compared with, for a stochastic engine.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Seed of every random number of a calibration: the replicates of a "
            "stochastic engine, the starts, a particle swarm and the methods"
        ),
    )

    @field_validator("loss_function")
    @classmethod
    def validate_loss_function(cls, v: str | None) -> str | None:
        if v is not None and v not in LossFunction:
            raise ValueError(
                f"Invalid loss_function: {v}. Must be one of {list(LossFunction)}"
            )
        return v

    @model_validator(mode="after")
    def validate_loss_or_models(self) -> Self:
        """Require a loss or observation models, exactly one."""
        if self.loss_function is not None and self.observation_models is not None:
            raise ValueError(
                "A problem has a loss_function or observation_models, not both: "
                "the fit and the posterior would answer different questions."
            )
        if self.loss_function is None and self.observation_models is None:
            raise ValueError(
                "A problem needs a loss_function (one of "
                f"{', '.join(LossFunction)}) or observation_models. There is no "
                "default: what to fit is a modelling choice."
            )
        if self.observation_models is not None and self.normalize_observations:
            raise ValueError(
                "normalize_observations needs a loss_function: the observation "
                "models set the scale of each series with their noise."
            )
        return self

    @model_validator(mode="after")
    def validate_unique_parameter_ids(self) -> Self:
        """Ensure parameter IDs are unique."""
        param_ids = [p.id for p in self.parameters]
        if len(param_ids) != len(set(param_ids)):
            duplicates = [id for id in set(param_ids) if param_ids.count(id) > 1]
            raise ValueError(f"Duplicate parameter IDs found: {duplicates}")
        return self
```

### Methods:

#### validate_loss_or_models

```python
validate_loss_or_models() -> Self
```

Require a loss or observation models, exactly one.

Source code in `commol/context/calibration.py`

```python
@model_validator(mode="after")
def validate_loss_or_models(self) -> Self:
    """Require a loss or observation models, exactly one."""
    if self.loss_function is not None and self.observation_models is not None:
        raise ValueError(
            "A problem has a loss_function or observation_models, not both: "
            "the fit and the posterior would answer different questions."
        )
    if self.loss_function is None and self.observation_models is None:
        raise ValueError(
            "A problem needs a loss_function (one of "
            f"{', '.join(LossFunction)}) or observation_models. There is no "
            "default: what to fit is a modelling choice."
        )
    if self.observation_models is not None and self.normalize_observations:
        raise ValueError(
            "normalize_observations needs a loss_function: the observation "
            "models set the scale of each series with their noise."
        )
    return self
```

#### validate_unique_parameter_ids

```python
validate_unique_parameter_ids() -> Self
```

Ensure parameter IDs are unique.

Source code in `commol/context/calibration.py`

```python
@model_validator(mode="after")
def validate_unique_parameter_ids(self) -> Self:
    """Ensure parameter IDs are unique."""
    param_ids = [p.id for p in self.parameters]
    if len(param_ids) != len(set(param_ids)):
        duplicates = [id for id in set(param_ids) if param_ids.count(id) > 1]
        raise ValueError(f"Duplicate parameter IDs found: {duplicates}")
    return self
```

## Quantile

Bases: `BaseModel`

A quantile of the replicates of a stochastic engine, for `CalibrationProblem.replicate_summary`.

It is the one quantile of commol, the one of `weighted_quantiles` and `SimulationReplicates.quantiles`: with equal weights, NumPy's `method="hazen"`. `Quantile(0.25)` of the replicates 1, 2, 3, 4, 5 is 1.75.

Attributes:

| Name    | Type    | Description                                             |
| ------- | ------- | ------------------------------------------------------- |
| `level` | `float` | The level, between 0 and 1, such as 0.5 for the median. |

Source code in `commol/context/calibration.py`

```python
class Quantile(BaseModel):
    """
    A quantile of the replicates of a stochastic engine, for
    ``CalibrationProblem.replicate_summary``.

    It is the one quantile of commol, the one of `weighted_quantiles` and
    `SimulationReplicates.quantiles`: with equal weights, NumPy's
    ``method="hazen"``. ``Quantile(0.25)`` of the replicates 1, 2, 3, 4, 5 is
    1.75.

    Attributes
    ----------
    level : float
        The level, between 0 and 1, such as 0.5 for the median.
    """

    model_config = ConfigDict(extra="forbid")

    level: float = Field(..., ge=0.0, le=1.0, description="Level of the quantile.")

    def __init__(self, level: float, **data: object) -> None:
        """
        A quantile of the replicates.

        Parameters
        ----------
        level : float
            The level, between 0 and 1.
        **data : object
            The other fields, as JSON holds them.
        """
        super().__init__(level=level, **data)
```

### Methods:

#### __init__

```python
__init__(level: float, **data: object) -> None
```

A quantile of the replicates.

Parameters:

| Name     | Type     | Description                           | Default    |
| -------- | -------- | ------------------------------------- | ---------- |
| `level`  | `float`  | The level, between 0 and 1.           | *required* |
| `**data` | `object` | The other fields, as JSON holds them. | `{}`       |

Source code in `commol/context/calibration.py`

```python
def __init__(self, level: float, **data: object) -> None:
    """
    A quantile of the replicates.

    Parameters
    ----------
    level : float
        The level, between 0 and 1.
    **data : object
        The other fields, as JSON holds them.
    """
    super().__init__(level=level, **data)
```

## ObservedDataPoint

Bases: `BaseModel`

Represents a single observed data point for calibration.

An observation compares one output, or the sum of several outputs, with an observed value at a position on the axis.

Attributes:

| Name        | Type        | Description                                                                                                                                                                                                    |
| ----------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `at`        | `float`     | Position of the observation on the axis. The engine decides which positions its model type accepts: a DifferentialEquations model any non-negative position, a DifferenceEquations model only an integer step. |
| `output`    | \`str       | None\`                                                                                                                                                                                                         |
| `outputs`   | \`list[str] | None\`                                                                                                                                                                                                         |
| `label`     | `str`       | Name of the observed series, used by reports, plots and the per-series normalization. Defaults to output.                                                                                                      |
| `value`     | `float`     | Observed value                                                                                                                                                                                                 |
| `weight`    | `float`     | How much this observation counts in the loss (default: 1.0). Every loss multiplies the squared or absolute residual by it: sse is Σ w·r², rmse is √(Σ w·r² / Σ w), mae is Σ w·                                 |
| `factor_id` | \`str       | None\`                                                                                                                                                                                                         |
| `window`    | \`float     | None\`                                                                                                                                                                                                         |
| `trials`    | \`int       | None\`                                                                                                                                                                                                         |

Source code in `commol/context/calibration.py`

```python
class ObservedDataPoint(BaseModel):
    """
    Represents a single observed data point for calibration.

    An observation compares one output, or the sum of several outputs, with an
    observed value at a position on the axis.

    Attributes
    ----------
    at : float
        Position of the observation on the axis. The engine decides which
        positions its model type accepts: a DifferentialEquations model any
        non-negative position, a DifferenceEquations model only an integer
        step.
    output : str | None
        The output being observed. Give either `output` or `outputs`.
    outputs : list[str] | None
        Outputs summed before comparing with this observation. Needs a
        `label`.
    label : str
        Name of the observed series, used by reports, plots and the
        per-series normalization. Defaults to `output`.
    value : float
        Observed value
    weight : float
        How much this observation counts in the loss (default: 1.0). Every
        loss multiplies the squared or absolute residual by it: ``sse`` is
        ``Σ w·r²``, ``rmse`` is ``√(Σ w·r² / Σ w)``, ``mae`` is
        ``Σ w·|r| / Σ w``. With observation models it is a power of the
        observation's likelihood, ``w·log p``: a weight of 2 counts the
        observation twice.
    factor_id : str | None
        Optional factor parameter ID that multiplies the model output before
        comparison
    window : float | None
        Optional window length for a cumulative output: the prediction is the
        output at `at` minus the output at ``at - window``.
    trials : int | None
        The trials of this observation, for a `BinomialObservations` model
        with no ``trials`` of its own (default: None).
    """

    model_config = ConfigDict(extra="forbid")

    at: float = Field(
        default=..., ge=0, allow_inf_nan=False, description="Position on the axis"
    )
    output: str | None = Field(
        default=None, min_length=1, description="The output being observed"
    )
    outputs: list[str] | None = Field(
        default=None,
        description="Outputs summed before comparison",
    )
    label: str = Field(
        default="",
        description="Name of the observed series; defaults to `output`",
    )
    value: float = Field(default=..., ge=0.0, description="Observed value")
    weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight for this observation in the loss function",
    )
    factor_id: str | None = Field(
        default=None,
        description="Optional factor parameter ID that multiplies the model output",
    )
    window: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Optional window length for cumulative-output observations. "
            "When set, prediction is value(at) - value(at - window)."
        ),
    )
    trials: int | None = Field(
        default=None,
        ge=0,
        description="Trials of a binomial observation model that reads them here",
    )

    @model_validator(mode="after")
    def validate_observed_outputs(self) -> Self:
        """Require one of `output` and `outputs`; name the series."""
        if (self.output is None) == (self.outputs is None):
            raise ValueError("Give exactly one of 'output' and 'outputs'.")
        if self.outputs is not None:
            if not self.outputs:
                raise ValueError("outputs must be non-empty when provided")
            if not self.label:
                raise ValueError(
                    "An observation of several outputs needs a 'label' that names "
                    "their sum."
                )
        elif not self.label:
            self.label = self.output or ""
        return self

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        """A window must not start before position 0."""
        if self.window is not None and self.window > self.at:
            raise ValueError(f"window ({self.window}) must be <= at ({self.at})")
        return self

    @property
    def observed_outputs(self) -> list[str]:
        """The outputs whose sum is compared: `outputs`, or `output` alone."""
        return list(self.outputs) if self.outputs is not None else [self.output or ""]
```

### Attributes

#### observed_outputs

```python
observed_outputs: list[str]
```

The outputs whose sum is compared: `outputs`, or `output` alone.

### Methods:

#### validate_observed_outputs

```python
validate_observed_outputs() -> Self
```

Require one of `output` and `outputs`; name the series.

Source code in `commol/context/calibration.py`

```python
@model_validator(mode="after")
def validate_observed_outputs(self) -> Self:
    """Require one of `output` and `outputs`; name the series."""
    if (self.output is None) == (self.outputs is None):
        raise ValueError("Give exactly one of 'output' and 'outputs'.")
    if self.outputs is not None:
        if not self.outputs:
            raise ValueError("outputs must be non-empty when provided")
        if not self.label:
            raise ValueError(
                "An observation of several outputs needs a 'label' that names "
                "their sum."
            )
    elif not self.label:
        self.label = self.output or ""
    return self
```

#### validate_window

```python
validate_window() -> Self
```

A window must not start before position 0.

Source code in `commol/context/calibration.py`

```python
@model_validator(mode="after")
def validate_window(self) -> Self:
    """A window must not start before position 0."""
    if self.window is not None and self.window > self.at:
        raise ValueError(f"window ({self.window}) must be <= at ({self.at})")
    return self
```

## Observation models

Guide: [Observation Models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/index.md).

## ObservationModel

```python
ObservationModel = Annotated[NormalObservations | LogNormalObservations | StudentTObservations | PoissonObservations | NegativeBinomialObservations | BinomialObservations, Field(discriminator=kind)]
```

One observation model of a series.

## NormalObservations

Bases: `_ObservationModel`

Normal noise: `y ~ N(μ, sd²)`, with `μ` the prediction.

For continuous values with errors of the same size everywhere. With a known `sd` the maximum-likelihood fit is the least-squares fit.

Attributes:

| Name | Type    | Description |
| ---- | ------- | ----------- |
| `sd` | \`float | str\`       |

Source code in `commol/context/observation_models.py`

```python
class NormalObservations(_ObservationModel):
    """
    Normal noise: ``y ~ N(μ, sd²)``, with ``μ`` the prediction.

    For continuous values with errors of the same size everywhere. With a
    known ``sd`` the maximum-likelihood fit is the least-squares fit.

    Attributes
    ----------
    sd : float | str
        The standard deviation, above 0, or the id of a noise parameter.
    """

    kind: Literal["normal"] = "normal"
    sd: NoiseArgument = Field(description="The standard deviation.")
```

## LogNormalObservations

Bases: `_ObservationModel`

Log-normal noise: `log y` is normal with standard deviation `sigma`.

For positive values with errors that grow with the value. `center` says what the prediction `μ` is, and has no default: `"mean"` makes `E[y] = μ` (`log y ~ N(log μ − σ²/2, σ²)`), `"median"` makes the median `μ` (`log y ~ N(log μ, σ²)`, the proportional error `y = μ·e^ε`). Every observed value must be above 0.

Attributes:

| Name     | Type             | Description             |
| -------- | ---------------- | ----------------------- |
| `sigma`  | \`float          | str\`                   |
| `center` | `{mean, median}` | What the prediction is. |

Source code in `commol/context/observation_models.py`

```python
class LogNormalObservations(_ObservationModel):
    """
    Log-normal noise: ``log y`` is normal with standard deviation ``sigma``.

    For positive values with errors that grow with the value. ``center``
    says what the prediction ``μ`` is, and has no default: ``"mean"`` makes
    ``E[y] = μ`` (``log y ~ N(log μ − σ²/2, σ²)``), ``"median"`` makes the
    median ``μ`` (``log y ~ N(log μ, σ²)``, the proportional error
    ``y = μ·e^ε``). Every observed value must be above 0.

    Attributes
    ----------
    sigma : float | str
        The standard deviation of ``log y``, above 0, or the id of a noise
        parameter.
    center : {"mean", "median"}
        What the prediction is.
    """

    kind: Literal["log_normal"] = "log_normal"
    sigma: NoiseArgument = Field(description="The standard deviation of log y.")
    center: Literal["mean", "median"] = Field(description="What the prediction is.")
```

## StudentTObservations

Bases: `_ObservationModel`

Student-t noise of center `μ`, the prediction: normal noise with heavier tails, so an outlier pulls the fit less.

Attributes:

| Name    | Type    | Description |
| ------- | ------- | ----------- |
| `df`    | \`float | str\`       |
| `scale` | \`float | str\`       |

Source code in `commol/context/observation_models.py`

```python
class StudentTObservations(_ObservationModel):
    """
    Student-t noise of center ``μ``, the prediction: normal noise with heavier
    tails, so an outlier pulls the fit less.

    Attributes
    ----------
    df : float | str
        The degrees of freedom, above 0, or the id of a noise parameter. The
        noise has a mean when ``df > 1`` and a variance when ``df > 2``.
    scale : float | str
        The scale, above 0, or the id of a noise parameter.
    """

    kind: Literal["student_t"] = "student_t"
    df: NoiseArgument = Field(description="The degrees of freedom.")
    scale: NoiseArgument = Field(description="The scale.")
```

## PoissonObservations

Bases: `_ObservationModel`

Poisson counts of mean `μ`, the prediction: variance equal to the mean.

Every observed value must be a whole number, at least 0.

Source code in `commol/context/observation_models.py`

```python
class PoissonObservations(_ObservationModel):
    """
    Poisson counts of mean ``μ``, the prediction: variance equal to the mean.

    Every observed value must be a whole number, at least 0.
    """

    kind: Literal["poisson"] = "poisson"
```

## NegativeBinomialObservations

Bases: `_ObservationModel`

Negative binomial counts of mean `μ`, the prediction, and variance `μ + φμ²`: counts that scatter more than a Poisson.

`φ = 0` is the Poisson, so bounds `[0, …]` of a noise parameter hold it. The dispersion `k` (`size` in R) is `1/φ`. Every observed value must be a whole number, at least 0.

Attributes:

| Name             | Type    | Description |
| ---------------- | ------- | ----------- |
| `overdispersion` | \`float | str\`       |

Source code in `commol/context/observation_models.py`

```python
class NegativeBinomialObservations(_ObservationModel):
    """
    Negative binomial counts of mean ``μ``, the prediction, and variance
    ``μ + φμ²``: counts that scatter more than a Poisson.

    ``φ = 0`` is the Poisson, so bounds ``[0, …]`` of a noise parameter hold
    it. The dispersion ``k`` (``size`` in R) is ``1/φ``.
    Every observed value must be a whole number, at least 0.

    Attributes
    ----------
    overdispersion : float | str
        ``φ``, at least 0, or the id of a noise parameter.
    """

    kind: Literal["negative_binomial"] = "negative_binomial"
    overdispersion: NoiseArgument = Field(description="The overdispersion φ.")
```

## BinomialObservations

Bases: `_ObservationModel`

Binomial counts of `n` trials and mean `μ`, the prediction: the probability is `μ / n`, so the prediction is an expected count, not a proportion.

`trials` gives `n` for every observation of the series; without it, each observation of the series gives its own (`ObservedDataPoint.trials`), for trials that change from one observation to the next. One source per series. Every observed value must be a whole number between 0 and its trials.

Attributes:

| Name     | Type  | Description |
| -------- | ----- | ----------- |
| `trials` | \`int | None\`      |

Source code in `commol/context/observation_models.py`

```python
class BinomialObservations(_ObservationModel):
    """
    Binomial counts of ``n`` trials and mean ``μ``, the prediction: the
    probability is ``μ / n``, so the prediction is an expected count, not a
    proportion.

    ``trials`` gives ``n`` for every observation of the series; without it,
    each observation of the series gives its own (``ObservedDataPoint.trials``),
    for trials that change from one observation to the next. One source per
    series. Every observed value must be a whole number between 0 and its
    trials.

    Attributes
    ----------
    trials : int | None
        The trials of every observation, or None (default) to read the
        trials of each observation.
    """

    kind: Literal["binomial"] = "binomial"
    trials: int | None = Field(default=None, ge=0, description="The trials.")
```

## Unknowns, optimizers and results

## CalibrationParameter

Bases: `BaseModel`

Defines a parameter or initial condition to be calibrated with its bounds.

Attributes:

| Name             | Type    | Description                                                             |
| ---------------- | ------- | ----------------------------------------------------------------------- |
| `id`             | `str`   | Identifier (parameter ID for parameters, bin ID for initial conditions) |
| `parameter_type` | `str`   | Type of value being calibrated                                          |
| `min_bound`      | `float` | Minimum allowed value for this parameter                                |
| `max_bound`      | `float` | Maximum allowed value for this parameter                                |
| `initial_guess`  | \`float | None\`                                                                  |

Source code in `commol/context/calibration.py`

```python
class CalibrationParameter(BaseModel):
    """
    Defines a parameter or initial condition to be calibrated with its bounds.

    Attributes
    ----------
    id : str
        Identifier (parameter ID for parameters, bin ID for initial conditions)
    parameter_type : str
        Type of value being calibrated
    min_bound : float
        Minimum allowed value for this parameter
    max_bound : float
        Maximum allowed value for this parameter
    initial_guess : float | None
        Optional starting value for optimization (if None, midpoint is used)
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default=..., min_length=1, description="Parameter or bin identifier"
    )
    parameter_type: str = Field(
        default=...,
        description="Type of value being calibrated",
    )
    min_bound: float = Field(default=..., description="Minimum allowed value")
    max_bound: float = Field(default=..., description="Maximum allowed value")
    initial_guess: float | None = Field(
        default=None, description="Optional starting value for optimization"
    )

    @field_validator("parameter_type")
    @classmethod
    def validate_parameter_type(cls, v: str) -> str:
        if v not in CalibrationParameterType:
            raise ValueError(
                f"Invalid parameter_type: {v}. "
                f"Must be one of {list(CalibrationParameterType)}"
            )
        return v

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Validate that max_bound > min_bound and initial_guess is within bounds."""
        if self.max_bound <= self.min_bound:
            raise ValueError(
                (
                    f"max_bound ({self.max_bound}) must be greater than "
                    f"min_bound ({self.min_bound}) for parameter '{self.id}'"
                )
            )
        if self.initial_guess is not None:
            if not (self.min_bound <= self.initial_guess <= self.max_bound):
                raise ValueError(
                    (
                        f"initial_guess ({self.initial_guess}) must be between "
                        f"min_bound ({self.min_bound}) and max_bound "
                        f"({self.max_bound}) for parameter '{self.id}'"
                    )
                )

        return self
```

### Methods:

#### validate_bounds

```python
validate_bounds() -> Self
```

Validate that max_bound > min_bound and initial_guess is within bounds.

Source code in `commol/context/calibration.py`

```python
@model_validator(mode="after")
def validate_bounds(self) -> Self:
    """Validate that max_bound > min_bound and initial_guess is within bounds."""
    if self.max_bound <= self.min_bound:
        raise ValueError(
            (
                f"max_bound ({self.max_bound}) must be greater than "
                f"min_bound ({self.min_bound}) for parameter '{self.id}'"
            )
        )
    if self.initial_guess is not None:
        if not (self.min_bound <= self.initial_guess <= self.max_bound):
            raise ValueError(
                (
                    f"initial_guess ({self.initial_guess}) must be between "
                    f"min_bound ({self.min_bound}) and max_bound "
                    f"({self.max_bound}) for parameter '{self.id}'"
                )
            )

    return self
```

## CalibrationConstraint

Bases: `BaseModel`

A constraint on calibration parameters defined as a mathematical expression.

Constraints are mathematical expressions that must evaluate to >= 0 for the constraint to be satisfied. When the expression evaluates to < 0, the constraint is violated and a penalty is applied during optimization.

Attributes:

| Name          | Type          | Description                                                                                                                                                                                                                |
| ------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`          | `str`         | Unique identifier for this constraint                                                                                                                                                                                      |
| `expression`  | `str`         | Mathematical expression that must evaluate >= 0 for constraint satisfaction. Can reference calibration parameters by their IDs. When at is specified, can also reference output values (A, B, C, etc.) at those positions. |
| `description` | \`str         | None\`                                                                                                                                                                                                                     |
| `weight`      | `float`       | Penalty weight multiplier (default: 1.0). Higher values make this constraint more important relative to others. The penalty for violating this constraint is: weight * violation^2                                         |
| `at`          | \`list[float] | None\`                                                                                                                                                                                                                     |

Examples:

```pycon
>>> # k1/k2 must be <= 5
>>> constraint = CalibrationConstraint(
...     id="ratio_bound",
...     expression="5.0 - k1/k2",
...     description="Ratio k1/k2 <= 5",
... )
```

```pycon
>>> # k1 must be >= k2
>>> constraint = CalibrationConstraint(
...     id="ordering",
...     expression="k1 - k2",
...     description="k1 >= k2",
... )
```

```pycon
>>> # Sum of parameters <= 1.0
>>> constraint = CalibrationConstraint(
...     id="sum_bound",
...     expression="1.0 - (param1 + param2 + param3)",
...     description="Sum of rates <= 1.0",
... )
```

```pycon
>>> # At positions: bin B stays at or below 500 at specific positions
>>> constraint = CalibrationConstraint(
...     id="peak_b",
...     expression="500.0 - B",
...     description="Peak B never exceeds 500",
...     at=[10, 20, 30, 40, 50],
... )
```

Source code in `commol/context/calibration.py`

```python
class CalibrationConstraint(BaseModel):
    """
    A constraint on calibration parameters defined as a mathematical expression.

    Constraints are mathematical expressions that must evaluate to >= 0 for the
    constraint to be satisfied. When the expression evaluates to < 0, the constraint
    is violated and a penalty is applied during optimization.

    Attributes
    ----------
    id : str
        Unique identifier for this constraint
    expression : str
        Mathematical expression that must evaluate >= 0 for constraint satisfaction.
        Can reference calibration parameters by their IDs. When `at` is specified,
        can also reference output values (A, B, C, etc.) at those positions.
    description : str | None
        Human-readable description of the constraint (optional)
    weight : float
        Penalty weight multiplier (default: 1.0). Higher values make this constraint
        more important relative to others. The penalty for violating this constraint
        is: weight * violation^2
    at : list[float] | None
        Optional positions on the axis at which to evaluate this constraint.
        The engine decides which positions its model type accepts: a
        DifferenceEquations model only integer steps. If None, the constraint
        is evaluated once before simulation using parameter values only. If
        given, it is evaluated at each position and can reference output values.

    Examples
    --------
    >>> # k1/k2 must be <= 5
    >>> constraint = CalibrationConstraint(
    ...     id="ratio_bound",
    ...     expression="5.0 - k1/k2",
    ...     description="Ratio k1/k2 <= 5",
    ... )

    >>> # k1 must be >= k2
    >>> constraint = CalibrationConstraint(
    ...     id="ordering",
    ...     expression="k1 - k2",
    ...     description="k1 >= k2",
    ... )

    >>> # Sum of parameters <= 1.0
    >>> constraint = CalibrationConstraint(
    ...     id="sum_bound",
    ...     expression="1.0 - (param1 + param2 + param3)",
    ...     description="Sum of rates <= 1.0",
    ... )

    >>> # At positions: bin B stays at or below 500 at specific positions
    >>> constraint = CalibrationConstraint(
    ...     id="peak_b",
    ...     expression="500.0 - B",
    ...     description="Peak B never exceeds 500",
    ...     at=[10, 20, 30, 40, 50],
    ... )
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default=..., min_length=1, description="Unique identifier for this constraint"
    )
    expression: str = Field(
        default=...,
        min_length=1,
        description="Mathematical expression that must evaluate >= 0",
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the constraint"
    )
    weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Penalty weight multiplier for violations",
    )
    at: list[float] | None = Field(
        default=None,
        description=(
            "Optional positions on the axis at which to evaluate the constraint; "
            "it can then reference output values."
        ),
    )

    @field_validator("expression", mode="before")
    @classmethod
    def validate_expression(cls, value: str) -> str:
        """Perform security validation on the constraint expression."""
        try:
            validate_expression_security(value)
        except ValueError as e:
            raise ValueError(
                f"Security validation failed for expression '{value}': {e}"
            )
        return value

    @model_validator(mode="after")
    def validate_positions(self) -> Self:
        """Validate that the positions are finite and non-negative."""
        if self.at is not None:
            if len(self.at) == 0:
                raise ValueError(
                    f"at for constraint '{self.id}' must not be empty if specified"
                )
            if any(position < 0 or not math.isfinite(position) for position in self.at):
                raise ValueError(
                    f"at for constraint '{self.id}' must all be finite and non-negative"
                )
        return self
```

### Methods:

#### validate_expression

```python
validate_expression(value: str) -> str
```

Perform security validation on the constraint expression.

Source code in `commol/context/calibration.py`

```python
@field_validator("expression", mode="before")
@classmethod
def validate_expression(cls, value: str) -> str:
    """Perform security validation on the constraint expression."""
    try:
        validate_expression_security(value)
    except ValueError as e:
        raise ValueError(
            f"Security validation failed for expression '{value}': {e}"
        )
    return value
```

#### validate_positions

```python
validate_positions() -> Self
```

Validate that the positions are finite and non-negative.

Source code in `commol/context/calibration.py`

```python
@model_validator(mode="after")
def validate_positions(self) -> Self:
    """Validate that the positions are finite and non-negative."""
    if self.at is not None:
        if len(self.at) == 0:
            raise ValueError(
                f"at for constraint '{self.id}' must not be empty if specified"
            )
        if any(position < 0 or not math.isfinite(position) for position in self.at):
            raise ValueError(
                f"at for constraint '{self.id}' must all be finite and non-negative"
            )
    return self
```

## NelderMeadConfig

Bases: `BaseModel`

Configuration for the Nelder-Mead optimization algorithm.

The Nelder-Mead method is a simplex-based derivative-free optimization algorithm, suitable for problems where gradients are not available.

Attributes:

| Name                   | Type    | Description                                                                                                                                                                                                                                     |
| ---------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_iterations`       | `int`   | Maximum number of iterations (default: 1000)                                                                                                                                                                                                    |
| `sd_tolerance`         | `float` | Convergence tolerance for standard deviation (default: 1e-6)                                                                                                                                                                                    |
| `initial_simplex_size` | `float` | Size of the initial simplex, as a fraction of each parameter's bound width, in (0, 1\] (default: 0.1). Nelder-Mead searches an unbounded coordinate u and scores x = lo + (hi - lo)(sin u + 1)/2, so every point it tries is inside the bounds. |
| `alpha`                | \`float | None\`                                                                                                                                                                                                                                          |
| `gamma`                | \`float | None\`                                                                                                                                                                                                                                          |
| `rho`                  | \`float | None\`                                                                                                                                                                                                                                          |
| `sigma`                | \`float | None\`                                                                                                                                                                                                                                          |
| `verbose`              | `bool`  | Enable verbose output during optimization (default: False)                                                                                                                                                                                      |
| `header_interval`      | `int`   | Number of iterations between table header repeats in verbose output (default: 100)                                                                                                                                                              |

Source code in `commol/context/calibration.py`

```python
class NelderMeadConfig(BaseModel):
    """
    Configuration for the Nelder-Mead optimization algorithm.

    The Nelder-Mead method is a simplex-based derivative-free optimization
    algorithm, suitable for problems where gradients are not available.

    Attributes
    ----------
    max_iterations : int
        Maximum number of iterations (default: 1000)
    sd_tolerance : float
        Convergence tolerance for standard deviation (default: 1e-6)
    initial_simplex_size : float
        Size of the initial simplex, as a fraction of each parameter's bound
        width, in (0, 1] (default: 0.1). Nelder-Mead searches an unbounded
        coordinate ``u`` and scores ``x = lo + (hi - lo)(sin u + 1)/2``, so
        every point it tries is inside the bounds.
    alpha : float | None
        Reflection coefficient (default: None, uses argmin's default)
    gamma : float | None
        Expansion coefficient (default: None, uses argmin's default)
    rho : float | None
        Contraction coefficient (default: None, uses argmin's default)
    sigma : float | None
        Shrink coefficient (default: None, uses argmin's default)
    verbose : bool
        Enable verbose output during optimization (default: False)
    header_interval: int
        Number of iterations between table header repeats in verbose output
        (default: 100)
    """

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(
        default=1000, gt=0, description="Maximum number of iterations"
    )
    sd_tolerance: float = Field(
        default=1e-6, gt=0.0, description="Convergence tolerance for standard deviation"
    )
    initial_simplex_size: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description=(
            "Size of the initial simplex, as a fraction of each parameter's bound width"
        ),
    )
    alpha: float | None = Field(
        default=None,
        gt=0.0,
        description="Reflection coefficient (default: None, uses argmin's default)",
    )
    gamma: float | None = Field(
        default=None,
        gt=0.0,
        description="Expansion coefficient (default: None, uses argmin's default)",
    )
    rho: float | None = Field(
        default=None,
        gt=0.0,
        description="Contraction coefficient (default: None, uses argmin's default)",
    )
    sigma: float | None = Field(
        default=None,
        gt=0.0,
        description="Shrink coefficient (default: None, uses argmin's default)",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output during optimization (default: False)",
    )
    header_interval: int = Field(
        default=100,
        gt=0,
        description=(
            "Number of iterations between table header repeats in verbose output "
            "(default: 100)"
        ),
    )
```

## ParticleSwarmConfig

Bases: `BaseModel`

Configuration for the Particle Swarm Optimization algorithm.

Attributes:

| Name             | Type   | Description                                           |
| ---------------- | ------ | ----------------------------------------------------- |
| `particles`      | `int`  | Number of particles in the swarm (default: 20)        |
| `max_iterations` | `int`  | Maximum number of iterations (default: 1000)          |
| `verbose`        | `bool` | Enable verbose output (default: False)                |
| `initialization` | `str`  | Particle initialization strategy (default: "uniform") |

Methods:

| Name               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `inertia`          | Set inertia weight strategy ("constant" or "chaotic")        |
| `acceleration`     | Set acceleration coefficients ("constant" or "time_varying") |
| `mutation`         | Enable mutation to escape local optima                       |
| `velocity`         | Configure velocity control                                   |
| `improvement_stop` | Stop when the best loss stops improving                      |
| `radius_stop`      | Stop when the swarm has gathered around its best position    |

Source code in `commol/context/calibration.py`

```python
class ParticleSwarmConfig(BaseModel):
    """
    Configuration for the Particle Swarm Optimization algorithm.

    Attributes
    ----------
    particles : int
        Number of particles in the swarm (default: 20)
    max_iterations : int
        Maximum number of iterations (default: 1000)
    verbose : bool
        Enable verbose output (default: False)
    initialization : str
        Particle initialization strategy (default: "uniform")

    Methods
    -------
    inertia(type, **kwargs)
        Set inertia weight strategy ("constant" or "chaotic")
    acceleration(type, **kwargs)
        Set acceleration coefficients ("constant" or "time_varying")
    mutation(strategy, scale, probability, application)
        Enable mutation to escape local optima
    velocity(clamp_factor, mutation_threshold)
        Configure velocity control
    improvement_stop(window, tol_abs, tol_rel)
        Stop when the best loss stops improving
    radius_stop(tol)
        Stop when the swarm has gathered around its best position
    """

    model_config = ConfigDict(extra="forbid")

    # Core parameters
    particles: int = Field(
        default=20, gt=0, description="Number of particles in the swarm"
    )
    max_iterations: int = Field(
        default=1000, gt=0, description="Maximum number of iterations"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output during optimization",
    )
    initialization: str = Field(
        default=PSOInitializationStrategy.UNIFORM,
        description=(
            "Particle initialization strategy: 'uniform' (default), 'latin_hypercube',"
            "or 'opposition_based'"
        ),
    )

    # Private attributes for optional configurations (set via fluent methods)
    _inertia: PSOConstantInertia | PSOChaoticInertia | None = PrivateAttr(default=None)
    _acceleration: PSOConstantAcceleration | PSOTimeVaryingAcceleration | None = (
        PrivateAttr(default=None)
    )
    _mutation: PSOMutationConfig | None = PrivateAttr(default=None)
    _velocity: PSOVelocityConfig | None = PrivateAttr(default=None)
    _improvement_stop: PSOImprovementStop | None = PrivateAttr(default=None)
    _radius_stop: PSORadiusStop | None = PrivateAttr(default=None)

    @field_validator("initialization")
    @classmethod
    def validate_initialization(cls, v: str) -> str:
        if v not in PSOInitializationStrategy:
            raise ValueError(
                f"Invalid initialization: {v}. "
                f"Must be one of {list(PSOInitializationStrategy)}"
            )
        return v

    @overload
    def inertia(self, type: str, *, factor: float = 0.721) -> Self: ...

    @overload
    def inertia(self, type: str, *, w_min: float, w_max: float) -> Self: ...

    def inertia(self, type: str, **kwargs: float) -> Self:
        """
        Set inertia weight strategy.

        Parameters
        ----------
        type : str
            Inertia strategy type: "constant" or "chaotic"
        **kwargs : float
            The settings of the strategy, listed under Other Parameters.

        Other Parameters
        ----------------
        factor : float, default=0.721
            Fixed inertia weight (canonical PSO: 1/(2*ln(2)) ≈ 0.721).
            Only used when type="constant".
        w_min : float
            Minimum inertia weight. Only used when type="chaotic".
        w_max : float
            Maximum inertia weight (must be > w_min). Only used when type="chaotic".

        Returns
        -------
        Self
            The config instance for method chaining
        """
        if type not in PSOInertiaType:
            raise ValueError(
                f"Invalid inertia type: {type}. Must be one of {list(PSOInertiaType)}"
            )
        if type == PSOInertiaType.CONSTANT:
            self._inertia = PSOConstantInertia(factor=kwargs.get("factor", 0.721))
        else:
            self._inertia = PSOChaoticInertia(
                w_min=kwargs["w_min"], w_max=kwargs["w_max"]
            )
        return self

    @overload
    def acceleration(
        self,
        type: str,
        *,
        cognitive: float = 1.193,
        social: float = 1.193,
    ) -> Self: ...

    @overload
    def acceleration(
        self,
        type: str,
        *,
        c1_initial: float,
        c1_final: float,
        c2_initial: float,
        c2_final: float,
    ) -> Self: ...

    def acceleration(self, type: str, **kwargs: float) -> Self:
        """
        Set acceleration coefficient strategy.

        Parameters
        ----------
        type : str
            Acceleration strategy type: "constant" or "time_varying"
        **kwargs : float
            The settings of the strategy, listed under Other Parameters.

        Other Parameters
        ----------------
        cognitive : float, default=1.193
            Cognitive coefficient (c1) - attraction to personal best.
            Only used when type="constant".
        social : float, default=1.193
            Social coefficient (c2) - attraction to swarm best.
            Only used when type="constant".
        c1_initial : float
            Initial cognitive factor (typically high, e.g., 2.5).
            Only used when type="time_varying".
        c1_final : float
            Final cognitive factor (typically low, e.g., 0.5).
            Only used when type="time_varying".
        c2_initial : float
            Initial social factor (typically low, e.g., 0.5).
            Only used when type="time_varying".
        c2_final : float
            Final social factor (typically high, e.g., 2.5).
            Only used when type="time_varying".

        Returns
        -------
        Self
            The config instance for method chaining
        """
        if type not in PSOAccelerationType:
            raise ValueError(
                f"Invalid acceleration type: {type}. "
                f"Must be one of {list(PSOAccelerationType)}"
            )
        if type == PSOAccelerationType.CONSTANT:
            self._acceleration = PSOConstantAcceleration(
                cognitive=kwargs.get("cognitive", 1.193),
                social=kwargs.get("social", 1.193),
            )
        else:
            self._acceleration = PSOTimeVaryingAcceleration(
                c1_initial=kwargs["c1_initial"],
                c1_final=kwargs["c1_final"],
                c2_initial=kwargs["c2_initial"],
                c2_final=kwargs["c2_final"],
            )
        return self

    def mutation(
        self,
        strategy: str,
        *,
        scale: float,
        probability: float,
        application: str,
    ) -> Self:
        """
        Enable mutation to help escape local optima.

        Parameters
        ----------
        strategy : str
            Mutation distribution: "gaussian" or "cauchy".
            Cauchy has heavier tails for larger jumps.
        scale : float
            Standard deviation (gaussian) or scale parameter (cauchy)
        probability : float
            Mutation probability per iteration (0.0 to 1.0)
        application : str
            Which particles to mutate:
            "global_best", "all_particles", or "below_average"

        Returns
        -------
        Self
            The config instance for method chaining
        """
        if strategy not in PSOMutationStrategy:
            raise ValueError(
                f"Invalid mutation strategy: {strategy}. "
                f"Must be one of {list(PSOMutationStrategy)}"
            )
        if application not in PSOMutationApplication:
            raise ValueError(
                f"Invalid mutation application: {application}. "
                f"Must be one of {list(PSOMutationApplication)}"
            )

        self._mutation = PSOMutationConfig(
            strategy=PSOMutationStrategy(strategy),
            scale=scale,
            probability=probability,
            application=PSOMutationApplication(application),
        )
        return self

    def velocity(
        self,
        *,
        clamp_factor: float | None = None,
        mutation_threshold: float | None = None,
    ) -> Self:
        """
        Configure velocity control.

        Parameters
        ----------
        clamp_factor : float, optional
            Velocity clamping as fraction of search space (0.0-1.0).
            Typically 0.1-0.2. Prevents explosive velocities.
        mutation_threshold : float, optional
            Reinitialize velocities below this threshold.
            Typically 0.001-0.01. Prevents stagnation.

        Returns
        -------
        Self
            The config instance for method chaining
        """
        self._velocity = PSOVelocityConfig(
            clamp_factor=clamp_factor,
            mutation_threshold=mutation_threshold,
        )
        return self

    def improvement_stop(self, *, window: int, tol_abs: float, tol_rel: float) -> Self:
        """
        Stop when the best loss stops improving.

        The run stops when the best loss improved by at most
        ``tol_abs + tol_rel * |best loss|`` over the last `window` iterations,
        and reports ``converged=True``. Off by default. A swarm can stall for
        some iterations and then improve again, so a short window can stop it
        early.

        Parameters
        ----------
        window : int
            Iterations over which the improvement is measured (> 0).
        tol_abs : float
            Absolute tolerance on the improvement (>= 0).
        tol_rel : float
            Tolerance on the improvement relative to the best loss (>= 0).

        Returns
        -------
        Self
            The config instance for method chaining
        """
        self._improvement_stop = PSOImprovementStop(
            window=window, tol_abs=tol_abs, tol_rel=tol_rel
        )
        return self

    def radius_stop(self, tol: float) -> Self:
        """
        Stop when the swarm has gathered around its best position.

        The run stops when the largest distance of a particle from the best
        position, divided by the size of the search box, is at most `tol`, and
        reports ``converged=True``. Off by default. Mutation and velocity
        mutation scatter the swarm again, so with them the radius can stay
        above `tol`; values that the data do not tell apart keep the swarm
        spread.

        Parameters
        ----------
        tol : float
            Tolerance on the radius of the swarm, relative to the size of the
            search box (>= 0).

        Returns
        -------
        Self
            The config instance for method chaining
        """
        self._radius_stop = PSORadiusStop(tol=tol)
        return self

    @property
    def inertia_config(self) -> PSOConstantInertia | PSOChaoticInertia | None:
        """Get the inertia configuration (for internal use)."""
        return self._inertia

    @property
    def acceleration_config(
        self,
    ) -> PSOConstantAcceleration | PSOTimeVaryingAcceleration | None:
        """Get the acceleration configuration (for internal use)."""
        return self._acceleration

    @property
    def mutation_config(self) -> PSOMutationConfig | None:
        """Get the mutation configuration (for internal use)."""
        return self._mutation

    @property
    def velocity_config(self) -> PSOVelocityConfig | None:
        """Get the velocity configuration (for internal use)."""
        return self._velocity

    @property
    def improvement_stop_config(self) -> PSOImprovementStop | None:
        """Get the improvement stop rule (for internal use)."""
        return self._improvement_stop

    @property
    def radius_stop_config(self) -> PSORadiusStop | None:
        """Get the radius stop rule (for internal use)."""
        return self._radius_stop
```

### Attributes

#### inertia_config

```python
inertia_config: PSOConstantInertia | PSOChaoticInertia | None
```

Get the inertia configuration (for internal use).

#### acceleration_config

```python
acceleration_config: PSOConstantAcceleration | PSOTimeVaryingAcceleration | None
```

Get the acceleration configuration (for internal use).

#### mutation_config

```python
mutation_config: PSOMutationConfig | None
```

Get the mutation configuration (for internal use).

#### velocity_config

```python
velocity_config: PSOVelocityConfig | None
```

Get the velocity configuration (for internal use).

#### improvement_stop_config

```python
improvement_stop_config: PSOImprovementStop | None
```

Get the improvement stop rule (for internal use).

#### radius_stop_config

```python
radius_stop_config: PSORadiusStop | None
```

Get the radius stop rule (for internal use).

### Methods:

#### inertia

```python
inertia(type: str, *, factor: float = 0.721) -> Self
```

```python
inertia(type: str, *, w_min: float, w_max: float) -> Self
```

```python
inertia(type: str, **kwargs: float) -> Self
```

Set inertia weight strategy.

Parameters:

| Name       | Type    | Description                                                  | Default    |
| ---------- | ------- | ------------------------------------------------------------ | ---------- |
| `type`     | `str`   | Inertia strategy type: "constant" or "chaotic"               | *required* |
| `**kwargs` | `float` | The settings of the strategy, listed under Other Parameters. | `{}`       |

Other Parameters:

| Name     | Type    | Description                                                                                 |
| -------- | ------- | ------------------------------------------------------------------------------------------- |
| `factor` | `float` | Fixed inertia weight (canonical PSO: 1/(2\*ln(2)) ≈ 0.721). Only used when type="constant". |
| `w_min`  | `float` | Minimum inertia weight. Only used when type="chaotic".                                      |
| `w_max`  | `float` | Maximum inertia weight (must be > w_min). Only used when type="chaotic".                    |

Returns:

| Type   | Description                             |
| ------ | --------------------------------------- |
| `Self` | The config instance for method chaining |

Source code in `commol/context/calibration.py`

```python
def inertia(self, type: str, **kwargs: float) -> Self:
    """
    Set inertia weight strategy.

    Parameters
    ----------
    type : str
        Inertia strategy type: "constant" or "chaotic"
    **kwargs : float
        The settings of the strategy, listed under Other Parameters.

    Other Parameters
    ----------------
    factor : float, default=0.721
        Fixed inertia weight (canonical PSO: 1/(2*ln(2)) ≈ 0.721).
        Only used when type="constant".
    w_min : float
        Minimum inertia weight. Only used when type="chaotic".
    w_max : float
        Maximum inertia weight (must be > w_min). Only used when type="chaotic".

    Returns
    -------
    Self
        The config instance for method chaining
    """
    if type not in PSOInertiaType:
        raise ValueError(
            f"Invalid inertia type: {type}. Must be one of {list(PSOInertiaType)}"
        )
    if type == PSOInertiaType.CONSTANT:
        self._inertia = PSOConstantInertia(factor=kwargs.get("factor", 0.721))
    else:
        self._inertia = PSOChaoticInertia(
            w_min=kwargs["w_min"], w_max=kwargs["w_max"]
        )
    return self
```

#### acceleration

```python
acceleration(type: str, *, cognitive: float = 1.193, social: float = 1.193) -> Self
```

```python
acceleration(type: str, *, c1_initial: float, c1_final: float, c2_initial: float, c2_final: float) -> Self
```

```python
acceleration(type: str, **kwargs: float) -> Self
```

Set acceleration coefficient strategy.

Parameters:

| Name       | Type    | Description                                                  | Default    |
| ---------- | ------- | ------------------------------------------------------------ | ---------- |
| `type`     | `str`   | Acceleration strategy type: "constant" or "time_varying"     | *required* |
| `**kwargs` | `float` | The settings of the strategy, listed under Other Parameters. | `{}`       |

Other Parameters:

| Name         | Type    | Description                                                                               |
| ------------ | ------- | ----------------------------------------------------------------------------------------- |
| `cognitive`  | `float` | Cognitive coefficient (c1) - attraction to personal best. Only used when type="constant". |
| `social`     | `float` | Social coefficient (c2) - attraction to swarm best. Only used when type="constant".       |
| `c1_initial` | `float` | Initial cognitive factor (typically high, e.g., 2.5). Only used when type="time_varying". |
| `c1_final`   | `float` | Final cognitive factor (typically low, e.g., 0.5). Only used when type="time_varying".    |
| `c2_initial` | `float` | Initial social factor (typically low, e.g., 0.5). Only used when type="time_varying".     |
| `c2_final`   | `float` | Final social factor (typically high, e.g., 2.5). Only used when type="time_varying".      |

Returns:

| Type   | Description                             |
| ------ | --------------------------------------- |
| `Self` | The config instance for method chaining |

Source code in `commol/context/calibration.py`

```python
def acceleration(self, type: str, **kwargs: float) -> Self:
    """
    Set acceleration coefficient strategy.

    Parameters
    ----------
    type : str
        Acceleration strategy type: "constant" or "time_varying"
    **kwargs : float
        The settings of the strategy, listed under Other Parameters.

    Other Parameters
    ----------------
    cognitive : float, default=1.193
        Cognitive coefficient (c1) - attraction to personal best.
        Only used when type="constant".
    social : float, default=1.193
        Social coefficient (c2) - attraction to swarm best.
        Only used when type="constant".
    c1_initial : float
        Initial cognitive factor (typically high, e.g., 2.5).
        Only used when type="time_varying".
    c1_final : float
        Final cognitive factor (typically low, e.g., 0.5).
        Only used when type="time_varying".
    c2_initial : float
        Initial social factor (typically low, e.g., 0.5).
        Only used when type="time_varying".
    c2_final : float
        Final social factor (typically high, e.g., 2.5).
        Only used when type="time_varying".

    Returns
    -------
    Self
        The config instance for method chaining
    """
    if type not in PSOAccelerationType:
        raise ValueError(
            f"Invalid acceleration type: {type}. "
            f"Must be one of {list(PSOAccelerationType)}"
        )
    if type == PSOAccelerationType.CONSTANT:
        self._acceleration = PSOConstantAcceleration(
            cognitive=kwargs.get("cognitive", 1.193),
            social=kwargs.get("social", 1.193),
        )
    else:
        self._acceleration = PSOTimeVaryingAcceleration(
            c1_initial=kwargs["c1_initial"],
            c1_final=kwargs["c1_final"],
            c2_initial=kwargs["c2_initial"],
            c2_final=kwargs["c2_final"],
        )
    return self
```

#### mutation

```python
mutation(strategy: str, *, scale: float, probability: float, application: str) -> Self
```

Enable mutation to help escape local optima.

Parameters:

| Name          | Type    | Description                                                                               | Default    |
| ------------- | ------- | ----------------------------------------------------------------------------------------- | ---------- |
| `strategy`    | `str`   | Mutation distribution: "gaussian" or "cauchy". Cauchy has heavier tails for larger jumps. | *required* |
| `scale`       | `float` | Standard deviation (gaussian) or scale parameter (cauchy)                                 | *required* |
| `probability` | `float` | Mutation probability per iteration (0.0 to 1.0)                                           | *required* |
| `application` | `str`   | Which particles to mutate: "global_best", "all_particles", or "below_average"             | *required* |

Returns:

| Type   | Description                             |
| ------ | --------------------------------------- |
| `Self` | The config instance for method chaining |

Source code in `commol/context/calibration.py`

```python
def mutation(
    self,
    strategy: str,
    *,
    scale: float,
    probability: float,
    application: str,
) -> Self:
    """
    Enable mutation to help escape local optima.

    Parameters
    ----------
    strategy : str
        Mutation distribution: "gaussian" or "cauchy".
        Cauchy has heavier tails for larger jumps.
    scale : float
        Standard deviation (gaussian) or scale parameter (cauchy)
    probability : float
        Mutation probability per iteration (0.0 to 1.0)
    application : str
        Which particles to mutate:
        "global_best", "all_particles", or "below_average"

    Returns
    -------
    Self
        The config instance for method chaining
    """
    if strategy not in PSOMutationStrategy:
        raise ValueError(
            f"Invalid mutation strategy: {strategy}. "
            f"Must be one of {list(PSOMutationStrategy)}"
        )
    if application not in PSOMutationApplication:
        raise ValueError(
            f"Invalid mutation application: {application}. "
            f"Must be one of {list(PSOMutationApplication)}"
        )

    self._mutation = PSOMutationConfig(
        strategy=PSOMutationStrategy(strategy),
        scale=scale,
        probability=probability,
        application=PSOMutationApplication(application),
    )
    return self
```

#### velocity

```python
velocity(*, clamp_factor: float | None = None, mutation_threshold: float | None = None) -> Self
```

Configure velocity control.

Parameters:

| Name                 | Type    | Description                                                                                                | Default |
| -------------------- | ------- | ---------------------------------------------------------------------------------------------------------- | ------- |
| `clamp_factor`       | `float` | Velocity clamping as fraction of search space (0.0-1.0). Typically 0.1-0.2. Prevents explosive velocities. | `None`  |
| `mutation_threshold` | `float` | Reinitialize velocities below this threshold. Typically 0.001-0.01. Prevents stagnation.                   | `None`  |

Returns:

| Type   | Description                             |
| ------ | --------------------------------------- |
| `Self` | The config instance for method chaining |

Source code in `commol/context/calibration.py`

```python
def velocity(
    self,
    *,
    clamp_factor: float | None = None,
    mutation_threshold: float | None = None,
) -> Self:
    """
    Configure velocity control.

    Parameters
    ----------
    clamp_factor : float, optional
        Velocity clamping as fraction of search space (0.0-1.0).
        Typically 0.1-0.2. Prevents explosive velocities.
    mutation_threshold : float, optional
        Reinitialize velocities below this threshold.
        Typically 0.001-0.01. Prevents stagnation.

    Returns
    -------
    Self
        The config instance for method chaining
    """
    self._velocity = PSOVelocityConfig(
        clamp_factor=clamp_factor,
        mutation_threshold=mutation_threshold,
    )
    return self
```

#### improvement_stop

```python
improvement_stop(*, window: int, tol_abs: float, tol_rel: float) -> Self
```

Stop when the best loss stops improving.

The run stops when the best loss improved by at most `tol_abs + tol_rel * |best loss|` over the last `window` iterations, and reports `converged=True`. Off by default. A swarm can stall for some iterations and then improve again, so a short window can stop it early.

Parameters:

| Name      | Type    | Description                                                    | Default    |
| --------- | ------- | -------------------------------------------------------------- | ---------- |
| `window`  | `int`   | Iterations over which the improvement is measured (> 0).       | *required* |
| `tol_abs` | `float` | Absolute tolerance on the improvement (>= 0).                  | *required* |
| `tol_rel` | `float` | Tolerance on the improvement relative to the best loss (>= 0). | *required* |

Returns:

| Type   | Description                             |
| ------ | --------------------------------------- |
| `Self` | The config instance for method chaining |

Source code in `commol/context/calibration.py`

```python
def improvement_stop(self, *, window: int, tol_abs: float, tol_rel: float) -> Self:
    """
    Stop when the best loss stops improving.

    The run stops when the best loss improved by at most
    ``tol_abs + tol_rel * |best loss|`` over the last `window` iterations,
    and reports ``converged=True``. Off by default. A swarm can stall for
    some iterations and then improve again, so a short window can stop it
    early.

    Parameters
    ----------
    window : int
        Iterations over which the improvement is measured (> 0).
    tol_abs : float
        Absolute tolerance on the improvement (>= 0).
    tol_rel : float
        Tolerance on the improvement relative to the best loss (>= 0).

    Returns
    -------
    Self
        The config instance for method chaining
    """
    self._improvement_stop = PSOImprovementStop(
        window=window, tol_abs=tol_abs, tol_rel=tol_rel
    )
    return self
```

#### radius_stop

```python
radius_stop(tol: float) -> Self
```

Stop when the swarm has gathered around its best position.

The run stops when the largest distance of a particle from the best position, divided by the size of the search box, is at most `tol`, and reports `converged=True`. Off by default. Mutation and velocity mutation scatter the swarm again, so with them the radius can stay above `tol`; values that the data do not tell apart keep the swarm spread.

Parameters:

| Name  | Type    | Description                                                                          | Default    |
| ----- | ------- | ------------------------------------------------------------------------------------ | ---------- |
| `tol` | `float` | Tolerance on the radius of the swarm, relative to the size of the search box (>= 0). | *required* |

Returns:

| Type   | Description                             |
| ------ | --------------------------------------- |
| `Self` | The config instance for method chaining |

Source code in `commol/context/calibration.py`

```python
def radius_stop(self, tol: float) -> Self:
    """
    Stop when the swarm has gathered around its best position.

    The run stops when the largest distance of a particle from the best
    position, divided by the size of the search box, is at most `tol`, and
    reports ``converged=True``. Off by default. Mutation and velocity
    mutation scatter the swarm again, so with them the radius can stay
    above `tol`; values that the data do not tell apart keep the swarm
    spread.

    Parameters
    ----------
    tol : float
        Tolerance on the radius of the swarm, relative to the size of the
        search box (>= 0).

    Returns
    -------
    Self
        The config instance for method chaining
    """
    self._radius_stop = PSORadiusStop(tol=tol)
    return self
```

## CalibrationResult

Bases: `BaseModel`

Result of a calibration: the best of its starts.

Attributes:

| Name                 | Type                 | Description                                                                                                                                                                                 |
| -------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `best_parameters`    | `dict[str, float]`   | The calibrated value of each parameter, in the order of the problem's parameters.                                                                                                           |
| `final_loss`         | `float`              | Loss at the best parameters.                                                                                                                                                                |
| `iterations`         | `int`                | Iterations of the best start.                                                                                                                                                               |
| `converged`          | `bool`               | Whether the best start converged.                                                                                                                                                           |
| `termination_reason` | `str`                | Why the best start stopped.                                                                                                                                                                 |
| `starts`             | `list[StartSummary]` | Every start that finished, in order; the first is the initial guess.                                                                                                                        |
| `failed`             | `list[str]`          | Each start that failed: "Start <index> failed: <reason>", also logged as a warning.                                                                                                         |
| `seed`               | `int`                | The seed of the calibration's random numbers (Calibrator.seed): problem.seed, or the seed the calibrator drew. Set as problem.seed, it repeats the calibration and its uncertainty methods. |

Source code in `commol/context/calibration.py`

```python
class CalibrationResult(BaseModel):
    """
    Result of a calibration: the best of its starts.

    Attributes
    ----------
    best_parameters : dict[str, float]
        The calibrated value of each parameter, in the order of the problem's
        parameters.
    final_loss : float
        Loss at the best parameters.
    iterations : int
        Iterations of the best start.
    converged : bool
        Whether the best start converged.
    termination_reason : str
        Why the best start stopped.
    starts : list[StartSummary]
        Every start that finished, in order; the first is the initial guess.
    failed : list[str]
        Each start that failed: ``"Start <index> failed: <reason>"``, also
        logged as a warning.
    seed : int
        The seed of the calibration's random numbers (`Calibrator.seed`):
        ``problem.seed``, or the seed the calibrator drew. Set as
        ``problem.seed``, it repeats the calibration and its uncertainty
        methods.
    """

    model_config = ConfigDict(extra="forbid")

    best_parameters: dict[str, float] = Field(
        default=..., description="Calibrated parameter values"
    )
    final_loss: float = Field(default=..., description="Final loss value")
    iterations: int = Field(
        default=..., ge=0, description="Number of iterations performed"
    )
    converged: bool = Field(default=..., description="Whether optimization converged")
    termination_reason: str = Field(
        default=..., description="Reason for optimization termination"
    )
    starts: list[StartSummary] = Field(
        default_factory=list, description="Every start that finished"
    )
    failed: list[str] = Field(
        default_factory=list, description="Each start that failed"
    )
    seed: int = Field(
        default=..., ge=0, description="Seed of the calibration's random numbers"
    )

    @override
    def __str__(self) -> str:
        """String representation of calibration result."""
        return (
            f"CalibrationResult(\n"
            f"  converged={self.converged},\n"
            f"  final_loss={self.final_loss:.6f},\n"
            f"  iterations={self.iterations},\n"
            f"  best_parameters={self.best_parameters},\n"
            f"  termination_reason='{self.termination_reason}',\n"
            f"  starts={len(self.starts)},\n"
            f"  failed={len(self.failed)},\n"
            f"  seed={self.seed}\n"
            f")"
        )
```

### Methods:

#### __str__

```python
__str__() -> str
```

String representation of calibration result.

Source code in `commol/context/calibration.py`

```python
@override
def __str__(self) -> str:
    """String representation of calibration result."""
    return (
        f"CalibrationResult(\n"
        f"  converged={self.converged},\n"
        f"  final_loss={self.final_loss:.6f},\n"
        f"  iterations={self.iterations},\n"
        f"  best_parameters={self.best_parameters},\n"
        f"  termination_reason='{self.termination_reason}',\n"
        f"  starts={len(self.starts)},\n"
        f"  failed={len(self.failed)},\n"
        f"  seed={self.seed}\n"
        f")"
    )
```

## StartSummary

Bases: `BaseModel`

One start of a multi-start optimization.

Distinct ends with equal losses are a sign that the data do not identify the parameters.

Attributes:

| Name        | Type               | Description                                          |
| ----------- | ------------------ | ---------------------------------------------------- |
| `start`     | `dict[str, float]` | The point the optimizer started from.                |
| `end`       | `dict[str, float]` | The best point it found.                             |
| `loss`      | `float`            | The loss at the end.                                 |
| `converged` | `bool`             | Whether the optimizer met its convergence criterion. |

Source code in `commol/context/calibration.py`

```python
class StartSummary(BaseModel):
    """
    One start of a multi-start optimization.

    Distinct ends with equal losses are a sign that the data do not identify
    the parameters.

    Attributes
    ----------
    start : dict[str, float]
        The point the optimizer started from.
    end : dict[str, float]
        The best point it found.
    loss : float
        The loss at the end.
    converged : bool
        Whether the optimizer met its convergence criterion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: dict[str, float] = Field(description="Start point")
    end: dict[str, float] = Field(description="Best point found")
    loss: float = Field(description="Loss at the end")
    converged: bool = Field(description="Whether the optimizer converged")
```

## CalibrationHistory

Bases: `BaseModel`

A calibration and every evaluation of a feasible point it made.

Attributes:

| Name          | Type                          | Description                                                        |
| ------------- | ----------------------------- | ------------------------------------------------------------------ |
| `result`      | `CalibrationResult`           | The result of the calibration.                                     |
| `evaluations` | `list[CalibrationEvaluation]` | Every evaluation of a feasible point, in the order they were made. |

Source code in `commol/context/calibration.py`

```python
class CalibrationHistory(BaseModel):
    """
    A calibration and every evaluation of a feasible point it made.

    Attributes
    ----------
    result : CalibrationResult
        The result of the calibration.
    evaluations : list[CalibrationEvaluation]
        Every evaluation of a feasible point, in the order they were made.
    """

    model_config = ConfigDict(extra="forbid")

    result: CalibrationResult = Field(description="The result")
    evaluations: list[CalibrationEvaluation] = Field(
        description="Every evaluation of a feasible point"
    )
```

## CalibrationEvaluation

Bases: `BaseModel`

One evaluation of a feasible point during an optimization.

Attributes:

| Name         | Type               | Description                      |
| ------------ | ------------------ | -------------------------------- |
| `parameters` | `dict[str, float]` | The values the objective scored. |
| `loss`       | `float`            | The loss at these values.        |

Source code in `commol/context/calibration.py`

```python
class CalibrationEvaluation(BaseModel):
    """
    One evaluation of a feasible point during an optimization.

    Attributes
    ----------
    parameters : dict[str, float]
        The values the objective scored.
    loss : float
        The loss at these values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameters: dict[str, float] = Field(description="Values the objective scored")
    loss: float = Field(description="Loss at these values")
```

## LossFunction

Bases: `StrEnum`

Available loss functions for calibration.

Values

SSE Sum of Squared Errors RMSE Root Mean Squared Error MAE Mean Absolute Error

Source code in `commol/context/constants.py`

```python
@unique
class LossFunction(StrEnum):
    """
    Available loss functions for calibration.

    Values
    ------
    SSE
        Sum of Squared Errors
    RMSE
        Root Mean Squared Error
    MAE
        Mean Absolute Error
    """

    SSE = "sse"
    RMSE = "rmse"
    MAE = "mae"
```

## OptimizationAlgorithm

Bases: `StrEnum`

Available optimization algorithms.

Values

NELDER_MEAD Nelder-Mead simplex algorithm PARTICLE_SWARM Particle Swarm Optimization

Source code in `commol/context/constants.py`

```python
@unique
class OptimizationAlgorithm(StrEnum):
    """
    Available optimization algorithms.

    Values
    ------
    NELDER_MEAD
        Nelder-Mead simplex algorithm
    PARTICLE_SWARM
        Particle Swarm Optimization
    """

    NELDER_MEAD = "nelder_mead"
    PARTICLE_SWARM = "particle_swarm"
```
