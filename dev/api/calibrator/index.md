# Calibration API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Fits a model to observations. Guide: [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md).

## Calibrator

A Facade for running parameter calibration from a defined CalibrationProblem.

Source code in `commol/api/calibrator.py`

```python
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
        self._engine: "EngineProtocol" = simulation.engine

        logger.info(
            (
                f"Calibration initialized with {len(problem.parameters)} parameters "
                f"and {len(problem.observed_data)} observed data points."
            )
        )

        # Validate calibration parameters against model
        self._validate_calibration_parameters()
        self._validate_observed_data()
        self._warn_constraint_positions_beyond_observations()

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
        # Determine algorithm name from config type
        algorithm_name = (
            OptimizationAlgorithm.NELDER_MEAD
            if isinstance(self.problem.optimization_config, NelderMeadConfig)
            else OptimizationAlgorithm.PARTICLE_SWARM
        )
        logger.info(
            (
                f"Starting calibration with "
                f"{algorithm_name} algorithm and "
                f"{self.problem.loss_function} loss function."
            )
        )

        rust_observed_data = engine_state.rust_observed_data(self.problem.observed_data)

        # Convert parameters to Rust types
        rust_parameters = self._rust_parameters()

        rust_constraints = engine_state.rust_constraints(self.problem.constraints)

        # Convert loss config to Rust type
        rust_loss_config = self._build_loss_config()

        # Convert optimization config to Rust type
        rust_optimization_config = self._build_optimization_config()

        logger.info("Converted problem definition to Rust types.")
        logger.info("Running optimization...")

        # Call the Rust calibrate function
        rust_result = rust_calibration.calibrate(
            self._engine,
            rust_observed_data,
            rust_parameters,
            rust_constraints,
            rust_loss_config,
            rust_optimization_config,
            initial_condition_remainder=self.problem.initial_condition_remainder,
        )

        # Convert result back to Python CalibrationResult
        result = CalibrationResult(
            best_parameters=rust_result.best_parameters,
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

    def run_with_history(self) -> "CalibrationResultWithHistoryProtocol":
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

        rust_observed_data = engine_state.rust_observed_data(self.problem.observed_data)

        # Convert parameters to Rust types
        rust_parameters = self._rust_parameters()

        rust_constraints = engine_state.rust_constraints(self.problem.constraints)

        # Convert loss config to Rust type
        rust_loss_config = self._build_loss_config()

        # Convert optimization config to Rust type
        rust_optimization_config = self._build_optimization_config()

        # Call the Rust calibrate_with_history function
        rust_result = rust_calibration.calibrate_with_history(
            self._engine,
            rust_observed_data,
            rust_parameters,
            rust_constraints,
            rust_loss_config,
            rust_optimization_config,
            initial_condition_remainder=self.problem.initial_condition_remainder,
        )

        logger.info(
            f"Calibration finished after {rust_result.iterations} iterations. "
            f"Final loss: {rust_result.final_loss:.6f}. "
            f"Collected {len(rust_result.evaluations)} evaluations."
        )

        return rust_result

    def _to_rust_parameter_type(
        self, param_type: str
    ) -> "CalibrationParameterTypeProtocol":
        """Convert Python CalibrationParameterType to Rust type."""
        if param_type == CalibrationParameterType.PARAMETER:
            return rust_calibration.CalibrationParameterType.Parameter
        elif param_type == CalibrationParameterType.INITIAL_CONDITION:
            return rust_calibration.CalibrationParameterType.InitialCondition
        elif param_type == CalibrationParameterType.FACTOR:
            return rust_calibration.CalibrationParameterType.Factor
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _build_loss_config(self) -> "LossConfigProtocol":
        """Convert Python loss function to Rust LossConfig."""
        loss_func = self.problem.loss_function

        if loss_func == LossFunction.SSE:
            loss_config = rust_calibration.LossConfig.sse()
        elif loss_func == LossFunction.RMSE:
            loss_config = rust_calibration.LossConfig.rmse()
        elif loss_func == LossFunction.MAE:
            loss_config = rust_calibration.LossConfig.mae()
        else:
            raise ValueError(f"Unsupported loss function: {loss_func}.")

        # Per-series normalization is orthogonal to the chosen metric.
        return loss_config.normalized(self.problem.normalize_observations)

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

        elif isinstance(opt_config, ParticleSwarmConfig):
            return self._build_pso_config(opt_config)

        else:
            raise ValueError(
                f"Unsupported optimization config type: {type(opt_config).__name__}"
            )

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

        ps_config = rust_calibration.ParticleSwarmConfig(
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
        return rust_calibration.OptimizationConfig.particle_swarm(ps_config)

    def _build_rust_inertia(
        self, inertia: PSOConstantInertia | PSOChaoticInertia | None
    ):
        """Convert Python inertia config to Rust type."""
        if inertia is None:
            return None
        if isinstance(inertia, PSOConstantInertia):
            return rust_calibration.PSOInertiaConstant(factor=inertia.factor)
        if isinstance(inertia, PSOChaoticInertia):
            return rust_calibration.PSOInertiaChaotic(
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
            return rust_calibration.PSOAccelerationConstant(
                cognitive=acceleration.cognitive,
                social=acceleration.social,
            )
        if isinstance(acceleration, PSOTimeVaryingAcceleration):
            return rust_calibration.PSOAccelerationTimeVarying(
                c1_initial=acceleration.c1_initial,
                c1_final=acceleration.c1_final,
                c2_initial=acceleration.c2_initial,
                c2_final=acceleration.c2_final,
            )
        return None

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

    def _rust_parameters(self) -> list["CalibrationParameterProtocol"]:
        """Calibration parameters as engine types, in problem order."""
        return [
            rust_calibration.CalibrationParameter(
                id=param.id,
                parameter_type=self._to_rust_parameter_type(param.parameter_type),
                min_bound=param.min_bound,
                max_bound=param.max_bound,
                initial_guess=param.initial_guess,
            )
            for param in self.problem.parameters
        ]

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

        for param in self.problem.parameters:
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

        # The structure and the engine check the values of the model kind,
        # such as initial conditions.
        engine_state.check_model_values(
            model,
            self._engine,
            self.problem.parameters,
            self.problem.initial_condition_remainder,
        )

    def _validate_observed_data(self) -> None:
        """Validate that observed data compartments exist and have valid steps.

        Raises
        ------
        ValueError
            If observed data contains invalid compartments or negative time steps.
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
        """Warn when time-dependent constraints extend beyond observed data."""
        if not self.problem.observed_data:
            return

        max_observed_position = max(obs.at for obs in self.problem.observed_data)
        for constraint in self.problem.constraints:
            if constraint.at and max(constraint.at) > max_observed_position:
                logger.warning(
                    "Constraint '%s' has positions beyond the maximum observed "
                    "position (%s); simulation will be extended to evaluate them.",
                    constraint.id,
                    max_observed_position,
                )

    def run_probabilistic(self):
        """Run probabilistic calibration.

        Returns
        -------
        ProbabilisticCalibrationResult
            Object containing the ensemble of parameter sets, statistics,
            predictions with confidence intervals, and coverage metrics.

        Raises
        ------
        ValueError
            If probabilistic_config is not set in the CalibrationProblem.
        RuntimeError
            If calibration or ensemble selection fails.
        """
        if self.problem.probabilistic_config is None:
            raise ValueError(
                "probabilistic_config must be set in CalibrationProblem to run "
                "probabilistic calibration. Please set problem.probabilistic_config "
                "to a ProbabilisticCalibrationConfig instance."
            )

        # Import ProbabilisticCalibrator implementation
        from commol.api.probabilistic_calibrator import ProbabilisticCalibrator

        # Create a probabilistic calibrator instance and delegate to it
        prob_calibrator = ProbabilisticCalibrator(self.simulation, self.problem)
        return prob_calibrator.run()
```

### Attributes

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

    # Validate calibration parameters against model
    self._validate_calibration_parameters()
    self._validate_observed_data()
    self._warn_constraint_positions_beyond_observations()
```

#### run

```python
run() -> CalibrationResult
```

Runs the calibration optimization.

Returns:

| Type                | Description                                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `CalibrationResult` | Object containing the optimized parameter values, final loss, convergence status, and other optimization statistics. |

Raises:

| Type           | Description                              |
| -------------- | ---------------------------------------- |
| `ImportError`  | If Rust extension is not available.      |
| `ValueError`   | If calibration problem setup is invalid. |
| `RuntimeError` | If optimization fails.                   |

Source code in `commol/api/calibrator.py`

```python
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
    # Determine algorithm name from config type
    algorithm_name = (
        OptimizationAlgorithm.NELDER_MEAD
        if isinstance(self.problem.optimization_config, NelderMeadConfig)
        else OptimizationAlgorithm.PARTICLE_SWARM
    )
    logger.info(
        (
            f"Starting calibration with "
            f"{algorithm_name} algorithm and "
            f"{self.problem.loss_function} loss function."
        )
    )

    rust_observed_data = engine_state.rust_observed_data(self.problem.observed_data)

    # Convert parameters to Rust types
    rust_parameters = self._rust_parameters()

    rust_constraints = engine_state.rust_constraints(self.problem.constraints)

    # Convert loss config to Rust type
    rust_loss_config = self._build_loss_config()

    # Convert optimization config to Rust type
    rust_optimization_config = self._build_optimization_config()

    logger.info("Converted problem definition to Rust types.")
    logger.info("Running optimization...")

    # Call the Rust calibrate function
    rust_result = rust_calibration.calibrate(
        self._engine,
        rust_observed_data,
        rust_parameters,
        rust_constraints,
        rust_loss_config,
        rust_optimization_config,
        initial_condition_remainder=self.problem.initial_condition_remainder,
    )

    # Convert result back to Python CalibrationResult
    result = CalibrationResult(
        best_parameters=rust_result.best_parameters,
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
```

#### run_with_history

```python
run_with_history() -> CalibrationResultWithHistoryProtocol
```

Runs the calibration optimization and returns evaluation history.

Returns all objective function evaluations that occurred during optimization, not just the final best result. This is useful for probabilistic calibration where we want to explore the parameter space.

Returns:

| Type                           | Description                                                              |
| ------------------------------ | ------------------------------------------------------------------------ |
| `CalibrationResultWithHistory` | Object containing optimized parameters, final loss, and all evaluations. |

Raises:

| Type           | Description                              |
| -------------- | ---------------------------------------- |
| `ImportError`  | If Rust extension is not available.      |
| `ValueError`   | If calibration problem setup is invalid. |
| `RuntimeError` | If optimization fails.                   |

Source code in `commol/api/calibrator.py`

```python
def run_with_history(self) -> "CalibrationResultWithHistoryProtocol":
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

    rust_observed_data = engine_state.rust_observed_data(self.problem.observed_data)

    # Convert parameters to Rust types
    rust_parameters = self._rust_parameters()

    rust_constraints = engine_state.rust_constraints(self.problem.constraints)

    # Convert loss config to Rust type
    rust_loss_config = self._build_loss_config()

    # Convert optimization config to Rust type
    rust_optimization_config = self._build_optimization_config()

    # Call the Rust calibrate_with_history function
    rust_result = rust_calibration.calibrate_with_history(
        self._engine,
        rust_observed_data,
        rust_parameters,
        rust_constraints,
        rust_loss_config,
        rust_optimization_config,
        initial_condition_remainder=self.problem.initial_condition_remainder,
    )

    logger.info(
        f"Calibration finished after {rust_result.iterations} iterations. "
        f"Final loss: {rust_result.final_loss:.6f}. "
        f"Collected {len(rust_result.evaluations)} evaluations."
    )

    return rust_result
```

#### run_probabilistic

```python
run_probabilistic()
```

Run probabilistic calibration.

Returns:

| Type                             | Description                                                                                                                |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `ProbabilisticCalibrationResult` | Object containing the ensemble of parameter sets, statistics, predictions with confidence intervals, and coverage metrics. |

Raises:

| Type           | Description                                                   |
| -------------- | ------------------------------------------------------------- |
| `ValueError`   | If probabilistic_config is not set in the CalibrationProblem. |
| `RuntimeError` | If calibration or ensemble selection fails.                   |

Source code in `commol/api/calibrator.py`

```python
def run_probabilistic(self):
    """Run probabilistic calibration.

    Returns
    -------
    ProbabilisticCalibrationResult
        Object containing the ensemble of parameter sets, statistics,
        predictions with confidence intervals, and coverage metrics.

    Raises
    ------
    ValueError
        If probabilistic_config is not set in the CalibrationProblem.
    RuntimeError
        If calibration or ensemble selection fails.
    """
    if self.problem.probabilistic_config is None:
        raise ValueError(
            "probabilistic_config must be set in CalibrationProblem to run "
            "probabilistic calibration. Please set problem.probabilistic_config "
            "to a ProbabilisticCalibrationConfig instance."
        )

    # Import ProbabilisticCalibrator implementation
    from commol.api.probabilistic_calibrator import ProbabilisticCalibrator

    # Create a probabilistic calibrator instance and delegate to it
    prob_calibrator = ProbabilisticCalibrator(self.simulation, self.problem)
    return prob_calibrator.run()
```

## CalibrationProblem

Bases: `BaseModel`

Defines a complete calibration problem.

This class encapsulates all the information needed to calibrate model parameters against observed data. It provides validation of the calibration setup but delegates the actual optimization to the Rust backend.

Attributes:

| Name                          | Type                             | Description                                                                                                                |
| ----------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `observed_data`               | `list[ObservedDataPoint]`        | List of observed data points to fit against                                                                                |
| `parameters`                  | `list[CalibrationParameter]`     | List of parameters to calibrate with their bounds                                                                          |
| `constraints`                 | `list[CalibrationConstraint]`    | List of constraints on calibration parameters (optional, default: empty list)                                              |
| `loss_function`               | `str`                            | Loss function to use for measuring fit quality: "sse" (default), "rmse" or "mae". Every loss uses the observation weights. |
| `normalize_observations`      | `bool`                           | When True, divide each residual by the RMS of its observation series before applying the loss metric (default: False).     |
| `optimization_config`         | `OptimizationConfig`             | Configuration for the optimization algorithm                                                                               |
| `initial_condition_remainder` | \`str                            | None\`                                                                                                                     |
| `probabilistic_config`        | \`ProbabilisticCalibrationConfig | None\`                                                                                                                     |
| `seed`                        | \`int                            | None\`                                                                                                                     |

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
    loss_function : str
        Loss function to use for measuring fit quality: "sse" (default),
        "rmse" or "mae". Every loss uses the observation weights.
    normalize_observations : bool
        When True, divide each residual by the RMS of its observation series
        before applying the loss metric (default: False).
    optimization_config : OptimizationConfig
        Configuration for the optimization algorithm
    initial_condition_remainder : str | None
        Expanded compartment or bin that absorbs the difference to the
        population size once the calibrated initial conditions are applied
        (default: None). It is not calibrated itself. With a remainder, the
        initial population always equals ``population_size``; without one,
        calibrated initial conditions are applied as given and only a total
        above ``population_size`` is penalized.
    probabilistic_config : ProbabilisticCalibrationConfig | None
        Optional configuration for probabilistic calibration (default: None).
        When provided, enables ensemble-based parameter estimation with
        uncertainty quantification instead of single-point optimization.
    seed : int | None
        Random seed for reproducibility across all stochastic processes
        (default: None, uses system entropy).
        Controls randomness in:
        - Optimization algorithms (e.g., Particle Swarm initialization)
        - Probabilistic calibration runs
        - Clustering algorithms
        - Ensemble selection
        When set, all random operations become deterministic and reproducible.
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
    loss_function: str = Field(
        default=LossFunction.SSE,
        description="Loss function to use for measuring fit quality",
    )
    normalize_observations: bool = Field(
        default=False,
        description=(
            "Normalize each residual by the RMS of its observation series before "
            "applying the loss metric. Keeps series measured on very different "
            "scales comparable in both the optimization loss and fit-gated "
            "ensemble selection."
        ),
    )
    optimization_config: NelderMeadConfig | ParticleSwarmConfig = Field(
        default=..., description="Optimization algorithm configuration"
    )
    initial_condition_remainder: str | None = Field(
        default=None,
        description=(
            "Expanded compartment or bin that absorbs the difference to the "
            "population size once the calibrated initial conditions are applied."
        ),
    )
    probabilistic_config: ProbabilisticCalibrationConfig | None = Field(
        default=None,
        description="Optional configuration for probabilistic calibration",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Random seed for reproducibility across all stochastic processes "
            "(optimization, probabilistic calibration, clustering, ensemble selection)"
        ),
    )

    @field_validator("loss_function")
    def validate_loss_function(cls, v: str) -> str:
        if v not in LossFunction:
            raise ValueError(
                f"Invalid loss_function: {v}. Must be one of {list(LossFunction)}"
            )
        return v

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
        ``Σ w·|r| / Σ w``.
    factor_id : str | None
        Optional factor parameter ID that multiplies the model output before
        comparison
    window : float | None
        Optional window length for a cumulative output: the prediction is the
        output at `at` minus the output at ``at - window``.
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
| `expression`  | `str`         | Mathematical expression that must evaluate >= 0 for constraint satisfaction. Can reference calibration parameters by their IDs. When at is specified, can also reference output values (S, I, R, etc.) at those positions. |
| `description` | \`str         | None\`                                                                                                                                                                                                                     |
| `weight`      | `float`       | Penalty weight multiplier (default: 1.0). Higher values make this constraint more important relative to others. The penalty for violating this constraint is: weight * violation^2                                         |
| `at`          | \`list[float] | None\`                                                                                                                                                                                                                     |

Examples:

```pycon
>>> # R0 = beta/gamma must be <= 5
>>> constraint = CalibrationConstraint(
...     id="r0_bound",
...     expression="5.0 - beta/gamma",
...     description="Basic reproduction number R0 <= 5",
... )
```

```pycon
>>> # beta must be >= gamma
>>> constraint = CalibrationConstraint(
...     id="ordering",
...     expression="beta - gamma",
...     description="Transmission rate >= recovery rate",
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
        can also reference output values (S, I, R, etc.) at those positions.
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
    >>> # R0 = beta/gamma must be <= 5
    >>> constraint = CalibrationConstraint(
    ...     id="r0_bound",
    ...     expression="5.0 - beta/gamma",
    ...     description="Basic reproduction number R0 <= 5",
    ... )

    >>> # beta must be >= gamma
    >>> constraint = CalibrationConstraint(
    ...     id="ordering",
    ...     expression="beta - gamma",
    ...     description="Transmission rate >= recovery rate",
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
| `num_particles`  | `int`  | Number of particles in the swarm (default: 20)        |
| `max_iterations` | `int`  | Maximum number of iterations (default: 1000)          |
| `verbose`        | `bool` | Enable verbose output (default: False)                |
| `initialization` | `str`  | Particle initialization strategy (default: "uniform") |

Methods:

| Name           | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `inertia`      | Set inertia weight strategy ("constant" or "chaotic")        |
| `acceleration` | Set acceleration coefficients ("constant" or "time_varying") |
| `mutation`     | Enable mutation to escape local optima                       |
| `velocity`     | Configure velocity control                                   |

Source code in `commol/context/calibration.py`

```python
class ParticleSwarmConfig(BaseModel):
    """
    Configuration for the Particle Swarm Optimization algorithm.

    Attributes
    ----------
    num_particles : int
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
    """

    model_config = ConfigDict(extra="forbid")

    # Core parameters
    num_particles: int = Field(
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

    @field_validator("initialization")
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
        elif type == PSOInertiaType.CHAOTIC:
            self._inertia = PSOChaoticInertia(
                w_min=kwargs["w_min"], w_max=kwargs["w_max"]
            )
        else:
            raise ValueError(
                f"Unknown inertia type: {type}. Use 'constant' or 'chaotic'"
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
        elif type == PSOAccelerationType.TIME_VARYING:
            self._acceleration = PSOTimeVaryingAcceleration(
                c1_initial=kwargs["c1_initial"],
                c1_final=kwargs["c1_final"],
                c2_initial=kwargs["c2_initial"],
                c2_final=kwargs["c2_final"],
            )
        else:
            raise ValueError(
                f"Unknown acceleration type: {type}. Use 'constant' or 'time_varying'"
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
            strategy=strategy,
            scale=scale,
            probability=probability,
            application=application,
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
    elif type == PSOInertiaType.CHAOTIC:
        self._inertia = PSOChaoticInertia(
            w_min=kwargs["w_min"], w_max=kwargs["w_max"]
        )
    else:
        raise ValueError(
            f"Unknown inertia type: {type}. Use 'constant' or 'chaotic'"
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
    elif type == PSOAccelerationType.TIME_VARYING:
        self._acceleration = PSOTimeVaryingAcceleration(
            c1_initial=kwargs["c1_initial"],
            c1_final=kwargs["c1_final"],
            c2_initial=kwargs["c2_initial"],
            c2_final=kwargs["c2_final"],
        )
    else:
        raise ValueError(
            f"Unknown acceleration type: {type}. Use 'constant' or 'time_varying'"
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
        strategy=strategy,
        scale=scale,
        probability=probability,
        application=application,
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

## CalibrationResult

Bases: `BaseModel`

Result of a calibration run.

This is a simple data class that holds the results returned from the Rust calibration function.

Attributes:

| Name                 | Type               | Description                                                 |
| -------------------- | ------------------ | ----------------------------------------------------------- |
| `best_parameters`    | `dict[str, float]` | Dictionary mapping parameter IDs to their calibrated values |
| `final_loss`         | `float`            | Final loss value achieved                                   |
| `iterations`         | `int`              | Number of iterations performed                              |
| `converged`          | `bool`             | Whether the optimization converged                          |
| `termination_reason` | `str`              | Explanation of why optimization terminated                  |

Source code in `commol/context/calibration.py`

```python
class CalibrationResult(BaseModel):
    """
    Result of a calibration run.

    This is a simple data class that holds the results returned from the
    Rust calibration function.

    Attributes
    ----------
    best_parameters : dict[str, float]
        Dictionary mapping parameter IDs to their calibrated values
    final_loss : float
        Final loss value achieved
    iterations : int
        Number of iterations performed
    converged : bool
        Whether the optimization converged
    termination_reason : str
        Explanation of why optimization terminated
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

    @override
    def __str__(self) -> str:
        """String representation of calibration result."""
        return (
            f"CalibrationResult(\n"
            f"  converged={self.converged},\n"
            f"  final_loss={self.final_loss:.6f},\n"
            f"  iterations={self.iterations},\n"
            f"  best_parameters={self.best_parameters},\n"
            f"  termination_reason='{self.termination_reason}'\n"
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
        f"  termination_reason='{self.termination_reason}'\n"
        f")"
    )
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
