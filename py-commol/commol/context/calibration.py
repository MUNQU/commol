"""Calibration problems, observations, unknowns, constraints, optimizers and results."""

import math
from collections.abc import Mapping
from typing import Self, overload, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from commol.context.constants import (
    CalibrationParameterType,
    LossFunction,
    OptimizationAlgorithm,
    PSOAccelerationType,
    PSOInertiaType,
    PSOInitializationStrategy,
    PSOMutationApplication,
    PSOMutationStrategy,
)
from commol.context.probabilistic_calibration import (
    ProbabilisticCalibrationConfig,
    ProbabilisticCalibrationResult,
)
from commol.utils.security import validate_expression_security


# Re-export for documentation
__all__ = ["OptimizationAlgorithm", "calibrated_values"]


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


class PSOInertiaStrategy(BaseModel):
    """
    Base class for PSO inertia weight strategies.

    Use either `PSOConstantInertia` or `PSOChaoticInertia`.
    """

    model_config = ConfigDict(extra="forbid")

    pass


class PSOConstantInertia(PSOInertiaStrategy):
    """
    Constant inertia weight for PSO.

    Attributes
    ----------
    factor : float
        Constant inertia weight factor (must be positive).
    """

    factor: float = Field(
        default=..., gt=0.0, description="Constant inertia weight factor"
    )


class PSOChaoticInertia(PSOInertiaStrategy):
    """
    Chaotic inertia weight for PSO using logistic map.

    The chaotic variation helps particles escape local optima through
    non-linear dynamics.

    Attributes
    ----------
    w_min : float
        Minimum inertia weight (must be positive)
    w_max : float
        Maximum inertia weight (must be > w_min)

    References
    ----------
    Liu, B., Wang, L., Jin, Y. H., Tang, F., & Huang, D. X. (2005).
    Improved particle swarm optimization combined with chaos.
    Chaos, Solitons & Fractals, 25(5), 1261-1271.
    """

    w_min: float = Field(default=..., gt=0.0, description="Minimum inertia weight")
    w_max: float = Field(default=..., gt=0.0, description="Maximum inertia weight")

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Validate that w_max > w_min."""
        if self.w_max <= self.w_min:
            raise ValueError(
                f"w_max ({self.w_max}) must be greater than w_min ({self.w_min})"
            )
        return self


class PSOAccelerationStrategy(BaseModel):
    """
    Base class for PSO acceleration coefficient strategies.

    Use either `PSOConstantAcceleration` or `PSOTimeVaryingAcceleration`.
    """

    model_config = ConfigDict(extra="forbid")

    pass


class PSOConstantAcceleration(PSOAccelerationStrategy):
    """
    Constant acceleration coefficients for PSO (canonical PSO).

    Attributes
    ----------
    cognitive : float
        Cognitive coefficient (c1) - attraction to personal best.
    social : float
        Social coefficient (c2) - attraction to swarm best.
    """

    cognitive: float = Field(
        default=..., gt=0.0, description="Cognitive coefficient (c1)"
    )
    social: float = Field(default=..., gt=0.0, description="Social coefficient (c2)")


class PSOTimeVaryingAcceleration(PSOAccelerationStrategy):
    """
    Time-Varying Acceleration Coefficients (TVAC) for PSO.

    TVAC adjusts cognitive and social coefficients over iterations to
    balance exploration and exploitation. Typically, c1 decreases and
    c2 increases over time.

    Attributes
    ----------
    c1_initial : float
        Initial cognitive factor
    c1_final : float
        Final cognitive factor
    c2_initial : float
        Initial social factor
    c2_final : float
        Final social factor

    References
    ----------
    Ratnaweera, A., Halgamuge, S. K., & Watson, H. C. (2004).
    Self-organizing hierarchical particle swarm optimizer with
    time-varying acceleration coefficients. IEEE Transactions on
    Evolutionary Computation, 8(3), 240-255.
    """

    c1_initial: float = Field(
        default=..., gt=0.0, description="Initial cognitive factor"
    )
    c1_final: float = Field(default=..., gt=0.0, description="Final cognitive factor")
    c2_initial: float = Field(default=..., gt=0.0, description="Initial social factor")
    c2_final: float = Field(default=..., gt=0.0, description="Final social factor")


class PSOMutationConfig(BaseModel):
    """
    Configuration for PSO particle mutation to help escape local optima.

    Mutation can use Gaussian or Cauchy distributions. Cauchy has heavier tails
    and provides larger jumps, making it more effective for escaping local optima.

    Attributes
    ----------
    strategy : str
        Mutation strategy: "gaussian" or "cauchy"
    scale : float
        Standard deviation (gaussian) or scale parameter (cauchy)
    probability : float
        Mutation probability per iteration (0.0 to 1.0)
    application : str
        Which particles to mutate:
        - "global_best": Only mutate global best (most efficient)
        - "all_particles": Mutate all particles (maximum diversity)
        - "below_average": Mutate only below-average particles (balanced)

    References
    ----------
    Stacey, A., Jancic, M., & Grundy, I. (2003). Particle swarm
    optimization with mutation. In Proceedings of the 2003 Congress
    on Evolutionary Computation (CEC 2003) (Vol. 2, pp. 1425-1430).
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(
        default=...,
        description='Mutation strategy: "gaussian" or "cauchy"',
    )
    scale: float = Field(
        default=..., gt=0.0, description="Standard deviation or scale parameter"
    )
    probability: float = Field(
        default=...,
        ge=0.0,
        le=1.0,
        description="Mutation probability per iteration (0.0-1.0)",
    )
    application: str = Field(
        default=...,
        description='Application: "global_best", "all_particles", or "below_average"',
    )

    @field_validator("strategy")
    def validate_strategy(cls, v: str) -> str:
        if v not in PSOMutationStrategy:
            raise ValueError(
                f"Invalid strategy: {v}. Must be one of {list(PSOMutationStrategy)}"
            )
        return v

    @field_validator("application")
    def validate_application(cls, v: str) -> str:
        if v not in PSOMutationApplication:
            raise ValueError(
                f"Invalid application: {v}. "
                f"Must be one of {list(PSOMutationApplication)}"
            )
        return v


class PSOVelocityConfig(BaseModel):
    """
    Configuration for PSO particle velocity control.

    Both clamping and mutation threshold can be used together.

    Attributes
    ----------
    clamp_factor : float | None
        Velocity clamping factor as fraction of search space range (0.0-1.0).
        Limits velocity to prevent explosive behavior.
        Typically 0.1 to 0.2. If None, no clamping is applied.
    mutation_threshold : float | None
        Velocity threshold for reinitialization when velocity approaches zero.
        Prevents particle stagnation.
        Typically 0.001 to 0.01. If None, no velocity mutation is applied.

    References
    ----------
    - Shi, Y., & Eberhart, R. C. (1998). A modified particle swarm optimizer.
    - Ratnaweera, A., Halgamuge, S. K., & Watson, H. C. (2004).
      Self-organizing hierarchical particle swarm optimizer.
    """

    model_config = ConfigDict(extra="forbid")

    clamp_factor: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Velocity clamping factor (0.0-1.0), typically 0.1-0.2",
    )
    mutation_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description="Velocity mutation threshold, typically 0.001-0.01",
    )


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


def calibrated_values(
    result: "CalibrationResult | ProbabilisticCalibrationResult | Mapping[str, float]",
) -> "Mapping[str, float]":
    """
    Return the parameter values carried by a calibration outcome.

    Parameters
    ----------
    result : CalibrationResult | ProbabilisticCalibrationResult | Mapping
        A calibration outcome, or a mapping of parameter id to value.

    Returns
    -------
    Mapping[str, float]
        The values. For a probabilistic result these are the point parameters
        of the selected ensemble.
    """
    if isinstance(result, CalibrationResult):
        return result.best_parameters
    if isinstance(result, ProbabilisticCalibrationResult):
        return result.selected_ensemble.point_parameters
    return result
