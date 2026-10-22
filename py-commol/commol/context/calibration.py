"""Calibration problems, observations, unknowns, constraints, optimizers and results."""

import math
from collections.abc import Mapping
from typing import Literal, Self, overload, override

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
from commol.context.observation_models import ObservationModel
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
    c2 increases over the iterations.

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

    strategy: PSOMutationStrategy = Field(
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
    application: PSOMutationApplication = Field(
        default=...,
        description='Application: "global_best", "all_particles", or "below_average"',
    )


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


class PSOImprovementStop(BaseModel):
    """
    The improvement stop rule of a particle swarm.

    The run stops when the best loss improved by at most
    ``tol_abs + tol_rel * |best loss|`` over the last `window` iterations.

    Attributes
    ----------
    window : int
        Iterations over which the improvement is measured (> 0).
    tol_abs : float
        Absolute tolerance on the improvement (>= 0).
    tol_rel : float
        Tolerance on the improvement relative to the best loss (>= 0).
    """

    model_config = ConfigDict(extra="forbid")

    window: int = Field(default=..., gt=0, description="Iterations of the window")
    tol_abs: float = Field(
        default=..., ge=0.0, description="Absolute tolerance on the improvement"
    )
    tol_rel: float = Field(
        default=..., ge=0.0, description="Relative tolerance on the improvement"
    )


class PSORadiusStop(BaseModel):
    """
    The radius stop rule of a particle swarm.

    The run stops when the largest distance of a particle from the best
    position, divided by the size of the search box (the norm of the widths of
    the bounds), is at most `tol` (Zielinski and Laur, 2007).

    Attributes
    ----------
    tol : float
        Tolerance on the radius of the swarm, relative to the size of the
        search box (>= 0).

    References
    ----------
    Zielinski, K., & Laur, R. (2007). Stopping criteria for a constrained
    single-objective particle swarm optimization algorithm. Informatica,
    31(1), 51-59.
    """

    model_config = ConfigDict(extra="forbid")

    tol: float = Field(
        default=..., ge=0.0, description="Tolerance on the radius of the swarm"
    )


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
    failed_runs : list[str]
        Why each start that failed did, with its number.
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
    failed_runs: list[str] = Field(
        default_factory=list, description="Why each failed start failed"
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
            f"  failed_runs={len(self.failed_runs)},\n"
            f"  seed={self.seed}\n"
            f")"
        )


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


class Quantile(BaseModel):
    """
    A quantile of the replicates of a stochastic engine, for
    ``CalibrationProblem.replicate_summary``.

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


type ReplicateSummary = Literal["mean", "median", "crps"] | Quantile
"""What the observations are compared with: the mean, the median or a
`Quantile` of the replicates, or their whole distribution (``"crps"``)."""


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


def calibrated_values(
    result: "CalibrationResult | Mapping[str, float]",
) -> "Mapping[str, float]":
    """
    Return the parameter values carried by a calibration outcome.

    Parameters
    ----------
    result : CalibrationResult | Mapping
        A calibration result, or a mapping of parameter id to value such as
        one draw of a `ParameterSample`.

    Returns
    -------
    Mapping[str, float]
        The values.

    Raises
    ------
    TypeError
        If `result` is a `ParameterSample`, which has no single set of
        parameters.
    """
    from commol.context.inference.sample import ParameterSample

    if isinstance(result, CalibrationResult):
        return result.best_parameters
    if isinstance(result, ParameterSample):
        raise TypeError(
            "A ParameterSample has no single set of parameters; pass one of "
            "sample.draws() instead."
        )
    return result
