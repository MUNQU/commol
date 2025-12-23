from typing import Self, override

from pydantic import BaseModel, Field, field_validator, model_validator

from commol.context.constants import (
    LOSS_SSE,
    OPT_ALG_NELDER_MEAD,
    OPT_ALG_PARTICLE_SWARM,
    CalibrationParameterType,
    LossFunction,
    OptimizationAlgorithm,
)
from commol.context.probabilistic_calibration import ProbabilisticCalibrationConfig
from commol.utils.security import validate_expression_security


class ObservedDataPoint(BaseModel):
    """
    Represents a single observed data point for calibration.

    Attributes
    ----------
    step : int
        Time step of the observation
    compartment : str
        Name of the compartment being observed
    value : float
        Observed value
    weight : float
        Weight for this observation in the loss function (default: 1.0)
    scale_id : str | None
        Optional scale parameter ID to apply to model output before comparison
    """

    step: int = Field(default=..., ge=0, description="Time step of the observation")
    compartment: str = Field(
        default=..., min_length=1, description="Name of the compartment being observed"
    )
    value: float = Field(default=..., ge=0.0, description="Observed value")
    weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight for this observation in the loss function",
    )
    scale_id: str | None = Field(
        default=None,
        description="Optional scale parameter ID to apply to model output",
    )


class CalibrationParameter(BaseModel):
    """
    Defines a parameter or initial condition to be calibrated with its bounds.

    Attributes
    ----------
    id : str
        Identifier (parameter ID for parameters, bin ID for initial conditions)
    parameter_type : CalibrationParameterType
        Type of value being calibrated (default: PARAMETER)
    min_bound : float
        Minimum allowed value for this parameter
    max_bound : float
        Maximum allowed value for this parameter
    initial_guess : float | None
        Optional starting value for optimization (if None, midpoint is used)
    """

    id: str = Field(
        default=..., min_length=1, description="Parameter or bin identifier"
    )
    parameter_type: CalibrationParameterType = Field(
        default=...,
        description="Type of value being calibrated",
    )
    min_bound: float = Field(default=..., description="Minimum allowed value")
    max_bound: float = Field(default=..., description="Maximum allowed value")
    initial_guess: float | None = Field(
        default=None, description="Optional starting value for optimization"
    )

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
        Can reference calibration parameters by their IDs. When time_steps is specified,
        can also reference compartment values (S, I, R, etc.) at those time steps.
    description : str | None
        Human-readable description of the constraint (optional)
    weight : float
        Penalty weight multiplier (default: 1.0). Higher values make this constraint
        more important relative to others. The penalty for violating this constraint
        is: weight * violation^2
    time_steps : list[int] | None
        Optional time steps at which to evaluate this constraint. If None, constraint
        is evaluated once before simulation using parameter values only. If specified,
        constraint is evaluated at each time step and can reference compartment values.

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

    >>> # Time-dependent: Infected compartment <= 500 at specific time steps
    >>> constraint = CalibrationConstraint(
    ...     id="peak_infected",
    ...     expression="500.0 - I",
    ...     description="Peak infected never exceeds 500",
    ...     time_steps=[10, 20, 30, 40, 50],
    ... )
    """

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
    time_steps: list[int] | None = Field(
        default=None,
        description=(
            "Optional time steps at which to evaluate (for time-dependent constraints)"
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
    def validate_time_steps(self) -> Self:
        """Validate that time steps are non-negative and sorted."""
        if self.time_steps is not None:
            if len(self.time_steps) == 0:
                raise ValueError(
                    f"time_steps for constraint '{self.id}' must not be empty if "
                    "specified"
                )
            if any(ts < 0 for ts in self.time_steps):
                raise ValueError(
                    f"time_steps for constraint '{self.id}' must all be non-negative"
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
    simplex_perturbation : float
        Multiplier for creating initial simplex vertices by perturbing each
        parameter dimension. A value of 1.1 means 10% perturbation. (default: 1.1)
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

    max_iterations: int = Field(
        default=1000, gt=0, description="Maximum number of iterations"
    )
    sd_tolerance: float = Field(
        default=1e-6, gt=0.0, description="Convergence tolerance for standard deviation"
    )
    simplex_perturbation: float = Field(
        default=1.1,
        gt=1.0,
        description=(
            "Multiplier for creating initial simplex vertices (e.g., 1.1 = 10% "
            "perturbation)"
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


class ParticleSwarmConfig(BaseModel):
    """
    Configuration for the Particle Swarm Optimization algorithm.

    Use the constructor to create an instance, then chain builder methods to configure:
    - `.with_inertia()` or `.with_chaotic_inertia()` for inertia control
    - `.with_acceleration()` or `.with_tvac()` for acceleration coefficients
    - `.with_initialization_strategy()` for particle initialization
    - `.with_velocity_clamping()` and `.with_velocity_mutation()` for velocity control
    - `.with_mutation()` for mutation settings

    Attributes
    ----------
    num_particles : int
        Number of particles in the swarm (default: 20)
    max_iterations : int
        Maximum number of iterations (default: 1000)
    target_cost : float | None
        Target cost for early stopping (default: None)
    verbose : bool
        Enable verbose output (default: False)
    header_interval : int
        Iterations between header repeats in verbose mode (default: 100)
    """

    num_particles: int = Field(
        default=20, gt=0, description="Number of particles in the swarm"
    )
    max_iterations: int = Field(
        default=1000, gt=0, description="Maximum number of iterations"
    )
    target_cost: float | None = Field(
        default=None, description="Target cost for early stopping (default: None)"
    )
    inertia_factor: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Inertia weight applied to velocity (default: None, uses argmin's default)"
        ),
    )
    cognitive_factor: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Attraction to personal best (default: None, uses argmin's default)"
        ),
    )
    social_factor: float | None = Field(
        default=None,
        gt=0.0,
        description="Attraction to swarm best (default: None, uses argmin's default)",
    )
    default_acceleration_coefficient: float = Field(
        default=1.1931471805599454,  # 0.5 + ln(2), standard PSO default
        gt=0.0,
        description=(
            "Default value for cognitive_factor or social_factor when only one is "
            "provided (default: 0.5 + ln(2) ≈ 1.193, standard PSO default)"
        ),
    )
    chaotic_inertia: tuple[float, float] | None = Field(
        default=None, description="Chaotic inertia weight range (w_min, w_max)"
    )
    tvac: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Time-varying acceleration coefficients (c1_i, c1_f, c2_i, c2_f)",
    )
    initialization_strategy: str | None = Field(
        default=None,
        description=(
            "Initialization strategy: "
            "'uniform', 'latin_hypercube', or 'opposition_based'"
        ),
    )
    velocity_clamp_factor: float | None = Field(
        default=None, gt=0.0, le=1.0, description="Velocity clamping factor (0.0-1.0)"
    )
    velocity_mutation_threshold: float | None = Field(
        default=None, ge=0.0, description="Velocity mutation threshold"
    )
    mutation_strategy: str | None = Field(
        default=None, description="Mutation strategy: 'gaussian' or 'cauchy'"
    )
    mutation_scale: float | None = Field(
        default=None, gt=0.0, description="Mutation scale parameter"
    )
    mutation_probability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Mutation probability (0.0-1.0)"
    )
    mutation_application: str | None = Field(
        default=None,
        description=(
            "Mutation application: 'global_best', 'all_particles', or 'below_average'"
        ),
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

    def with_inertia(self, inertia: float) -> Self:
        """
        Set constant inertia weight factor.

        Parameters
        ----------
        inertia : float
            Inertia weight (must be positive)

        Returns
        -------
        Self
            Updated configuration for method chaining
        """
        self.inertia_factor = inertia
        self.chaotic_inertia = None  # Clear conflicting setting
        return self

    def with_chaotic_inertia(self, w_min: float, w_max: float) -> Self:
        """
        Enable chaotic inertia weight using logistic map.

        Parameters
        ----------
        w_min : float
            Minimum inertia weight (must be positive)
        w_max : float
            Maximum inertia weight (must be > w_min)

        Returns
        -------
        Self
            Updated configuration for method chaining
        """
        self.chaotic_inertia = (w_min, w_max)
        self.inertia_factor = None  # Clear conflicting setting
        return self

    def with_acceleration(self, cognitive: float, social: float) -> Self:
        """
        Set constant cognitive and social acceleration factors.

        Parameters
        ----------
        cognitive : float
            Attraction to personal best (must be positive)
        social : float
            Attraction to swarm best (must be positive)

        Returns
        -------
        Self
            Updated configuration for method chaining
        """
        self.cognitive_factor = cognitive
        self.social_factor = social
        self.tvac = None  # Clear conflicting setting
        return self

    def with_tvac(
        self, c1_initial: float, c1_final: float, c2_initial: float, c2_final: float
    ) -> Self:
        """
        Enable Time-Varying Acceleration Coefficients (TVAC).

        Parameters
        ----------
        c1_initial : float
            Initial cognitive factor (must be positive)
        c1_final : float
            Final cognitive factor (must be positive)
        c2_initial : float
            Initial social factor (must be positive)
        c2_final : float
            Final social factor (must be positive)

        Returns
        -------
        Self
            Updated configuration for method chaining
        """
        self.tvac = (c1_initial, c1_final, c2_initial, c2_final)
        self.cognitive_factor = None  # Clear conflicting settings
        self.social_factor = None
        return self

    def with_initialization_strategy(self, strategy: str) -> Self:
        """
        Set particle initialization strategy.

        Parameters
        ----------
        strategy : str
            One of: "uniform", "latin_hypercube", "opposition_based"

        Returns
        -------
        Self
            Updated configuration for method chaining
        """
        self.initialization_strategy = strategy
        return self

    def with_velocity_clamping(self, clamp_factor: float) -> Self:
        """
        Enable velocity clamping.

        Parameters
        ----------
        clamp_factor : float
            Fraction of search space range (typically 0.1 to 0.2)

        Returns
        -------
        Self
            Updated configuration for method chaining
        """
        self.velocity_clamp_factor = clamp_factor
        return self

    def with_velocity_mutation(self, threshold: float) -> Self:
        """
        Enable velocity mutation when velocity approaches zero.

        Parameters
        ----------
        threshold : float
            Velocity threshold for reinitialization (typically 0.001 to 0.01)

        Returns
        -------
        Self
            Updated configuration for method chaining
        """
        self.velocity_mutation_threshold = threshold
        return self

    def with_mutation(
        self, strategy: str, scale: float, probability: float, application: str
    ) -> Self:
        """
        Enable mutation to help escape local optima.

        Parameters
        ----------
        strategy : str
            Either "gaussian" or "cauchy"
        scale : float
            Standard deviation (gaussian) or scale parameter (cauchy)
        probability : float
            Mutation probability per iteration (0.0 to 1.0)
        application : str
            One of: "global_best", "all_particles", "below_average"

        Returns
        -------
        Self
            Updated configuration for method chaining
        """
        self.mutation_strategy = strategy
        self.mutation_scale = scale
        self.mutation_probability = probability
        self.mutation_application = application
        return self


class LossConfig(BaseModel):
    """
    Configuration for the loss function used in calibration.

    Attributes
    ----------
    function : LossFunction
        The loss function to use for measuring fit quality
    """

    function: LossFunction = Field(
        default=LOSS_SSE,
        description="Loss function for measuring fit quality",
    )


class OptimizationConfig(BaseModel):
    """
    Configuration for the optimization algorithm.

    Attributes
    ----------
    algorithm : Literal["nelder_mead", "particle_swarm"]
        The optimization algorithm to use
    config : NelderMeadConfig | ParticleSwarmConfig
        Configuration for the selected algorithm
    """

    algorithm: OptimizationAlgorithm = Field(
        default=..., description="Optimization algorithm to use"
    )
    config: NelderMeadConfig | ParticleSwarmConfig = Field(
        default=..., description="Algorithm-specific configuration"
    )

    @model_validator(mode="after")
    def validate_algorithm_config(self) -> Self:
        """Ensure the config type matches the selected algorithm."""
        if self.algorithm == OPT_ALG_NELDER_MEAD:
            if not isinstance(self.config, NelderMeadConfig):
                raise ValueError(
                    (
                        f"Algorithm '{OPT_ALG_NELDER_MEAD}' requires NelderMeadConfig, "
                        f"but got {type(self.config).__name__}"
                    )
                )
        elif self.algorithm == OPT_ALG_PARTICLE_SWARM:
            if not isinstance(self.config, ParticleSwarmConfig):
                raise ValueError(
                    (
                        f"Algorithm '{OPT_ALG_PARTICLE_SWARM}' requires "
                        f"ParticleSwarmConfig, but got {type(self.config).__name__}"
                    )
                )

        return self


class CalibrationResult(BaseModel):
    """
    Result of a calibration run.

    This is a simple data class that holds the results returned from the
    Rust calibration function.

    Attributes
    ----------
    best_parameters : dict[str, float]
        Dictionary mapping parameter IDs to their calibrated values
    parameter_names : list[str]
        Ordered list of parameter names
    best_parameters_list : list[float]
        Ordered list of parameter values (matches parameter_names order)
    final_loss : float
        Final loss value achieved
    iterations : int
        Number of iterations performed
    converged : bool
        Whether the optimization converged
    termination_reason : str
        Explanation of why optimization terminated
    """

    best_parameters: dict[str, float] = Field(
        default=..., description="Calibrated parameter values"
    )
    parameter_names: list[str] = Field(
        default=..., description="Ordered list of parameter names"
    )
    best_parameters_list: list[float] = Field(
        default=..., description="Ordered list of parameter values"
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
    loss_config : LossConfig
        Configuration for the loss function
    optimization_config : OptimizationConfig
        Configuration for the optimization algorithm
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
    loss_config: LossConfig = Field(
        default_factory=lambda: LossConfig(function=LOSS_SSE),
        description="Loss function configuration",
    )
    optimization_config: OptimizationConfig = Field(
        default=..., description="Optimization algorithm configuration"
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

    @model_validator(mode="after")
    def validate_unique_parameter_ids(self) -> Self:
        """Ensure parameter IDs are unique."""
        param_ids = [p.id for p in self.parameters]
        if len(param_ids) != len(set(param_ids)):
            duplicates = [id for id in set(param_ids) if param_ids.count(id) > 1]
            raise ValueError(f"Duplicate parameter IDs found: {duplicates}")
        return self
