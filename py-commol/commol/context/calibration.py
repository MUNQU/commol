from typing import Self, override
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class LossFunction(str, Enum):
    """
    Available loss functions for calibration.

    Attributes
    ----------
    SSE : str
        Sum of Squared Errors
    RMSE : str
        Root Mean Squared Error
    MAE : str
        Mean Absolute Error
    WEIGHTED_SSE : str
        Weighted Sum of Squared Errors
    """

    SSE = "sum_squared_error"
    RMSE = "root_mean_squared_error"
    MAE = "mean_absolute_error"
    WEIGHTED_SSE = "weighted_sse"


class OptimizationAlgorithm(str, Enum):
    """
    Available optimization algorithms.

    Attributes
    ----------
    NELDER_MEAD : str
        Nelder-Mead simplex algorithm
    PARTICLE_SWARM : str
        Particle Swarm Optimization
    """

    NELDER_MEAD = "nelder_mead"
    PARTICLE_SWARM = "particle_swarm"


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


class CalibrationParameter(BaseModel):
    """
    Defines a parameter to be calibrated with its bounds.

    Attributes
    ----------
    id : str
        Parameter identifier (must match a model parameter ID)
    min_bound : float
        Minimum allowed value for this parameter
    max_bound : float
        Maximum allowed value for this parameter
    initial_guess : float | None
        Optional starting value for optimization (if None, midpoint is used)
    """

    id: str = Field(default=..., min_length=1, description="Parameter identifier")
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

    Particle Swarm Optimization (PSO) is a population-based metaheuristic
    inspired by social behavior of bird flocking or fish schooling.

    Attributes
    ----------
    num_particles : int
        Number of particles in the swarm (default: 40)
    max_iterations : int
        Maximum number of iterations (default: 1000)
    target_cost : float | None
        Target cost for early stopping (optional)
    inertia_factor : float | None
        Inertia weight applied to velocity (default: None, uses argmin's default)
    cognitive_factor : float | None
        Attraction to personal best (default: None, uses argmin's default)
    social_factor : float | None
        Attraction to swarm best (default: None, uses argmin's default)
    verbose : bool
        Enable verbose output during optimization (default: False)
    header_interval: int
        Number of iterations between table header repeats in verbose output
        (default: 100)
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


class LossConfig(BaseModel):
    """
    Configuration for the loss function used in calibration.

    Attributes
    ----------
    function : LossFunction
        The loss function to use for measuring fit quality
    """

    function: LossFunction = Field(
        default=LossFunction.SSE, description="Loss function for measuring fit quality"
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
        if self.algorithm == "nelder_mead":
            if not isinstance(self.config, NelderMeadConfig):
                raise ValueError(
                    (
                        f"Algorithm 'nelder_mead' requires NelderMeadConfig, "
                        f"but got {type(self.config).__name__}"
                    )
                )
        elif self.algorithm == "particle_swarm":
            if not isinstance(self.config, ParticleSwarmConfig):
                raise ValueError(
                    (
                        f"Algorithm 'particle_swarm' requires ParticleSwarmConfig, "
                        f"but got {type(self.config).__name__}"
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
    loss_config : LossConfig
        Configuration for the loss function
    optimization_config : OptimizationConfig
        Configuration for the optimization algorithm
    """

    observed_data: list[ObservedDataPoint] = Field(
        default=..., min_length=1, description="Observed data points"
    )
    parameters: list[CalibrationParameter] = Field(
        default=..., min_length=1, description="Parameters to calibrate"
    )
    loss_config: LossConfig = Field(
        default_factory=lambda: LossConfig(function=LossFunction.SSE),
        description="Loss function configuration",
    )
    optimization_config: OptimizationConfig = Field(
        default=..., description="Optimization algorithm configuration"
    )

    @model_validator(mode="after")
    def validate_unique_parameter_ids(self) -> Self:
        """Ensure parameter IDs are unique."""
        param_ids = [p.id for p in self.parameters]
        if len(param_ids) != len(set(param_ids)):
            duplicates = [id for id in set(param_ids) if param_ids.count(id) > 1]
            raise ValueError(f"Duplicate parameter IDs found: {duplicates}")
        return self
