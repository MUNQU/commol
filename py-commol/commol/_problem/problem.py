"""Calibration problems: observations, unknowns, constraints and optimizers."""

from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from commol._data.engine_json import engine_json
from commol._data.ranges import RangeChecked
from commol._fit.result import CalibrationResult
from commol.commol_rs import _commol_rs as commol_rs
from commol.observations.models import _ObservationModel

__all__ = ["calibrated_values"]


class ObservedDataPoint(RangeChecked):
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
    scale : float
        A known positive number that multiplies the model output, with the
        factor if there is one, before comparison (default: 1.0): the known
        share of the model output that the observation sees, such as the
        share of the units that report. A count model then reads the
        observed count with the mean ``scale · model``.
    window : float | None
        Optional window length for a cumulative output: the prediction is the
        output at `at` minus the output at ``at - window``.
    trials : int | None
        The trials of this observation, for a `BinomialObservations` model
        with no ``trials`` of its own (default: None).
    """

    model_config = ConfigDict(extra="forbid")

    at: float = Field(default=..., description="Position on the axis")
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
    value: float = Field(default=..., description="Observed value")
    weight: float = Field(
        default=1.0,
        description="Weight for this observation in the loss function",
    )
    factor_id: str | None = Field(
        default=None,
        description="Optional factor parameter ID that multiplies the model output",
    )
    scale: float = Field(
        default=1.0,
        description="Known positive number that multiplies the model output",
    )
    window: float | None = Field(
        default=None,
        description=(
            "Optional window length for cumulative-output observations. "
            "When set, prediction is value(at) - value(at - window)."
        ),
    )
    trials: int | None = Field(
        default=None,
        description="Trials of a binomial observation model that reads them here",
    )

    @model_validator(mode="after")
    def _validate_observed_outputs(self) -> Self:
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
    def _validate_window(self) -> Self:
        """
        A window must not start before position 0, with the rule of the
        engine (`check_window_end` of
        `crates/commol-calibration/src/observation.rs`).
        """
        if self.window is not None:
            commol_rs.calibration.check_window_end(self.at, self.window)
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
    parameter_type : {"parameter", "initial_condition", "factor", "noise"}
        What the value is: a parameter of the model; the initial amount of a
        compartment, a bin or a category; a factor that multiplies the model
        output of an observation; or an argument of an observation model, such
        as its standard deviation, which never reaches the engine
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
    parameter_type: Literal["parameter", "initial_condition", "factor", "noise"] = (
        Field(
            default=...,
            description="Type of value being calibrated",
        )
    )
    min_bound: float = Field(default=..., description="Minimum allowed value")
    max_bound: float = Field(default=..., description="Maximum allowed value")
    initial_guess: float | None = Field(
        default=None, description="Optional starting value for optimization"
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
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


class CalibrationConstraint(RangeChecked):
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
        Can reference calibration parameters by their IDs, and the entries
        ``<M>_<i>_<j>`` and the spectral radius ``<M>_radius`` of a matrix
        between categories, scaled by the values of the point. When `at` is
        specified, can also reference output values (A, B, C, etc.) at those
        positions.
    description : str | None
        Human-readable description of the constraint (optional)
    weight : float
        Penalty weight multiplier (default: 1.0). Higher values make this constraint
        more important relative to others. The amount of a violation of this
        constraint is weight * |violation|, summed over its positions; the loss
        of a point that violates constraints is P * (1 + the sum of the amounts),
        with P much larger than any fit.
    at : list[float] | None
        Optional positions on the axis at which to evaluate this constraint.
        The engine decides which positions its model type accepts: a
        DifferenceEquations model only integer steps. If None, the constraint
        is evaluated once before simulation from the parameter values and the
        matrices at them. If given, it is evaluated at each position and can
        reference output values.

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
        description="Mathematical expression that must evaluate >= 0",
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the constraint"
    )
    weight: float = Field(
        default=1.0,
        description="Penalty weight multiplier for violations",
    )
    at: list[float] | None = Field(
        default=None,
        description=(
            "Optional positions on the axis at which to evaluate the constraint; "
            "it can then reference output values."
        ),
    )

    @field_validator("expression")
    @classmethod
    def _validate_expression(cls, value: str) -> str:
        """Check the expression with the parser of the engine."""
        commol_rs.core.MathExpression(value).check()
        return value

    @model_validator(mode="after")
    def _validate_positions(self) -> Self:
        """Validate that the positions, when given, are not empty."""
        if self.at is not None and len(self.at) == 0:
            raise ValueError(
                f"at for constraint '{self.id}' must not be empty if specified"
            )
        return self


class NelderMeadConfig(RangeChecked):
    """
    Configuration for the Nelder-Mead optimization algorithm.

    The Nelder-Mead method is a simplex-based derivative-free optimization
    algorithm, suitable for problems where gradients are not available.

    Attributes
    ----------
    kind : "nelder_mead"
        The optimizer, which the engine reads.
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

    kind: Literal["nelder_mead"] = Field(
        default="nelder_mead", description="The optimizer"
    )
    max_iterations: int = Field(
        default=1000, description="Maximum number of iterations"
    )
    sd_tolerance: float = Field(
        default=1e-6, description="Convergence tolerance for standard deviation"
    )
    initial_simplex_size: float = Field(
        default=0.1,
        description=(
            "Size of the initial simplex, as a fraction of each parameter's bound width"
        ),
    )
    alpha: float | None = Field(
        default=None,
        description="Reflection coefficient (default: None, uses argmin's default)",
    )
    gamma: float | None = Field(
        default=None,
        description="Expansion coefficient (default: None, uses argmin's default)",
    )
    rho: float | None = Field(
        default=None,
        description="Contraction coefficient (default: None, uses argmin's default)",
    )
    sigma: float | None = Field(
        default=None,
        description="Shrink coefficient (default: None, uses argmin's default)",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output during optimization (default: False)",
    )
    header_interval: int = Field(
        default=100,
        description=(
            "Number of iterations between table header repeats in verbose output "
            "(default: 100)"
        ),
    )


class ParticleSwarmConfig(RangeChecked):
    """
    Configuration for the Particle Swarm Optimization algorithm.

    Each strategy of the swarm is a name, and its numbers are fields of their
    own. A strategy left as None is that of canonical PSO; a number set
    without its strategy, or with another strategy, is an error, so a setting
    is never ignored.

    Attributes
    ----------
    kind : "particle_swarm"
        The optimizer, which the engine reads.
    particles : int
        Number of particles in the swarm (default: 20).
    max_iterations : int
        Maximum number of iterations (default: 1000).
    verbose : bool
        Enable verbose output (default: False).
    header_interval : int
        Number of iterations between table header repeats in verbose output
        (default: 100).
    initialization : {"uniform", "latin_hypercube", "opposition_based"}
        How the particles start (default: "uniform").
    inertia : {"constant", "chaotic"} | None
        The inertia weight: "constant" uses `inertia_factor`, "chaotic" moves
        between `inertia_min` and `inertia_max` on a logistic map (Liu et al.,
        2005). None is the weight of canonical PSO, 1/(2 ln 2) ≈ 0.72135.
    inertia_factor : float | None
        The constant weight, above 0; None is that of canonical PSO. Only with
        ``inertia="constant"``.
    inertia_min : float | None
        The lowest chaotic weight, above 0. Needed with ``inertia="chaotic"``.
    inertia_max : float | None
        The highest chaotic weight, above `inertia_min`. Needed with
        ``inertia="chaotic"``.
    acceleration : {"constant", "time_varying"} | None
        The acceleration coefficients: "constant" uses `cognitive` and
        `social`, "time_varying" moves each from its initial to its final
        value (Ratnaweera et al., 2004). None is the coefficients of canonical
        PSO, 0.5 + ln 2 ≈ 1.19315.
    cognitive : float | None
        The attraction to the best point of the particle (c1), above 0; None
        is that of canonical PSO. Only with ``acceleration="constant"``.
    social : float | None
        The attraction to the best point of the swarm (c2), above 0; None is
        that of canonical PSO. Only with ``acceleration="constant"``.
    cognitive_initial, cognitive_final : float | None
        The cognitive coefficient at the first and the last iteration, above
        0. Needed with ``acceleration="time_varying"``.
    social_initial, social_final : float | None
        The social coefficient at the first and the last iteration, above 0.
        Needed with ``acceleration="time_varying"``.
    mutation : {"gaussian", "cauchy"} | None
        The distribution of a mutation step, to escape a local optimum
        (Stacey et al., 2003); Cauchy has heavier tails. None mutates no
        particle.
    mutation_scale : float | None
        The standard deviation (gaussian) or the scale (cauchy) of a step,
        above 0. Needed with a mutation.
    mutation_probability : float | None
        The probability of a mutation at each iteration, from 0 to 1. Needed
        with a mutation.
    mutation_application : {"global_best", "all_particles", "below_average"} | None
        The particles a mutation moves. Needed with a mutation.
    velocity_clamp_factor : float | None
        The largest velocity, as a fraction of each bound width, from 0 to 1;
        None clamps no velocity.
    velocity_mutation_threshold : float | None
        The speed below which a velocity is drawn again, above 0; None draws
        none again.
    improvement_window : int | None
        Stop when the best loss improved by at most
        ``improvement_tol_abs + improvement_tol_rel * |best loss|`` over the
        last `improvement_window` iterations, and report ``converged=True``.
        Give the three fields or none. A swarm can stall for some iterations
        and then improve again, so a short window can stop it early.
    improvement_tol_abs : float | None
        The absolute tolerance of the improvement, at least 0.
    improvement_tol_rel : float | None
        The tolerance of the improvement relative to the best loss, at least
        0.
    radius_tol : float | None
        Stop when the largest distance of a particle from the best position,
        divided by the size of the search box (the norm of the widths of the
        bounds), is at most `radius_tol`, and report ``converged=True``
        (Zielinski and Laur, 2007). Mutation and velocity mutation scatter the
        swarm again, so with them the radius can stay above it.

    References
    ----------
    Liu, B., Wang, L., Jin, Y. H., Tang, F., & Huang, D. X. (2005).
    Improved particle swarm optimization combined with chaos.
    Chaos, Solitons & Fractals, 25(5), 1261-1271.

    Ratnaweera, A., Halgamuge, S. K., & Watson, H. C. (2004).
    Self-organizing hierarchical particle swarm optimizer with
    time-varying acceleration coefficients. IEEE Transactions on
    Evolutionary Computation, 8(3), 240-255.

    Stacey, A., Jancic, M., & Grundy, I. (2003). Particle swarm
    optimization with mutation. In Proceedings of the 2003 Congress
    on Evolutionary Computation (CEC 2003) (Vol. 2, pp. 1425-1430).

    Zielinski, K., & Laur, R. (2007). Stopping criteria for a constrained
    single-objective particle swarm optimization algorithm. Informatica,
    31(1), 51-59.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["particle_swarm"] = Field(
        default="particle_swarm", description="The optimizer"
    )
    particles: int = Field(default=20, description="Number of particles in the swarm")
    max_iterations: int = Field(
        default=1000, description="Maximum number of iterations"
    )
    verbose: bool = Field(
        default=False, description="Enable verbose output during optimization"
    )
    header_interval: int = Field(
        default=100,
        description="Number of iterations between table header repeats",
    )
    initialization: Literal["uniform", "latin_hypercube", "opposition_based"] = Field(
        default="uniform", description="How the particles start"
    )
    inertia: Literal["constant", "chaotic"] | None = Field(
        default=None, description="The inertia weight"
    )
    inertia_factor: float | None = Field(
        default=None, description="The constant inertia weight"
    )
    inertia_min: float | None = Field(
        default=None, description="The lowest chaotic inertia weight"
    )
    inertia_max: float | None = Field(
        default=None, description="The highest chaotic inertia weight"
    )
    acceleration: Literal["constant", "time_varying"] | None = Field(
        default=None, description="The acceleration coefficients"
    )
    cognitive: float | None = Field(
        default=None, description="The constant cognitive coefficient (c1)"
    )
    social: float | None = Field(
        default=None, description="The constant social coefficient (c2)"
    )
    cognitive_initial: float | None = Field(
        default=None, description="The cognitive coefficient at the first iteration"
    )
    cognitive_final: float | None = Field(
        default=None, description="The cognitive coefficient at the last iteration"
    )
    social_initial: float | None = Field(
        default=None, description="The social coefficient at the first iteration"
    )
    social_final: float | None = Field(
        default=None, description="The social coefficient at the last iteration"
    )
    mutation: Literal["gaussian", "cauchy"] | None = Field(
        default=None, description="The distribution of a mutation step"
    )
    mutation_scale: float | None = Field(
        default=None, description="The scale of a mutation step"
    )
    mutation_probability: float | None = Field(
        default=None, description="The probability of a mutation per iteration"
    )
    mutation_application: (
        Literal["global_best", "all_particles", "below_average"] | None
    ) = Field(default=None, description="The particles a mutation moves")
    velocity_clamp_factor: float | None = Field(
        default=None, description="The largest velocity, as a share of the bounds"
    )
    velocity_mutation_threshold: float | None = Field(
        default=None, description="The speed below which a velocity is drawn again"
    )
    improvement_window: int | None = Field(
        default=None, description="The iterations of the improvement stop rule"
    )
    improvement_tol_abs: float | None = Field(
        default=None, description="The absolute tolerance of the improvement"
    )
    improvement_tol_rel: float | None = Field(
        default=None, description="The relative tolerance of the improvement"
    )
    radius_tol: float | None = Field(
        default=None, description="The tolerance of the radius stop rule"
    )

    @model_validator(mode="after")
    def _check_strategies(self) -> Self:
        """
        Check each number against its strategy, as the engine reads them.

        Returns
        -------
        Self
            The configuration.

        Raises
        ------
        ValueError
            If a strategy lacks a number it needs, a number is set without
            its strategy or with another one, or the chaotic weights are not
            in order; the message is that of the engine.
        """
        commol_rs.json.as_read("ParticleSwarmConfig", engine_json(self))
        return self


def check_quantile_level(level: float) -> float:
    """
    Check a quantile level with the one check of commol.

    Parameters
    ----------
    level : float
        The level, such as 0.5 for the median.

    Returns
    -------
    float
        The level.

    Raises
    ------
    ValueError
        If ``level`` is not from 0 to 1.
    """
    commol_rs.inference.check_levels([level])
    return level


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

    level: Annotated[float, AfterValidator(check_quantile_level)] = Field(
        ..., description="Level of the quantile."
    )

    def __init__(self, level: float) -> None:
        """
        A quantile of the replicates.

        Parameters
        ----------
        level : float
            The level, between 0 and 1.
        """
        super().__init__(level=level)


class CalibrationProblem(RangeChecked):
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
    observation_models : dict[str, observation model] | None
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
    optimization_config : NelderMeadConfig | ParticleSwarmConfig
        Configuration for the optimization algorithm, by its ``kind``
    initial_condition_remainder : str | None
        Expanded compartment or bin that takes what the calibrated initial
        conditions leave of the amounts the model states (default: None). It
        is not calibrated itself: it takes the sum of the amounts the model
        states, less every other compartment, so that sum does not change.
        A value that would make it negative is penalized. Without a
        remainder, calibrated initial conditions are applied as given, and
        the other compartments keep their amounts.
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
        the engine draws when the calibrator is built, which `Calibrator.seed`
        gives and the calibrator logs):
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
    loss_function: Literal["sse", "rmse", "mae"] | None = Field(
        default=None,
        description="Loss function to use for measuring fit quality",
    )
    observation_models: dict[str, _ObservationModel] | None = Field(
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
    optimization_config: Annotated[
        NelderMeadConfig | ParticleSwarmConfig, Field(discriminator="kind")
    ] = Field(default=..., description="Optimization algorithm configuration")
    initial_condition_remainder: str | None = Field(
        default=None,
        description=(
            "Expanded compartment or bin that takes what the calibrated "
            "initial conditions leave of the amounts the model states."
        ),
    )
    replicates: int | None = Field(
        default=None,
        description="Replicates per evaluation of a stochastic engine.",
    )
    replicate_summary: Literal["mean", "median", "crps"] | Quantile | None = Field(
        default=None,
        description="What the observations are compared with, for a stochastic engine.",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Seed of every random number of a calibration: the replicates of a "
            "stochastic engine, the starts, a particle swarm and the methods"
        ),
    )

    @model_validator(mode="after")
    def _validate_loss_or_models(self) -> Self:
        """Require a loss or observation models, exactly one."""
        if self.loss_function is not None and self.observation_models is not None:
            raise ValueError(
                "A problem has a loss_function or observation_models, not both: "
                "the fit and the posterior would answer different questions."
            )
        if self.loss_function is None and self.observation_models is None:
            raise ValueError(
                "A problem needs a loss_function (sse, rmse or mae) or "
                "observation_models. There is no default: what to fit is a "
                "modelling choice."
            )
        if self.observation_models is not None and self.normalize_observations:
            raise ValueError(
                "normalize_observations needs a loss_function: the observation "
                "models set the scale of each series with their noise."
            )
        return self

    @model_validator(mode="after")
    def _validate_unique_parameter_ids(self) -> Self:
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
        If `result` is neither, such as a `ParameterSample`, which has no
        single set of parameters.
    """
    if isinstance(result, CalibrationResult):
        return result.best_parameters
    if not isinstance(result, Mapping):
        raise TypeError(
            f"A {type(result).__name__} has no single set of parameters; pass one "
            "of sample.draws() instead."
        )
    return result
