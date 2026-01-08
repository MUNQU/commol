from enum import StrEnum, unique


@unique
class CalibrationParameterType(StrEnum):
    """
    Type of value being calibrated.

    Values
    ------
    PARAMETER
        Model parameter
    INITIAL_CONDITION
        Initial population in a compartment
    SCALE
        Scaling factor for observed data
    """

    PARAMETER = "parameter"
    INITIAL_CONDITION = "initial_condition"
    SCALE = "scale"


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
    WEIGHTED_SSE
        Weighted Sum of Squared Errors
    """

    SSE = "sse"
    RMSE = "rmse"
    MAE = "mae"
    WEIGHTED_SSE = "weighted_sse"


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


@unique
class PSOInitializationStrategy(StrEnum):
    """
    Particle Swarm Optimization initialization strategies.

    Values
    ------
    UNIFORM
        Uniform random initialization (default)
    LATIN_HYPERCUBE
        Latin Hypercube Sampling for better coverage
    OPPOSITION_BASED
        Opposition-based learning initialization
    """

    UNIFORM = "uniform"
    LATIN_HYPERCUBE = "latin_hypercube"
    OPPOSITION_BASED = "opposition_based"


@unique
class PSOInertiaType(StrEnum):
    """
    Particle Swarm Optimization inertia weight strategies.

    Values
    ------
    CONSTANT
        Fixed inertia weight (canonical PSO)
    CHAOTIC
        Chaotic inertia using logistic map
    """

    CONSTANT = "constant"
    CHAOTIC = "chaotic"


@unique
class PSOAccelerationType(StrEnum):
    """
    Particle Swarm Optimization acceleration coefficient strategies.

    Values
    ------
    CONSTANT
        Fixed cognitive and social coefficients (canonical PSO)
    TIME_VARYING
        Time-Varying Acceleration Coefficients (TVAC)
    """

    CONSTANT = "constant"
    TIME_VARYING = "time_varying"


@unique
class PSOMutationStrategy(StrEnum):
    """
    Particle Swarm Optimization mutation distribution strategies.

    Values
    ------
    GAUSSIAN
        Gaussian distribution mutation
    CAUCHY
        Cauchy distribution mutation (heavier tails for larger jumps)
    """

    GAUSSIAN = "gaussian"
    CAUCHY = "cauchy"


@unique
class PSOMutationApplication(StrEnum):
    """
    Particle Swarm Optimization mutation application targets.

    Values
    ------
    GLOBAL_BEST
        Only mutate global best particle (most efficient)
    ALL_PARTICLES
        Mutate all particles (maximum diversity)
    BELOW_AVERAGE
        Mutate only below-average particles (balanced)
    """

    GLOBAL_BEST = "global_best"
    ALL_PARTICLES = "all_particles"
    BELOW_AVERAGE = "below_average"
