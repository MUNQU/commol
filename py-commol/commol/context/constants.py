"""Enumerations of calibration settings."""

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
        Initial amount of a compartment, bin or category
    FACTOR
        Factor that multiplies the model output of an observation
    NOISE
        Argument of an observation model, such as its standard deviation; it
        never reaches the engine
    """

    PARAMETER = "parameter"
    INITIAL_CONDITION = "initial_condition"
    FACTOR = "factor"
    NOISE = "noise"


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
