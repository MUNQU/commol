import logging
from typing import TextIO

from commol import constants
from commol.api import Calibrator, ModelBuilder, ModelLoader, Simulation
from commol.context import Model
from commol.context.calibration import (
    CalibrationParameter,
    CalibrationProblem,
    CalibrationResult,
    LossConfig,
    LossFunction,
    NelderMeadConfig,
    ObservedDataPoint,
    OptimizationAlgorithm,
    OptimizationConfig,
    ParticleSwarmConfig,
)
from commol.context.parameter import Parameter


logging.getLogger(__name__).addHandler(logging.NullHandler())


def add_stderr_logger(level: int = logging.INFO) -> logging.StreamHandler[TextIO]:
    logger = logging.getLogger(__name__)
    handler: logging.StreamHandler[TextIO] = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return handler


__all__ = [
    "CalibrationParameter",
    "CalibrationProblem",
    "CalibrationResult",
    "Calibrator",
    "constants",
    "LossConfig",
    "LossFunction",
    "Model",
    "ModelBuilder",
    "ModelLoader",
    "NelderMeadConfig",
    "ObservedDataPoint",
    "OptimizationAlgorithm",
    "OptimizationConfig",
    "Parameter",
    "ParticleSwarmConfig",
    "Simulation",
]
