import logging
from typing import TextIO

from commol import constants
from commol.api import (
    Calibrator,
    ModelBuilder,
    ModelLoader,
    Simulation,
    SimulationPlotter,
)
from commol.context import Model
from commol.context.calibration import (
    CalibrationConstraint,
    CalibrationParameter,
    CalibrationParameterType,
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
from commol.context.visualization import (
    PlotConfig,
    SeabornContext,
    SeabornStyle,
    SeabornStyleConfig,
)

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
    "CalibrationConstraint",
    "CalibrationParameter",
    "CalibrationParameterType",
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
    "PlotConfig",
    "SeabornContext",
    "SeabornStyle",
    "SeabornStyleConfig",
    "Simulation",
    "SimulationPlotter",
]
