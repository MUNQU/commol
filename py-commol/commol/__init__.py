import logging
from typing import TextIO

from commol.api import (
    Calibrator,
    ModelBuilder,
    Simulation,
    SimulationPlotter,
)
from commol.context import Model
from commol.context.calibration import (
    CalibrationConstraint,
    CalibrationParameter,
    CalibrationProblem,
    CalibrationResult,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
)
from commol.context.probabilistic_calibration import (
    CalibrationEvaluation,
    ParameterSetStatistics,
    ProbabilisticCalibrationConfig,
    ProbabilisticCalibrationResult,
    ProbClusteringConfig,
    ProbEnsembleConfig,
    ProbEvaluationFilterConfig,
    ProbRepresentativeConfig,
)
from commol.context.visualization import PlotConfig

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
    "CalibrationEvaluation",
    "CalibrationParameter",
    "CalibrationProblem",
    "CalibrationResult",
    "Calibrator",
    "Model",
    "ModelBuilder",
    "NelderMeadConfig",
    "ObservedDataPoint",
    "ParameterSetStatistics",
    "ParticleSwarmConfig",
    "PlotConfig",
    "ProbabilisticCalibrationConfig",
    "ProbabilisticCalibrationResult",
    "ProbClusteringConfig",
    "ProbEnsembleConfig",
    "ProbEvaluationFilterConfig",
    "ProbRepresentativeConfig",
    "Simulation",
    "SimulationPlotter",
]
