"""Compartment models with a Rust engine: build, simulate and calibrate them."""

import logging
from typing import TextIO

from commol.api import (
    Calibrator,
    ModelBuilder,
    Simulation,
    SimulationPlotter,
    SimulationResults,
    AxisPattern,
    applied_parameters_report,
    ci_percentiles,
    simulation_report,
    member_statistics,
    window_end,
    window_ends,
    window_index,
    window_start,
    windowed_totals,
)
from commol.commol_rs._commol_rs import engine as _engine
from commol.context import CompartmentalStructure, Model, ModelStructure
from commol.context.axis_units import AxisUnits
from commol.context.engine import OdeSolver
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
    EnsembleSolution,
    EnsembleSelectionSummary,
    ParameterSetStatistics,
    ProbabilisticCalibrationConfig,
    ProbabilisticCalibrationResult,
    ProbClusteringConfig,
    ProbEvaluationFilterConfig,
    ProbGreedyLocalSearchConfig,
    ProbNsga2Config,
    ProbRepresentativeConfig,
)
from commol.context.visualization import PlotConfig

InfeasibleStateError = _engine.InfeasibleStateError
"""A run left the values its model allows, such as an amount below zero.

A difference-equations step that leaves a compartment below zero raises it.
It is a `ValueError`; a calibration treats such a run as infeasible.
"""

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
    "EnsembleSolution",
    "EnsembleSelectionSummary",
    "InfeasibleStateError",
    "CalibrationParameter",
    "CalibrationProblem",
    "CalibrationResult",
    "Calibrator",
    "CompartmentalStructure",
    "Model",
    "ModelBuilder",
    "ModelStructure",
    "NelderMeadConfig",
    "ObservedDataPoint",
    "OdeSolver",
    "ParameterSetStatistics",
    "ParticleSwarmConfig",
    "PlotConfig",
    "ProbabilisticCalibrationConfig",
    "ProbabilisticCalibrationResult",
    "ProbClusteringConfig",
    "ProbEvaluationFilterConfig",
    "ProbGreedyLocalSearchConfig",
    "ProbNsga2Config",
    "ProbRepresentativeConfig",
    "Simulation",
    "SimulationPlotter",
    "SimulationResults",
    "AxisPattern",
    "AxisUnits",
    "applied_parameters_report",
    "ci_percentiles",
    "member_statistics",
    "simulation_report",
    "window_end",
    "window_ends",
    "window_index",
    "window_start",
    "windowed_totals",
]
