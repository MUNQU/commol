"""User-facing workflows: building, simulating, calibrating, plotting and reporting."""

from .calibrator import Calibrator
from .model_builder import ModelBuilder
from .plotter import SimulationPlotter
from .probabilistic.intervals import ci_percentiles, member_statistics
from .reporting import applied_parameters_report, simulation_report
from .results import SimulationResults
from .simulation import Simulation
from .patterns import AxisPattern
from .windows import (
    window_end,
    window_ends,
    window_index,
    window_start,
    windowed_totals,
)

__all__ = [
    "Calibrator",
    "applied_parameters_report",
    "ModelBuilder",
    "SimulationPlotter",
    "Simulation",
    "SimulationResults",
    "AxisPattern",
    "ci_percentiles",
    "member_statistics",
    "simulation_report",
    "window_end",
    "window_ends",
    "window_index",
    "window_start",
    "windowed_totals",
]
