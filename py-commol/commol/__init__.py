"""Compartment models with a Rust engine: build, simulate and calibrate them.

The root holds the main flow: build a model, choose its structure, simulate
it and calibrate it. Every other public name lives in the module of its page
of the guide, such as `commol.resampling` or `commol.plotting`.
"""

import logging

from commol._build.info import build_info
from commol._calibration.calibrator import Calibrator
from commol._model.aggregate import Aggregate
from commol._model.axis_units import AxisUnits
from commol._model.builder import ModelBuilder
from commol._model.model import Model
from commol._model.neighbors import Complete, Groups
from commol._model.parameter import SeriesValue
from commol._model.patterns import AxisPattern
from commol._model.stratification_condition import StratificationCondition
from commol._model.transition import StratifiedRate
from commol._problem.problem import (
    CalibrationConstraint,
    CalibrationParameter,
    CalibrationProblem,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Quantile,
)
from commol._simulation.simulation import Simulation
from commol.commol_rs._commol_rs import engine as _engine

InfeasibleStateError = _engine.InfeasibleStateError
"""A run left the values its model allows, such as an amount below zero.

A difference-equations step or an impulse that leaves a compartment below
zero raises it. A stochastic engine raises it for a rate it cannot draw, such
as a negative rate, an impulse fraction outside 0 to 1, or an absolute
impulse that is not a whole number. It is a `ValueError`; a calibration
treats such a run as infeasible.
"""

__version__: str = build_info().version
"""The PEP 440 version of the installed commol; `build_info` adds its commit."""

logging.getLogger("commol").addHandler(logging.NullHandler())


__all__ = [
    # Build
    "ModelBuilder",
    "Model",
    "StratificationCondition",
    "StratifiedRate",
    "SeriesValue",
    "AxisUnits",
    "AxisPattern",
    # Structures and neighborhoods
    "Aggregate",
    "Complete",
    "Groups",
    # Simulate
    "Simulation",
    "InfeasibleStateError",
    # Calibrate
    "Calibrator",
    "CalibrationProblem",
    "CalibrationParameter",
    "CalibrationConstraint",
    "ObservedDataPoint",
    "NelderMeadConfig",
    "ParticleSwarmConfig",
    "Quantile",
    # Package
    "build_info",
]
