"""Compartment models with a Rust engine: build, simulate and calibrate them."""

import logging

from commol.api import (
    AxisPattern,
    Calibrator,
    ModelBuilder,
    SensitivityAnalysis,
    Simulation,
    member_statistics,
    weighted_quantiles,
)
from commol.api.build_info import build_info
from commol.api.inference import Predictive, Scenarios
from commol.api.parallelism import get_levels, get_threads, set_threads, threads
from commol.api.plotting.compared import plot_compared
from commol.commol_rs._commol_rs import engine as _engine
from commol.context import Aggregate, Complete, Groups, Model
from commol.context.axis_units import AxisUnits
from commol.context.calibration import (
    CalibrationConstraint,
    CalibrationParameter,
    CalibrationProblem,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Quantile,
)
from commol.context.engine import (
    ODESolver,
    StochasticEventsSettings,
    StochasticStepsSettings,
)
from commol.context.inference import (
    ABCRejection,
    ABCSMC,
    AdaptiveConformal,
    BayesianBootstrap,
    BayesianPosterior,
    Beta,
    BlockJackknife,
    BootstrapCoverageEta,
    Collinearity,
    DistributionNoise,
    EnbPI,
    Gain,
    Gamma,
    GibbsPosterior,
    GibbsTarget,
    Increments,
    InformationMatchingEta,
    InputSpace,
    JackknifePlus,
    Laplace,
    LaplaceMixture,
    Level,
    LikelihoodTarget,
    LogNormal,
    LogUniform,
    LossDistance,
    Morris,
    NoNoise,
    Normal,
    ParameterSample,
    ParametricBootstrap,
    Peak,
    PeakPosition,
    PolynomialChaos,
    PRCC,
    PredictionProfile,
    PredictionSpec,
    Priors,
    ProfileLikelihood,
    Propagation,
    Ratio,
    ResidualBootstrap,
    ResidualNoise,
    RollingConformal,
    SampleIdentifiability,
    SampleMethod,
    Sandwich,
    SeriesDistance,
    Sobol,
    SplitConformal,
    StartsIdentifiability,
    StructuralRank,
    Threshold,
    Trajectory,
    Uniform,
    ValueAt,
    WildBootstrap,
    quantity_schedule,
)
from commol.context.observation_models import (
    BinomialObservations,
    LogNormalObservations,
    NegativeBinomialObservations,
    NormalObservations,
    PoissonObservations,
    StudentTObservations,
)
from commol.context.parameter import SeriesValue
from commol.context.stratification_condition import StratificationCondition
from commol.context.transition import StratifiedRate
from commol.context.visualization import PlotConfig

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

logging.getLogger(__name__).addHandler(logging.NullHandler())


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
    # Engines
    "ODESolver",
    "StochasticEventsSettings",
    "StochasticStepsSettings",
    # Plotting
    "plot_compared",
    "PlotConfig",
    # Parallelism
    "set_threads",
    "threads",
    "get_threads",
    "get_levels",
    # Distributions
    "Uniform",
    "LogUniform",
    "Normal",
    "LogNormal",
    "Beta",
    "Gamma",
    "Priors",
    # Quantities of interest
    "Level",
    "Increments",
    "ValueAt",
    "Gain",
    "Peak",
    "PeakPosition",
    "Trajectory",
    "Ratio",
    "quantity_schedule",
    # Observation models and noise
    "NormalObservations",
    "LogNormalObservations",
    "StudentTObservations",
    "PoissonObservations",
    "NegativeBinomialObservations",
    "BinomialObservations",
    "NoNoise",
    "ResidualNoise",
    "DistributionNoise",
    # Uncertainty
    "ParameterSample",
    "SampleMethod",
    "Predictive",
    "Scenarios",
    "weighted_quantiles",
    "member_statistics",
    # Resampling
    "BayesianBootstrap",
    "ParametricBootstrap",
    "ResidualBootstrap",
    "WildBootstrap",
    "BlockJackknife",
    # Profiles
    "ProfileLikelihood",
    "PredictionProfile",
    "PredictionSpec",
    "Threshold",
    # Local approximations
    "Laplace",
    "Sandwich",
    "LaplaceMixture",
    "LikelihoodTarget",
    "GibbsTarget",
    # Sequential Monte Carlo
    "GibbsPosterior",
    "BayesianPosterior",
    "ABCSMC",
    "ABCRejection",
    "InformationMatchingEta",
    "BootstrapCoverageEta",
    "LossDistance",
    "SeriesDistance",
    # Conformal prediction
    "SplitConformal",
    "RollingConformal",
    "JackknifePlus",
    "EnbPI",
    "AdaptiveConformal",
    # Identifiability
    "StructuralRank",
    "Collinearity",
    "SampleIdentifiability",
    "StartsIdentifiability",
    # Sensitivity
    "SensitivityAnalysis",
    "InputSpace",
    "Propagation",
    "PRCC",
    "Morris",
    "Sobol",
    "PolynomialChaos",
]
