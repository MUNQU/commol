"""Whether the values of each series of a problem are dependent, and how
strongly: of any values, one per observation, and of the residuals of a
predictive (guide: Uncertainty, Dependence)."""

from commol.dependence.analysis import residual_dependence, series_dependence
from commol.dependence.references import AtParameters, Refitted
from commol.dependence.results import (
    DependenceSettings,
    DrawPValues,
    ProblemDependence,
    SeriesCorrelation,
    SeriesDependence,
    SeriesStatistics,
)


__all__ = [
    "AtParameters",
    "Refitted",
    "residual_dependence",
    "series_dependence",
    # Results
    "DependenceSettings",
    "DrawPValues",
    "ProblemDependence",
    "SeriesCorrelation",
    "SeriesDependence",
    "SeriesStatistics",
]
