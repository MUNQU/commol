"""Coverage studies: does a method's interval cover the truth at its level?

A study runs a method on `K` synthetic data sets made from known parameters
and noise, and counts how often each interval holds the true value. A
calibrated interval of level `c` covers it in about `c·K` of them: the count
is binomial, so the study accepts an empirical coverage within `z` standard
errors of `c`, `z·√(c(1 − c)/K)`.

Tests that run a study carry the marker ``coverage_study``: ``pytest``
excludes them, ``pytest -m coverage_study`` runs them, and
``.github/workflows/coverage.yaml`` runs them for a pull request that changes
the uncertainty methods and before a release.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

Interval = tuple[float, float]
"""The lower and upper end of an interval."""

StudyMethod = Callable[[int, np.random.Generator], Mapping[str, Interval]]
"""A method in a study: given the index of a data set and the generator of
that data set, it makes the data, runs, and returns an interval per
quantity."""


@dataclass(frozen=True)
class CoverageStudy:
    """
    The outcome of a coverage study.

    Attributes
    ----------
    confidence_level : float
        The nominal level of the intervals.
    data_sets : int
        The number of synthetic data sets, `K`.
    hits : dict[str, int]
        For each quantity, how many intervals held its true value.
    tolerance : float
        The largest accepted distance between empirical and nominal coverage.
    """

    confidence_level: float
    data_sets: int
    hits: dict[str, int]
    tolerance: float

    def coverage(self, name: str) -> float:
        """The empirical coverage of the intervals of `name`."""
        return self.hits[name] / self.data_sets

    def calibrated(self) -> bool:
        """Whether every quantity's coverage is within the tolerance."""
        return all(
            abs(self.coverage(name) - self.confidence_level) <= self.tolerance
            for name in self.hits
        )


def coverage_study(
    method: StudyMethod,
    truth: Mapping[str, float],
    confidence_level: float,
    data_sets: int,
    seed: int,
    z: float = 3.29,
) -> CoverageStudy:
    """
    Run `method` on `data_sets` data sets and count its hits.

    Parameters
    ----------
    method : StudyMethod
        Makes data set `i` from its generator, runs the method, and returns
        an interval per quantity of `truth`.
    truth : Mapping[str, float]
        The true value of each quantity.
    confidence_level : float
        The nominal level of the intervals.
    data_sets : int
        The number of data sets, `K`.
    seed : int
        Seed of the generators: data set `i` uses ``seed + i``.
    z : float, optional
        Standard errors of tolerance; the default, 3.29, rejects a calibrated
        method in 1 study of 1000.

    Returns
    -------
    CoverageStudy
        The hits of each quantity and the tolerance.
    """
    hits = dict.fromkeys(truth, 0)
    for index in range(data_sets):
        intervals = method(index, np.random.default_rng(seed + index))
        for name, value in truth.items():
            lower, upper = intervals[name]
            hits[name] += int(lower <= value <= upper)
    tolerance = z * math.sqrt(confidence_level * (1 - confidence_level) / data_sets)
    return CoverageStudy(confidence_level, data_sets, hits, tolerance)
