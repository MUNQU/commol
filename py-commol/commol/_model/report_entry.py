"""The spread of an entry of a report of applied values, from a sample."""

from collections.abc import Mapping
from typing import Protocol, TypedDict


class EntrySummary(Protocol):
    """The weighted statistics of one entry across the draws of a sample, as
    `ParameterSample.summary` gives them."""

    @property
    def mean(self) -> float:
        """The weighted mean."""
        ...

    @property
    def median(self) -> float:
        """The weighted median."""
        ...

    @property
    def lower(self) -> float:
        """The lower end of the central interval."""
        ...

    @property
    def upper(self) -> float:
        """The upper end of the central interval."""
        ...

    @property
    def min(self) -> float:
        """The smallest draw."""
        ...

    @property
    def max(self) -> float:
        """The largest draw."""
        ...

    @property
    def std(self) -> float:
        """The weighted standard deviation."""
        ...


class SummarizedSample(Protocol):
    """A sample of parameters that gives the statistics of each of them, as a
    `ParameterSample` does."""

    def summary(self, confidence_level: float) -> Mapping[str, EntrySummary]:
        """The statistics of each parameter, at `confidence_level`."""
        ...


class Interval(TypedDict):
    """Weighted spread of one reported entry across the draws of a sample."""

    mean: float
    median: float
    ci_lower: float
    ci_upper: float
    min: float
    max: float
    std: float


def entry_interval(
    entry_id: str,
    statistics: Mapping[str, EntrySummary] | None,
) -> Interval | None:
    """
    Weighted spread of one reported entry across the draws of a sample.

    Parameters
    ----------
    entry_id : str
        Id of the entry.
    statistics : Mapping[str, EntrySummary] | None
        Statistics of each id from a sample, or None without one.

    Returns
    -------
    Interval | None
        The spread, or None when there is no sample or no spread for the id.
    """
    stats = (statistics or {}).get(entry_id)
    if stats is None:
        return None
    return {
        "mean": stats.mean,
        "median": stats.median,
        "ci_lower": stats.lower,
        "ci_upper": stats.upper,
        "min": stats.min,
        "max": stats.max,
        "std": stats.std,
    }
