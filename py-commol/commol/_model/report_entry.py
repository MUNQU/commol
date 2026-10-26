"""The spread of an entry of a report of applied values, from a sample."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from commol.uncertainty.sample import ParameterSummary


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
    statistics: "Mapping[str, ParameterSummary] | None",
) -> Interval | None:
    """
    Weighted spread of one reported entry across the draws of a sample.

    Parameters
    ----------
    entry_id : str
        Id of the entry.
    statistics : Mapping[str, ParameterSummary] | None
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
