"""The spread of an entry of a report of applied values, from a sample."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from commol.context.inference.sample import ParameterSummary


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
    scale: float = 1.0,
) -> Interval | None:
    """
    Weighted spread of one reported entry across the draws of a sample.

    Parameters
    ----------
    entry_id : str
        Id of the entry.
    statistics : Mapping[str, ParameterSummary] | None
        Statistics of each id from a sample, or None without one.
    scale : float, optional
        Factor every statistic is multiplied by.

    Returns
    -------
    Interval | None
        The spread, or None when there is no sample or no spread for the id.
    """
    stats = (statistics or {}).get(entry_id)
    if stats is None:
        return None
    return {
        "mean": stats.mean * scale,
        "median": stats.median * scale,
        "ci_lower": stats.lower * scale,
        "ci_upper": stats.upper * scale,
        "min": stats.min * scale,
        "max": stats.max * scale,
        "std": stats.std * scale,
    }
