"""The ensemble spread of an entry of a report of applied values."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from commol.context.probabilistic_calibration import ParameterSetStatistics


class Interval(TypedDict):
    """Ensemble spread of one reported entry."""

    mean: float
    median: float
    ci_lower: float
    ci_upper: float
    min: float
    max: float
    std: float


def entry_interval(
    entry_id: str,
    statistics: "Mapping[str, ParameterSetStatistics] | None",
    scale: float = 1.0,
) -> Interval | None:
    """
    Ensemble spread of one reported entry.

    Parameters
    ----------
    entry_id : str
        Id of the entry.
    statistics : Mapping[str, ParameterSetStatistics] | None
        Ensemble spread per id, or None without an ensemble.
    scale : float, optional
        Factor every statistic is multiplied by.

    Returns
    -------
    Interval | None
        The spread, or None when there is no ensemble or no spread for the id.
    """
    stats = (statistics or {}).get(entry_id)
    if stats is None:
        return None
    return {
        "mean": stats.mean * scale,
        "median": stats.median * scale,
        "ci_lower": stats.percentile_lower * scale,
        "ci_upper": stats.percentile_upper * scale,
        "min": stats.min * scale,
        "max": stats.max * scale,
        "std": stats.std * scale,
    }
