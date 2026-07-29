"""Confidence-interval helpers for ensemble statistics."""

from collections.abc import Sequence
from typing import overload

import numpy as np
from numpy.typing import NDArray

type EnsembleValues = Sequence[float] | Sequence[Sequence[float]] | NDArray[np.float64]


def ci_percentiles(confidence_level: float) -> tuple[float, float]:
    """
    Lower and upper percentile points of a two-sided confidence interval.

    Parameters
    ----------
    confidence_level : float
        Confidence level of the interval, strictly between 0 and 1.

    Returns
    -------
    tuple[float, float]
        Lower and upper percentile points, on the 0-100 scale.

    Raises
    ------
    ValueError
        If ``confidence_level`` is not strictly between 0 and 1.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be strictly between 0 and 1, got "
            f"{confidence_level}. Note this is a fraction, not a percentage: "
            f"use 0.95 for a 95% interval."
        )
    return (
        (1.0 - confidence_level) / 2.0 * 100.0,
        (1.0 + confidence_level) / 2.0 * 100.0,
    )


@overload
def member_statistics(
    per_member: Sequence[float],
    confidence_level: float,
) -> dict[str, float]: ...


@overload
def member_statistics(
    per_member: Sequence[Sequence[float]],
    confidence_level: float,
) -> dict[str, list[float]]: ...


@overload
def member_statistics(
    per_member: NDArray[np.float64],
    confidence_level: float,
) -> dict[str, float | list[float]]: ...


def member_statistics(
    per_member: EnsembleValues,
    confidence_level: float,
) -> dict[str, float] | dict[str, list[float]] | dict[str, float | list[float]]:
    """
    Spread of one quantity across the members of an ensemble.

    Each member must already be reduced to the quantity of interest, so that
    the reported interval is a percentile of member values.

    Parameters
    ----------
    per_member : Sequence[float] | Sequence[Sequence[float]] | NDArray
        One entry per ensemble member. Entries may be scalars, in which case
        every returned statistic is a scalar, or equal-length series, in which
        case every returned statistic is a series reduced across members at
        each position. A numpy array is accepted, but its rank is not visible
        to a type checker; pass a list of lists for a precise return type.
    confidence_level : float
        Confidence level of the reported interval, strictly between 0 and 1.

    Returns
    -------
    dict[str, float] | dict[str, list[float]]
        Keys ``mean``, ``median``, ``ci_lower``, ``ci_upper``, ``min`` and
        ``max``. Scalar members give scalar statistics, series members give
        series statistics.

    Raises
    ------
    ValueError
        If ``confidence_level`` is out of range, or ``per_member`` is empty.
    """
    lower_percentile, upper_percentile = ci_percentiles(confidence_level)
    values = np.asarray(per_member, dtype=float)
    if values.size == 0:
        raise ValueError("per_member must hold at least one ensemble member.")

    return {
        "mean": np.mean(values, axis=0).tolist(),
        "median": np.percentile(values, 50.0, axis=0).tolist(),
        "ci_lower": np.percentile(values, lower_percentile, axis=0).tolist(),
        "ci_upper": np.percentile(values, upper_percentile, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
    }
