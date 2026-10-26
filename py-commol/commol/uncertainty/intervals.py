"""The one confidence-interval convention, for a user: weighted quantiles and
the spread of the members of an ensemble, computed in Rust."""

from collections.abc import Sequence
from typing import overload

import numpy as np
from numpy.typing import NDArray

from commol._statistics.members import quantiles, spread


def weighted_quantiles(
    values: Sequence[float],
    levels: Sequence[float],
    weights: Sequence[float] | None = None,
) -> list[float]:
    """
    Quantiles of weighted values, by the one convention of commol.

    The quantile is the inverse of the weighted empirical distribution,
    interpolated between the midpoints of the cumulative weights. With equal
    weights it equals ``numpy.percentile(..., method="hazen")``. A weight of
    zero removes a value.

    Parameters
    ----------
    values : Sequence[float]
        The values.
    levels : Sequence[float]
        Quantile levels, each between 0 and 1.
    weights : Sequence[float] | None, optional
        One non-negative weight per value, of any scale. None gives equal
        weights.

    Returns
    -------
    list[float]
        One quantile per level.

    Raises
    ------
    ValueError
        If the lengths differ, no weight is positive or a level is not in
        ``[0, 1]``.
    """
    return quantiles(values, levels, weights)


@overload
def member_statistics(
    per_member: Sequence[float],
    confidence_level: float,
    weights: Sequence[float] | None = None,
) -> dict[str, float]: ...


@overload
def member_statistics(
    per_member: Sequence[Sequence[float]],
    confidence_level: float,
    weights: Sequence[float] | None = None,
) -> dict[str, list[float]]: ...


@overload
def member_statistics(
    per_member: NDArray[np.float64],
    confidence_level: float,
    weights: Sequence[float] | None = None,
) -> dict[str, float | list[float]]: ...


def member_statistics(
    per_member: Sequence[float] | Sequence[Sequence[float]] | NDArray[np.float64],
    confidence_level: float,
    weights: Sequence[float] | None = None,
) -> dict[str, float] | dict[str, list[float]] | dict[str, float | list[float]]:
    """
    Spread of one quantity across the members of an ensemble.

    Each member must already be reduced to the quantity of interest, so that
    the reported interval is a quantile of member values
    (:func:`weighted_quantiles`).

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
    weights : Sequence[float] | None, optional
        One weight per member, such as the weights of a
        :class:`~commol.uncertainty.Predictive`. None gives equal weights.

    Returns
    -------
    dict[str, float] | dict[str, list[float]]
        Keys ``mean``, ``median``, ``ci_lower``, ``ci_upper``, ``min`` and
        ``max``. Scalar members give scalar statistics, series members give
        series statistics. ``min`` and ``max`` are over the members of
        positive weight.

    Raises
    ------
    ValueError
        If ``confidence_level`` is out of range, ``per_member`` is empty, or
        the weights do not match the members.
    """
    return spread(per_member, confidence_level, weights)
