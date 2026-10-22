"""The one confidence-interval convention: weighted quantiles computed in Rust."""

from collections.abc import Sequence
from typing import overload

import numpy as np
from numpy.typing import NDArray

from commol.commol_rs import _commol_rs as commol_rs

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
    lower, upper = commol_rs.inference.interval_levels(confidence_level)
    return lower * 100.0, upper * 100.0


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
    values = [float(value) for value in values]
    if weights is None:
        weights = [1.0] * len(values)
    return commol_rs.inference.weighted_quantiles(
        values, [float(w) for w in weights], [float(level) for level in levels]
    )


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
    per_member: EnsembleValues,
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
        :class:`~commol.Predictive`. None gives equal weights.

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
    lower_point, upper_point = ci_percentiles(confidence_level)
    values = np.asarray(per_member, dtype=float)
    if values.size == 0:
        raise ValueError("per_member must hold at least one ensemble member.")
    member_weights = (
        [1.0] * values.shape[0] if weights is None else [float(w) for w in weights]
    )
    if len(member_weights) != values.shape[0]:
        raise ValueError(
            f"One weight per member: {values.shape[0]} members, "
            f"{len(member_weights)} weights."
        )
    levels = [0.0, lower_point / 100.0, 0.5, upper_point / 100.0, 1.0]
    columns = values.reshape(values.shape[0], -1).T
    table = [
        weighted_quantiles(column.tolist(), levels, member_weights)
        for column in columns
    ]
    means = [
        commol_rs.inference.weighted_mean(column.tolist(), member_weights)
        for column in columns
    ]

    def shaped(per_column: list[float]) -> float | list[float]:
        return per_column[0] if values.ndim == 1 else per_column

    return {
        "mean": shaped(means),
        "median": shaped([row[2] for row in table]),
        "ci_lower": shaped([row[1] for row in table]),
        "ci_upper": shaped([row[3] for row in table]),
        "min": shaped([row[0] for row in table]),
        "max": shaped([row[4] for row in table]),
    }
