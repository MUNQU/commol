"""The weighted statistics of commol, over the engine (`commol-statistics`):
the quantiles of weighted values and the spread of a quantity across the
members of an ensemble. The results of a simulation read them, and
`commol.uncertainty` gives them to a user (`weighted_quantiles`,
`member_statistics`)."""

from collections.abc import Sequence
from typing import overload

import numpy as np
from numpy.typing import NDArray

from commol.commol_rs import _commol_rs as commol_rs


def quantiles(
    values: Sequence[float],
    levels: Sequence[float],
    weights: Sequence[float] | None = None,
) -> list[float]:
    """The quantiles of weighted values at `levels`; `weighted_quantiles`
    documents them."""
    values = [float(value) for value in values]
    if weights is None:
        weights = [1.0] * len(values)
    return commol_rs.inference.weighted_quantiles(
        values, [float(w) for w in weights], [float(level) for level in levels]
    )


@overload
def spread(
    per_member: Sequence[float],
    confidence_level: float,
    weights: Sequence[float] | None = None,
) -> dict[str, float]: ...


@overload
def spread(
    per_member: Sequence[Sequence[float]],
    confidence_level: float,
    weights: Sequence[float] | None = None,
) -> dict[str, list[float]]: ...


@overload
def spread(
    per_member: NDArray[np.float64],
    confidence_level: float,
    weights: Sequence[float] | None = None,
) -> dict[str, float | list[float]]: ...


def spread(
    per_member: Sequence[float] | Sequence[Sequence[float]] | NDArray[np.float64],
    confidence_level: float,
    weights: Sequence[float] | None = None,
) -> dict[str, float] | dict[str, list[float]] | dict[str, float | list[float]]:
    """The spread of one quantity across the members of an ensemble;
    `member_statistics` documents it."""
    # The level is checked before the values, also when there are none.
    commol_rs.inference.interval_levels(confidence_level)
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
    columns = values.reshape(values.shape[0], -1).T
    summaries = [
        commol_rs.inference.weighted_summary(
            column.tolist(), member_weights, confidence_level
        )
        for column in columns
    ]

    def shaped(per_column: list[float]) -> float | list[float]:
        return per_column[0] if values.ndim == 1 else per_column

    return {
        "mean": shaped([summary.mean for summary in summaries]),
        "median": shaped([summary.median for summary in summaries]),
        "ci_lower": shaped([summary.lower for summary in summaries]),
        "ci_upper": shaped([summary.upper for summary in summaries]),
        "min": shaped([summary.minimum for summary in summaries]),
        "max": shaped([summary.maximum for summary in summaries]),
    }
