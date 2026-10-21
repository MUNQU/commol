"""Windows on the axis, and the increments of cumulative series over them.

Window ``w`` of length ``window`` covers ``(w * window, (w + 1) * window]``:
a quantity observed over it is the change between its start and its end.
"""

from collections.abc import Iterable, Sequence

from commol.api.results import POSITION_TOLERANCE, index_at
from commol.context._windows import check_window, window_ends


def windowed_totals(
    series: Sequence[float],
    positions: Sequence[float],
    window: float,
    at: Iterable[float] | None = None,
) -> list[float]:
    """
    Amount a cumulative series gained over each window.

    Each value is the series at the end of the window minus the series at its
    start, ``at - window``: the same quantity a calibration compares an
    observation against when that observation sets ``window``.

    Parameters
    ----------
    series : Sequence[float]
        A cumulative series, one value per position.
    positions : Sequence[float]
        Positions of the series, ascending, such as
        ``SimulationResults.positions``.
    window : float
        Length of one window on the axis.
    at : Iterable[float] | None, optional
        Positions at which windows close. Defaults to every complete window of
        the series (:func:`window_ends`). Pass the observation positions to
        reproduce exactly the values a calibration used.

    Returns
    -------
    list[float]
        One increment per requested position.

    Raises
    ------
    ValueError
        If `window` is not positive, a window starts before position 0, or the
        series has no value at the end or the start of a requested window.
    """
    check_window(window)
    ends = (
        window_ends(window, positions[-1] if positions else 0.0)
        if at is None
        else list(at)
    )

    totals = []
    for end in ends:
        start = end - window
        if start < -POSITION_TOLERANCE * max(abs(end), 1.0):
            raise ValueError(
                f"Position {end} has no complete window of {window} before it."
            )
        totals.append(
            series[index_at(positions, end)]
            - series[index_at(positions, max(start, 0.0))]
        )
    return totals
