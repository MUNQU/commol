"""
The window grid in steps: the start, the end and the window of a position.

Window ``w`` of length ``window`` covers ``(w * window, (w + 1) * window]``.
`AxisUnits` gives the grid of a width in a unit; reports and `windowed_totals`
read it in steps, with a window of any positive length.
"""

import math

WINDOW_COUNT_TOLERANCE = 1e-9
"""Relative tolerance with which a number of windows is a whole number."""


def check_window(window: float) -> None:
    """Reject a window that is not a finite, positive length."""
    if not (math.isfinite(window) and window > 0):
        raise ValueError(f"window must be a finite, positive length, got {window}.")


def _windows_in(at: float, window: float) -> float:
    """
    Number of windows in `at`, snapped to the nearest whole number within
    ``1e-9``, so ``0.3 / 0.1`` counts as 3.
    """
    count = at / window
    nearest = round(count)
    if math.isclose(count, nearest, rel_tol=WINDOW_COUNT_TOLERANCE, abs_tol=1e-9):
        return float(nearest)
    return count


def window_start(index: int, window: float) -> float:
    """
    Position at which a window opens.

    Parameters
    ----------
    index : int
        Zero-based window index.
    window : float
        Length of one window on the axis.

    Returns
    -------
    float
        ``index * window``.

    Raises
    ------
    ValueError
        If `window` is not positive or `index` is negative.
    """
    check_window(window)
    if index < 0:
        raise ValueError(f"index must not be negative, got {index}.")
    return index * window


def window_end(index: int, window: float) -> float:
    """
    Position at which a window closes, equal to the start of the next.

    Parameters
    ----------
    index : int
        Zero-based window index.
    window : float
        Length of one window on the axis.

    Returns
    -------
    float
        ``(index + 1) * window``.

    Raises
    ------
    ValueError
        If `window` is not positive or `index` is negative.
    """
    return window_start(index + 1, window)


def window_index(at: float, window: float) -> int:
    """
    Window a position belongs to.

    The exact inverse of :func:`window_end`, and the containing window for
    any position in between: ``max(ceil(at / window) - 1, 0)``. Position 0
    belongs to window 0. :func:`commol.api.reporting.observed_report` places
    each observation with this rule.

    Parameters
    ----------
    at : float
        A position on the axis.
    window : float
        Length of one window on the axis.

    Returns
    -------
    int
        Zero-based window index.

    Raises
    ------
    ValueError
        If `window` is not positive or `at` is negative.
    """
    check_window(window)
    if not at >= 0:
        raise ValueError(f"at must not be negative, got {at}.")
    return max(math.ceil(_windows_in(at, window)) - 1, 0)


def window_ends(window: float, end: float) -> list[float]:
    """
    Positions closing each complete window up to a position.

    Parameters
    ----------
    window : float
        Length of one window on the axis.
    end : float
        Last position covered, such as the last position of a run.

    Returns
    -------
    list[float]
        ``window``, ``2 * window``, ... up to `end`. A trailing partial window
        is not included.

    Raises
    ------
    ValueError
        If `window` is not positive.
    """
    check_window(window)
    # A non-positive end has no window: range() of a count below 1 is empty.
    count = math.floor(_windows_in(end, window))
    return [window_end(index, window) for index in range(count)]
