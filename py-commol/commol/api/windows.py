"""Windows on the axis, and the increments of cumulative series over them.

Window ``w`` of length ``window`` covers ``(w * window, (w + 1) * window]``:
a quantity observed over it is the change between its start and its end.
The engine computes every window (`crates/commol-calibration/src/observation.rs`).
"""

from collections.abc import Iterable, Sequence

from commol.commol_rs import _commol_rs as commol_rs


def _windowed_totals(
    series: Sequence[float],
    positions: Sequence[float],
    window: float,
    at: Iterable[float] | None = None,
) -> list[float]:
    """
    Amount a cumulative series gained over each window.

    Each value is the series at the end of the window minus the series at its
    start, ``at - window``, each found with the tolerance of the axis: the
    rule of a windowed observation (`window_increments` of
    `crates/commol-calibration/src/observation.rs`), so the value is the
    prediction a calibration compares an observation against when that
    observation sets ``window``.

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
        the series (`window_ends` of the engine, as `AxisUnits.window_ends`
        gives it). Pass the observation positions to reproduce exactly the
        values a calibration used.

    Returns
    -------
    list[float]
        One increment per requested position.

    Raises
    ------
    ValueError
        If `window` is not a finite, positive length (the message of the
        engine), a window starts before position 0, or the series has no value
        at the end or the start of a requested window.
    """
    ends = (
        commol_rs.calibration.window_ends(window, positions[-1] if positions else 0.0)
        if at is None
        else list(at)
    )
    return commol_rs.calibration.window_increments(
        [float(value) for value in series],
        [float(position) for position in positions],
        float(window),
        [float(end) for end in ends],
    )
