"""Reduction of cumulative series to per-window increments."""

from collections.abc import Iterable, Sequence


def window_end_steps(window_steps: int, num_steps: int) -> list[int]:
    """
    Steps closing each complete window of a run.

    Parameters
    ----------
    window_steps : int
        Length of one window, in simulation steps.
    num_steps : int
        Number of steps the run covers, excluding the initial state.

    Returns
    -------
    list[int]
        Steps ``window_steps``, ``2 * window_steps``, ... up to ``num_steps``.
        A trailing partial window is not included.

    Raises
    ------
    ValueError
        If ``window_steps`` is not positive.
    """
    if window_steps <= 0:
        raise ValueError(f"window_steps must be positive, got {window_steps}.")
    return list(range(window_steps, num_steps + 1, window_steps))


def windowed_totals(
    series: Sequence[float],
    window_steps: int,
    at_steps: Iterable[int] | None = None,
) -> list[float]:
    """
    Amount a cumulative series gained over each window.

    Each value is ``series[step] - series[step - window_steps]``, the same
    quantity a calibration compares an observation against when that
    observation sets ``window_steps``.

    Parameters
    ----------
    series : Sequence[float]
        A cumulative series, indexed by simulation step.
    window_steps : int
        Length of one window, in simulation steps.
    at_steps : Iterable[int] | None, optional
        Steps at which windows close. Defaults to every complete window of
        `series`. Pass the observation steps to reproduce exactly the values a
        calibration used.

    Returns
    -------
    list[float]
        One increment per requested step.

    Raises
    ------
    ValueError
        If ``window_steps`` is not positive, or a requested step has no
        complete window inside `series`.
    """
    if window_steps <= 0:
        raise ValueError(f"window_steps must be positive, got {window_steps}.")

    steps = (
        window_end_steps(window_steps, len(series) - 1)
        if at_steps is None
        else list(at_steps)
    )

    for step in steps:
        if step - window_steps < 0:
            raise ValueError(
                f"Step {step} has no complete window of {window_steps} steps before it."
            )
        if step >= len(series):
            raise ValueError(
                f"Step {step} is outside a series of {len(series)} points "
                f"(steps 0 to {len(series) - 1})."
            )

    return [series[step] - series[step - window_steps] for step in steps]
