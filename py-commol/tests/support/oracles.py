"""
Oracles for results that a model predicts but does not output.

A reproduction number, a final size or an equilibrium is not an output of a
run. These helpers compute the stated value independently of commol, or check
what it predicts about a run: an infection grows above the threshold and dies
out below it, and a run settles where the formula says.
"""

import math
from collections.abc import Callable, Sequence

from commol import Simulation
from commol._model.model import Model


def bisect(
    function: Callable[[float], float], low: float, high: float, *, tol: float = 1e-15
) -> float:
    """
    The root of ``function`` between ``low`` and ``high``.

    Raises
    ------
    ValueError
        If ``function`` has the same sign at both ends.
    """
    f_low = function(low)
    if f_low * function(high) > 0:
        raise ValueError(f"no sign change between {low} and {high}")
    while high - low > tol * max(1.0, abs(low), abs(high)):
        middle = 0.5 * (low + high)
        f_middle = function(middle)
        if f_middle == 0:
            return middle
        if (f_middle > 0) == (f_low > 0):
            low, f_low = middle, f_middle
        else:
            high = middle
    return 0.5 * (low + high)


def final_size(reproduction_number: float, s0: float, extra: float = 0.0) -> float:
    """
    The susceptible fraction left by an epidemic.

    Parameters
    ----------
    reproduction_number : float
        ``R`` in the final size relation.
    s0 : float
        The initial susceptible fraction, ``S0 / total``.
    extra : float
        A constant term, such as the infection caused by an initial pathogen.

    Returns
    -------
    float
        The root ``s`` in ``(0, s0)`` of ``log(s0 / s) = R (1 - s) + extra``.
    """
    return bisect(
        lambda s: math.log(s0 / s) - reproduction_number * (1 - s) - extra,
        1e-300,
        s0,
    )


def total_at(
    model: Model, outputs: Sequence[str], positions: Sequence[float]
) -> list[float]:
    """The sum of ``outputs`` at each position."""
    results = Simulation(model).run_at(list(positions))
    return [
        sum(results[name][index] for name in outputs) for index in range(len(positions))
    ]


def growth_ratio(model: Model, infected: Sequence[str], horizon: float) -> float:
    """``I(horizon) / I(horizon / 2)``, where ``I`` sums the ``infected`` outputs."""
    half, full = total_at(model, infected, [horizon / 2, horizon])
    return full / half


def assert_threshold(
    model_at: Callable[[float], Model],
    infected: Sequence[str],
    horizon: float,
    *,
    margin: float = 0.05,
) -> None:
    """
    Assert that a reproduction number of one is the threshold of invasion.

    Parameters
    ----------
    model_at : Callable[[float], Model]
        Returns the model whose stated reproduction number is its argument,
        started with a small infection.
    infected : Sequence[str]
        The outputs that hold the infection.
    horizon : float
        A time long enough for the dominant growth rate to set the trend from
        ``horizon / 2`` to ``horizon``, and short enough for the susceptibles to
        stay near their initial value.
    margin : float
        The reproduction numbers tried are ``1 + margin`` and ``1 - margin``.
    """
    above = growth_ratio(model_at(1 + margin), infected, horizon)
    below = growth_ratio(model_at(1 - margin), infected, horizon)
    assert above > 1, f"R = {1 + margin}: the infection fell by {above}"
    assert below < 1, f"R = {1 - margin}: the infection grew by {below}"


def settled(
    model: Model, horizon: float, *, rel: float = 1e-6, abs: float = 1e-9
) -> dict[str, float]:
    """
    The outputs at ``2 * horizon``, once they have settled.

    Asserts that no output moves by more than the tolerance between
    ``horizon`` and ``2 * horizon``, so the value read is the limit of the run
    to that tolerance.
    """
    results = Simulation(model).run_at([horizon, 2 * horizon])
    for name, (first, second) in results.items():
        assert math.isclose(first, second, rel_tol=rel, abs_tol=abs), (
            f"{name} still moves after {horizon}: {first!r} -> {second!r}"
        )
    return {name: series[1] for name, series in results.items()}
