"""
An environmentally driven infection that couples the within-host and the
between-host dynamics.

Within a host, healthy cells ``T``, infected cells ``T*`` and parasites ``V``
follow ``T' = Lambda - k V T - m T``, ``T*' = k V T - (m + d) T*`` and
``V' = a E + p T* - c V`` for a constant contamination ``E``: a unique
equilibrium ``U(E)`` attracts every positive state. Between hosts, with ``V`` at
its equilibrium ``V(E)``, ``I' = lambda E (N - I) - mu I`` and
``E' = theta I V(E) (1 - E) - gamma E``. The infection-free state is stable if
``Rb = theta V(0) lambda N / (mu gamma) < 1``. A positive equilibrium has
``I = lambda E N / (lambda E + mu)`` and ``E`` a root in ``(0, 1)`` of
``H(E) = (1 - E) V(E) - gamma E / (theta N) - mu gamma / (theta lambda N)``: none,
one or two roots, and with two roots the larger one is stable.

References
----------
.. [1] Feng, Z., Velasco-Hernandez, J., & Tapia-Santos, B. (2013). A
   mathematical model for coupling within-host and between-host dynamics in an
   environmentally-driven infectious disease. *Mathematical Biosciences*, 241,
   49-55. https://doi.org/10.1016/j.mbs.2012.09.004
"""

import math

import numpy as np
import pytest

from commol._model.model import Model
from tests.support import (
    ODE_REL,
    Flow,
    assert_close,
    assert_threshold,
    bisect,
    flow_model,
    settled,
)

# Cell and parasite parameters.
CELLS = {"Lam": 6e3, "k": 1.5e-6, "m": 0.3, "d": 0.2, "c": 50.0}
# Host and environment parameters.
N, MU, THETA = 1e4, 4e-4, 1e-10


def _coefficients(p: float, a: float) -> tuple[float, float, float, float]:
    """
    ``V(E)`` is the positive root of ``A V^2 + (B0 - B1 E) V - C1 E = 0``.

    It comes from setting the three within-host derivatives to zero.
    """
    lam, k, m, d, c = CELLS.values()
    return (
        c * k * (m + d),
        c * m * (m + d) - p * k * lam,
        a * k * (m + d),
        a * m * (m + d),
    )


def _parasites(e: float, p: float, a: float) -> float:
    """``V(E)``."""
    quadratic, b0, b1, c1 = _coefficients(p, a)
    linear, constant = b0 - b1 * e, -c1 * e
    if constant == 0:
        return max(0.0, -linear / quadratic)
    discriminant = linear * linear - 4 * quadratic * constant
    return (-linear + math.sqrt(discriminant)) / (2 * quadratic)


def test_within_host_cells_settle_at_their_unique_equilibrium() -> None:
    e, p, a = 0.1, 908.0, 4e5
    lam, k, m, d, c = CELLS.values()
    v = bisect(
        lambda v: a * e + p * k * v * lam / ((m + d) * (k * v + m)) - c * v, 0, 1e7
    )
    t = lam / (k * v + m)
    expected = {"T": t, "Ti": k * v * t / (m + d), "V": v}
    flows = [
        Flow("supply", None, "T", "Lam"),
        Flow("infection", "T", "Ti", "k * V * T"),
        Flow("t_deaths", "T", None, "m * T"),
        Flow("ti_deaths", "Ti", None, "(m + d) * Ti"),
        Flow("ingestion", None, "V", "a * E"),
        Flow("release", None, "V", "p * Ti"),
        Flow("clearance", "V", None, "c * V"),
    ]
    parameters = CELLS | {"p": p, "a": a, "E": e}

    for start in ({"T": 20000, "Ti": 0, "V": 0}, {"T": 5000, "Ti": 1000, "V": 100}):
        limit = settled(flow_model(start, flows, parameters), 200)

        assert_close(limit, expected, rel=ODE_REL)


def _hosts(
    values: dict[str, float], p: float, a: float, lam: float, gamma: float
) -> Model:
    """The between-host model with S = N - I as a bin, so ``N`` stays the host total."""
    quadratic, b0, b1, c1 = _coefficients(p, a)
    linear = f"({b0} - {b1} * E)"
    parasites = (
        f"((-{linear} + sqrt({linear} * {linear} + 4 * {quadratic} * {c1} * E))"
        f" / (2 * {quadratic}))"
    )
    flows = [
        Flow("infection", "S", "I", "lam * E * S"),
        Flow("turnover", "I", "S", "mu * I"),
        Flow("shedding", None, "E", f"theta * I * {parasites} * (1 - E)"),
        Flow("decay", "E", None, "gamma * E"),
    ]
    parameters = {"lam": lam, "mu": MU, "theta": THETA, "gamma": gamma}
    return flow_model(values, flows, parameters)


def _equilibria(p: float, a: float, lam: float, gamma: float) -> list[dict[str, float]]:
    """The positive equilibria, from the sign changes of ``H`` on a fine grid."""

    def h(e: float) -> float:
        g = gamma * e / (THETA * N) + MU * gamma / (THETA * lam * N)
        return (1 - e) * _parasites(e, p, a) - g

    grid = np.linspace(1e-9, 1 - 1e-9, 20001)
    signs = np.sign([h(e) for e in grid])
    roots = [
        bisect(h, grid[j], grid[j + 1]) for j in np.flatnonzero(signs[:-1] != signs[1:])
    ]
    return [{"I": lam * e * N / (lam * e + MU), "E": e} for e in roots]


SMALL = {"S": N - 1, "I": 1, "E": 0}
LARGE = {"S": N - 1000, "I": 1000, "E": 1}
# Parameters of a case with two positive equilibria.
P, A, GAMMA, LAMBDA = 850.0, 5e5, 0.015, 5.5e-4


def test_the_infection_free_state_loses_stability_at_rb_one() -> None:
    p, a, gamma = 908.0, 4e5, 0.02

    def model_at(rb: float) -> Model:
        lam = rb * MU * gamma / (THETA * _parasites(0, p, a) * N)
        return _hosts({"S": N - 1e-3, "I": 1e-3, "E": 0}, p, a, lam, gamma)

    assert_threshold(model_at, ["I"], 4e4)


@pytest.mark.parametrize(
    ("scale", "count", "start"), [(0.5, 0, LARGE), (1.0, 2, LARGE), (3.0, 1, SMALL)]
)
def test_the_number_of_positive_equilibria_decides_the_outcome(
    scale: float, count: int, start: dict[str, float]
) -> None:
    # Rb is 0.18, 0.37 and 1.1. With no positive equilibrium even a large
    # infection dies out; otherwise the run ends at the largest one.
    lam = scale * LAMBDA
    equilibria = _equilibria(P, A, lam, GAMMA)

    limit = settled(_hosts(start, P, A, lam, GAMMA), 4e5)

    assert len(equilibria) == count
    expected = equilibria[-1] if equilibria else {"I": 0.0, "E": 0.0}
    assert_close({"I": limit["I"], "E": limit["E"]}, expected, rel=ODE_REL, abs=1e-9)


def test_with_two_equilibria_a_small_infection_still_dies_out() -> None:
    rb = THETA * _parasites(0, P, A) * LAMBDA * N / (MU * GAMMA)

    limit = settled(_hosts(SMALL, P, A, LAMBDA, GAMMA), 4e5)

    assert round(rb, 2) == 0.37
    assert len(_equilibria(P, A, LAMBDA, GAMMA)) == 2
    assert_close({"I": limit["I"], "E": limit["E"]}, {"I": 0.0, "E": 0.0}, abs=1e-9)
