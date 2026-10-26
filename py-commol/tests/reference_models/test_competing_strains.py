"""
A discrete-time SIS model with two competing strains and recruitment ``f(T)``.

In a generation a member survives with probability ``e^{-mu}``, a susceptible
escapes strain ``i`` with probability ``e^{-alpha_i I_i}`` and an infective of
strain ``j`` recovers with probability ``1 - e^{-gamma_j}``. The total
``T = S + I1 + I2`` follows the demographic equation ``T_{n+1} = f(T_n) + T_n e^{-mu}``,
which tends to ``Lambda / (1 - e^{-mu})`` for ``f = Lambda`` and to
``k (r + e^{-mu} - 1) / r`` for ``f(T) = r T (1 - T / k)``. At a demographic
equilibrium the infection-free state is stable if both
``R_i = e^{-mu} T alpha_i / (1 - e^{-(mu + gamma_i)})`` are below one.

References
----------
.. [1] Castillo-Chavez, C., & Yakubu, A.-A. (2001). Discrete-time S-I-S models
   with complex dynamics. *Nonlinear Analysis: Theory, Methods &
   Applications*, 47, 4753-4762. https://doi.org/10.1016/S0362-546X(01)00587-9
.. [2] Castillo-Chavez, C., & Yakubu, A.-A. (2001). Epidemics on attractors.
   *Contemporary Mathematics*, 284, 23-42. https://doi.org/10.1090/conm/284/04697
"""

import math
from collections.abc import Callable

import pytest

from commol import Simulation
from commol._model.model import Model
from tests.support import (
    ENGINE_REL,
    Flow,
    assert_close,
    assert_threshold,
    flow_model,
    settled,
)

MU, GAMMA1, GAMMA2 = 0.1, 0.5, 0.3
LAMBDA = 1000 * (1 - math.exp(-MU))
"""Constant recruitment whose demographic equilibrium is 1000."""


def _strains(
    values: dict[str, float],
    alpha1: float = 0.003,
    alpha2: float = 0.002,
    recruitment: str = "Lam",
) -> Model:
    """[1]_, [2]_; the strain shares of new infections are gated at no infection."""
    force = "a1 * I1 + a2 * I2"
    infected = f"exp(-mu) * S * (1 - exp(-({force})))"
    flows = [
        Flow("recruitment", None, "S", recruitment),
        Flow("s_deaths", "S", None, "(1 - exp(-mu)) * S"),
        Flow(
            "strain1",
            "S",
            "I1",
            f"if({force} > 0, {infected} * a1 * I1 / ({force}), 0)",
        ),
        Flow(
            "strain2",
            "S",
            "I2",
            f"if({force} > 0, {infected} * a2 * I2 / ({force}), 0)",
        ),
        Flow("recovery1", "I1", "S", "exp(-mu) * (1 - exp(-g1)) * I1"),
        Flow("recovery2", "I2", "S", "exp(-mu) * (1 - exp(-g2)) * I2"),
        Flow("deaths1", "I1", None, "(1 - exp(-mu)) * I1"),
        Flow("deaths2", "I2", None, "(1 - exp(-mu)) * I2"),
    ]
    parameters = {
        "mu": MU,
        "a1": alpha1,
        "a2": alpha2,
        "g1": GAMMA1,
        "g2": GAMMA2,
        "Lam": LAMBDA,
        "r": 0.5,
        "k": 1000.0,
    }
    return flow_model(values, flows, parameters, ode=False)


def test_the_assumptions_give_the_stated_map() -> None:
    a1, a2, steps = 0.003, 0.002, 50
    s, i1, i2 = 990.0, 5.0, 5.0

    results = Simulation(_strains({"S": s, "I1": i1, "I2": i2}, a1, a2)).run(steps)

    expected = {"S": [s], "I1": [i1], "I2": [i2]}
    survive = math.exp(-MU)
    for _ in range(steps):
        force = a1 * i1 + a2 * i2
        infected = s * survive * (1 - math.exp(-force))
        s, i1, i2 = (
            LAMBDA
            + s * survive * math.exp(-force)
            + sum(
                i * survive * (1 - math.exp(-g))
                for i, g in ((i1, GAMMA1), (i2, GAMMA2))
            ),
            a1 * i1 / force * infected + i1 * survive * math.exp(-GAMMA1),
            a2 * i2 / force * infected + i2 * survive * math.exp(-GAMMA2),
        )
        for name, value in (("S", s), ("I1", i1), ("I2", i2)):
            expected[name].append(value)
    assert_close(results, expected, rel=ENGINE_REL)


@pytest.mark.parametrize(
    ("recruitment", "demography"),
    [
        ("Lam", lambda _: LAMBDA),
        ("r * total * (1 - total / k)", lambda t: 0.5 * t * (1 - t / 1000)),
    ],
)
def test_the_total_follows_the_demographic_equation(
    recruitment: str, demography: Callable[[float], float]
) -> None:
    steps = 40

    results = Simulation(
        _strains({"S": 500, "I1": 5, "I2": 5}, recruitment=recruitment)
    ).run(steps)

    totals, t = [], 510.0
    for _ in range(steps + 1):
        totals.append(t)
        t = demography(t) + t * math.exp(-MU)
    held = [sum(results[n][j] for n in ("S", "I1", "I2")) for j in range(steps + 1)]
    assert_close(held, totals, rel=ENGINE_REL)


def test_without_infection_only_the_demography_is_left() -> None:
    results = Simulation(_strains({"S": 500, "I1": 0, "I2": 0})).run(30)

    s = [500.0]
    for _ in range(30):
        s.append(LAMBDA + s[-1] * math.exp(-MU))
    assert_close(results["S"], s, rel=ENGINE_REL)
    assert results["I1"] == results["I2"] == [0.0] * 31


@pytest.mark.parametrize(
    ("recruitment", "equilibrium"),
    [
        ("Lam", LAMBDA / (1 - math.exp(-MU))),
        ("r * total * (1 - total / k)", 1000 * (0.5 + math.exp(-MU) - 1) / 0.5),
    ],
)
def test_the_total_tends_to_the_demographic_equilibrium(
    recruitment: str, equilibrium: float
) -> None:
    limit = settled(
        _strains({"S": 510, "I1": 0, "I2": 0}, recruitment=recruitment), 400
    )

    assert_close(limit["S"], equilibrium, rel=ENGINE_REL)


@pytest.mark.parametrize("strain", [1, 2])
def test_each_strain_invades_only_above_its_reproduction_number(strain: int) -> None:
    # At T = 1000, R_i = e^-mu T alpha_i / (1 - e^-(mu + gamma_i)): alpha_i is
    # set from R_i, and the other strain is absent.
    gamma = GAMMA1 if strain == 1 else GAMMA2
    total, i0 = 1000.0, 1e-6

    def model_at(r: float) -> Model:
        alpha = r * (1 - math.exp(-(MU + gamma))) / (math.exp(-MU) * total)
        values = {"S": total - i0, "I1": 0.0, "I2": 0.0} | {f"I{strain}": i0}
        return _strains(values, alpha1=alpha, alpha2=alpha)

    assert_threshold(model_at, [f"I{strain}"], 200)
