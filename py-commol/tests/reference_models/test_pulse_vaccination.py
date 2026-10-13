"""
Pulse vaccination in an SIR model with births and deaths.

``S' = mu K - beta S I - mu S`` and ``I' = beta S I - (mu + alpha) I``. With
``R0 = beta K / (mu + alpha)``, herd immunity needs a fraction ``1 - 1 / R0`` of
the susceptibles vaccinated. Without infection, ``S' = mu K - mu S``: from
``(1 - p) Gamma``, ``S`` stays below ``Gamma`` until
``T(p) = log(1 + p Gamma / (K - Gamma)) / mu``. Vaccinating a fraction ``p`` of
the susceptibles every ``T`` gives the map
``S_{n+1} = (1 - p) [K (1 - e^{-mu T}) + S_n e^{-mu T}]``, whose fixed point
``S* = K (1 - p e^{mu T} / (e^{mu T} - (1 - p)))`` starts a periodic orbit with
mean ``K ((mu T - p)(e^{mu T} - 1) + p mu T) / (mu T (e^{mu T} - (1 - p)))``.

References
----------
.. [1] Agur, Z., Cojocaru, L., Mazor, G., Anderson, R. M., & Danon, Y. L.
   (1993). Pulse mass measles vaccination across age cohorts. *Proceedings of
   the National Academy of Sciences*, 90, 11698-11702.
   https://doi.org/10.1073/pnas.90.24.11698
.. [2] Shulgin, B., Stone, L., & Agur, Z. (1998). Pulse vaccination strategy in
   the SIR epidemic model. *Bulletin of Mathematical Biology*, 60, 1123-1148.
   https://doi.org/10.1016/S0092-8240(98)90005-2
.. [3] Engbert, R., & Drepper, F. R. (1994). Chance and chaos in population
   biology: models of recurrent epidemics and food chain dynamics. *Chaos,
   Solitons & Fractals*, 4, 1147-1169.
   https://doi.org/10.1016/0960-0779(94)90028-0
"""

import math

import pytest

from commol import Simulation
from commol.context.model import Model
from tests.support import ODE_REL, Flow, assert_close, assert_threshold, flow_model

# Measles, per year, with the carrying capacity normalized to one [3].
MU, BETA, ALPHA, K = 0.02, 1800.0, 100.0, 1.0
R0 = BETA * K / (MU + ALPHA)
GAMMA = (MU + ALPHA) / BETA
"""The susceptible level below which the infection falls."""


def test_measles_needs_about_95_percent_vaccinated() -> None:
    # A fraction p of the susceptibles vaccinated at t = 0 leaves R0 (1 - p):
    # the infection grows just below p = 1 - 1 / R0 and falls just above it.
    def vaccinated(r: float) -> Model:
        s0, i0 = r / R0 * K, 1e-9
        flows = [
            Flow("births", None, "S", "mu * K"),
            Flow("infection", "S", "I", "beta * S * I"),
            Flow("s_deaths", "S", None, "mu * S"),
            Flow("recovery", "I", "R", "alpha * I"),
            Flow("i_deaths", "I", None, "mu * I"),
            Flow("r_deaths", "R", None, "mu * R"),
        ]
        parameters = {"mu": MU, "beta": BETA, "alpha": ALPHA, "K": K}
        return flow_model({"S": s0, "I": i0, "R": K - s0 - i0}, flows, parameters)

    # 0.05 years is five infective periods; births move S by under 2 %.
    assert_threshold(vaccinated, ["I"], 0.05)
    assert round(R0) == 18
    assert abs((1 - 1 / R0) - 0.95) < 0.01


def _no_infection(s0: float, *, p: float = 0.0, period: float | None = None) -> Model:
    """``S' = mu K - mu S``, moving a fraction ``p`` to V every ``period`` [2]_."""
    flows = [
        Flow("births", None, "S", "mu * K"),
        Flow("s_deaths", "S", None, "mu * S", accumulators=("s_deaths",)),
        Flow("v_deaths", "V", None, "mu * V"),
    ]
    if period is not None:
        flows.append(Flow("pulse", "S", "V", "p", impulse_every=(period, period)))
    return flow_model(
        {"S": s0, "V": K - s0},
        flows,
        {"mu": MU, "K": K, "p": p},
        accumulators=["s_deaths"],
    )


@pytest.mark.parametrize("m", range(1, 11))
def test_susceptibles_reach_the_threshold_at_t_of_p(m: int) -> None:
    p = m / 10
    t_of_p = math.log(1 + p * GAMMA / (K - GAMMA)) / MU

    results = Simulation(_no_infection((1 - p) * GAMMA)).run_at([t_of_p])

    assert_close(results["S"][0], GAMMA, rel=ODE_REL)


P, PERIOD = 0.5, 2.0
DECAY = math.exp(-MU * PERIOD)


def test_pulses_follow_the_map_and_the_solution_between_them() -> None:
    s0 = 0.3
    pulses = [n * PERIOD for n in range(1, 6)]
    halfway = [t + PERIOD / 2 for t in pulses]

    results = Simulation(_no_infection(s0, p=P, period=PERIOD)).run_at(
        sorted(pulses + halfway)
    )

    after_pulse, s = [], s0
    for _ in pulses:
        s = (1 - P) * K * (1 - DECAY) + (1 - P) * s * DECAY
        after_pulse.append(s)
    between = [
        K * (1 - math.exp(-MU * PERIOD / 2)) + s_n * math.exp(-MU * PERIOD / 2)
        for s_n in after_pulse
    ]
    assert_close(results["S"][0::2], after_pulse, rel=ODE_REL)
    assert_close(results["S"][1::2], between, rel=ODE_REL)


def test_the_fixed_point_starts_the_periodic_orbit() -> None:
    growth = math.exp(MU * PERIOD)
    c = P * growth / (growth - (1 - P))
    s_star = K * (1 - c)
    offsets = [0.5, 1.5, 2.0]
    times = [n * PERIOD + tau for n in (0, 4) for tau in offsets]

    results = Simulation(_no_infection(s_star, p=P, period=PERIOD)).run_at(times)

    orbit = [K * (1 - c * math.exp(-MU * tau)) for tau in offsets]
    orbit[-1] = s_star
    assert_close(results["S"], orbit + orbit, rel=ODE_REL)


def test_the_periodic_orbit_has_the_stated_mean() -> None:
    # The accumulator counts the deaths mu S, so it integrates S over a period.
    growth = math.exp(MU * PERIOD)
    s_star = K * (1 - P * growth / (growth - (1 - P)))

    results = Simulation(_no_infection(s_star, p=P, period=PERIOD)).run_at([PERIOD])

    mean = results["s_deaths"][0] / (MU * PERIOD)
    mu_t = MU * PERIOD
    expected = K * ((mu_t - P) * (growth - 1) + P * mu_t) / (mu_t * (growth - (1 - P)))
    assert_close(mean, expected, rel=ODE_REL)
