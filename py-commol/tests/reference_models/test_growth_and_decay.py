"""
Exponential growth, radioactive decay and a compartment with constant input.

``y' = a y`` grows or decays at a constant rate per member, and
``y' = b - a y`` adds a constant input. Every check is the closed form
``y(t) = y0 e^{a t}`` or ``b / a + (y0 - b / a) e^{-a t}``.

References
----------
.. [1] Malthus, T. R. (1798). *An Essay on the Principle of Population*.
   London: J. Johnson.
.. [2] Rutherford, E., & Soddy, F. (1902). The cause and nature of
   radioactivity, Part II. *The London, Edinburgh, and Dublin Philosophical
   Magazine and Journal of Science*, 4, 569-585.
   https://doi.org/10.1080/14786440209462881
.. [3] Libby, W. F., Anderson, E. C., & Arnold, J. R. (1949). Age determination
   by radiocarbon content: world-wide assay of natural radiocarbon. *Science*,
   109, 227-228. https://doi.org/10.1126/science.109.2827.227
.. [4] Teorell, T. (1937). Kinetics of distribution of substances administered
   to the body. I. The extravascular modes of administration. *Archives
   Internationales de Pharmacodynamie et de Therapie*, 57, 205-225.
"""

import math

import pytest

from commol import Simulation
from tests.support import ODE_REL, Flow, assert_close, flow_model


def _growth(y0: float, rate: float) -> Simulation:
    """``y' = rate * y`` from ``y0``: growth [1]_ or, if negative, decay [2]_."""
    flows = [Flow("growth", None, "Y", "a * Y")]
    return Simulation(flow_model({"Y": y0}, flows, {"a": rate}))


def test_exponential_growth_from_two_members() -> None:
    # 0.6 per member per day from two members: y(10) = 2 e^6.
    results = _growth(2, 0.6).run_at([10])

    assert_close(results["Y"][0], 2 * math.exp(6), rel=ODE_REL)


def test_doubling_in_ten_days_quadruples_in_twenty() -> None:
    # 50 members double in 10 days, so the rate is log(2) / 10.
    results = _growth(50, math.log(2) / 10).run_at([10, 20])

    assert_close(results["Y"], [100, 200], rel=ODE_REL)


def test_strontium_halves_after_its_half_life() -> None:
    # Decay constant 2.48e-2 per year; the half-life log(2) / k is 27.9 years.
    k = 2.48e-2
    half_life = math.log(2) / k

    results = _growth(1000, -k).run_at([half_life])

    assert round(half_life, 1) == 27.9
    assert_close(results["Y"][0], 500, rel=ODE_REL)


def test_radium_keeps_three_quarters_and_one_quarter() -> None:
    # Half-life 1620 years: 3/4 is left after 1620 log2(4/3) = 672 years, and
    # 1/4 after two half-lives.
    half_life = 1620
    k = math.log(2) / half_life
    three_quarters = half_life * math.log2(4 / 3)

    results = _growth(1000, -k).run_at([three_quarters, 2 * half_life])

    assert round(three_quarters) == 672
    assert_close(results["Y"], [750, 250], rel=ODE_REL)


def test_radiocarbon_dates_a_sample() -> None:
    # Radiocarbon dating [3]: 6e10 atoms of carbon-14 per gram decay at
    # 1.244e-4 per year, so a sample with 3.33e10 left is
    # log(6 / 3.33) / k = 4733 years old.
    k = 1.244e-4
    age = math.log(6 / 3.33) / k

    results = _growth(6e10, -k).run_at([age])

    assert round(age) == 4733
    assert_close(results["Y"][0], 3.33e10, rel=ODE_REL)


@pytest.mark.parametrize("a", [0.5, -0.5])
@pytest.mark.parametrize("y0", [1, 10])
def test_constant_input_tends_to_its_equilibrium_or_diverges(
    a: float, y0: float
) -> None:
    # y' = b - a y [4]: every solution tends to b / a if a > 0 and moves away
    # from it if a < 0.
    b = 3.0
    flows = [Flow("input", None, "Y", "b"), Flow("output", "Y", None, "a * Y")]
    times = [2.0, 8.0, 40.0]

    results = Simulation(flow_model({"Y": y0}, flows, {"a": a, "b": b})).run_at(times)

    expected = [b / a + (y0 - b / a) * math.exp(-a * t) for t in times]
    assert_close(results["Y"], expected, rel=ODE_REL)
    if a > 0:
        assert_close(results["Y"][-1], b / a, rel=ODE_REL)
