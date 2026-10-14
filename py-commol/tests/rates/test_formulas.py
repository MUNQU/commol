"""
Rate formulas: every operator and function gives the flow it computes.

Each case runs an SIR model and compares the whole trajectory with the same
formula evaluated by hand in Python (``tests.support.reference_run``).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from commol import Simulation
from tests.support import (
    ENGINE_REL,
    SIR_START,
    assert_close,
    assert_total_conserved,
    build,
    reference_run,
    sir_builder,
    sir_flows,
    total,
)
from tests.support.reference import FlowRate, State


@dataclass(frozen=True)
class Point:
    """The values a rate reads at one step: ``s``, ``i``, ``r``, ``n``, ``t``."""

    s: float
    i: float
    r: float
    n: float
    t: int


ByHand = Callable[[Point], float]


def _flow(by_hand: ByHand) -> FlowRate:
    def rate(state: State, step: int) -> float:
        return by_hand(Point(state["S"], state["I"], state["R"], total(state), step))

    return rate


def _matches_by_hand(
    parameters: dict[str, float],
    infection: str | float,
    recovery: str | float | None,
    by_hand_infection: ByHand,
    by_hand_recovery: ByHand | None,
    steps: int,
) -> None:
    model = build(
        sir_builder(parameters=parameters, infection=infection, recovery=recovery)
    )

    result = Simulation(model).run(steps)

    expected = reference_run(
        SIR_START,
        sir_flows(
            _flow(by_hand_infection),
            _flow(by_hand_recovery) if by_hand_recovery else None,
        ),
        steps,
    )
    assert_close(result, expected, rel=ENGINE_REL)
    assert_total_conserved(result, 1000.0)


RATE_FAMILIES = [
    pytest.param(
        {"beta": 0.3, "gamma": 0.1},
        0.3,
        0.1,
        lambda x: 0.3 * x.s,
        lambda x: 0.1 * x.i,
        10,
        id="constant-numbers-are-per-capita",
    ),
    pytest.param(
        {"transmission_rate": 0.25, "recovery_rate": 0.05},
        "transmission_rate",
        "recovery_rate",
        lambda x: 0.25 * x.s,
        lambda x: 0.05 * x.i,
        10,
        id="a-parameter-alone-is-per-capita",
    ),
    pytest.param(
        {"beta": 0.0003, "gamma": 0.05},
        "beta * S * I",
        "gamma * I",
        lambda x: 0.0003 * x.s * x.i,
        lambda x: 0.05 * x.i,
        10,
        id="mass-action",
    ),
    pytest.param(
        {"beta": 0.3, "gamma": 0.05},
        "beta * S * I / total",
        "gamma * I",
        lambda x: 0.3 * x.s * x.i / x.n,
        lambda x: 0.05 * x.i,
        10,
        id="frequency-dependent",
    ),
    pytest.param(
        {"beta_avg": 0.25, "seasonal_amp": 0.2, "gamma": 0.05},
        "(beta_avg * (1 + seasonal_amp * sin(2 * pi * step / 365))) * S * I / total",
        "gamma * I",
        lambda x: (
            0.25 * (1 + 0.2 * math.sin(2 * math.pi * x.t / 365)) * x.s * x.i / x.n
        ),
        lambda x: 0.05 * x.i,
        60,
        id="seasonal-sin-of-step",
    ),
    pytest.param(
        {"beta_0": 0.4, "decay_rate": 0.01, "gamma": 0.05},
        "beta_0 * exp(-decay_rate * step) * S * I / total",
        "gamma * I",
        lambda x: 0.4 * math.exp(-0.01 * x.t) * x.s * x.i / x.n,
        lambda x: 0.05 * x.i,
        30,
        id="exponential-decay",
    ),
    pytest.param(
        {
            "beta_pre": 0.3,
            "beta_post": 0.1,
            "intervention_time": 5,
            "gamma": 0.05,
        },
        "(beta_pre + (beta_post - beta_pre) * "
        "(1 / (1 + exp(-10 * (t - intervention_time))))) * S * I / total",
        "gamma * I",
        lambda x: (
            (0.3 + (0.1 - 0.3) / (1 + math.exp(-10 * (x.t - 5)))) * x.s * x.i / x.n
        ),
        lambda x: 0.05 * x.i,
        10,
        id="sigmoid-step-of-t",
    ),
    pytest.param(
        {"max_beta": 0.4, "steepness": 0.1, "midpoint": 30, "gamma": 0.05},
        "max_beta * (1 / (1 + exp(-steepness * (step - midpoint)))) * S * I / total",
        "gamma * I",
        lambda x: 0.4 / (1 + math.exp(-0.1 * (x.t - 30))) * x.s * x.i / x.n,
        lambda x: 0.05 * x.i,
        60,
        id="logistic",
    ),
    pytest.param(
        {"beta": 0.25, "gamma_0": 0.05, "power": 0.5},
        "beta * S * I / total",
        "gamma_0 * pow(I, power)",
        lambda x: 0.25 * x.s * x.i / x.n,
        lambda x: 0.05 * x.i**0.5,
        30,
        id="power-law-recovery",
    ),
    pytest.param(
        {"beta_base": 0.2, "amplitude_1": 0.1, "amplitude_2": 0.05, "gamma": 0.05},
        "(beta_base + amplitude_1 * sin(2 * pi * step / 365) + "
        "amplitude_2 * cos(4 * pi * step / 365)) * S * I / total",
        "gamma * I",
        lambda x: (
            (
                0.2
                + 0.1 * math.sin(2 * math.pi * x.t / 365)
                + 0.05 * math.cos(4 * math.pi * x.t / 365)
            )
            * x.s
            * x.i
            / x.n
        ),
        lambda x: 0.05 * x.i,
        60,
        id="sin-and-cos",
    ),
    pytest.param(
        {"tiny_beta": 1e-6, "tiny_gamma": 1e-7},
        "tiny_beta * S * I / total",
        "tiny_gamma * I",
        lambda x: 1e-6 * x.s * x.i / x.n,
        lambda x: 1e-7 * x.i,
        1000,
        id="very-small-rates",
    ),
    pytest.param(
        {"beta_max": 0.4, "beta_min": 0.1, "threshold": 100, "gamma": 0.05},
        "max(beta_min, min(beta_max, beta_max * (1 - I / threshold))) * S * I / total",
        "gamma * I",
        lambda x: max(0.1, min(0.4, 0.4 * (1 - x.i / 100))) * x.s * x.i / x.n,
        lambda x: 0.05 * x.i,
        100,
        id="min-and-max",
    ),
    pytest.param(
        {"beta": 0.25, "gamma_base": 0.05, "step_scale": 10},
        "beta * S * I / total",
        "gamma_base * (1 + 0.1 * floor(step / step_scale)) * I",
        lambda x: 0.25 * x.s * x.i / x.n,
        lambda x: 0.05 * (1 + 0.1 * math.floor(x.t / 10)) * x.i,
        50,
        id="floor-of-step",
    ),
    pytest.param(
        {"beta": 0.1, "gamma": 0.05},
        "beta * S * I / max(1, S + I + R)",
        "gamma * I",
        lambda x: 0.1 * x.s * x.i / max(1, x.s + x.i + x.r),
        lambda x: 0.05 * x.i,
        10,
        id="guarded-division",
    ),
]


@pytest.mark.parametrize(
    (
        "parameters",
        "infection",
        "recovery",
        "by_hand_infection",
        "by_hand_recovery",
        "steps",
    ),
    RATE_FAMILIES,
)
def test_a_rate_formula_gives_the_flow_it_computes(
    parameters: dict[str, float],
    infection: str | float,
    recovery: str | float,
    by_hand_infection: ByHand,
    by_hand_recovery: ByHand,
    steps: int,
) -> None:
    _matches_by_hand(
        parameters, infection, recovery, by_hand_infection, by_hand_recovery, steps
    )


def test_without_a_recovery_transition_nobody_recovers() -> None:
    _matches_by_hand(
        {"beta": 0.2},
        "beta * S * I / total",
        None,
        lambda x: 0.2 * x.s * x.i / x.n,
        None,
        50,
    )


BASE_RATE, SCALE, OFFSET = 0.1, 100, 10


def _round_half_up(value: float) -> float:
    """The engine's ``round``: halves round up."""
    return math.floor(value + 0.5)


FUNCTIONS = [
    (
        "exp",
        "exp(-I / scale) * S * I / total",
        lambda x: math.exp(-x.i / SCALE) * x.s * x.i / x.n,
    ),
    (
        "log",
        "log(1 + I / offset) * S * I / total",
        lambda x: math.log(1 + x.i / OFFSET) * x.s * x.i / x.n,
    ),
    (
        "log10",
        "log10(1 + I / offset) * S * I / total",
        lambda x: math.log10(1 + x.i / OFFSET) * x.s * x.i / x.n,
    ),
    (
        "log2",
        "log2(1 + I / offset) * S * I / total",
        lambda x: math.log2(1 + x.i / OFFSET) * x.s * x.i / x.n,
    ),
    (
        "ln",
        "ln(1 + I / offset) * S * I / total",
        lambda x: math.log(1 + x.i / OFFSET) * x.s * x.i / x.n,
    ),
    (
        "sin",
        "(1 + 0.1 * sin(I / scale)) * S * I / total",
        lambda x: (1 + 0.1 * math.sin(x.i / SCALE)) * x.s * x.i / x.n,
    ),
    (
        "cos",
        "(1 + 0.1 * cos(I / scale)) * S * I / total",
        lambda x: (1 + 0.1 * math.cos(x.i / SCALE)) * x.s * x.i / x.n,
    ),
    (
        "tan",
        "(1 + 0.01 * tan(I / (10 * scale))) * S * I / total",
        lambda x: (1 + 0.01 * math.tan(x.i / (10 * SCALE))) * x.s * x.i / x.n,
    ),
    (
        "sinh",
        "(1 + 0.01 * sinh(I / scale)) * S * I / total",
        lambda x: (1 + 0.01 * math.sinh(x.i / SCALE)) * x.s * x.i / x.n,
    ),
    (
        "cosh",
        "cosh(I / scale) * S * I / total",
        lambda x: math.cosh(x.i / SCALE) * x.s * x.i / x.n,
    ),
    (
        "tanh",
        "tanh(I / scale) * S * I / total",
        lambda x: math.tanh(x.i / SCALE) * x.s * x.i / x.n,
    ),
    ("abs", "abs(I - offset) * S / total", lambda x: abs(x.i - OFFSET) * x.s / x.n),
    ("sqrt", "sqrt(I + 1) * S / total", lambda x: math.sqrt(x.i + 1) * x.s / x.n),
    ("pow", "pow(I + 1, 0.5) * S / total", lambda x: (x.i + 1) ** 0.5 * x.s / x.n),
    (
        "floor",
        "(1 + floor(I / offset) * 0.01) * S * I / total",
        lambda x: (1 + math.floor(x.i / OFFSET) * 0.01) * x.s * x.i / x.n,
    ),
    (
        "ceil",
        "(1 + ceil(I / offset) * 0.01) * S * I / total",
        lambda x: (1 + math.ceil(x.i / OFFSET) * 0.01) * x.s * x.i / x.n,
    ),
    (
        "round",
        "(1 + round(I / offset) * 0.01) * S * I / total",
        lambda x: (1 + _round_half_up(x.i / OFFSET) * 0.01) * x.s * x.i / x.n,
    ),
    (
        "asin",
        "asin(I / (10 * scale)) * S / total",
        lambda x: math.asin(x.i / (10 * SCALE)) * x.s / x.n,
    ),
    (
        "acos",
        "acos(I / (10 * scale)) * S / total",
        lambda x: math.acos(x.i / (10 * SCALE)) * x.s / x.n,
    ),
    (
        "atan",
        "atan(I / scale) * S / total",
        lambda x: math.atan(x.i / SCALE) * x.s / x.n,
    ),
    ("atan2", "atan2(I, S) * I / total", lambda x: math.atan2(x.i, x.s) * x.i / x.n),
]


@pytest.mark.parametrize(
    ("formula", "by_hand"),
    [pytest.param(formula, by_hand, id=name) for name, formula, by_hand in FUNCTIONS],
)
def test_a_function_gives_the_value_it_computes(formula: str, by_hand: ByHand) -> None:
    _matches_by_hand(
        {"base_rate": BASE_RATE, "scale": SCALE, "offset": OFFSET},
        f"base_rate * {formula}",
        "base_rate * I",
        lambda x: BASE_RATE * by_hand(x),
        lambda x: BASE_RATE * x.i,
        30,
    )


class TestInvalidFormulas:
    @pytest.mark.parametrize(
        "formula", ["S * I /", "beta * * gamma", "sin()", "S +", "/ beta"]
    )
    def test_an_incomplete_expression_is_rejected(self, formula: str) -> None:
        with pytest.raises((ValueError, SyntaxError)):
            build(
                sir_builder(
                    parameters={"beta": 0.1, "gamma": 0.05},
                    infection=formula,
                    recovery=None,
                )
            )

    def test_an_undefined_parameter_is_rejected_at_build(self) -> None:
        builder = sir_builder(
            parameters={"beta": 0.1},
            infection="beta * S * I",
            recovery="undefined_gamma * I",
        )

        with pytest.raises(ValueError, match="Undefined variables in transition"):
            build(builder)
