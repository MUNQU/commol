"""
Rate formulas: every operator and function gives the flow it computes.

Each case runs the model of ``abc_builder`` and compares the whole trajectory
with the same formula evaluated by hand in Python (``tests.support.reference_run``).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from commol import Simulation
from tests.support import (
    ENGINE_REL,
    ABC_START,
    assert_close,
    assert_total_conserved,
    build,
    reference_run,
    abc_builder,
    abc_flows,
    total,
)
from tests.support.reference import FlowRate, State


@dataclass(frozen=True)
class Point:
    """The values a rate reads at one step: ``a``, ``b``, ``c``, ``n``, ``t``."""

    a: float
    b: float
    c: float
    n: float
    t: int


ByHand = Callable[[Point], float]


def _flow(by_hand: ByHand) -> FlowRate:
    def rate(state: State, step: int) -> float:
        return by_hand(Point(state["A"], state["B"], state["C"], total(state), step))

    return rate


def _matches_by_hand(
    parameters: dict[str, float],
    ab: str | float,
    bc: str | float | None,
    by_hand_ab: ByHand,
    by_hand_bc: ByHand | None,
    steps: int,
) -> None:
    model = build(abc_builder(parameters=parameters, ab=ab, bc=bc))

    result = Simulation(model).run(steps)

    expected = reference_run(
        ABC_START,
        abc_flows(
            _flow(by_hand_ab),
            _flow(by_hand_bc) if by_hand_bc else None,
        ),
        steps,
    )
    assert_close(result, expected, rel=ENGINE_REL)
    assert_total_conserved(result, 1000.0)


RATE_FAMILIES = [
    pytest.param(
        {"k1": 0.3, "k2": 0.1},
        0.3,
        0.1,
        lambda x: 0.3 * x.a,
        lambda x: 0.1 * x.b,
        10,
        id="constant-numbers-are-proportional",
    ),
    pytest.param(
        {"rate_ab": 0.25, "rate_bc": 0.05},
        "rate_ab",
        "rate_bc",
        lambda x: 0.25 * x.a,
        lambda x: 0.05 * x.b,
        10,
        id="a-parameter-alone-is-proportional",
    ),
    pytest.param(
        {"k1": 0.0003, "k2": 0.05},
        "k1 * A * B",
        "k2 * B",
        lambda x: 0.0003 * x.a * x.b,
        lambda x: 0.05 * x.b,
        10,
        id="product-of-amounts",
    ),
    pytest.param(
        {"k1": 0.3, "k2": 0.05},
        "k1 * A * B / total",
        "k2 * B",
        lambda x: 0.3 * x.a * x.b / x.n,
        lambda x: 0.05 * x.b,
        10,
        id="frequency-dependent",
    ),
    pytest.param(
        {"k1_avg": 0.25, "amplitude": 0.2, "k2": 0.05},
        "(k1_avg * (1 + amplitude * sin(2 * pi * step / 365))) * A * B / total",
        "k2 * B",
        lambda x: (
            0.25 * (1 + 0.2 * math.sin(2 * math.pi * x.t / 365)) * x.a * x.b / x.n
        ),
        lambda x: 0.05 * x.b,
        60,
        id="sin-of-step",
    ),
    pytest.param(
        {"k1_0": 0.4, "decay_rate": 0.01, "k2": 0.05},
        "k1_0 * exp(-decay_rate * step) * A * B / total",
        "k2 * B",
        lambda x: 0.4 * math.exp(-0.01 * x.t) * x.a * x.b / x.n,
        lambda x: 0.05 * x.b,
        30,
        id="exponential-decay",
    ),
    pytest.param(
        {
            "k1_pre": 0.3,
            "k1_post": 0.1,
            "t_switch": 5,
            "k2": 0.05,
        },
        "(k1_pre + (k1_post - k1_pre) * "
        "(1 / (1 + exp(-10 * (t - t_switch))))) * A * B / total",
        "k2 * B",
        lambda x: (
            (0.3 + (0.1 - 0.3) / (1 + math.exp(-10 * (x.t - 5)))) * x.a * x.b / x.n
        ),
        lambda x: 0.05 * x.b,
        10,
        id="sigmoid-step-of-t",
    ),
    pytest.param(
        {"max_k1": 0.4, "steepness": 0.1, "midpoint": 30, "k2": 0.05},
        "max_k1 * (1 / (1 + exp(-steepness * (step - midpoint)))) * A * B / total",
        "k2 * B",
        lambda x: 0.4 / (1 + math.exp(-0.1 * (x.t - 30))) * x.a * x.b / x.n,
        lambda x: 0.05 * x.b,
        60,
        id="logistic",
    ),
    pytest.param(
        {"k1": 0.25, "k2_0": 0.05, "power": 0.5},
        "k1 * A * B / total",
        "k2_0 * pow(B, power)",
        lambda x: 0.25 * x.a * x.b / x.n,
        lambda x: 0.05 * x.b**0.5,
        30,
        id="power-law-bc",
    ),
    pytest.param(
        {"k1_base": 0.2, "amplitude_1": 0.1, "amplitude_2": 0.05, "k2": 0.05},
        "(k1_base + amplitude_1 * sin(2 * pi * step / 365) + "
        "amplitude_2 * cos(4 * pi * step / 365)) * A * B / total",
        "k2 * B",
        lambda x: (
            (
                0.2
                + 0.1 * math.sin(2 * math.pi * x.t / 365)
                + 0.05 * math.cos(4 * math.pi * x.t / 365)
            )
            * x.a
            * x.b
            / x.n
        ),
        lambda x: 0.05 * x.b,
        60,
        id="sin-and-cos",
    ),
    pytest.param(
        {"tiny_k1": 1e-6, "tiny_k2": 1e-7},
        "tiny_k1 * A * B / total",
        "tiny_k2 * B",
        lambda x: 1e-6 * x.a * x.b / x.n,
        lambda x: 1e-7 * x.b,
        1000,
        id="very-small-rates",
    ),
    pytest.param(
        {"k1_max": 0.4, "k1_min": 0.1, "threshold": 100, "k2": 0.05},
        "max(k1_min, min(k1_max, k1_max * (1 - B / threshold))) * A * B / total",
        "k2 * B",
        lambda x: max(0.1, min(0.4, 0.4 * (1 - x.b / 100))) * x.a * x.b / x.n,
        lambda x: 0.05 * x.b,
        100,
        id="min-and-max",
    ),
    pytest.param(
        {"k1": 0.25, "k2_base": 0.05, "step_scale": 10},
        "k1 * A * B / total",
        "k2_base * (1 + 0.1 * floor(step / step_scale)) * B",
        lambda x: 0.25 * x.a * x.b / x.n,
        lambda x: 0.05 * (1 + 0.1 * math.floor(x.t / 10)) * x.b,
        50,
        id="floor-of-step",
    ),
    pytest.param(
        {"k1": 0.1, "k2": 0.05},
        "k1 * A * B / max(1, A + B + C)",
        "k2 * B",
        lambda x: 0.1 * x.a * x.b / max(1, x.a + x.b + x.c),
        lambda x: 0.05 * x.b,
        10,
        id="guarded-division",
    ),
]


@pytest.mark.parametrize(
    (
        "parameters",
        "ab",
        "bc",
        "by_hand_ab",
        "by_hand_bc",
        "steps",
    ),
    RATE_FAMILIES,
)
def test_a_rate_formula_gives_the_flow_it_computes(
    parameters: dict[str, float],
    ab: str | float,
    bc: str | float,
    by_hand_ab: ByHand,
    by_hand_bc: ByHand,
    steps: int,
) -> None:
    _matches_by_hand(parameters, ab, bc, by_hand_ab, by_hand_bc, steps)


def test_without_bc_nothing_reaches_c() -> None:
    _matches_by_hand(
        {"k1": 0.2},
        "k1 * A * B / total",
        None,
        lambda x: 0.2 * x.a * x.b / x.n,
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
        "exp(-B / scale) * A * B / total",
        lambda x: math.exp(-x.b / SCALE) * x.a * x.b / x.n,
    ),
    (
        "log",
        "log(1 + B / offset) * A * B / total",
        lambda x: math.log(1 + x.b / OFFSET) * x.a * x.b / x.n,
    ),
    (
        "log10",
        "log10(1 + B / offset) * A * B / total",
        lambda x: math.log10(1 + x.b / OFFSET) * x.a * x.b / x.n,
    ),
    (
        "log2",
        "log2(1 + B / offset) * A * B / total",
        lambda x: math.log2(1 + x.b / OFFSET) * x.a * x.b / x.n,
    ),
    (
        "ln",
        "ln(1 + B / offset) * A * B / total",
        lambda x: math.log(1 + x.b / OFFSET) * x.a * x.b / x.n,
    ),
    (
        "sin",
        "(1 + 0.1 * sin(B / scale)) * A * B / total",
        lambda x: (1 + 0.1 * math.sin(x.b / SCALE)) * x.a * x.b / x.n,
    ),
    (
        "cos",
        "(1 + 0.1 * cos(B / scale)) * A * B / total",
        lambda x: (1 + 0.1 * math.cos(x.b / SCALE)) * x.a * x.b / x.n,
    ),
    (
        "tan",
        "(1 + 0.01 * tan(B / (10 * scale))) * A * B / total",
        lambda x: (1 + 0.01 * math.tan(x.b / (10 * SCALE))) * x.a * x.b / x.n,
    ),
    (
        "sinh",
        "(1 + 0.01 * sinh(B / scale)) * A * B / total",
        lambda x: (1 + 0.01 * math.sinh(x.b / SCALE)) * x.a * x.b / x.n,
    ),
    (
        "cosh",
        "cosh(B / scale) * A * B / total",
        lambda x: math.cosh(x.b / SCALE) * x.a * x.b / x.n,
    ),
    (
        "tanh",
        "tanh(B / scale) * A * B / total",
        lambda x: math.tanh(x.b / SCALE) * x.a * x.b / x.n,
    ),
    ("abs", "abs(B - offset) * A / total", lambda x: abs(x.b - OFFSET) * x.a / x.n),
    ("sqrt", "sqrt(B + 1) * A / total", lambda x: math.sqrt(x.b + 1) * x.a / x.n),
    ("pow", "pow(B + 1, 0.5) * A / total", lambda x: (x.b + 1) ** 0.5 * x.a / x.n),
    (
        "floor",
        "(1 + floor(B / offset) * 0.01) * A * B / total",
        lambda x: (1 + math.floor(x.b / OFFSET) * 0.01) * x.a * x.b / x.n,
    ),
    (
        "ceil",
        "(1 + ceil(B / offset) * 0.01) * A * B / total",
        lambda x: (1 + math.ceil(x.b / OFFSET) * 0.01) * x.a * x.b / x.n,
    ),
    (
        "round",
        "(1 + round(B / offset) * 0.01) * A * B / total",
        lambda x: (1 + _round_half_up(x.b / OFFSET) * 0.01) * x.a * x.b / x.n,
    ),
    (
        "asin",
        "asin(B / (10 * scale)) * A / total",
        lambda x: math.asin(x.b / (10 * SCALE)) * x.a / x.n,
    ),
    (
        "acos",
        "acos(B / (10 * scale)) * A / total",
        lambda x: math.acos(x.b / (10 * SCALE)) * x.a / x.n,
    ),
    (
        "atan",
        "atan(B / scale) * A / total",
        lambda x: math.atan(x.b / SCALE) * x.a / x.n,
    ),
    ("atan2", "atan2(B, A) * B / total", lambda x: math.atan2(x.b, x.a) * x.b / x.n),
]


@pytest.mark.parametrize(
    ("formula", "by_hand"),
    [pytest.param(formula, by_hand, id=name) for name, formula, by_hand in FUNCTIONS],
)
def test_a_function_gives_the_value_it_computes(formula: str, by_hand: ByHand) -> None:
    _matches_by_hand(
        {"base_rate": BASE_RATE, "scale": SCALE, "offset": OFFSET},
        f"base_rate * {formula}",
        "base_rate * B",
        lambda x: BASE_RATE * by_hand(x),
        lambda x: BASE_RATE * x.b,
        30,
    )


class TestInvalidFormulas:
    @pytest.mark.parametrize(
        "formula", ["A * B /", "k1 * * k2", "sin()", "A +", "/ k1"]
    )
    def test_an_incomplete_expression_is_rejected(self, formula: str) -> None:
        with pytest.raises((ValueError, SyntaxError)):
            build(
                abc_builder(
                    parameters={"k1": 0.1, "k2": 0.05},
                    ab=formula,
                    bc=None,
                )
            )

    def test_a_rate_that_does_not_parse_names_itself(self) -> None:
        with pytest.raises(ValueError, match=r"Validation failed for rate 'k1 \+\* 2'"):
            abc_builder(parameters={"k1": 0.1, "k2": 0.05}, ab="k1 +* 2")

    def test_an_undefined_parameter_is_rejected_at_build(self) -> None:
        builder = abc_builder(
            parameters={"k1": 0.1},
            ab="k1 * A * B",
            bc="undefined_k2 * B",
        )

        with pytest.raises(ValueError, match="Undefined variables in transition"):
            build(builder)
