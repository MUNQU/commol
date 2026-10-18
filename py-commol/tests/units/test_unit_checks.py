"""Unit checks of a single equation, and the helpers behind them."""

import pint
import pytest

from commol.context.parameter import Parameter
from commol.utils.equations import (
    check_all_parameters_have_units,
    check_equation_units,
    get_predefined_variable_units,
    ureg,
)

ITEM_RATE = "item/day"


@pytest.fixture(scope="module", autouse=True)
def common_units() -> None:
    """
    Define ``item`` and ``time``, which a model registers for itself.

    These tests call ``check_equation_units`` without a model, so nothing
    registers the units they use.
    """
    for name, definition in (("item", "[item]"), ("time", "[time_abstract]")):
        try:
            ureg(name)
        except pint.UndefinedUnitError:
            ureg.define(f"{name} = {definition}")


CONSISTENT = [
    pytest.param("A + -5", {"A": "item"}, "item", id="negative-number-in-a-sum"),
    pytest.param("max(1, 2) * k", {"k": "1/day"}, "1/day", id="numbers-alone-in-a-max"),
    pytest.param("k * A", {"k": "1", "A": "item"}, "item", id="unit-written-as-1"),
    pytest.param("2", {}, "dimensionless", id="number-alone"),
    pytest.param(
        "k1 * A * B / total",
        {"k1": "1/day", "A": "item", "B": "item", "total": "item"},
        ITEM_RATE,
        id="product-of-amounts",
    ),
    pytest.param(
        "k1 * exp(-decay * step) * A * B / total",
        {
            "k1": "1/day",
            "decay": "dimensionless",
            "step": "dimensionless",
            "A": "item",
            "B": "item",
            "total": "item",
        },
        ITEM_RATE,
        id="exponential-decay",
    ),
    pytest.param(
        "rate * B", {"rate": "1/time", "B": "item"}, "item/time", id="abstract-time"
    ),
    pytest.param(
        "k1 * sin(B / total) * A",
        {"k1": "1/day", "B": "item", "total": "item", "A": "item"},
        ITEM_RATE,
        id="division-inside-a-function",
    ),
    pytest.param(
        "k1 * (1 + amplitude * sin(2 * pi * step / 365)) * A * B / total",
        {
            "k1": "1/day",
            "amplitude": "dimensionless",
            "pi": "dimensionless",
            "step": "dimensionless",
            "A": "item",
            "B": "item",
            "total": "item",
        },
        ITEM_RATE,
        id="sin-of-step",
    ),
    pytest.param(
        "k1 * (1 + amplitude * cos(2 * pi * step / 365)) * A * B / total",
        {
            "k1": "1/day",
            "amplitude": "dimensionless",
            "pi": "dimensionless",
            "step": "dimensionless",
            "A": "item",
            "B": "item",
            "total": "item",
        },
        ITEM_RATE,
        id="cos-of-step",
    ),
    pytest.param(
        "base_rate * log(1 + B / scale) * A * B / total",
        {
            "base_rate": "1/day",
            "B": "item",
            "scale": "item",
            "A": "item",
            "total": "item",
        },
        ITEM_RATE,
        id="log",
    ),
    pytest.param(
        "base_rate * sqrt(B / threshold) * A",
        {"base_rate": "1/day", "B": "item", "threshold": "item", "A": "item"},
        ITEM_RATE,
        id="sqrt",
    ),
    pytest.param(
        "base_rate * pow(B / threshold, exponent) * A",
        {
            "base_rate": "1/day",
            "B": "item",
            "threshold": "item",
            "exponent": "dimensionless",
            "A": "item",
        },
        ITEM_RATE,
        id="pow",
    ),
    pytest.param(
        "max(k1_min, min(k1_max, k1_max * (1 - B / threshold))) * A",
        {
            "k1_min": "1/day",
            "k1_max": "1/day",
            "B": "item",
            "threshold": "item",
            "A": "item",
        },
        ITEM_RATE,
        id="min-max",
    ),
    pytest.param("abs(rate) * B", {"rate": "1/day", "B": "item"}, ITEM_RATE, id="abs"),
    pytest.param(
        "base * exp(-decay * step) * (1 + amp1 * sin(2 * pi * step / 365) "
        "+ amp2 * cos(4 * pi * step / 365)) * A * B / total",
        {
            "base": "1/day",
            "amp1": "dimensionless",
            "amp2": "dimensionless",
            "decay": "dimensionless",
            "step": "dimensionless",
            "pi": "dimensionless",
            "A": "item",
            "B": "item",
            "total": "item",
        },
        ITEM_RATE,
        id="combined-functions",
    ),
]

INCONSISTENT = [
    pytest.param(
        "k * True", {"k": "1/day"}, "1/day", "unsupported operand", id="boolean"
    ),
    pytest.param(
        "k, A",
        {"k": "1/day", "A": "item"},
        "item",
        "Unexpected result type <class 'tuple'>",
        id="tuple",
    ),
    pytest.param(
        "k",
        {"k": "1/day"},
        "blorp",
        "Unit error in equation 'k': 'blorp' is not defined",
        id="unknown-expected-unit",
    ),
    pytest.param(
        "k1 * A",
        {"k1": "1/day", "A": "item"},
        "item",
        "Unit mismatch",
        id="mismatch",
    ),
    pytest.param(
        "rate * B",
        {"rate": "1/week", "B": "item"},
        ITEM_RATE,
        "Unit mismatch",
        id="convertible-time-units-are-not-equal",
    ),
    pytest.param(
        "k1 * (1 + amp * sin(2 * pi * step / 365)) * A * B / total",
        {
            "k1": "1/day",
            "amp": "dimensionless",
            "step": "dimensionless",
            "pi": "dimensionless",
            "A": "item",
            "B": "item",
            "total": "item",
        },
        "item",
        "Unit mismatch",
        id="result-of-a-function-has-the-wrong-unit",
    ),
    pytest.param(
        "k1 * sin(B) * A",
        {"k1": "1/day", "B": "item", "A": "item"},
        ITEM_RATE,
        "Cannot convert|Error",
        id="dimensional-sin",
    ),
    pytest.param(
        "base_rate * log(B) * A",
        {"base_rate": "1/day", "B": "item", "A": "item"},
        ITEM_RATE,
        None,
        id="dimensional-log",
    ),
    pytest.param(
        "k1 * exp(time) * A * B / total",
        {
            "k1": "1/day",
            "time": "day",
            "A": "item",
            "B": "item",
            "total": "item",
        },
        ITEM_RATE,
        "Cannot convert|Error",
        id="dimensional-exp",
    ),
    pytest.param(
        "rate * sqrt(area)",
        {"rate": "1/day", "area": "item**2"},
        ITEM_RATE,
        "Cannot convert|Error",
        id="dimensional-sqrt",
    ),
    pytest.param(
        "rate * atan2(y, x) * A",
        {"rate": "1/day", "y": "item", "x": "day", "A": "item"},
        ITEM_RATE,
        None,
        id="atan2-of-different-units",
    ),
    pytest.param(
        "rate * pow(base, exponent) * A",
        {
            "rate": "1/day",
            "base": "dimensionless",
            "exponent": "item",
            "A": "item",
        },
        ITEM_RATE,
        None,
        id="dimensional-exponent",
    ),
    pytest.param(
        "rate * exp(sin(x)) * A",
        {"rate": "1/day", "x": "item", "A": "item"},
        ITEM_RATE,
        None,
        id="nested-functions",
    ),
    pytest.param(
        "max(rate1, rate2) * A",
        {"rate1": "1/day", "rate2": "item", "A": "item"},
        ITEM_RATE,
        None,
        id="max-of-different-units",
    ),
    pytest.param(
        "rate * floor(A)",
        {"rate": "1/day", "A": "item"},
        ITEM_RATE,
        "Cannot convert|Error",
        id="dimensional-floor",
    ),
]


class TestCheckEquationUnits:
    @pytest.mark.parametrize(("equation", "units", "expected"), CONSISTENT)
    def test_a_consistent_equation_passes(
        self, equation: str, units: dict[str, str], expected: str
    ) -> None:
        assert check_equation_units(equation, units, expected) == (True, None)

    @pytest.mark.parametrize(
        "time_unit",
        ["second", "minute", "hour", "day", "week", "month", "fortnight", "year"],
    )
    def test_every_unit_of_time_is_known(self, time_unit: str) -> None:
        units = {"rate": f"1/{time_unit}", "B": "item"}

        assert check_equation_units("rate * B", units, f"item/{time_unit}") == (
            True,
            None,
        )

    @pytest.mark.parametrize(("equation", "units", "expected", "message"), INCONSISTENT)
    def test_an_inconsistent_equation_is_reported(
        self,
        equation: str,
        units: dict[str, str],
        expected: str,
        message: str | None,
    ) -> None:
        is_consistent, error = check_equation_units(equation, units, expected)

        assert not is_consistent
        assert error is not None
        if message is not None:
            assert any(part in error for part in message.split("|")), error


class TestHelpers:
    def test_check_all_parameters_have_units(self) -> None:
        assert check_all_parameters_have_units(
            [
                Parameter(id="k1", value=0.5, unit="1/day"),
                Parameter(id="k2", value=0.1, unit="1/day"),
            ]
        )
        assert not check_all_parameters_have_units(
            [
                Parameter(id="k1", value=0.5, unit="1/day"),
                Parameter(id="k2", value=0.1, unit=None),
            ]
        )

    def test_the_total_and_the_subgroup_totals_have_the_bin_unit(self) -> None:
        units = get_predefined_variable_units(
            ["A", "A_g1", "total_g2", "total_g1", "total_g1_t1"],
            bin_unit="item",
        )

        assert units == {
            "total": "item",
            "total_g2": "item",
            "total_g1": "item",
            "total_g1_t1": "item",
        }

    def test_without_a_bin_unit_there_are_no_units(self) -> None:
        assert get_predefined_variable_units(["total_g1"], bin_unit=None) == {}
