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

PERSON_RATE = "person/day"


@pytest.fixture(scope="module", autouse=True)
def common_units() -> None:
    """
    Define ``person`` and ``time``, which a model registers for itself.

    These tests call ``check_equation_units`` without a model, so nothing
    registers the units they use.
    """
    for name, definition in (("person", "[person]"), ("time", "[time_abstract]")):
        try:
            ureg(name)
        except pint.UndefinedUnitError:
            ureg.define(f"{name} = {definition}")


CONSISTENT = [
    pytest.param(
        "beta * S * I / total",
        {"beta": "1/day", "S": "person", "I": "person", "total": "person"},
        PERSON_RATE,
        id="mass-action",
    ),
    pytest.param(
        "beta * exp(-decay * step) * S * I / total",
        {
            "beta": "1/day",
            "decay": "dimensionless",
            "step": "dimensionless",
            "S": "person",
            "I": "person",
            "total": "person",
        },
        PERSON_RATE,
        id="exponential-decay",
    ),
    pytest.param(
        "rate * I", {"rate": "1/time", "I": "person"}, "person/time", id="abstract-time"
    ),
    pytest.param(
        "beta * sin(I / total) * S",
        {"beta": "1/day", "I": "person", "total": "person", "S": "person"},
        PERSON_RATE,
        id="division-inside-a-function",
    ),
    pytest.param(
        "beta * (1 + amplitude * sin(2 * pi * step / 365)) * S * I / total",
        {
            "beta": "1/day",
            "amplitude": "dimensionless",
            "pi": "dimensionless",
            "step": "dimensionless",
            "S": "person",
            "I": "person",
            "total": "person",
        },
        PERSON_RATE,
        id="seasonal-sin",
    ),
    pytest.param(
        "beta * (1 + amplitude * cos(2 * pi * step / 365)) * S * I / total",
        {
            "beta": "1/day",
            "amplitude": "dimensionless",
            "pi": "dimensionless",
            "step": "dimensionless",
            "S": "person",
            "I": "person",
            "total": "person",
        },
        PERSON_RATE,
        id="seasonal-cos",
    ),
    pytest.param(
        "base_rate * log(1 + I / scale) * S * I / total",
        {
            "base_rate": "1/day",
            "I": "person",
            "scale": "person",
            "S": "person",
            "total": "person",
        },
        PERSON_RATE,
        id="log",
    ),
    pytest.param(
        "base_rate * sqrt(I / threshold) * S",
        {"base_rate": "1/day", "I": "person", "threshold": "person", "S": "person"},
        PERSON_RATE,
        id="sqrt",
    ),
    pytest.param(
        "base_rate * pow(I / threshold, exponent) * S",
        {
            "base_rate": "1/day",
            "I": "person",
            "threshold": "person",
            "exponent": "dimensionless",
            "S": "person",
        },
        PERSON_RATE,
        id="pow",
    ),
    pytest.param(
        "max(beta_min, min(beta_max, beta_max * (1 - I / threshold))) * S",
        {
            "beta_min": "1/day",
            "beta_max": "1/day",
            "I": "person",
            "threshold": "person",
            "S": "person",
        },
        PERSON_RATE,
        id="min-max",
    ),
    pytest.param(
        "abs(rate) * I", {"rate": "1/day", "I": "person"}, PERSON_RATE, id="abs"
    ),
    pytest.param(
        "base * exp(-decay * step) * (1 + amp1 * sin(2 * pi * step / 365) "
        "+ amp2 * cos(4 * pi * step / 365)) * S * I / total",
        {
            "base": "1/day",
            "amp1": "dimensionless",
            "amp2": "dimensionless",
            "decay": "dimensionless",
            "step": "dimensionless",
            "pi": "dimensionless",
            "S": "person",
            "I": "person",
            "total": "person",
        },
        PERSON_RATE,
        id="combined-functions",
    ),
]

INCONSISTENT = [
    pytest.param(
        "beta * S",
        {"beta": "1/day", "S": "person"},
        "person",
        "Unit mismatch",
        id="mismatch",
    ),
    pytest.param(
        "rate * I",
        {"rate": "1/week", "I": "person"},
        PERSON_RATE,
        "Unit mismatch",
        id="convertible-time-units-are-not-equal",
    ),
    pytest.param(
        "beta * (1 + amp * sin(2 * pi * step / 365)) * S * I / total",
        {
            "beta": "1/day",
            "amp": "dimensionless",
            "step": "dimensionless",
            "pi": "dimensionless",
            "S": "person",
            "I": "person",
            "total": "person",
        },
        "person",
        "Unit mismatch",
        id="result-of-a-function-has-the-wrong-unit",
    ),
    pytest.param(
        "beta * sin(I) * S",
        {"beta": "1/day", "I": "person", "S": "person"},
        PERSON_RATE,
        "Cannot convert|Error",
        id="dimensional-sin",
    ),
    pytest.param(
        "base_rate * log(I) * S",
        {"base_rate": "1/day", "I": "person", "S": "person"},
        PERSON_RATE,
        None,
        id="dimensional-log",
    ),
    pytest.param(
        "beta * exp(time) * S * I / total",
        {
            "beta": "1/day",
            "time": "day",
            "S": "person",
            "I": "person",
            "total": "person",
        },
        PERSON_RATE,
        "Cannot convert|Error",
        id="dimensional-exp",
    ),
    pytest.param(
        "rate * sqrt(area)",
        {"rate": "1/day", "area": "person**2"},
        PERSON_RATE,
        "Cannot convert|Error",
        id="dimensional-sqrt",
    ),
    pytest.param(
        "rate * atan2(y, x) * S",
        {"rate": "1/day", "y": "person", "x": "day", "S": "person"},
        PERSON_RATE,
        None,
        id="atan2-of-different-units",
    ),
    pytest.param(
        "rate * pow(base, exponent) * S",
        {
            "rate": "1/day",
            "base": "dimensionless",
            "exponent": "person",
            "S": "person",
        },
        PERSON_RATE,
        None,
        id="dimensional-exponent",
    ),
    pytest.param(
        "rate * exp(sin(x)) * S",
        {"rate": "1/day", "x": "person", "S": "person"},
        PERSON_RATE,
        None,
        id="nested-functions",
    ),
    pytest.param(
        "max(rate1, rate2) * S",
        {"rate1": "1/day", "rate2": "person", "S": "person"},
        PERSON_RATE,
        None,
        id="max-of-different-units",
    ),
    pytest.param(
        "rate * floor(S)",
        {"rate": "1/day", "S": "person"},
        PERSON_RATE,
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
        units = {"rate": f"1/{time_unit}", "I": "person"}

        assert check_equation_units("rate * I", units, f"person/{time_unit}") == (
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
                Parameter(id="beta", value=0.5, unit="1/day"),
                Parameter(id="gamma", value=0.1, unit="1/day"),
            ]
        )
        assert not check_all_parameters_have_units(
            [
                Parameter(id="beta", value=0.5, unit="1/day"),
                Parameter(id="gamma", value=0.1, unit=None),
            ]
        )

    def test_the_total_and_the_subgroup_totals_have_the_bin_unit(self) -> None:
        units = get_predefined_variable_units(
            ["A", "A_young", "total_old", "total_young", "total_young_urban"],
            bin_unit="person",
        )

        assert units == {
            "total": "person",
            "total_old": "person",
            "total_young": "person",
            "total_young_urban": "person",
        }

    def test_without_a_bin_unit_there_are_no_units(self) -> None:
        assert get_predefined_variable_units(["total_young"], bin_unit=None) == {}
