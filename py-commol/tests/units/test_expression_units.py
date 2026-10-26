"""The unit check reads the language of formulas: its power operators and
each function of the engine."""

import pytest

from commol.commol_rs import _commol_rs as commol_rs
from commol.utils import equations
from commol.utils.equations import define_units, parse_equation_unit, same_unit

UNITS = {"k1": "1/(item*day)", "A": "item", "total": "item"}


def _unit(equation: str) -> str:
    define_units("item")
    return str(parse_equation_unit(equation, UNITS).units)


def test_a_caret_is_a_power() -> None:
    assert same_unit(_unit("k1 * A ^ 2 / total"), "1/day")
    assert _unit("k1 * A ^ 2 / total") == _unit("k1 * A ** 2 / total")


def test_cbrt_takes_the_cube_root_of_a_unit() -> None:
    assert same_unit(_unit("cbrt(k1 ** 3) * A"), _unit("k1 * A"))
    assert parse_equation_unit("cbrt(8)", {}).magnitude == 2.0


def test_the_unit_check_knows_every_function_of_the_engine() -> None:
    rules = equations._FUNCTION_UNITS  # pyright: ignore[reportPrivateUsage]

    assert set(rules) == set(commol_rs.core.FUNCTIONS)


def test_the_unit_follows_the_grouping_of_the_parser() -> None:
    """The unit walks the tree of the engine: a chain of powers groups from
    the right, and ``!`` and a name that Python reserves, such as ``lambda``,
    which the grammar of Python does not read, have the meaning of the
    language."""
    units = {"A": "item", "lambda": "1/day", "t": "day"}

    def unit(equation: str) -> str:
        define_units("item")
        return str(parse_equation_unit(equation, units).units)

    assert same_unit(unit("A ^ 2 ^ -1"), "item ** 0.5")
    assert same_unit(unit("(A ^ 2) ^ -1"), "1 / item ** 2")
    assert same_unit(unit("lambda * A"), "item/day")
    assert same_unit(unit("if(!(t > 2), A, 0)"), "item")


def test_a_variable_without_a_unit_is_named() -> None:
    define_units("item")

    with pytest.raises(ValueError, match="the variable 'x' has no unit"):
        parse_equation_unit("k1 * A ^ x", UNITS)
