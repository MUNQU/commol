"""The unit check reads the language of formulas: its power operators and
each function of the engine."""

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
