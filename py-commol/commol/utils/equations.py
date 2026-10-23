"""Dimensional analysis of model equations with pint."""

import ast
import math
import re
from collections.abc import Callable, Iterable

import pint
from pint.facets.plain import PlainQuantity

from commol.commol_rs import _commol_rs as commol_rs
from commol.context.parameter import Parameter
from commol.constants import SUBGROUP_TOTAL_PREFIX, TOTAL_VARIABLE

DEFAULT_STEP_UNIT = "step"
"""Step unit of a model that declares none: the default increment of any axis."""

ureg: pint.UnitRegistry[float] = pint.UnitRegistry()
ureg.define(f"{DEFAULT_STEP_UNIT} = [{DEFAULT_STEP_UNIT}]")


class UnitConsistencyError(Exception):
    """Raised when unit consistency check fails."""

    pass


def define_units(expression: str) -> None:
    """
    Define every unit name in `expression` that `pint` does not know.

    A unit means what `pint` defines. A name that `pint` does not define
    (``item``) becomes the one unit of its own dimension
    (``[item_dimension]``), for time as for any other quantity.

    Parameters
    ----------
    expression : str
        A unit expression, such as ``"item/day"`` or ``"1/km**2"``.
    """
    for part in re.split(r"[*/\s()]+", expression):
        part = part.strip()
        if not part or part.replace(".", "").replace("-", "").isdigit():
            continue
        try:
            ureg(part)
        except pint.UndefinedUnitError:
            ureg.define(f"{part} = [{part}_dimension]")


def check_equation_units(
    equation: str,
    variable_units: dict[str, str],
    expected_unit: str,
) -> tuple[bool, str | None]:
    """
    Check if an equation is unit-consistent.

    Parameters
    ----------
    equation : str
        The mathematical equation to check (e.g., "k1 * A * B / total").
    variable_units : dict[str, str]
        Mapping of variable names to their units
        (e.g., {"k1": "1/day", "A": "item"}).
    expected_unit : str
        The expected unit of the equation result (e.g., "item/day").

    Returns
    -------
    tuple[bool, str | None]
        A tuple of (is_consistent, error_message).
        If consistent, returns (True, None).
        If inconsistent, returns (False, error_message).

    Notes
    -----
    The unit must be equal to ``expected_unit``, not only convertible to it:
    ``item/week`` does not match ``item/day``, because the engine never
    converts values between units.
    """
    try:
        # Parse the equation to build a pint Quantity
        equation_unit = parse_equation_unit(equation, variable_units)

        if not same_unit(str(equation_unit.units), expected_unit):
            return (
                False,
                (
                    f"Unit mismatch: equation '{equation}' has unit "
                    f"'{equation_unit.units}' but expected '{expected_unit}'"
                ),
            )

        return (True, None)

    except (pint.UndefinedUnitError, pint.DimensionalityError) as e:
        return (False, f"Unit error in equation '{equation}': {str(e)}")
    except Exception as e:
        return (False, f"Error checking units for equation '{equation}': {str(e)}")


def same_unit(unit: str, other: str) -> bool:
    """
    Whether two units are equal: convertible, with a conversion factor of 1.

    Parameters
    ----------
    unit, other : str
        Unit expressions, such as ``"1/day"`` and ``"1/d"``.

    Returns
    -------
    bool
        True for ``"1/day"`` and ``"1/d"``; False for ``"1/day"`` and
        ``"1/week"``, which are convertible but not equal.
    """
    quantity = ureg.Quantity(1.0, ureg.parse_units(unit))
    target = ureg.parse_units(other)
    if not quantity.is_compatible_with(target):
        return False
    return math.isclose(float(quantity.to(target).magnitude), 1.0, rel_tol=1e-9)


class _Literal(float):
    """A number written in an equation, whose unit the check may read."""


type Operand = PlainQuantity | float
"""What the check reads in place of a value: a quantity with its unit (a
`pint.Quantity`, or the `PlainQuantity` that pint's arithmetic gives), or a
number, written (`_Literal`) or computed."""


def _match_units(*operands: Operand) -> list[Operand]:
    """
    Give each written number the unit of the first quantity among `operands`.

    The engine reads a number added to, subtracted from or compared with a
    value in the unit of that value: ``t >= 10`` with ``t`` in days compares
    with 10 days. The check reads it the same way. A number that a function
    or an operation computed stays dimensionless.

    Raises
    ------
    pint.DimensionalityError
        If two quantities among `operands` have different dimensions.
    """
    quantities = [operand for operand in operands if isinstance(operand, pint.Quantity)]
    if not quantities:
        return list(operands)
    units = quantities[0].units
    matched: list[Operand] = []
    for operand in operands:
        if isinstance(operand, _Literal):
            operand = ureg.Quantity(float(operand), units)
        if not isinstance(operand, pint.Quantity):
            operand = ureg.Quantity(float(operand), ureg.dimensionless)
        if not operand.is_compatible_with(units):
            raise pint.DimensionalityError(operand.units, units)
        matched.append(operand)
    return matched


def _add(left: Operand, right: Operand) -> Operand:
    a, b = _match_units(left, right)
    return a + b


def _subtract(left: Operand, right: Operand) -> Operand:
    a, b = _match_units(left, right)
    return a - b


def _compare(*operands: Operand) -> float:
    """A comparison: its operands share a unit, and its value is a number."""
    _match_units(*operands)
    return 1.0


def _if(_condition: Operand, then: Operand, otherwise: Operand) -> Operand:
    """``if(condition, then, otherwise)``: both branches share a unit."""
    return _match_units(then, otherwise)[0]


def _min(*operands: Operand) -> Operand:
    return _match_units(*operands)[0]


def _cbrt(operand: Operand) -> Operand:
    """``cbrt(x)``: the cube root of a quantity is the cube root of its unit."""
    if isinstance(operand, pint.Quantity):
        return operand ** (1 / 3)
    return math.cbrt(operand)


_FUNCTION_UNITS: dict[str, Callable[..., Operand]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sqrt": math.sqrt,
    "cbrt": _cbrt,
    "pow": math.pow,
    "hypot": math.hypot,
    "min": _min,
    "max": _min,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
}
"""The rule of each function: a function of `math` takes and gives numbers
without a unit; ``abs`` and ``round`` keep the unit of their argument,
``cbrt`` takes its cube root, and ``min`` and ``max`` need one unit."""

_FUNCTIONS: dict[str, Callable[..., Operand]] = {
    name: _FUNCTION_UNITS[name] for name in commol_rs.core.FUNCTIONS
}
"""The rule of each function of the engine (``commol_rs.core.FUNCTIONS``)."""


class _ReadWrittenNumbers(ast.NodeTransformer):
    """Mark written numbers, and route sums, differences and comparisons."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return _call("_literal", [node])
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(
            node.operand, ast.Constant
        ):
            return _call("_literal", [node])
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Add):
            return _call("_add", [node.left, node.right])
        if isinstance(node.op, ast.Sub):
            return _call("_subtract", [node.left, node.right])
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        return _call("_compare", [node.left, *node.comparators])


def _call(name: str, arguments: list) -> ast.Call:
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=arguments, keywords=[])


_IF_CALL = re.compile(r"\bif\s*\(")


def parse_equation_unit(equation: str, variable_units: dict[str, str]) -> pint.Quantity:
    """
    Parse an equation and compute its resulting unit.

    The equation is evaluated with every variable as a `pint` quantity of
    value 1 in its unit. ``^`` and ``**`` are a power. A number written in the
    equation and added to, subtracted from or compared with a quantity is read
    in that quantity's unit, as the engine reads it; the two branches of
    ``if(c, a, b)`` and the arguments of ``min`` and ``max`` share one unit.
    ``cbrt`` takes the cube root of a unit, ``abs`` and ``round`` keep it,
    and the other functions take and give numbers without a unit.

    Parameters
    ----------
    equation : str
        The equation to parse.
    variable_units : dict[str, str]
        Mapping of variable names to their units.

    Returns
    -------
    pint.Quantity
        A pint Quantity representing the unit of the equation.
    """
    namespace: dict[str, Operand | Callable[..., Operand]] = {}

    for var, unit_str in variable_units.items():
        namespace[var] = ureg(unit_str)

    namespace.update(_FUNCTIONS)
    namespace.update(
        {
            "pi": math.pi,
            "e": math.e,
            "_literal": _Literal,
            "_add": _add,
            "_subtract": _subtract,
            "_compare": _compare,
            "_if": _if,
        }
    )

    try:
        python = _IF_CALL.sub("_if(", equation).replace("^", "**")
        tree = ast.parse(python, mode="eval")
        tree = ast.fix_missing_locations(_ReadWrittenNumbers().visit(tree))
        eval_result: Operand = eval(
            compile(tree, "<equation>", "eval"), {"__builtins__": {}}, namespace
        )
    except Exception as e:
        raise ValueError(f"Failed to evaluate equation '{equation}': {str(e)}")

    if isinstance(eval_result, pint.Quantity):
        return eval_result
    if isinstance(eval_result, (int, float)):
        return ureg.Quantity(float(eval_result), ureg.dimensionless)
    raise ValueError(
        f"Unexpected result type {type(eval_result)} from equation '{equation}'"
    )


def get_predefined_variable_units(
    subgroup_totals: Iterable[str],
    bin_unit: str | None = None,
) -> dict[str, str]:
    """
    Get units for the predefined variables ``total`` and ``total_<cat>…``.

    Parameters
    ----------
    subgroup_totals : Iterable[str]
        Names of the subgroup totals of the model, as the engine provides them
        (``engine_state.subgroup_total_names``). Names without the
        ``total_`` prefix, such as bin totals, are skipped: they take the unit
        of their bin.
    bin_unit : str | None
        The unit for bins. If None, predefined variables will not be added to the
        units mapping.

    Returns
    -------
    dict[str, str]
        Mapping of predefined variable names to their units.
    """
    if bin_unit is None:
        return {}
    names = [name for name in subgroup_totals if name.startswith(SUBGROUP_TOTAL_PREFIX)]
    return {name: bin_unit for name in (TOTAL_VARIABLE, *names)}


def check_all_parameters_have_units(parameters: list[Parameter]) -> bool:
    """
    Check if all parameters have units specified.

    Parameters
    ----------
    parameters : list
        List of Parameter objects.

    Returns
    -------
    bool
        True if all parameters have units, False otherwise.
    """
    return all(param.unit is not None for param in parameters)
