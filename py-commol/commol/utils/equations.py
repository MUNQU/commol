"""Dimensional analysis of model equations with pint."""

import math
from collections.abc import Callable, Iterable
from typing import cast

import pint

from commol.context.parameter import Parameter
from commol.constants import SUBGROUP_TOTAL_PREFIX, TOTAL_VARIABLE

ureg: pint.UnitRegistry[float] = pint.UnitRegistry()
ureg.define("step = [step]")


class UnitConsistencyError(Exception):
    """Raised when unit consistency check fails."""

    pass


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
        The mathematical equation to check (e.g., "beta * S * I / total").
    variable_units : dict[str, str]
        Mapping of variable names to their units
        (e.g., {"beta": "1/day", "S": "person"}).
    expected_unit : str
        The expected unit of the equation result (e.g., "person/day").

    Returns
    -------
    tuple[bool, str | None]
        A tuple of (is_consistent, error_message).
        If consistent, returns (True, None).
        If inconsistent, returns (False, error_message).

    Notes
    -----
    The unit must be equal to ``expected_unit``, not only convertible to it:
    ``person/week`` does not match ``person/day``, because the engine never
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


def parse_equation_unit(equation: str, variable_units: dict[str, str]) -> pint.Quantity:
    """
    Parse an equation and compute its resulting unit.

    This function evaluates the equation symbolically using pint quantities
    to determine the final unit.

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
    # Create a namespace with all variables as pint quantities with value 1
    # The namespace also includes math functions which are callables
    namespace: dict[
        str,
        pint.Quantity[float]
        | float
        | Callable[[float], float]
        | Callable[[float, float], float]
        | Callable[..., float]
        | type[int]
        | type[float],
    ] = {}

    # Add variables with their units
    for var, unit_str in variable_units.items():
        unit_obj = ureg(unit_str)
        if isinstance(unit_obj, pint.Quantity):
            namespace[var] = unit_obj
        else:
            # Handle cases where ureg() returns Unit instead of Quantity
            namespace[var] = 1.0 * unit_obj

    # Add mathematical constants (dimensionless)
    namespace["pi"] = math.pi
    namespace["e"] = math.e

    # Add mathematical functions
    # Pint automatically handles dimensionless quantities for these functions
    # by extracting magnitudes when needed. Functions return plain floats which
    # are then treated as dimensionless quantities.

    # Trigonometric functions
    namespace["sin"] = math.sin
    namespace["cos"] = math.cos
    namespace["tan"] = math.tan
    namespace["asin"] = math.asin
    namespace["acos"] = math.acos
    namespace["atan"] = math.atan
    namespace["atan2"] = math.atan2
    namespace["sinh"] = math.sinh
    namespace["cosh"] = math.cosh
    namespace["tanh"] = math.tanh
    namespace["asinh"] = math.asinh
    namespace["acosh"] = math.acosh
    namespace["atanh"] = math.atanh

    # Exponential and logarithmic functions
    namespace["exp"] = math.exp
    namespace["log"] = math.log
    namespace["log10"] = math.log10
    namespace["log2"] = math.log2
    namespace["ln"] = math.log  # alias for natural log

    # Power and root functions
    namespace["sqrt"] = math.sqrt
    namespace["pow"] = math.pow
    namespace["hypot"] = math.hypot

    # Rounding and absolute value functions
    namespace["abs"] = abs
    namespace["floor"] = math.floor
    namespace["ceil"] = math.ceil
    namespace["round"] = round

    # Min/max functions
    namespace["min"] = min
    namespace["max"] = max

    # Evaluate the equation
    try:
        eval_result: object = cast(
            object, eval(equation, {"__builtins__": {}}, namespace)
        )

        if isinstance(eval_result, pint.Quantity):
            return eval_result
        elif isinstance(eval_result, (int, float)):
            # If result is a number, it's dimensionless
            numeric_value = float(eval_result)
            unit_result = ureg.dimensionless

            if isinstance(unit_result, pint.Unit):
                # Multiply numeric value by Unit to get Quantity
                quantity_result: object = cast(object, numeric_value * unit_result)
                # The result should be a Quantity
                if isinstance(quantity_result, pint.Quantity):
                    return quantity_result
                else:
                    raise ValueError(
                        (
                            f"Multiplying float by Unit did not produce "
                            f"Quantity: {type(quantity_result)}"
                        )
                    )
            else:
                # ureg.dimensionless might be a Quantity itself
                quantity_result = cast(object, numeric_value * unit_result)
                if isinstance(quantity_result, pint.Quantity):
                    return quantity_result
                else:
                    raise ValueError(
                        (
                            f"Multiplying float by dimensionless did not "
                            f"produce Quantity: {type(quantity_result)}"
                        )
                    )
        else:
            raise ValueError(
                f"Unexpected result type {type(eval_result)} from equation '{equation}'"
            )
    except Exception as e:
        raise ValueError(f"Failed to evaluate equation '{equation}': {str(e)}")


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
