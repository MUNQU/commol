import math
from collections.abc import Callable
from typing import cast

import pint

from commol.context.parameter import Parameter
from commol.context.stratification import Stratification

ureg: pint.UnitRegistry[float] = pint.UnitRegistry()


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
        The mathematical equation to check (e.g., "beta * S * I / N").
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
    """
    try:
        # Parse the equation to build a pint Quantity
        equation_unit = parse_equation_unit(equation, variable_units)

        # Check if result matches expected unit
        expected_quantity = ureg(expected_unit)

        if not equation_unit.is_compatible_with(expected_quantity):
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
    stratifications: list[Stratification],
    bin_unit: str | None = None,
) -> dict[str, str]:
    """
    Get units for predefined variables like N, N_young, etc.

    Parameters
    ----------
    stratifications : list
        List of stratification objects with categories.
    bin_unit : str | None
        The unit for bins. If None, predefined variables will not be added to the
        units mapping.

    Returns
    -------
    dict[str, str]
        Mapping of predefined variable names to their units.
    """
    predefined_units: dict[str, str] = {}

    # Only add predefined variables if bin_unit is specified
    if bin_unit is None:
        return predefined_units

    # Total population N
    predefined_units["N"] = bin_unit

    # Add N_{category} for each category in stratifications
    for strat in stratifications:
        for category in strat.categories:
            predefined_units[f"N_{category}"] = bin_unit

    # Add N_{category1}_{category2} for combinations
    # This handles cases like N_young_urban, etc.
    if len(stratifications) > 1:
        from itertools import combinations, product

        category_groups = [s.categories for s in stratifications]

        # All possible combinations of categories across different stratifications
        full_category_combos = product(*category_groups)

        for combo_tuple in full_category_combos:
            # For each combo, find all non-empty subsets
            for i in range(1, len(combo_tuple) + 1):
                for subset in combinations(combo_tuple, i):
                    var_name = f"N_{'_'.join(subset)}"
                    predefined_units[var_name] = bin_unit

    return predefined_units


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
