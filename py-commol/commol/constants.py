"""Enumerations and constants shared across the package."""

from enum import StrEnum, unique

TOTAL_VARIABLE = "total"
"""The sum of every compartment, accumulators excluded, in every expression."""

RESERVED_VARIABLE_NAMES: tuple[str, ...] = (TOTAL_VARIABLE, "step", "t", "pi", "e")
"""Variables the engine provides in every expression."""

SUBGROUP_TOTAL_PREFIX = f"{TOTAL_VARIABLE}_"
"""Prefix of the subgroup totals ``total_<cat>…`` the engine provides."""


@unique
class PrintEquationsOutputFormat(StrEnum):
    """Output formats of `Model.print_equations`."""

    TEXT = "text"
    LATEX = "latex"
