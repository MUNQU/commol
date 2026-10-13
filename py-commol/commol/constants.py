"""Enumerations and constants shared across the package."""

from enum import StrEnum, unique

RESERVED_VARIABLE_NAMES: tuple[str, ...] = ("N", "step", "t", "pi", "e")
"""Variables the engine provides in every expression."""


@unique
class ModelTypes(StrEnum):
    """Model typologies. Only `DIFFERENCE_EQUATIONS` can be simulated."""

    DIFFERENCE_EQUATIONS = "DifferenceEquations"
    DIFFERENTIAL_EQUATIONS = "DifferentialEquations"


@unique
class PrintEquationsOutputFormat(StrEnum):
    """Output formats of `Model.print_equations`."""

    TEXT = "text"
    LATEX = "latex"
