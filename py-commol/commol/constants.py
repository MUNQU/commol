"""Enumerations shared across the package."""

from enum import StrEnum, unique


@unique
class LogicOperators(StrEnum):
    """Logic and comparison operators of a transition `Condition`."""

    AND = "and"
    OR = "or"
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GET = "get"
    LT = "lt"
    LET = "let"


@unique
class ModelTypes(StrEnum):
    """Model typologies. Only `DIFFERENCE_EQUATIONS` can be simulated."""

    DIFFERENCE_EQUATIONS = "DifferenceEquations"
    DIFFERENTIAL_EQUATIONS = "DifferentialEquations"


@unique
class VariablePrefixes(StrEnum):
    """Prefixes of a `Rule` variable: `state:<bin>` or `strat:<stratification>`."""

    STATE = "state"
    STRAT = "strat"


@unique
class PrintEquationsOutputFormat(StrEnum):
    """Output formats of `Model.print_equations`."""

    TEXT = "text"
    LATEX = "latex"
