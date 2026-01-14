from enum import StrEnum, unique


@unique
class LogicOperators(StrEnum):
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
    DIFFERENCE_EQUATIONS = "DifferenceEquations"
    DIFFERENTIAL_EQUATIONS = "DifferentialEquations"


@unique
class VariablePrefixes(StrEnum):
    STATE = "state"
    STRAT = "strat"


@unique
class PrintEquationsOutputFormat(StrEnum):
    TEXT = "text"
    LATEX = "latex"
