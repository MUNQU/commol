from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from epimodel.constants import LogicOperators, ModelTypes, VariablePrefixes
from epimodel.utils.security import validate_expression_security


PREFIX_SEPARATOR: str = ":"
VALID_PREFIXES = [el.value for el in VariablePrefixes]


class Rule(BaseModel):
    """
    A simple logical rule that compares a variable with a value.

    Attributes
    ----------
    variable : str
        The variable to evaluate. Must follow the format '<prefix>:<variable_id>'. The
        allowed prefixes are: ['states', 'strati'].
    operator : Literal["eq", "neq", "gt", "get", "lt", "let"]
        The comparison operator.
    value : Union[str, int, float, bool]
        The value to which the variable is compared.
    """

    variable: str = Field(
        ...,
        description=(
            "Variable to evaluate. Must follow the format '<prefix>:<variable_id>'. "
            "The allowed prefixes are: ['states', 'strati']."
        ),
    )
    operator: Literal[
        LogicOperators.EQ,
        LogicOperators.NEQ,
        LogicOperators.GT,
        LogicOperators.GET,
        LogicOperators.LT,
        LogicOperators.LET,
    ] = Field(..., description="Comparison operator.")
    value: str | int | float | bool = Field(
        ..., description="Value to which the variable is compared."
    )

    @model_validator(mode="after")
    def validate_variable_predecessor(self) -> Self:
        """
        Enforces that the 'variable' field is correctly formatted as
        '<prefix>:<id>' and uses an allowed prefix.
        """
        parts = self.variable.split(PREFIX_SEPARATOR, 1)

        if len(parts) != 2:
            raise ValueError(
                (
                    f"Variable '{self.variable}' must contain exactly one "
                    f"'{PREFIX_SEPARATOR}' to separate the predecessor and the id."
                )
            )

        predecessor, var_id = parts

        if predecessor not in VALID_PREFIXES:
            raise ValueError(
                (
                    f"Variable predecessor must be one of {VALID_PREFIXES}. "
                    f"Found '{predecessor}' in variable '{self.variable}'."
                )
            )

        if not var_id:
            raise ValueError(
                f"Variable id must not be empty. Found empty id in '{self.variable}'."
            )

        return self


class Condition(BaseModel):
    """
    Defines a set of logical restrictions for a transition.

    Attributes
    ----------
    logic : Literal["and", "or"]
        How to combine the rules.
    rules : List[Rule]
        A list of rules that make up the condition.
    """

    logic: Literal[LogicOperators.AND, LogicOperators.OR] = Field(
        ..., description="How to combine the rules. Allowed operators: ['and', 'or']"
    )
    rules: list[Rule]


class Transition(BaseModel):
    """
    Defines a rule for system evolution.

    Attributes
    ----------
    id : str
        Id of the transition.
    source : list[str]
        The origin compartments.
    target : list[str]
        The destination compartments.
    rate : str | None
        Mathematical formula, parameter name, or constant value for the flow between
        compartments. Numeric values are automatically converted to strings during
        validation.

        Operators: +, -, *, /, % (modulo), ^ or ** (power)
        Functions: sin, cos, tan, exp, ln, sqrt, abs, min, max, if, etc.
        Constants: pi, e

        Note: Both ^ and ** are supported for exponentiation (** is converted to ^).

        Examples:
        - "beta" (parameter reference)
        - "0.5" (constant, can also be passed as float 0.5)
        - "beta * S * I / N" (mathematical formula)
        - "0.3 * sin(2 * pi * t / 365)" (time-dependent formula)
        - "2^10" or "2**10" (power: both syntaxes work)
    condition : Condition | None
        Logical restrictions for the transition.
    """

    id: str = Field(..., description="Id of the transition.")
    source: list[str] = Field(..., description="Origin compartments.")
    target: list[str] = Field(..., description="Destination compartments.")

    rate: str | None = Field(
        None,
        description=(
            "Mathematical formula, parameter name, or constant value for the flow. "
            "Can be a parameter reference (e.g., 'beta'), a constant (e.g., '0.5'), "
            "or a mathematical expression (e.g., 'beta * S * I / N'). "
            "Numeric values are automatically converted to strings during validation."
        ),
    )

    condition: Condition | None = Field(
        None, description="Logical restrictions for the transition."
    )

    @field_validator("rate", mode="before")
    @classmethod
    def validate_rate(cls, value: str | None) -> str | None:
        """Convert numeric rates to strings and perform security validation."""
        if value is None:
            return value
        try:
            validate_expression_security(value)
        except ValueError as e:
            raise ValueError(f"Security validation failed for rate '{value}': {e}")
        return value


class Dynamics(BaseModel):
    """
    Defines how the system evolves.

    Attributes
    ----------
    typology : Literal["DifferenceEquations"]
        The type of model.
    transitions : List[Transition]
        A list of rules for state changes.
    """

    typology: Literal[ModelTypes.DIFFERENCE_EQUATIONS]
    transitions: list[Transition]
