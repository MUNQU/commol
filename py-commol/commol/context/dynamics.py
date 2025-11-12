from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from commol.commol_rs import commol_rs
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e
from commol.constants import LogicOperators, ModelTypes, VariablePrefixes
from commol.utils.security import validate_expression_security

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
        default=...,
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
    ] = Field(default=..., description="Comparison operator.")
    value: str | int | float | bool = Field(
        default=..., description="Value to which the variable is compared."
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
        default=...,
        description="How to combine the rules. Allowed operators: ['and', 'or']",
    )
    rules: list[Rule]


class StratificationCondition(BaseModel):
    """
    Specifies a category within a stratification for rate matching.

    Attributes
    ----------
    stratification : str
        The ID of the stratification (e.g., "age", "location")
    category : str
        The category within that stratification (e.g., "young", "urban")
    """

    stratification: str = Field(default=..., description="ID of the stratification")
    category: str = Field(default=..., description="Category within the stratification")


class StratifiedRate(BaseModel):
    """
    Defines a rate for specific stratification categories.

    Attributes
    ----------
    conditions : list[StratificationCondition]
        List of stratification-category pairs that must match
    rate : str
        Rate expression for compartments matching these conditions
    """

    conditions: list[StratificationCondition] = Field(
        default=..., description="Stratification conditions that must match"
    )
    rate: str = Field(
        default=..., description="Rate expression for matching compartments"
    )

    @field_validator("conditions")
    @classmethod
    def validate_not_empty(
        cls, v: list[StratificationCondition]
    ) -> list[StratificationCondition]:
        """Ensure at least one stratification condition is specified."""
        if not v:
            raise ValueError(
                "conditions must contain at least one stratification condition"
            )
        return v

    @model_validator(mode="after")
    def validate_unique_stratifications(self) -> Self:
        """Ensure no duplicate stratification IDs."""
        strat_ids = [sc.stratification for sc in self.conditions]
        if len(strat_ids) != len(set(strat_ids)):
            duplicates = [s for s in strat_ids if strat_ids.count(s) > 1]
            raise ValueError(
                (
                    f"Duplicate stratifications found: {list(set(duplicates))}. "
                    f"Each stratification can only be specified once per rate."
                )
            )
        return self

    @field_validator("rate", mode="before")
    @classmethod
    def validate_rate(cls, value: str) -> str:
        """Perform security and syntax validation on the rate expression."""
        try:
            validate_expression_security(value)
            commol_rs.core.MathExpression(value).validate()
        except ValueError as e:
            raise ValueError(f"Validation failed for rate '{value}': {e}")
        return value


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
        Default mathematical formula, parameter name, or constant value for the flow.
        Used when no stratified rate matches. Numeric values are automatically
        converted to strings during validation.

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
    stratified_rates : list[StratifiedRate] | None
        Stratification-specific rates. Each rate applies to compartments that match
        all specified stratification conditions.
    condition : Condition | None
        Logical restrictions for the transition.
    """

    id: str = Field(default=..., description="Id of the transition.")
    source: list[str] = Field(default=..., description="Origin compartments.")
    target: list[str] = Field(default=..., description="Destination compartments.")

    rate: str | None = Field(
        None,
        description=(
            "Default rate expression (fallback when no stratified rate matches). "
            "Can be a parameter reference (e.g., 'beta'), a constant (e.g., '0.5'), "
            "or a mathematical expression (e.g., 'beta * S * I / N'). "
            "Numeric values are automatically converted to strings during validation."
        ),
    )

    stratified_rates: list[StratifiedRate] | None = Field(
        default=None, description="List of stratification-specific rates"
    )

    condition: Condition | None = Field(
        default=None, description="Logical restrictions for the transition."
    )

    @field_validator("rate", mode="before")
    @classmethod
    def validate_rate(cls, value: str | None) -> str | None:
        """
        Convert numeric rates to strings and perform security and syntax validation.
        """
        if value is None:
            return value
        try:
            validate_expression_security(value)
            commol_rs.core.MathExpression(value).validate()
        except ValueError as e:
            raise ValueError(f"Validation failed for rate '{value}': {e}")
        return value


class Dynamics(BaseModel):
    """
    Defines how the system evolves.

    Attributes
    ----------
    typology : Literal["DifferenceEquations", "DifferentialEquations"]
        The type of model.
    transitions : List[Transition]
        A list of rules for state changes.
    """

    typology: Literal[
        ModelTypes.DIFFERENCE_EQUATIONS, ModelTypes.DIFFERENTIAL_EQUATIONS
    ]
    transitions: list[Transition]

    @field_validator("transitions")
    @classmethod
    def validate_transitions_not_empty(cls, v: list[Transition]) -> list[Transition]:
        if not v:
            raise ValueError("At least one transition must be defined.")
        return v

    @model_validator(mode="after")
    def validate_unique_transition_ids(self) -> Self:
        """
        Validates that transition IDs are unique.
        """
        transition_ids = [t.id for t in self.transitions]
        if len(transition_ids) != len(set(transition_ids)):
            duplicates = [
                item for item in set(transition_ids) if transition_ids.count(item) > 1
            ]
            raise ValueError(f"Duplicate transition IDs found: {duplicates}")
        return self
